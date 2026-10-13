#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <cstdio>
#include <algorithm>
#include <dlfcn.h>

// ============================================================
// 通用辅助：从 HOST 内存读取任意类型
// args[i] 就是 host 端指向第 i 个参数值的指针，直接 memcpy 即可
// ============================================================
template<typename T>
static inline T read_arg(void* arg_ptr) {
    T v;
    std::memcpy(&v, arg_ptr, sizeof(T));
    return v;
}

template<typename T>
static inline T field_at(const void* base, size_t offset) {
    T v;
    std::memcpy(&v, reinterpret_cast<const uint8_t*>(base) + offset, sizeof(T));
    return v;
}

// ============================================================
// IntDivider<uint32_t> 实际大小由 IntegerDivider.cuh 决定
// 已验证版本（PyTorch 2.x / CUDA 12.x）：
//   struct IntDivider { uint32_t divisor; uint32_t m1; uint32_t shift; uint32_t sign; }
//   sizeof = 16B
// ============================================================
static constexpr size_t kIntDividerSize = 16;
static constexpr int    kMaxDims        = 25;

// OffsetCalculator<NARGS, uint32_t> 大小（unsigned strides）：
//   4(dims) + kMaxDims*kIntDividerSize(sizes) + kMaxDims*NARGS*4(strides)
static constexpr size_t kOC1_size = 4 + kMaxDims * kIntDividerSize + kMaxDims * 1 * 4; // 4+400+100=504
static constexpr size_t kOC2_size = 4 + kMaxDims * kIntDividerSize + kMaxDims * 2 * 4; // 4+400+200=604

// 注意：如果 IntDivider 实际为 12B，则 kOC1_size=404, kOC2_size=504
// 程序启动时通过 verifyReduceOpLayout() 确认

// ReduceOp<float, MeanOps<f,f,f,f>, uint32_t, float, 4> 精确偏移
// MeanOps 唯一成员：float factor
// 成员按源码顺序排列（ops, ident, config, input_calc, output_calc, src, dst, ...）
struct ReduceOpOffsets {
    size_t factor;        // ops.factor
    size_t ident;
    size_t num_inputs;    // config 内部
    size_t num_outputs;
    size_t block_width;
    size_t block_height;
    size_t num_threads;
    size_t vectorize_input;
    size_t output_vec_size;
    size_t ctas_per_output;
    size_t input_mult0;   // !=0 → block_x_reduce
    size_t input_mult1;   // !=0 → block_y_reduce
    size_t input_mult2;   // !=0 → global_reduce
    size_t input_dims;
    size_t output_dims;
    size_t src;
    size_t dst0;
    size_t dst1;
    size_t acc_buf;
    size_t semaphores;
    size_t base_idx;
    size_t accumulate;
    size_t final_output;
    size_t noutputs;
};

// 根据 IntDivider 实际大小计算
static ReduceOpOffsets computeReduceOpOffsets(size_t intdiv_size) {
    ReduceOpOffsets o;
    const size_t oc1 = 4 + kMaxDims * intdiv_size + kMaxDims * 1 * 4;
    const size_t oc2 = 4 + kMaxDims * intdiv_size + kMaxDims * 2 * 4;

    // ReduceConfig 成员（从 Reduce.cuh 源码逐一计算）
    const size_t cfg = 8;  // ops(4) + ident(4)
    o.factor        = 0;
    o.ident         = 4;
    o.num_inputs    = cfg + 4;   // element_size_bytes(4) | num_inputs
    o.num_outputs   = cfg + 8;
    o.ctas_per_output = cfg + 20;
    o.input_mult0   = cfg + 24;
    o.input_mult1   = cfg + 28;
    o.input_mult2   = cfg + 32;
    o.block_width   = cfg + 44;
    o.block_height  = cfg + 48;
    o.num_threads   = cfg + 52;
    o.vectorize_input = cfg + 56;
    o.output_vec_size = cfg + 60; // cfg+64 = end of ReduceConfig (64B)

    const size_t ic_off = cfg + 64;       // input_calc 起始
    const size_t oc_off = ic_off + oc1;   // output_calc 起始
    o.input_dims  = ic_off;
    o.output_dims = oc_off;

    // src 需要 8 字节对齐
    size_t after_oc = oc_off + oc2;
    size_t src_off  = (after_oc + 7) & ~size_t(7);
    o.src         = src_off;
    o.dst0        = src_off + 8;
    o.dst1        = src_off + 16;
    o.acc_buf     = src_off + 24;
    o.semaphores  = src_off + 40;
    o.base_idx    = src_off + 48;
    o.accumulate  = src_off + 56;
    o.final_output= src_off + 57;
    // 2B padding
    o.noutputs    = src_off + 60;
    return o;
}

static ReduceOpOffsets kReduceOpOffsets;
static bool            kReduceOpOffsetReady = false;

// 启动时调用一次，自动探测 IntDivider 实际大小
static void detectReduceOpLayout(void* arg0) {
    // isz=12 先尝试（PyTorch 2.x 实际使用 12B）
    for (size_t isz : {size_t(12), size_t(16)}) {
        auto o = computeReduceOpOffsets(isz);
        // ── 检查 1：factor × N ≈ 1.0 ──────────────────────────
        float factor = field_at<float>(arg0, o.factor);
        int   n_in   = field_at<int>(arg0, o.num_inputs);
        int   n_out  = field_at<int>(arg0, o.num_outputs);
        if (n_in  <= 0 || n_in  > 100000000) continue;
        if (n_out <= 0 || n_out > n_in)       continue;
        if (factor <= 0.0f || factor > 1.0f)  continue;
        if (fabsf(factor * (float)n_in - 1.0f) > 0.02f) continue;
        // ── 检查 2：input_dims 和 output_dims 必须在 [0, MAX_DIMS] ──
        int idim = field_at<int>(arg0, o.input_dims);
        int odim = field_at<int>(arg0, o.output_dims);
        if (idim < 0 || idim > kMaxDims) continue;  // ← 关键新增
        if (odim < 0 || odim > kMaxDims) continue;  // ← 关键新增
        // ── 检查 3：src 指针基本合理性 ───────────────────────
        uintptr_t src_val = field_at<uintptr_t>(arg0, o.src);
        if (src_val == 0)        continue;
        if (src_val < 0x10000UL) continue;
        kReduceOpOffsets     = o;
        kReduceOpOffsetReady = true;
        LOG_INFO(CUDART,
            "[detect] IntDivider=%zuB  src_offset=%zu  "
            "factor=%.6f  N=%d→%d  dims(in=%d out=%d)",
            isz, o.src, (double)factor, n_in, n_out, idim, odim);
        return;
    }
    // 回退：用 12B（PyTorch 2.x 默认）
    LOG_INFO(CUDART, "[detect] 回退到 IntDivider=12B");
    kReduceOpOffsets     = computeReduceOpOffsets(12);
    kReduceOpOffsetReady = true;
}

// ============================================================
// Rank 7: reduce_kernel + MeanOps<float,float,float,float>
// args[0] → ReduceOp<float, MeanOps, uint32_t, float, 4>（HOST端结构体）
// ============================================================
static std::string parseReduceMean(void** args) {
    if (!args || !args[0]) return "[reduce/MeanOps] null";
    void* p = args[0];

    if (!kReduceOpOffsetReady) detectReduceOpLayout(p);
    if (!kReduceOpOffsetReady) return "[reduce/MeanOps] layout unknown";

    const auto& o = kReduceOpOffsets;
    float   factor    = field_at<float>(p, o.factor);
    float   ident     = field_at<float>(p, o.ident);
    int     n_in      = field_at<int>(p, o.num_inputs);
    int     n_out     = field_at<int>(p, o.num_outputs);
    int     bw        = field_at<int>(p, o.block_width);
    int     bh        = field_at<int>(p, o.block_height);
    int     nt        = field_at<int>(p, o.num_threads);
    bool    vec       = field_at<bool>(p, o.vectorize_input);
    int     vsz       = field_at<int>(p, o.output_vec_size);
    int     ctas      = field_at<int>(p, o.ctas_per_output);
    int     bx        = field_at<int>(p, o.input_mult0);
    int     by        = field_at<int>(p, o.input_mult1);
    int     gl        = field_at<int>(p, o.input_mult2);
    int     idim      = field_at<int>(p, o.input_dims);
    int     odim      = field_at<int>(p, o.output_dims);
    const void* src   = field_at<const void*>(p, o.src);
    const void* dst0  = field_at<const void*>(p, o.dst0);
    const void* dst1  = field_at<const void*>(p, o.dst1);
    bool   accum      = field_at<bool>(p, o.accumulate);
    bool   final_out  = field_at<bool>(p, o.final_output);
    int    nout       = field_at<int>(p, o.noutputs);

    char buf[512];
    snprintf(buf, sizeof(buf),
        "[reduce_kernel/MeanOps<float>] "
        "N=%d→%d  factor=%.6f(≈1/%d)  ident=%.4f\n"
        "  src=%-16p  dst[0]=%-16p  dst[1]=%p\n"
        "  block=(%d,%d)  threads=%d  vec=%d/%d  ctas=%d\n"
        "  strategy: block_x=%d block_y=%d global=%d  dims: in=%d out=%d\n"
        "  accumulate=%d  final=%d  noutputs=%d",
        n_in, n_out, (double)factor,
        (factor > 0 ? (int)roundf(1.0f / factor) : 0),
        (double)ident,
        src, dst0, dst1,
        bw, bh, nt, (int)vec, vsz, ctas,
        (bx != 0), (by != 0), (gl != 0), idim, odim,
        (int)accum, (int)final_out, nout);
    return buf;
}

// ============================================================
// Rank 9: flash_fwd_kernel
// args[0] → Flash_fwd_params（HOST端结构体，from flash.h）
//
// 偏移基于 Qkv_params + Flash_fwd_params 成员顺序精确计算
// ============================================================
namespace FFP {
    // Qkv_params（全部 int64_t stride + int head counts）
    constexpr size_t q_ptr          = 0;
    constexpr size_t k_ptr          = 8;
    constexpr size_t v_ptr          = 16;
    constexpr size_t q_batch_stride = 24;
    constexpr size_t k_batch_stride = 32;
    constexpr size_t v_batch_stride = 40;
    constexpr size_t q_row_stride   = 48;
    constexpr size_t k_row_stride   = 56;
    constexpr size_t v_row_stride   = 64;
    constexpr size_t q_head_stride  = 72;
    constexpr size_t k_head_stride  = 80;
    constexpr size_t v_head_stride  = 88;
    constexpr size_t h              = 96;   // int
    constexpr size_t h_k           = 100;  // int
    constexpr size_t h_h_k_ratio   = 104;  // int  +4B pad → base=112
    // Flash_fwd_params 自有字段
    constexpr size_t o_ptr          = 112;
    constexpr size_t oaccum_ptr     = 120;
    constexpr size_t o_batch_stride = 128;
    constexpr size_t o_row_stride   = 136;
    constexpr size_t o_head_stride  = 144;
    constexpr size_t p_ptr          = 152;
    constexpr size_t softmax_lse    = 160;
    constexpr size_t softmax_lseacc = 168;
    constexpr size_t b              = 176;  // int batch_size
    constexpr size_t seqlen_q       = 180;
    constexpr size_t seqlen_k       = 184;
    constexpr size_t seqlen_knew    = 188;
    constexpr size_t d              = 192;  // int head_dim
    constexpr size_t seqlen_q_r     = 196;
    constexpr size_t seqlen_k_r     = 200;
    constexpr size_t d_rounded      = 204;
    constexpr size_t rotary_dim     = 208;
    constexpr size_t total_q        = 212;
    constexpr size_t scale_softmax  = 216;  // float
    constexpr size_t scale_log2     = 220;  // float
    // cu_seqlens_q(8) cu_seqlens_k(8) leftpad_k(8) seqused_k(8) blockmask(8) = 40B
    constexpr size_t cu_seqlens_q   = 224;
    constexpr size_t cu_seqlens_k   = 232;
    // knew/vnew strides...（略，这里只关心核心参数）
    // p_dropout 在大量指针字段之后
    // 精确位置：cu_seqlens_q(8)+cu_seqlens_k(8)+leftpad_k(8)+seqused_k(8)+blockmask(8)
    //           +knew_ptr(8)+vnew_ptr(8)+knew/vnew strides(6×8=48)
    //           +rotary_cos(8)+rotary_sin(8)+cache_batch_idx(8)
    //           +block_table(8)+block_table_batch_stride(8)+page_block_size(4) = 162B
    constexpr size_t p_dropout       = 224 + 40 + 8+8+48+8+8+8+8+8+4; // = 372
    constexpr size_t p_dropout_u8    = 376; // uint8_t
    // 3B pad
    constexpr size_t rp_dropout      = 380; // float
    constexpr size_t scale_rp        = 384; // float
    constexpr size_t window_left     = 388; // int
    constexpr size_t window_right    = 392; // int
    constexpr size_t softcap         = 396; // float
    // PhiloxCudaState(24B) + rng_state(8B) = 32B
    constexpr size_t is_bf16         = 432; // bool
    constexpr size_t is_causal       = 433; // bool
    constexpr size_t num_splits      = 436; // int
    constexpr size_t alibi_slopes    = 440; // void*
    constexpr size_t unpadded_lse    = 456; // bool
}

static std::string parseFlashFwd(void** args) {
    if (!args || !args[0]) return "[flash_fwd] null";
    const uint8_t* p = reinterpret_cast<const uint8_t*>(args[0]);

    void*   q   = field_at<void*>(p, FFP::q_ptr);
    void*   k   = field_at<void*>(p, FFP::k_ptr);
    void*   v   = field_at<void*>(p, FFP::v_ptr);
    void*   o   = field_at<void*>(p, FFP::o_ptr);
    int     b   = field_at<int>(p, FFP::b);
    int     sq  = field_at<int>(p, FFP::seqlen_q);
    int     sk  = field_at<int>(p, FFP::seqlen_k);
    int     h   = field_at<int>(p, FFP::h);
    int     hk  = field_at<int>(p, FFP::h_k);
    int     d   = field_at<int>(p, FFP::d);
    int     tq  = field_at<int>(p, FFP::total_q);
    float   sc  = field_at<float>(p, FFP::scale_softmax);
    float   cap = field_at<float>(p, FFP::softcap);
    float   drp = field_at<float>(p, FFP::p_dropout);
    bool    cau = field_at<bool>(p, FFP::is_causal);
    bool    bf  = field_at<bool>(p, FFP::is_bf16);
    int     ns  = field_at<int>(p, FFP::num_splits);
    int     wl  = field_at<int>(p, FFP::window_left);
    int     wr  = field_at<int>(p, FFP::window_right);
    int64_t qbs = field_at<int64_t>(p, FFP::q_batch_stride);
    int64_t qrs = field_at<int64_t>(p, FFP::q_row_stride);
    int64_t qhs = field_at<int64_t>(p, FFP::q_head_stride);

    char buf[512];
    snprintf(buf, sizeof(buf),
        "[flash_fwd_kernel]\n"
        "  batch=%-4d  seqlen_q=%-6d  seqlen_k=%-6d  total_q=%d\n"
        "  heads=%-4d  heads_k=%-4d   head_dim=%d\n"
        "  Q=%-16p K=%-16p V=%-16p O=%p\n"
        "  Q strides: batch=%-8ld row=%-8ld head=%ld\n"
        "  scale=%.6f  causal=%-2d  softcap=%.4f  dropout=%.4f\n"
        "  bf16=%-2d  num_splits=%-3d  window=[%d,%d]",
        b, sq, sk, tq, h, hk, d,
        q, k, v, o,
        qbs, qrs, qhs,
        (double)sc, (int)cau, (double)cap, (double)drp,
        (int)bf, ns, wl, wr);
    return buf;
}

// ============================================================
// Rank 6: elementwise_kernel<128, 4, lambda>(int N, lambda f)
// args[0] → int N
// args[1] → lambda（内含 TensorIterator 的指针数组，无法直接解析）
// ============================================================
// ============================================================
// 从 kernel readableName 提取关键信息的辅助函数
// ============================================================
// 从模板参数名提取 array_t 的指针数量
// "std::array<char*, 3ul>" → 3
static int extractArraySize(const std::string& name) {
    // 找 "std::array<char*," 后的数字
    auto pos = name.find("std::array<char*,");
    if (pos == std::string::npos) return -1;
    pos += strlen("std::array<char*,");
    while (pos < name.size() && name[pos] == ' ') pos++;
    int n = 0;
    while (pos < name.size() && isdigit(name[pos]))
        n = n * 10 + (name[pos++] - '0');
    return n > 0 ? n : -1;
}
// 从 kernel 名推断操作类型
static const char* inferOpType(const std::string& name) {
    if (name.find("FillFunctor")           != std::string::npos) return "fill";
    if (name.find("MulFunctor")            != std::string::npos) return "mul";
    if (name.find("CUDAFunctor_add")       != std::string::npos) return "add(inplace)";
    if (name.find("CUDAFunctorOnSelf_add") != std::string::npos) return "add_";
    if (name.find("AddFunctor")            != std::string::npos) return "add";
    if (name.find("direct_copy")           != std::string::npos) return "copy";
    if (name.find("LoadWithCast")          != std::string::npos) return "cast";
    if (name.find("float16_copy")          != std::string::npos) return "f32→f16";
    if (name.find("cos_kernel")            != std::string::npos) return "cos";
    if (name.find("sin_kernel")            != std::string::npos) return "sin";
    if (name.find("exp_kernel")            != std::string::npos) return "exp";
    if (name.find("rsqrt_kernel")          != std::string::npos) return "rsqrt";
    if (name.find("pow_tensor_scalar")     != std::string::npos) return "pow(scalar)";
    if (name.find("arange")               != std::string::npos) return "arange";
    if (name.find("silu")                  != std::string::npos) return "silu";
    if (name.find("sigmoid")              != std::string::npos) return "sigmoid";
    if (name.find("gelu")                  != std::string::npos) return "gelu";
    if (name.find("tanh")                  != std::string::npos) return "tanh";
    return "unknown";
}

// ============================================================
// 从数组参数读取张量指针列表
// arg_ptr 即 args[2]（vectorized 和 unrolled 路径相同位置）
// ============================================================
static int appendTensorPtrs(char* buf, int pos, int buf_sz,
                              void* arg_ptr, int arr_sz) {
    if (!arg_ptr || arr_sz <= 0) return pos;
    const char* role[] = {"output", "input1", "input2", "input3"};
    pos += snprintf(buf + pos, buf_sz - pos, "\n  tensors(%d):", arr_sz);
    for (int i = 0; i < arr_sz && i < 4; i++) {
        void* ptr = field_at<void*>(arg_ptr, i * sizeof(void*));
        pos += snprintf(buf + pos, buf_sz - pos,
            "\n    [%d] %-7s = %p", i,
            (i < 4 ? role[i] : "inputN"), ptr);
    }
    return pos;
}

// ============================================================
// FillFunctor<T> 布局：唯一成员是填充值
// struct FillFunctor { T fill_value; };
// ============================================================
static std::string parseFillFunctor(void* arg1, const std::string& name) {
    if (!arg1) return "";
    char tmp[64] = {};
    if (name.find("FillFunctor<float>") != std::string::npos) {
        float v = field_at<float>(arg1, 0);
        snprintf(tmp, sizeof(tmp), "  fill_value(float)=%.6f", (double)v);
    } else if (name.find("FillFunctor<c10::Half>") != std::string::npos ||
               name.find("FillFunctor<at::Half>")  != std::string::npos) {
        uint16_t h = field_at<uint16_t>(arg1, 0);
        // half → float
        uint32_t f = ((uint32_t)(h & 0x8000u) << 16)
                   | ((uint32_t)((h & 0x7C00u) + 0x1C000u) << 13)
                   | ((uint32_t)(h & 0x03FFu) << 13);
        float v; std::memcpy(&v, &f, 4);
        snprintf(tmp, sizeof(tmp), "  fill_value(half)=%.6f", (double)v);
    } else if (name.find("FillFunctor<int>") != std::string::npos) {
        int v = field_at<int>(arg1, 0);
        snprintf(tmp, sizeof(tmp), "  fill_value(int)=%d", v);
    }
    return tmp;
}

// ============================================================
// 从 lambda 捕获列表读取标量参数（HOST 端，直接可读）
// ============================================================
static int appendScalarCaptures(char* buf, int pos, int buf_sz,
                                  void* lambda_arg, const std::string& name) {
    if (!lambda_arg) return pos;
    // ── pow_tensor_scalar：lambda = { float exponent } ──────
    if (name.find("pow_tensor_scalar") != std::string::npos) {
        float exp = field_at<float>(lambda_arg, 0);
        pos += snprintf(buf + pos, buf_sz - pos,
            "\n  exponent = %.6f", (double)exp);
        return pos;
    }
    // ── AUnaryFunctor<T, T, T, MulFunctor<T>>
    //    结构体：{ MulFunctor op (空，1B+pad); opmath_t b; }
    //    MulFunctor 是空结构体，ABI 下占 1B，对齐到 4B → b 在 offset=4
    if (name.find("AUnaryFunctor")  != std::string::npos &&
        name.find("MulFunctor")      != std::string::npos) {
        float scalar = field_at<float>(lambda_arg, 4);
        // 如果是 NaN/Inf 说明偏移不对，尝试 offset=0
        if (scalar != scalar || scalar > 1e20f) // NaN or huge
            scalar = field_at<float>(lambda_arg, 0);
        pos += snprintf(buf + pos, buf_sz - pos,
            "\n  scalar_mul = %.6f", (double)scalar);
        return pos;
    }
    // ── arange：lambda 捕获 { scalar_t start; scalar_t step; }
    if (name.find("arange") != std::string::npos) {
        // 模板参数中有 result_type，从名称判断类型
        if (name.find("<int,") != std::string::npos ||
            name.find("result_type*") != std::string::npos) {
            // 尝试 float（arange 内部用 float 精度）
            float start = field_at<float>(lambda_arg, 0);
            float step  = field_at<float>(lambda_arg, 4);
            pos += snprintf(buf + pos, buf_sz - pos,
                "\n  start=%.4f  step=%.4f", (double)start, (double)step);
        }
        return pos;
    }
    // ── FillFunctor：已有函数处理 ───────────────────────────
    if (name.find("FillFunctor") != std::string::npos) {
        auto s = parseFillFunctor(lambda_arg, name);
        pos += snprintf(buf + pos, buf_sz - pos, "\n%s", s.c_str());
        return pos;
    }
    return pos;
}

// ============================================================
// 主解析函数（替换原来的 parseElementwiseKernel）
// ============================================================
static std::string parseElementwiseKernel(void** args, const std::string& name) {
    if (!args || !args[0]) return "[elementwise] null";
    int         N      = read_arg<int>(args[0]);
    const char* op     = inferOpType(name);
    bool is_vec      = name.find("vectorized_elementwise_kernel")  != std::string::npos;
    bool is_unrolled = name.find("unrolled_elementwise_kernel")    != std::string::npos;
    bool is_indexed  = name.find("elementwise_kernel_with_index")  != std::string::npos;
    char buf[512];
    int pos = snprintf(buf, sizeof(buf), "[elementwise/%s]  N=%d", op, N);
    if (is_vec || is_unrolled) {
        // args: [0]=N  [1]=lambda  [2]=std::array<char*, K>
        int arr_sz = extractArraySize(name);
        pos = appendTensorPtrs(buf, pos, sizeof(buf), args[2], arr_sz);
        pos = appendScalarCaptures(buf, pos, sizeof(buf), args[1], name);
    }
    else if (is_indexed) {
        // args: [0]=N  [1]=lambda  [2]=scalar_t* output（单指针）
        void* out = args[2] ? read_arg<void*>(args[2]) : nullptr;
        pos += snprintf(buf + pos, sizeof(buf) - pos,
            "\n  output = %p", out);
        pos = appendScalarCaptures(buf, pos, sizeof(buf), args[1], name);
    }
    else {
        pos += snprintf(buf + pos, sizeof(buf) - pos,
            "\n  (unknown elementwise variant)");
    }
    return buf;
}

// ============================================================
// Rank 8: CatArrayBatchedCopy
// args[0] → T1* output（设备指针值）
// args[1] → CatArrInputTensorMetadata struct（HOST端）
// args[3] → int concatDim
// args[4] → uint32_t dimStride
//
// CatArrInputTensorMetadata<OpaqueType<2>, uint32_t, 4, 64>:
//   void*    inputPtr[64]  →  0B   (512B)
//   uint32_t offset[64]   →  512B (256B)
//   uint32_t dimSize[64]  →  768B (256B)
//   uint32_t nElements[64]→  1024B(256B)
// ============================================================
static std::string parseCatArray(void** args) {
    if (!args) return "[CatArray] null";
    void*    out  = args[0] ? read_arg<void*>(args[0]) : nullptr;
    int      cdim = args[3] ? read_arg<int>(args[3])    : -1;
    uint32_t dstr = args[4] ? read_arg<uint32_t>(args[4]): 0;

    void* in0 = nullptr, *in1 = nullptr, *in2 = nullptr;
    uint32_t off0=0, off1=0, dsz0=0, dsz1=0, nel0=0;
    if (args[1]) {
        const uint8_t* m = reinterpret_cast<const uint8_t*>(args[1]);
        in0  = field_at<void*>(m, 0);
        in1  = field_at<void*>(m, 8);
        in2  = field_at<void*>(m, 16);
        off0 = field_at<uint32_t>(m, 512);
        off1 = field_at<uint32_t>(m, 516);
        dsz0 = field_at<uint32_t>(m, 768);
        dsz1 = field_at<uint32_t>(m, 772);
        nel0 = field_at<uint32_t>(m, 1024);
    }
    char buf[256];
    snprintf(buf, sizeof(buf),
        "[CatArrayBatchedCopy<u16>]\n"
        "  output=%p  concatDim=%d  dimStride=%u\n"
        "  in[0]=%p offset=%-6u dimSize=%u nElements=%u\n"
        "  in[1]=%p offset=%-6u dimSize=%u\n"
        "  in[2]=%p",
        out, cdim, dstr,
        in0, off0, dsz0, nel0,
        in1, off1, dsz1,
        in2);
    return buf;
}

// ============================================================
// 主分发函数
// ============================================================
static void dispatchKernelArgs(const std::string& name,
                                void** args,
                                dim3 grid, dim3 block, size_t smem)
{
    if (!fakecuda::capture::generates_trace()) return;
    fakecuda::capture::count(fakecuda::capture::Counter::KernelArgParsers);
    std::string detail;
    bool matched = true;
    if (name.find("reduce_kernel") != std::string::npos &&
        name.find("MeanOps")       != std::string::npos)
        detail = parseReduceMean(args);
    else if (name.find("flash_fwd_kernel") != std::string::npos)
        detail = parseFlashFwd(args);
    // ↓ 修改：传入 name
    else if (name.find("vectorized_elementwise_kernel") != std::string::npos ||
         name.find("unrolled_elementwise_kernel")   != std::string::npos ||
         name.find("elementwise_kernel_with_index") != std::string::npos ||
         name.find("elementwise_kernel")            != std::string::npos)
        detail = parseElementwiseKernel(args, name);
    else if (name.find("CatArrayBatchedCopy") != std::string::npos)
        detail = parseCatArray(args);
    else
        matched = false;
    if (matched)
        LOG_INFO(CUDART, "  args: %s", detail.c_str());
}
