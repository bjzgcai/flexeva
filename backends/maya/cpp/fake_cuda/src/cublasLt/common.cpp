#include "fake_types.h"
#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "cublas.hpp"
#include "../../include/common/ras_capture.h"
#include <algorithm>
#include <climits>
#include <cstring>

namespace {
std::unordered_set<cublasLtHandle_t> handles;
std::unordered_set<FakeDesc*> descriptors;
std::unordered_set<FakeLayout*> layouts;
struct Preference { uint64_t workspace_bytes = 0; uint32_t alignment[4] = {256,256,256,256}; };
std::unordered_set<Preference*> preferences;

cublasStatus_t invalid(const char* message) {
    if (fakecuda::ras::active()) fakecuda_ras_fail(message);
    return CUBLAS_STATUS_INVALID_VALUE;
}
cublasStatus_t unsupported(const char* message) {
    if (fakecuda::ras::active()) fakecuda_ras_fail(message);
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
template <class T> bool contains(const std::unordered_set<T*>& set, T* p) {
    return p && set.find(p) != set.end();
}
// PyTorch obtains an Lt view from its regular cuBLAS handle. Both native
// registries validate the address before any state is dereferenced.
bool valid_handle(cublasLtHandle_t handle) {
    return contains(handles, handle) || fake_valid_blas_handle(reinterpret_cast<cublasHandle_t>(handle));
}
template <class T> bool read_attribute(T& out, const void* buf, size_t size) {
    if (!buf || size != sizeof(T)) return false;
    std::memcpy(&out, buf, sizeof(T)); return true;
}
template <class T> T* create(std::unordered_set<T*>& set) {
    auto* p = new (std::nothrow) T{};
    if (!p) return nullptr;
    try { set.insert(p); } catch (...) { delete p; return nullptr; }
    return p;
}
}

API cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle) {
    if (!lightHandle) return invalid("Invalid cuBLASLt handle output");
    auto* ctx = create(handles);
    if (!ctx) return CUBLAS_STATUS_ALLOC_FAILED;
    ctx->magic_number = 0xBEEF; ctx->stream = nullptr; *lightHandle = ctx;
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_BLAS_CREATE);
        r.handle = fakecuda::ras::address(ctx);
        if (!fakecuda::ras::emit(r)) { handles.erase(ctx); delete ctx; *lightHandle = nullptr; return CUBLAS_STATUS_INTERNAL_ERROR; }
        return CUBLAS_STATUS_SUCCESS;
    }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtDestroy(cublasLtHandle_t handle) {
    if (!contains(handles, handle)) return invalid("Invalid cuBLASLt handle");
    auto id = fakecuda::ras::address(handle);
    handles.erase(handle); delete handle;
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_BLAS_DESTROY); r.handle = id;
        return fakecuda::ras::emit(r) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* output,
    cublasComputeType_t computeType, cudaDataType_t scaleType) {
    if (!output) return invalid("Invalid cuBLASLt descriptor output");
    auto* p = create(descriptors);
    if (!p) return CUBLAS_STATUS_ALLOC_FAILED;
    p->compute_type = computeType; p->scaleType = scaleType;
    *output = reinterpret_cast<cublasLtMatmulDesc_t>(p);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t descriptor) {
    auto* p = reinterpret_cast<FakeDesc*>(descriptor);
    if (!contains(descriptors, p)) return invalid("Invalid cuBLASLt descriptor");
    descriptors.erase(p); delete p; return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t descriptor,
    cublasLtMatmulDescAttributes_t attr, const void* buf, size_t size) {
    auto* p = reinterpret_cast<FakeDesc*>(descriptor);
    if (!contains(descriptors, p)) return invalid("Invalid cuBLASLt descriptor");
    bool valid = false;
    switch (attr) {
    case CUBLASLT_MATMUL_DESC_TRANSA: valid = read_attribute(p->transa, buf, size); break;
    case CUBLASLT_MATMUL_DESC_TRANSB: valid = read_attribute(p->transb, buf, size); break;
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE: valid = read_attribute(p->scaleType, buf, size); break;
    case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE: valid = read_attribute(p->compute_type, buf, size); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE: valid = read_attribute(p->epilogue, buf, size); break;
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER: valid = read_attribute(p->bias, buf, size); break;
    case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: valid = read_attribute(p->bias_type, buf, size); break;
    case CUBLASLT_MATMUL_DESC_POINTER_MODE: valid = read_attribute(p->pointer_mode, buf, size); break;
    default: return unsupported("Unsupported cuBLASLt matmul descriptor attribute");
    }
    return valid ? CUBLAS_STATUS_SUCCESS : invalid("Invalid cuBLASLt descriptor attribute size");
}

API cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* output,
    cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld) {
    if (!output || ld < 0) return invalid("Invalid cuBLASLt layout output");
    auto* p = create(layouts);
    if (!p) return CUBLAS_STATUS_ALLOC_FAILED;
    p->type = type; p->rows = rows; p->cols = cols; p->ld = ld;
    *output = reinterpret_cast<cublasLtMatrixLayout_t>(p); return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t layout) {
    auto* p = reinterpret_cast<FakeLayout*>(layout);
    if (!contains(layouts, p)) return invalid("Invalid cuBLASLt layout");
    layouts.erase(p); delete p; return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t layout,
    cublasLtMatrixLayoutAttribute_t attr, const void* buf, size_t size) {
    auto* p = reinterpret_cast<FakeLayout*>(layout);
    if (!contains(layouts, p)) return invalid("Invalid cuBLASLt layout");
    bool valid = false;
    switch (attr) {
    case CUBLASLT_MATRIX_LAYOUT_TYPE: valid = read_attribute(p->type, buf, size); break;
    case CUBLASLT_MATRIX_LAYOUT_ROWS: valid = read_attribute(p->rows, buf, size); break;
    case CUBLASLT_MATRIX_LAYOUT_COLS: valid = read_attribute(p->cols, buf, size); break;
    case CUBLASLT_MATRIX_LAYOUT_LD: valid = read_attribute(p->ld, buf, size); break;
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: valid = read_attribute(p->batch, buf, size); break;
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: valid = read_attribute(p->stride, buf, size); break;
    case CUBLASLT_MATRIX_LAYOUT_ORDER: {
        int32_t order = -1;
        if (!read_attribute(order, buf, size)) return invalid("Invalid cuBLASLt layout order size");
        return order == 0 ? CUBLAS_STATUS_SUCCESS : unsupported("Only column-major cuBLASLt layout is supported");
    }
    default: return unsupported("Unsupported cuBLASLt matrix layout attribute");
    }
    return valid ? CUBLAS_STATUS_SUCCESS : invalid("Invalid cuBLASLt layout attribute size");
}

API cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* output) {
    if (!output) return invalid("Invalid cuBLASLt preference output");
    auto* p = create(preferences);
    if (!p) return CUBLAS_STATUS_ALLOC_FAILED;
    *output = reinterpret_cast<cublasLtMatmulPreference_t>(p); return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t preference) {
    auto* p = reinterpret_cast<Preference*>(preference);
    if (!contains(preferences, p)) return invalid("Invalid cuBLASLt preference");
    preferences.erase(p); delete p; return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t preference,
    cublasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t size) {
    auto* p = reinterpret_cast<Preference*>(preference);
    if (!contains(preferences, p)) return invalid("Invalid cuBLASLt preference");
    bool valid = false;
    if (attr == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES) valid = read_attribute(p->workspace_bytes, buf, size);
    else if (attr >= CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES && attr <= CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES)
        valid = read_attribute(p->alignment[attr - CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES], buf, size);
    else return unsupported("Unsupported cuBLASLt preference attribute");
    return valid ? CUBLAS_STATUS_SUCCESS : invalid("Invalid cuBLASLt preference attribute size");
}

API cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b,
    cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d, cublasLtMatmulPreference_t preference,
    int requested, cublasLtMatmulHeuristicResult_t results[], int* returned) {
    if (!valid_handle(handle)) return invalid("Invalid cuBLASLt heuristic handle");
    if (!contains(descriptors, reinterpret_cast<FakeDesc*>(operation))) return invalid("Invalid cuBLASLt heuristic descriptor");
    if (!contains(layouts, reinterpret_cast<FakeLayout*>(a))) return invalid("Invalid cuBLASLt heuristic A layout");
    if (!contains(layouts, reinterpret_cast<FakeLayout*>(b))) return invalid("Invalid cuBLASLt heuristic B layout");
    if (!contains(layouts, reinterpret_cast<FakeLayout*>(c))) return invalid("Invalid cuBLASLt heuristic C layout");
    if (!contains(layouts, reinterpret_cast<FakeLayout*>(d))) return invalid("Invalid cuBLASLt heuristic D layout");
    if (!contains(preferences, reinterpret_cast<Preference*>(preference))) return invalid("Invalid cuBLASLt heuristic preference");
    if (!returned || requested < 0 || (requested && !results)) return invalid("Invalid cuBLASLt heuristic results output");
    *returned = requested ? 1 : 0;
    if (requested) std::memset(results, 0, sizeof(*results));
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t descriptor,
    const void* alpha, const void* A, cublasLtMatrixLayout_t a_desc,
    const void* B, cublasLtMatrixLayout_t b_desc, const void* beta,
    const void* C, cublasLtMatrixLayout_t c_desc, void* D, cublasLtMatrixLayout_t d_desc,
    const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    auto* md = reinterpret_cast<FakeDesc*>(descriptor);
    auto* a = reinterpret_cast<FakeLayout*>(a_desc); auto* b = reinterpret_cast<FakeLayout*>(b_desc);
    auto* c = reinterpret_cast<FakeLayout*>(c_desc); auto* d = reinterpret_cast<FakeLayout*>(d_desc);
    if (!valid_handle(handle) || !contains(descriptors, md) || !contains(layouts, a) ||
        !contains(layouts, b) || !contains(layouts, c) || !contains(layouts, d) ||
        !fake_valid_stream(stream) || !alpha || !beta || (!workspace && workspace_bytes))
        return invalid("Invalid cuBLASLt matmul arguments");
    if (md->transa < CUBLAS_OP_N || md->transa > CUBLAS_OP_C ||
        md->transb < CUBLAS_OP_N || md->transb > CUBLAS_OP_C || md->pointer_mode != 0)
        return unsupported("Unsupported cuBLASLt transpose or pointer mode");
    // Default, bias, ReLU/GELU, and their bias forms need no auxiliary matrix.
    if (md->epilogue != 1 && md->epilogue != 2 && md->epilogue != 4 &&
        md->epilogue != 6 && md->epilogue != 32 && md->epilogue != 36)
        return unsupported("Unsupported cuBLASLt epilogue");
    const uint64_t m = md->transa == CUBLAS_OP_N ? a->rows : a->cols;
    const uint64_t k = md->transa == CUBLAS_OP_N ? a->cols : a->rows;
    const uint64_t n = md->transb == CUBLAS_OP_N ? b->cols : b->rows;
    if ((md->transb == CUBLAS_OP_N ? b->rows : b->cols) != k ||
        c->rows != m || c->cols != n || d->rows != m || d->cols != n ||
        a->batch < 1 || b->batch != a->batch || c->batch != a->batch || d->batch != a->batch ||
        a->ld < static_cast<int64_t>(a->rows) || b->ld < static_cast<int64_t>(b->rows) ||
        c->ld < static_cast<int64_t>(c->rows) || d->ld < static_cast<int64_t>(d->rows) ||
        m > INT_MAX || n > INT_MAX || k > INT_MAX || a->ld > INT_MAX || b->ld > INT_MAX ||
        c->ld > INT_MAX || d->ld > INT_MAX || (m && n && (!A || !B || !C || !D)))
        return invalid("Invalid cuBLASLt dimensions or operand");
    if (!m || !n) return CUBLAS_STATUS_SUCCESS;
    fake_stream_mark_work_enqueued(stream);
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record_v2(FAKECUDA_RAS_GEMM);
        r.api_kind = FAKECUDA_RAS_API_CUBLASLT_MATMUL;
        r.base.handle = fakecuda::ras::address(handle); r.base.stream = fakecuda::ras::address(stream);
        r.base.a = fakecuda::ras::address(A); r.base.b = fakecuda::ras::address(B);
        r.base.c = fakecuda::ras::address(C); r.d = fakecuda::ras::address(D);
        r.aux = fakecuda::ras::address(md->bias);
        if (!fakecuda::ras::wants_payload(FAKECUDA_RAS_GEMM)) r.flags_v2 |= FAKECUDA_RAS_PAYLOAD_ELIDED;
        else {
            r.base.m = m; r.base.n = n; r.base.k = k;
            r.base.lda = a->ld; r.base.ldb = b->ld; r.base.ldc = c->ld; r.ldd = d->ld;
            r.base.trans_a = md->transa; r.base.trans_b = md->transb;
            r.base.a_type = a->type; r.base.b_type = b->type; r.base.c_type = c->type; r.d_type = d->type;
            r.base.compute_type = md->compute_type; r.base.algorithm = -1;
            r.batch_count = a->batch; r.stride_a = a->stride; r.stride_b = b->stride;
            r.stride_c = c->stride; r.stride_d = d->stride;
            r.epilogue = md->epilogue; r.pointer_mode = md->pointer_mode;
        }
        return fakecuda::ras::emit(r) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return CUBLAS_STATUS_SUCCESS;
}
