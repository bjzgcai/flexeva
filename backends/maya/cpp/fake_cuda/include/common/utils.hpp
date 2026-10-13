#pragma once
#include <mutex>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <time.h>
#include <sys/mman.h>
#include "fake_types.h"
#include "capture_mode.hpp"
#include "../device/fake_device_core.h"

#define METADATA_THRESHOLD (256 * 1024)  // 数据传输阈值256KB，尽可能的只传输meta data

// API 可见性
#define API extern "C" __attribute__((visibility("default")))

// 全局记录 Host 内存分配大小
static std::map<void*, size_t> g_host_allocations;
static std::mutex g_host_mutex;

// 日志系统
#ifndef FAKECUDA_LOG_H
#define FAKECUDA_LOG_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum LogLevel {
    LOG_NONE = 0,
    LOG_ERROR = 1,
    LOG_WARN = 2,
    LOG_INFO = 3,
    LOG_DEBUG = 4
};
// 库标识
enum LibType {
    CUDA,
    CUDART,
    CUBLAS,
    CUBLASLT,
    NVML,
    NCCL,
    LIB_COUNT
};
class Logger {
public:
    static const char* libNames[LIB_COUNT];
    static bool shouldLog(LibType lib, LogLevel level) {
        if (fakecuda_ras_active()) return false;
        init();
        return levels[lib] >= level;
    }
private:
    static LogLevel levels[LIB_COUNT];
    static bool initialized;
    static void init(){
        if (initialized) return;
        // 1. 读取全局日志级别
        LogLevel globalLevel = LOG_NONE;
        const char* globalEnv = getenv("FAKECUDA_LOG_LEVEL");
        if (globalEnv) {
            globalLevel = (LogLevel)atoi(globalEnv);
        }

        // 2. 为每个库设置日志级别（优先读取库特定的环境变量）
        const char* envVars[] = {
            "CUDA_LOG_LEVEL",
            "CUDART_LOG_LEVEL",
            "CUBLAS_LOG_LEVEL",
            "CUBLASLT_LOG_LEVEL",
            "NVML_LOG_LEVEL",
            "NCCL_LOG_LEVEL"
        };

        for (int i = 0; i < LIB_COUNT; i++) {
            const char* env = getenv(envVars[i]);
            if (env) {
                levels[i] = (LogLevel)atoi(env);
            } else {
                levels[i] = globalLevel;  // 使用全局级别
            }
        }
        initialized = true;
    }
};
// 静态成员初始化
inline LogLevel Logger::levels[LIB_COUNT] = {LOG_NONE};
inline bool Logger::initialized = false;
inline const char* Logger::libNames[LIB_COUNT] = {
    "cuda","cudart","cublas","cublaslt","nvml","nccl"
};

namespace fakecuda::trace {

inline bool trace_capture_enabled() {
    if (!fakecuda::capture::generates_trace()) return false;
    static bool initialized = false;
    static bool enabled = false;
    if (!initialized) {
        const char* raw = getenv("FAKECUDA_TRACE");
        enabled = raw != nullptr && strcmp(raw, "1") == 0;
        initialized = true;
    }
    return enabled;
}

inline const char* canonical_debug_trace_api_name(const char* api) {
    if (!api) {
        return api;
    }
    if (strcmp(api, "cudaEventRecordWithFlags") == 0) {
        return "cudaEventRecord";
    }
    if (strcmp(api, "cudaEventCreate") == 0) {
        return "cudaEventCreateWithFlags";
    }
    if (strcmp(api, "cudaStreamCreateWithFlags") == 0 || strcmp(api, "cudaStreamCreateWithPriority") == 0) {
        return "cudaStreamCreate";
    }
    if (strcmp(api, "ncclBcast") == 0) {
        return "ncclBroadcast";
    }
    return api;
}

inline long long debug_entry_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

inline long long& trace_writer_overhead_ns() {
    thread_local long long overhead_ns = 0LL;
    return overhead_ns;
}

inline long long current_trace_writer_overhead_ns() {
    return trace_writer_overhead_ns();
}

inline long long logical_host_time_ns(long long raw_ns) {
    const long long overhead_ns = current_trace_writer_overhead_ns();
    if (raw_ns <= overhead_ns) {
        return 0LL;
    }
    return raw_ns - overhead_ns;
}

inline void add_trace_writer_overhead_ns(long long overhead_ns) {
    if (overhead_ns <= 0LL) {
        return;
    }
    trace_writer_overhead_ns() += overhead_ns;
}

inline std::unordered_map<std::string, std::vector<long long>>& wrapper_entry_start_ns() {
    thread_local std::unordered_map<std::string, std::vector<long long>> starts;
    return starts;
}

inline long long sanitize_duration_ns(long long duration_ns) {
    return duration_ns > 0LL ? duration_ns : 0LL;
}

inline std::string parse_debug_entry_api(const char* fmt) {
    if (!fmt) {
        return "";
    }
    std::string text(fmt);
    const std::size_t first = text.find_first_not_of(" \t");
    if (first == std::string::npos) {
        return "";
    }
    text.erase(0, first);

    const std::string called_suffix = " called.";
    std::size_t suffix_pos = text.rfind(called_suffix);
    if (suffix_pos == std::string::npos || suffix_pos + called_suffix.size() != text.size()) {
        return "";
    }

    std::string api = text.substr(0, suffix_pos);
    while (!api.empty() && (api.back() == ' ' || api.back() == '\t')) {
        api.pop_back();
    }
    if (api.size() >= 2 && api.substr(api.size() - 2) == "()") {
        api.erase(api.size() - 2);
    }
    if (api.empty()) {
        return "";
    }
    for (const unsigned char ch : api) {
        if (!(std::isalnum(ch) || ch == '_' || ch == ':')) {
            return "";
        }
    }
    return std::string(canonical_debug_trace_api_name(api.c_str()));
}

inline void maybe_record_wrapper_entry(const char* fmt) {
    if (!trace_capture_enabled()) {
        return;
    }
    const long long overhead_start_ns = debug_entry_now_ns();
    const std::string api = parse_debug_entry_api(fmt);
    add_trace_writer_overhead_ns(debug_entry_now_ns() - overhead_start_ns);
    if (api.empty()) {
        return;
    }
    fakecuda::capture::count(fakecuda::capture::Counter::WrapperEntries);
    wrapper_entry_start_ns()[api].push_back(debug_entry_now_ns());
}

inline void maybe_record_wrapper_entry_api(const char* api) {
    if (!trace_capture_enabled() || api == nullptr || api[0] == '\0') {
        return;
    }
    fakecuda::capture::count(fakecuda::capture::Counter::WrapperEntries);
    wrapper_entry_start_ns()[canonical_debug_trace_api_name(api)].push_back(debug_entry_now_ns());
}

inline bool take_wrapper_entry_start_ns(const char* api, long long* start_ns_out) {
    if (!trace_capture_enabled() || api == nullptr || start_ns_out == nullptr) {
        return false;
    }
    const std::string canonical_api = canonical_debug_trace_api_name(api);
    auto& starts = wrapper_entry_start_ns();
    auto it = starts.find(canonical_api);
    if (it == starts.end() || it->second.empty()) {
        return false;
    }
    *start_ns_out = it->second.back();
    it->second.pop_back();
    if (it->second.empty()) {
        starts.erase(it);
    }
    return true;
}

}  // namespace fakecuda::trace

// 日志宏定义
#define LOG_ERROR(lib, fmt, ...) \
    do { \
        if (Logger::shouldLog(lib, LOG_ERROR)) { \
            fprintf(stderr, "[fake%s][error] " fmt "\n", Logger::libNames[lib], ##__VA_ARGS__); \
        } \
    } while(0)
#define LOG_WARN(lib, fmt, ...) \
    do { \
        if (Logger::shouldLog(lib, LOG_WARN)) { \
            fprintf(stderr, "[fake%s][warn] " fmt "\n", Logger::libNames[lib], ##__VA_ARGS__); \
        } \
    } while(0)
#define LOG_INFO(lib, fmt, ...) \
    do { \
        if (Logger::shouldLog(lib, LOG_INFO)) { \
            fprintf(stderr, "[fake%s][info] " fmt "\n", Logger::libNames[lib], ##__VA_ARGS__); \
        } \
    } while(0)
#define LOG_DEBUG(lib, fmt, ...) \
    do { \
        fakecuda::trace::maybe_record_wrapper_entry(fmt); \
        if (Logger::shouldLog(lib, LOG_DEBUG)) { \
            fprintf(stderr, "[fake%s][debug] " fmt "\n", Logger::libNames[lib], ##__VA_ARGS__); \
        } \
    } while(0)

#define LOG_DEBUG_NOENTRY(lib, fmt, ...) \
    do { \
        if (Logger::shouldLog(lib, LOG_DEBUG)) { \
            fprintf(stderr, "[fake%s][debug] " fmt "\n", Logger::libNames[lib], ##__VA_ARGS__); \
        } \
    } while(0)
#endif

#include <cstring>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <cxxabi.h>

// ==================== 常量定义 ====================
#define FATBIN_MAGIC         0xBA55ED50
#define FATBINC_MAGIC        0x466243B1
#define FATBINC_VERSION      1
#define FATBINC_LINK_VERSION 2
enum { FATBIN_KIND_PTX = 1, FATBIN_KIND_CUBIN = 2 };

// ==================== 辅助读取 ====================
static inline uint32_t u32(const void* p) {
    uint32_t v; std::memcpy(&v, p, 4); return v;
}
static inline uint16_t u16(const void* p) {
    uint16_t v; std::memcpy(&v, p, 2); return v;
}

// ==================== Wrapper ====================
struct __fatBinC_Wrapper_t {
    int magic;
    int version;
    const unsigned long long* data;
    void* filename_or_fatbins;
};
static_assert(sizeof(__fatBinC_Wrapper_t) == 24, "wrapper layout mismatch");

// ==================== Fatbin Header ====================
#pragma pack(push, 1)
struct FatbinHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t headerSize;
    uint32_t dataSize;
    uint32_t flags;
};
#pragma pack(pop)

// ==================== ELF 结构 ====================
struct Elf64_Ehdr {
    unsigned char e_ident[16];
    uint16_t e_type; uint16_t e_machine; uint32_t e_version;
    uint64_t e_entry; uint64_t e_phoff; uint64_t e_shoff;
    uint32_t e_flags; uint16_t e_ehsize; uint16_t e_phentsize;
    uint16_t e_phnum; uint16_t e_shentsize; uint16_t e_shnum; uint16_t e_shstrndx;
};
struct Elf64_Shdr {
    uint32_t sh_name; uint32_t sh_type; uint64_t sh_flags; uint64_t sh_addr;
    uint64_t sh_offset; uint64_t sh_size; uint32_t sh_link; uint32_t sh_info;
    uint64_t sh_addralign; uint64_t sh_entsize;
};

struct KernelInfo {
    void* hostFunction = nullptr;
    std::string deviceName;
    std::string readableName;
    int threadLimit = 0;
    void** fatCubinHandle = nullptr;
    cudaFuncAttributes attributes;
    bool attributesParsed = false;
};

struct LaunchConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem = 0;
    void* stream = nullptr;
};

struct CubinData {
    int sm_major = 0;
    int sm_minor = 0;
    size_t size = 0;
    std::map<std::string, cudaFuncAttributes> kernelAttrs;
};

struct FatBinaryInfo {
    void* fatCubin = nullptr;
    std::vector<CubinData> cubins;
    int ptxSmVersion = 0;
    bool parsed = false;
};

// ==================== demangle ====================
static std::string demangle(const char* mangled_name) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        free(demangled);
        return result;
    }
    return mangled_name;
}

// ==================== resolve fatbin base ====================
static inline const uint8_t* resolve_fatbin_base(const void* p) {
    if (!p) return nullptr;
    auto* w = (const __fatBinC_Wrapper_t*)p;
    if (w->magic == FATBINC_MAGIC &&
        (w->version == FATBINC_VERSION || w->version == FATBINC_LINK_VERSION) &&
        w->data && u32(w->data) == FATBIN_MAGIC) {
        return (const uint8_t*)w->data;
    }
    if (u32(p) == FATBIN_MAGIC) return (const uint8_t*)p;
    return nullptr;
}

// ==================== 获取目标 SM 版本 ====================
static int getTargetSM() {
    static int cachedSM = 0;
    if (cachedSM == 0) {
        int major = fake_getDevProps(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
        int minor = fake_getDevProps(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
        cachedSM = major * 10 + minor;
    }
    return cachedSM;
}

// ==================== SM 版本选择 ====================
struct CubinCandidate {
    int smVersion;
    const uint8_t* payload;
    size_t compressedSz;
    size_t payloadSize;
    uint8_t smMajor, smMinor;
};

static int selectBestCubin(const std::vector<CubinCandidate>& candidates, int targetSM) {
    int targetMajor = targetSM / 10;
    int bestIdx = -1, bestSM = -1, bestPriority = -1;

    for (int i = 0; i < (int)candidates.size(); i++) {
        int sm = candidates[i].smVersion;
        int priority;

        if (sm == targetSM)                                priority = 3; // 精确匹配
        else if (sm / 10 == targetMajor && sm < targetSM)  priority = 2; // 同主版本
        else if (sm < targetSM)                            priority = 1; // 低主版本
        else                                               priority = 0; // 高于目标

        if (priority > bestPriority || (priority == bestPriority && sm > bestSM)) {
            bestPriority = priority;
            bestSM = sm;
            bestIdx = i;
        }
        if (priority == 3) break;
    }
    return bestIdx;
}

// ==================== KernelRegistry ====================
class KernelRegistry {
public:
    static KernelRegistry& getInstance() {
        static KernelRegistry instance;
        return instance;
    }
    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;

    LaunchConfig& getLaunchConfig() { return launchConfig; }

    // ==================== 注册（延迟，不解析） ====================
    void** RegisterFatbin(void* fatCubin) {
        void** handle = new void*(fatCubin);
        fatBinaries[handle] = {fatCubin, {}, 0, false};
        return handle;
    }

    void RegisterFunction(void** fatCubinHandle, const void* hostFun,
                          const char* deviceFun, const char* deviceName,
                          int thread_limit) {
        KernelInfo ki;
        ki.hostFunction = (void*)hostFun;
        ki.deviceName = deviceName ? deviceName : "";
        ki.readableName = demangle(ki.deviceName.c_str());
        ki.threadLimit = thread_limit;
        ki.fatCubinHandle = fatCubinHandle;
        ki.attributesParsed = false;
        kernels[(void*)hostFun] = std::move(ki);
    }

    void UnregisterFatbin(void** handle) {
        auto it = fatBinaries.find(handle);
        if (it != fatBinaries.end()) {
            // 移除该 fatbin 下所有 kernel
            for (auto kit = kernels.begin(); kit != kernels.end(); ) {
                if (kit->second.fatCubinHandle == handle)
                    kit = kernels.erase(kit);
                else
                    ++kit;
            }
            fatBinaries.erase(it);
            delete handle;
        }
    }
    void RegisterVar(void** fatCubinHandle, void* hostVar,
                     const char* deviceName, size_t size,
                     int constant, int global) {
        DeviceVarInfo vi;
        vi.hostVar        = hostVar;
        vi.deviceName     = deviceName ? deviceName : "";
        vi.size           = size;
        vi.isConstant     = constant;
        vi.isGlobal       = global;
        vi.fatCubinHandle = fatCubinHandle;
        deviceVars[hostVar] = std::move(vi);
    }

    // ==================== 查找（按需解析） ====================
    KernelInfo* FindByHost(const void* hostFun, bool needAttrs = false) {
        auto it = kernels.find((void*)hostFun);
        if (it == kernels.end()) return nullptr;
        if (needAttrs && !it->second.attributesParsed)
            ensureParsed(it->second);
        return &it->second;
    }

private:
    KernelRegistry() = default;

    // ==================== LZ4 解压 ====================
    static std::vector<uint8_t> decompressLZ4(const uint8_t* src, size_t srcLen) {
        size_t capacity = srcLen * 4;
        if (capacity > 32 * 1024 * 1024) capacity = 32 * 1024 * 1024;

        std::vector<uint8_t> dst(capacity);
        size_t si = 0, di = 0;

        while (si < srcLen) {
            uint8_t token = src[si++];

            // Literal
            size_t litLen = (token >> 4) & 0x0F;
            if (litLen == 15) {
                uint8_t b;
                do {
                    if (si >= srcLen) goto done;
                    b = src[si++];
                    litLen += b;
                } while (b == 255);
            }

            if (si + litLen > srcLen) goto done;
            while (di + litLen > capacity) {
                capacity *= 2;
                if (capacity > 64 * 1024 * 1024) goto done;
                dst.resize(capacity);
            }
            memcpy(dst.data() + di, src + si, litLen);
            si += litLen;
            di += litLen;

            if (si >= srcLen) break;

            // Match
            if (si + 2 > srcLen) goto done;
            size_t offset = src[si] | ((size_t)src[si + 1] << 8);
            si += 2;
            if (offset == 0 || offset > di) goto done;

            size_t matchLen = (token & 0x0F) + 4;
            if ((token & 0x0F) == 15) {
                uint8_t b;
                do {
                    if (si >= srcLen) goto done;
                    b = src[si++];
                    matchLen += b;
                } while (b == 255);
            }

            while (di + matchLen > capacity) {
                capacity *= 2;
                if (capacity > 64 * 1024 * 1024) goto done;
                dst.resize(capacity);
            }
            for (size_t j = 0; j < matchLen; j++)
                dst[di + j] = dst[di - offset + j];
            di += matchLen;
        }
    done:
        dst.resize(di);
        if (di >= 4 && dst[0] == 0x7f && dst[1] == 'E' && dst[2] == 'L' && dst[3] == 'F')
            return dst;
        return {};
    }

    // ==================== ELF 解析 ====================
    static void parseCubinElf(const uint8_t* data, size_t size, CubinData& cubin) {
        if (size < sizeof(Elf64_Ehdr)) return;
        auto* eh = (const Elf64_Ehdr*)data;

        if (eh->e_shoff == 0 || eh->e_shnum == 0) return;
        if (eh->e_shoff + (size_t)eh->e_shnum * sizeof(Elf64_Shdr) > size) return;
        if (eh->e_shstrndx >= eh->e_shnum) return;

        auto* sh = (const Elf64_Shdr*)(data + eh->e_shoff);
        if (sh[eh->e_shstrndx].sh_offset >= size) return;
        const char* shstr = (const char*)(data + sh[eh->e_shstrndx].sh_offset);

        int smVer = cubin.sm_major * 10 + cubin.sm_minor;

        for (uint16_t i = 0; i < eh->e_shnum; ++i) {
            if (sh[i].sh_offset + sh[i].sh_size > size) continue;
            const char* sname = shstr + sh[i].sh_name;

            if (strncmp(sname, ".nv.info.", 9) == 0) {
                std::string kname = sname + 9;
                cubin.kernelAttrs[kname].binaryVersion = smVer;
            }
            else if (strncmp(sname, ".text.", 6) == 0) {
                std::string kname = sname + 6;
                uint32_t regCount = (uint32_t)sh[i].sh_info & 0xFF;
                if (regCount > 0 && regCount <= 255)
                    cubin.kernelAttrs[kname].numRegs = regCount;
            }
            else if (strncmp(sname, ".nv.shared.", 11) == 0) {
                std::string kname = sname + 11;
                if (kname.find("reserved") != std::string::npos) continue;
                cubin.kernelAttrs[kname].sharedSizeBytes = sh[i].sh_size;
            }
            else if (strncmp(sname, ".nv.constant0.", 14) == 0) {
                std::string kname = sname + 14;
                cubin.kernelAttrs[kname].constSizeBytes = sh[i].sh_size;
            }
        }

        // 根据寄存器数计算 maxThreadsPerBlock
        for (auto& [kname, attrs] : cubin.kernelAttrs) {
            int hwMax = 1024;
            if (attrs.numRegs > 0) {
                int regsPerWarp = ((attrs.numRegs * 32 + 255) / 256) * 256;
                int maxWarps = (regsPerWarp > 0) ? (65536 / regsPerWarp) : 64;
                int maxByRegs = maxWarps * 32;
                if (maxByRegs < hwMax) hwMax = maxByRegs;
            }
            if (hwMax > 1024) hwMax = 1024;
            attrs.maxThreadsPerBlock = hwMax;
        }
    }

    // ==================== Fatbin 解析（SM 筛选 + 单 CUBIN 解压） ====================
    void parseFatbin(FatBinaryInfo& fb) {
        const uint8_t* base = resolve_fatbin_base(fb.fatCubin);
        if (!base) return;

        auto* h = (const FatbinHeader*)base;
        if (h->magic != FATBIN_MAGIC) return;

        uint32_t hdrSize = h->headerSize;
        if (hdrSize < sizeof(FatbinHeader)) hdrSize = sizeof(FatbinHeader);

        const uint8_t* p   = base + hdrSize;
        const uint8_t* end = base + hdrSize + h->dataSize;
        int targetSM = getTargetSM();
        std::vector<CubinCandidate> candidates;

        // 扫描所有 entry，只收集元信息
        while (p + 8 <= end) {
            uint16_t kind       = u16(p);
            uint32_t entryHdrSz = u32(p + 4);
            if (kind == 0 || entryHdrSz == 0 || p + entryHdrSz > end) break;

            uint64_t payloadSize = 0;
            uint32_t compressedSz = 0;
            uint8_t  smMajor = 0, smMinor = 0;

            if (entryHdrSz >= 0x10) std::memcpy(&payloadSize, p + 0x08, 8);
            if (entryHdrSz >= 0x14) compressedSz = u32(p + 0x10);
            if (entryHdrSz >= 0x1A) { smMajor = p[0x18]; smMinor = p[0x19]; }

            const uint8_t* payload = p + entryHdrSz;
            uint64_t actualSize = compressedSz > 0 ? compressedSz : payloadSize;

            if (kind == FATBIN_KIND_CUBIN && payloadSize > 0 && payload + actualSize <= end) {
                candidates.push_back({
                    (int)(smMajor * 10 + smMinor), payload,
                    (size_t)compressedSz, (size_t)payloadSize,
                    smMajor, smMinor
                });
            } else if (kind == FATBIN_KIND_PTX) {
                fb.ptxSmVersion = smMajor * 10 + smMinor;
            }

            p = payload + actualSize;
            p = (const uint8_t*)(((uintptr_t)p + 7) & ~7UL);
        }

        // 选择最佳 SM 匹配，只解压一个
        int bestIdx = selectBestCubin(candidates, targetSM);
        if (bestIdx < 0) return;

        auto& best = candidates[bestIdx];
        CubinData cubin;
        cubin.sm_major = best.smMajor;
        cubin.sm_minor = best.smMinor;

        if (best.compressedSz > 0) {
            auto elf = decompressLZ4(best.payload, best.compressedSz);
            if (elf.empty()) return;
            cubin.size = elf.size();
            parseCubinElf(elf.data(), elf.size(), cubin);
        } else {
            cubin.size = best.payloadSize;
            parseCubinElf(best.payload, best.payloadSize, cubin);
        }

        // 填补 ptxVersion
        for (auto& [kname, attrs] : cubin.kernelAttrs) {
            if (attrs.ptxVersion == 0 && fb.ptxSmVersion > 0)
                attrs.ptxVersion = fb.ptxSmVersion;
        }

        fb.cubins.push_back(std::move(cubin));
    }

    // ==================== 延迟解析 ====================
    void ensureParsed(KernelInfo& ki) {
        if (ki.attributesParsed) return;
        ki.attributesParsed = true;

        auto fbIt = fatBinaries.find(ki.fatCubinHandle);
        if (fbIt == fatBinaries.end()) return;

        FatBinaryInfo& fb = fbIt->second;
        if (!fb.parsed) {
            parseFatbin(fb);
            fb.parsed = true;
        }

        for (auto& cubin : fb.cubins) {
            auto it = cubin.kernelAttrs.find(ki.deviceName);
            if (it != cubin.kernelAttrs.end()) {
                ki.attributes = it->second;
                break;
            }
        }

        LOG_INFO(CUDART, "Parsed %s → regs=%d maxThreads=%d shared=%zu const=%zu binary=%d",
                 ki.readableName.c_str(), ki.attributes.numRegs,
                 ki.attributes.maxThreadsPerBlock, ki.attributes.sharedSizeBytes,
                 ki.attributes.constSizeBytes, ki.attributes.binaryVersion);
    }
    struct DeviceVarInfo {
        void*       hostVar = nullptr;
        std::string deviceName;
        size_t      size = 0;
        int         isConstant = 0;
        int         isGlobal = 0;
        void**      fatCubinHandle = nullptr;
    };
    std::map<void**, FatBinaryInfo> fatBinaries;
    std::map<void*, KernelInfo> kernels;
    std::map<void*, DeviceVarInfo>  deviceVars;
    thread_local static LaunchConfig launchConfig;
};

inline thread_local LaunchConfig KernelRegistry::launchConfig;
