#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <time.h>

#include "utils.hpp"

namespace fakecuda::host_timing {

enum class Mode {
    kNone,
    kTrace,
    kMeasure,
    kSleep,
};

enum class DispatchScope {
    kThread,
    kProcess,
    kHostMachine,
};

enum class ScheduleSurface {
    kSupported,
    kSemantic,
};

inline const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::kTrace:
            return "trace";
        case Mode::kMeasure:
            return "measure";
        case Mode::kSleep:
            return "sleep";
        case Mode::kNone:
        default:
            return "none";
    }
}

inline const char* dispatch_scope_name(DispatchScope scope) {
    switch (scope) {
        case DispatchScope::kHostMachine:
            return "host_machine";
        case DispatchScope::kProcess:
            return "process";
        case DispatchScope::kThread:
        default:
            return "thread";
    }
}

inline bool uses_shared_schedule_state(DispatchScope scope) {
    return scope == DispatchScope::kProcess || scope == DispatchScope::kHostMachine;
}

inline bool uses_synthetic_profile_shaping(Mode mode) {
    return mode == Mode::kTrace || mode == Mode::kSleep;
}

inline std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

inline long long us_to_ns(double value_us) {
    return static_cast<long long>(std::llround(value_us * 1000.0));
}

inline const char* canonicalize_api_name(const char* api) {
    if (api == nullptr) {
        return nullptr;
    }
    if (std::strcmp(api, "cudaEventRecordWithFlags") == 0) {
        return "cudaEventRecord";
    }
    if (std::strcmp(api, "cudaEventCreate") == 0) {
        return "cudaEventCreateWithFlags";
    }
    if (std::strcmp(api, "cudaStreamCreateWithFlags") == 0 ||
        std::strcmp(api, "cudaStreamCreateWithPriority") == 0) {
        return "cudaStreamCreate";
    }
    if (std::strcmp(api, "ncclBcast") == 0) {
        return "ncclBroadcast";
    }
    return api;
}

inline std::string canonicalize_api_name_copy(const std::string& api) {
    return std::string(canonicalize_api_name(api.c_str()));
}

inline bool is_host_timing_event(const char* api, const char* type) {
    api = canonicalize_api_name(api);
    if (type != nullptr) {
        if (std::strcmp(type, "kernel_launch") == 0 ||
            std::strcmp(type, "blas_compute") == 0 ||
            std::strcmp(type, "mem_copy") == 0 ||
            std::strcmp(type, "mem_alloc") == 0 ||
            std::strcmp(type, "nccl_collective") == 0) {
            return true;
        }
    }
    if (api == nullptr) {
        return false;
    }
    return std::strcmp(api, "cudaLaunchKernel") == 0 ||
           std::strcmp(api, "cudaMemcpy") == 0 ||
           std::strcmp(api, "cudaMemcpyAsync") == 0 ||
           std::strcmp(api, "cudaMalloc") == 0 ||
           std::strcmp(api, "cudaMallocAsync") == 0 ||
           std::strcmp(api, "cudaFree") == 0 ||
           std::strcmp(api, "cudaFreeAsync") == 0 ||
           std::strcmp(api, "cudaMemGetInfo") == 0 ||
           std::strcmp(api, "cudaDeviceSynchronize") == 0 ||
           std::strcmp(api, "cudaStreamSynchronize") == 0 ||
           std::strcmp(api, "cudaStreamWaitEvent") == 0 ||
           std::strcmp(api, "cudaEventRecord") == 0 ||
           std::strcmp(api, "cudaEventQuery") == 0 ||
           std::strcmp(api, "cudaEventCreateWithFlags") == 0 ||
           std::strcmp(api, "cudaEventDestroy") == 0 ||
           std::strcmp(api, "cudaEventSynchronize") == 0 ||
           std::strcmp(api, "cudaGetDevice") == 0 ||
           std::strcmp(api, "cudaGetDeviceCount") == 0 ||
           std::strcmp(api, "cudaGetDeviceProperties") == 0 ||
           std::strcmp(api, "cudaGetLastError") == 0 ||
           std::strcmp(api, "cudaSetDevice") == 0 ||
           std::strcmp(api, "cudaStreamCreate") == 0 ||
           std::strcmp(api, "cudaStreamDestroy") == 0 ||
           std::strcmp(api, "cublasSetStream_v2") == 0 ||
           std::strcmp(api, "cublasCreate_v2") == 0 ||
           std::strcmp(api, "cublasDestroy_v2") == 0 ||
           std::strcmp(api, "cublasSetMathMode") == 0 ||
           std::strcmp(api, "cublasSetWorkspace_v2") == 0 ||
           std::strcmp(api, "cublasLtCreate") == 0 ||
           std::strcmp(api, "cublasLtDestroy") == 0 ||
           std::strcmp(api, "cublasLtMatmulDescCreate") == 0 ||
           std::strcmp(api, "cublasLtMatmulDescSetAttribute") == 0 ||
           std::strcmp(api, "cublasLtMatmulDescDestroy") == 0 ||
           std::strcmp(api, "cublasLtMatmulPreferenceCreate") == 0 ||
           std::strcmp(api, "cublasLtMatmulPreferenceSetAttribute") == 0 ||
           std::strcmp(api, "cublasLtMatmulPreferenceDestroy") == 0 ||
           std::strcmp(api, "cublasLtMatrixLayoutCreate") == 0 ||
           std::strcmp(api, "cublasLtMatrixLayoutDestroy") == 0 ||
           std::strcmp(api, "cublasGemmEx") == 0 ||
           std::strcmp(api, "cublasGemmStridedBatchedEx") == 0 ||
           std::strcmp(api, "cublasGemmBatchedEx") == 0 ||
           std::strcmp(api, "cublasLtMatmul") == 0 ||
           std::strcmp(api, "ncclCommInitRank") == 0 ||
           std::strcmp(api, "ncclCommInitRankConfig") == 0 ||
           std::strcmp(api, "ncclCommDestroy") == 0 ||
           std::strcmp(api, "ncclCommGetAsyncError") == 0 ||
           std::strcmp(api, "ncclGroupStart") == 0 ||
           std::strcmp(api, "ncclGroupEnd") == 0 ||
           std::strcmp(api, "ncclGetVersion") == 0 ||
           std::strcmp(api, "ncclGetUniqueId") == 0 ||
           std::strcmp(api, "ncclAllGather") == 0 ||
           std::strcmp(api, "ncclAllReduce") == 0 ||
           std::strcmp(api, "ncclAllToAll") == 0 ||
           std::strcmp(api, "ncclAllToAllv") == 0 ||
           std::strcmp(api, "ncclBroadcast") == 0 ||
           std::strcmp(api, "ncclReduce") == 0 ||
           std::strcmp(api, "ncclReduceScatter") == 0 ||
           std::strcmp(api, "ncclSend") == 0 ||
           std::strcmp(api, "ncclRecv") == 0;
}

inline bool is_semantic_schedule_event(const char* api, const char* type) {
    api = canonicalize_api_name(api);
    if (type != nullptr) {
        if (std::strcmp(type, "kernel_launch") == 0 ||
            std::strcmp(type, "blas_compute") == 0 ||
            std::strcmp(type, "mem_copy") == 0 ||
            std::strcmp(type, "mem_alloc") == 0 ||
            std::strcmp(type, "nccl_collective") == 0) {
            return true;
        }
    }
    if (api == nullptr) {
        return false;
    }
    return std::strcmp(api, "cudaLaunchKernel") == 0 ||
           std::strcmp(api, "cudaMemcpy") == 0 ||
           std::strcmp(api, "cudaMemcpyAsync") == 0 ||
           std::strcmp(api, "cudaMalloc") == 0 ||
           std::strcmp(api, "cudaMallocAsync") == 0 ||
           std::strcmp(api, "cudaFree") == 0 ||
           std::strcmp(api, "cudaFreeAsync") == 0 ||
           std::strcmp(api, "cudaMemGetInfo") == 0 ||
           std::strcmp(api, "cudaDeviceSynchronize") == 0 ||
           std::strcmp(api, "cudaStreamSynchronize") == 0 ||
           std::strcmp(api, "cudaStreamWaitEvent") == 0 ||
           std::strcmp(api, "cudaEventRecord") == 0 ||
           std::strcmp(api, "cudaEventQuery") == 0 ||
           std::strcmp(api, "cudaEventSynchronize") == 0 ||
           std::strcmp(api, "cublasSetStream_v2") == 0 ||
           std::strcmp(api, "cublasGemmEx") == 0 ||
           std::strcmp(api, "cublasGemmStridedBatchedEx") == 0 ||
           std::strcmp(api, "cublasGemmBatchedEx") == 0 ||
           std::strcmp(api, "cublasLtMatmul") == 0 ||
           std::strcmp(api, "ncclCommInitRank") == 0 ||
           std::strcmp(api, "ncclCommInitRankConfig") == 0 ||
           std::strcmp(api, "ncclAllGather") == 0 ||
           std::strcmp(api, "ncclAllReduce") == 0 ||
           std::strcmp(api, "ncclAllToAll") == 0 ||
           std::strcmp(api, "ncclAllToAllv") == 0 ||
           std::strcmp(api, "ncclBroadcast") == 0 ||
           std::strcmp(api, "ncclReduce") == 0 ||
           std::strcmp(api, "ncclReduceScatter") == 0 ||
           std::strcmp(api, "ncclSend") == 0 ||
           std::strcmp(api, "ncclRecv") == 0;
}

inline std::string normalize_profile_key(std::string key) {
    if (key == "default" || key.rfind("type:", 0) == 0 || key.rfind("lib:", 0) == 0) {
        return key;
    }
    if (key.rfind("threadstartocc:", 0) == 0) {
        const std::string prefix = "threadstartocc:";
        const std::size_t occurrence_delim = key.rfind('#');
        if (occurrence_delim == std::string::npos || occurrence_delim <= prefix.size()) {
            return key;
        }
        const std::string api = key.substr(prefix.size(), occurrence_delim - prefix.size());
        return prefix + canonicalize_api_name_copy(api) + key.substr(occurrence_delim);
    }
    if (key.rfind("threadstart:", 0) == 0) {
        const std::string prefix = "threadstart:";
        return prefix + canonicalize_api_name_copy(key.substr(prefix.size()));
    }
    if (key.rfind("pairocc:", 0) == 0 || key.rfind("pair:", 0) == 0) {
        const bool has_occurrence = key.rfind("pairocc:", 0) == 0;
        const std::string prefix = has_occurrence ? "pairocc:" : "pair:";
        const std::size_t body_start = prefix.size();
        const std::size_t arrow = key.find("->", body_start);
        if (arrow == std::string::npos) {
            return key;
        }
        std::size_t current_end = key.size();
        std::string suffix;
        if (has_occurrence) {
            const std::size_t occurrence_delim = key.rfind('#');
            if (occurrence_delim == std::string::npos || occurrence_delim <= arrow + 2) {
                return key;
            }
            current_end = occurrence_delim;
            suffix = key.substr(occurrence_delim);
        }
        const std::string previous_api = key.substr(body_start, arrow - body_start);
        const std::string current_api = key.substr(arrow + 2, current_end - (arrow + 2));
        return prefix + canonicalize_api_name_copy(previous_api) + "->" +
               canonicalize_api_name_copy(current_api) + suffix;
    }
    return canonicalize_api_name_copy(key);
}

class HostTimingConfig {
public:
    static const HostTimingConfig& instance() {
        static HostTimingConfig cfg;
        return cfg;
    }

    Mode mode() const {
        return mode_;
    }

    DispatchScope dispatch_scope() const {
        return dispatch_scope_;
    }

    ScheduleSurface schedule_surface() const {
        return schedule_surface_;
    }

    long long lookup_delay_ns(
        const char* previous_api,
        const char* previous_type,
        const char* previous_mod_name,
        const char* current_api,
        std::size_t pair_occurrence_index
    ) const {
        if (mode_ == Mode::kNone) {
            return 0LL;
        }
        previous_api = canonicalize_api_name(previous_api);
        current_api = canonicalize_api_name(current_api);
        if (previous_api != nullptr && current_api != nullptr) {
            auto occurrence_it = pair_occurrence_delays_ns_.find(
                std::string("pairocc:") + previous_api + "->" + current_api + "#" + std::to_string(pair_occurrence_index)
            );
            if (occurrence_it != pair_occurrence_delays_ns_.end()) {
                return occurrence_it->second;
            }
            auto it = pair_delays_ns_.find(
                std::string("pair:") + previous_api + "->" + current_api
            );
            if (it != pair_delays_ns_.end()) {
                return it->second;
            }
        }
        if (previous_api != nullptr) {
            auto it = delays_ns_.find(std::string(previous_api));
            if (it != delays_ns_.end()) {
                return it->second;
            }
        }
        if (previous_type != nullptr) {
            auto it = delays_ns_.find(std::string("type:") + previous_type);
            if (it != delays_ns_.end()) {
                return it->second;
            }
        }
        if (previous_mod_name != nullptr) {
            auto it = delays_ns_.find(std::string("lib:") + previous_mod_name);
            if (it != delays_ns_.end()) {
                return it->second;
            }
        }
        return default_delay_ns_;
    }

    long long lookup_thread_start_delay_ns(
        const char* current_api,
        std::size_t occurrence_index
    ) const {
        if (mode_ == Mode::kNone) {
            return 0LL;
        }
        current_api = canonicalize_api_name(current_api);
        if (current_api != nullptr) {
            auto occurrence_it = thread_start_occurrence_delays_ns_.find(
                std::string("threadstartocc:") + current_api + "#" + std::to_string(occurrence_index)
            );
            if (occurrence_it != thread_start_occurrence_delays_ns_.end()) {
                return occurrence_it->second;
            }
            auto it = thread_start_delays_ns_.find(std::string("threadstart:") + current_api);
            if (it != thread_start_delays_ns_.end()) {
                return it->second;
            }
        }
        return 0LL;
    }

private:
    HostTimingConfig() {
        initialize();
    }

    void initialize() {
        const char* raw_mode = std::getenv("FAKECUDA_HOST_TIMING_MODE");
        std::string mode_value = trim_copy(raw_mode ? raw_mode : "");
        if (mode_value == "trace") {
            mode_ = Mode::kTrace;
        } else if (mode_value == "measure") {
            mode_ = Mode::kMeasure;
        } else if (mode_value == "sleep") {
            mode_ = Mode::kSleep;
        } else {
            mode_ = Mode::kNone;
        }

        const char* raw_dispatch_scope = std::getenv("FAKECUDA_HOST_TIMING_DISPATCH_SCOPE");
        std::string dispatch_scope_value = trim_copy(raw_dispatch_scope ? raw_dispatch_scope : "");
        if (dispatch_scope_value == "host_machine") {
            dispatch_scope_ = DispatchScope::kHostMachine;
        } else if (dispatch_scope_value == "process") {
            dispatch_scope_ = DispatchScope::kProcess;
        } else {
            dispatch_scope_ = DispatchScope::kThread;
        }

        const char* raw_schedule_surface = std::getenv("FAKECUDA_HOST_TIMING_SCHEDULE_SURFACE");
        std::string schedule_surface_value = trim_copy(raw_schedule_surface ? raw_schedule_surface : "");
        if (schedule_surface_value == "semantic") {
            schedule_surface_ = ScheduleSurface::kSemantic;
        } else {
            schedule_surface_ = ScheduleSurface::kSupported;
        }

        const char* raw_default_us = std::getenv("FAKECUDA_HOST_TIMING_DEFAULT_US");
        if (raw_default_us != nullptr && raw_default_us[0] != '\0') {
            default_delay_ns_ = us_to_ns(std::atof(raw_default_us));
        }

        const char* raw_profile = std::getenv("FAKECUDA_HOST_TIMING_PROFILE");
        if (raw_profile != nullptr && raw_profile[0] != '\0') {
            load_profile(raw_profile);
        }
    }

    void load_profile(const char* path) {
        std::ifstream handle(path);
        if (!handle.is_open()) {
            return;
        }
        std::string line;
        while (std::getline(handle, line)) {
            std::string trimmed = trim_copy(line);
            if (trimmed.empty() || trimmed[0] == '#') {
                continue;
            }
            size_t delimiter = trimmed.find('=');
            if (delimiter == std::string::npos) {
                continue;
            }
            std::string key = trim_copy(trimmed.substr(0, delimiter));
            std::string value_text = trim_copy(trimmed.substr(delimiter + 1));
            if (key.empty() || value_text.empty()) {
                continue;
            }
            key = normalize_profile_key(std::move(key));
            double delay_us = std::atof(value_text.c_str());
            if (key == "default") {
                default_delay_ns_ = us_to_ns(delay_us);
            } else if (key.rfind("threadstartocc:", 0) == 0) {
                thread_start_occurrence_delays_ns_[key] = us_to_ns(delay_us);
            } else if (key.rfind("threadstart:", 0) == 0) {
                thread_start_delays_ns_[key] = us_to_ns(delay_us);
            } else if (key.rfind("pairocc:", 0) == 0) {
                pair_occurrence_delays_ns_[key] = us_to_ns(delay_us);
            } else if (key.rfind("pair:", 0) == 0) {
                pair_delays_ns_[key] = us_to_ns(delay_us);
            } else {
                delays_ns_[key] = us_to_ns(delay_us);
            }
        }
    }

    Mode mode_ = Mode::kNone;
    DispatchScope dispatch_scope_ = DispatchScope::kThread;
    ScheduleSurface schedule_surface_ = ScheduleSurface::kSupported;
    long long default_delay_ns_ = 0LL;
    std::unordered_map<std::string, long long> thread_start_occurrence_delays_ns_;
    std::unordered_map<std::string, long long> thread_start_delays_ns_;
    std::unordered_map<std::string, long long> pair_occurrence_delays_ns_;
    std::unordered_map<std::string, long long> pair_delays_ns_;
    std::unordered_map<std::string, long long> delays_ns_;
};

inline long long real_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

inline bool env_value_enabled(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "yes") == 0 ||
           std::strcmp(value, "YES") == 0 ||
           std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "ON") == 0;
}

inline bool env_value_disabled(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") == 0 ||
           std::strcmp(value, "false") == 0 ||
           std::strcmp(value, "FALSE") == 0 ||
           std::strcmp(value, "no") == 0 ||
           std::strcmp(value, "NO") == 0 ||
           std::strcmp(value, "off") == 0 ||
           std::strcmp(value, "OFF") == 0;
}

inline bool logical_trace_timestamp_overhead_adjustment_enabled() {
    static const bool enabled = [] {
        const char* raw_disable =
            std::getenv("MAYA_DISABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT");
        if (env_value_enabled(raw_disable)) {
            return false;
        }
        raw_disable =
            std::getenv("FLEXSIM_MAYA_DISABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT");
        if (env_value_enabled(raw_disable)) {
            return false;
        }

        const char* raw_enable =
            std::getenv("MAYA_ENABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT");
        if (raw_enable != nullptr && raw_enable[0] != '\0') {
            if (env_value_disabled(raw_enable)) {
                return false;
            }
            if (env_value_enabled(raw_enable)) {
                return true;
            }
        }
        raw_enable =
            std::getenv("FLEXSIM_MAYA_ENABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT");
        if (raw_enable != nullptr && raw_enable[0] != '\0') {
            if (env_value_disabled(raw_enable)) {
                return false;
            }
            if (env_value_enabled(raw_enable)) {
                return true;
            }
        }

        return true;
    }();
    return enabled;
}

inline long long trace_timestamp_now_ns_for_mode(long long raw_now_ns, Mode mode) {
    if (mode == Mode::kMeasure && logical_trace_timestamp_overhead_adjustment_enabled()) {
        return fakecuda::trace::logical_host_time_ns(raw_now_ns);
    }
    return raw_now_ns;
}

inline std::mutex& global_schedule_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline long long& trace_origin_ns() {
    static long long value = 0LL;
    return value;
}

inline std::unordered_map<std::string, std::size_t>& thread_start_occurrence_counts() {
    static std::unordered_map<std::string, std::size_t> counts;
    return counts;
}

inline std::atomic<std::size_t>& step_epoch() {
    static std::atomic<std::size_t> epoch{0};
    return epoch;
}

struct ThreadTimingState {
    long long last_emitted_ns = 0LL;
    long long previous_semantic_dispatch_ns = 0LL;
    std::string previous_semantic_api;
    std::string previous_semantic_type;
    std::string previous_semantic_mod_name;
    std::unordered_map<std::string, std::size_t> pair_occurrence_counts;
    std::size_t observed_step_epoch = 0;
};

inline ThreadTimingState& current_thread_state() {
    thread_local ThreadTimingState state;
    return state;
}

struct ProcessTimingState {
    long long last_emitted_ns = 0LL;
    long long previous_semantic_dispatch_ns = 0LL;
    std::string previous_semantic_api;
    std::string previous_semantic_type;
    std::string previous_semantic_mod_name;
    std::unordered_map<std::string, std::size_t> pair_occurrence_counts;
    std::size_t observed_step_epoch = 0;
};

inline ProcessTimingState& current_process_state() {
    static ProcessTimingState state;
    return state;
}

inline void clear_thread_state() {
    ThreadTimingState& state = current_thread_state();
    state.last_emitted_ns = 0LL;
    state.previous_semantic_dispatch_ns = 0LL;
    state.previous_semantic_api.clear();
    state.previous_semantic_type.clear();
    state.previous_semantic_mod_name.clear();
    state.pair_occurrence_counts.clear();
    state.observed_step_epoch = 0;
}

inline void clear_process_state() {
    ProcessTimingState& state = current_process_state();
    state.last_emitted_ns = 0LL;
    state.previous_semantic_dispatch_ns = 0LL;
    state.previous_semantic_api.clear();
    state.previous_semantic_type.clear();
    state.previous_semantic_mod_name.clear();
    state.pair_occurrence_counts.clear();
    state.observed_step_epoch = 0;
}

inline void begin_step_epoch() {
    {
        std::lock_guard<std::mutex> lock(global_schedule_mutex());
        trace_origin_ns() = 0LL;
        thread_start_occurrence_counts().clear();
        clear_process_state();
    }
    step_epoch().fetch_add(1, std::memory_order_release);
}

inline void maybe_block_measure_before_trace(
    const char* api,
    const char* type,
    const char* mod_name
) {
    const HostTimingConfig& cfg = HostTimingConfig::instance();
    if (cfg.mode() != Mode::kMeasure) {
        return;
    }
    (void)api;
    (void)type;
    (void)mod_name;
}

inline bool should_defer_trace_until_wrapper_exit() {
    return HostTimingConfig::instance().mode() == Mode::kMeasure;
}

inline long long reserve_trace_timestamp_us(
    const char* api,
    const char* type,
    const char* mod_name
) {
    const long long raw_now_ns = real_now_ns();
    const HostTimingConfig& cfg = HostTimingConfig::instance();
    const long long now_ns = trace_timestamp_now_ns_for_mode(raw_now_ns, cfg.mode());
    const char* canonical_api = canonicalize_api_name(api);
    const bool recorded_timing_event = is_host_timing_event(canonical_api, type);
    const bool scheduled_event = cfg.schedule_surface() == ScheduleSurface::kSemantic
        ? is_semantic_schedule_event(canonical_api, type)
        : recorded_timing_event;
    if (cfg.mode() == Mode::kNone) {
        return now_ns / 1000LL;
    }

    if (cfg.mode() == Mode::kMeasure) {
        if (uses_shared_schedule_state(cfg.dispatch_scope())) {
            const std::size_t active_step_epoch = step_epoch().load(std::memory_order_acquire);
            std::lock_guard<std::mutex> lock(global_schedule_mutex());
            if (trace_origin_ns() <= 0LL) {
                trace_origin_ns() = now_ns;
            }
            ProcessTimingState& state = current_process_state();
            if (state.observed_step_epoch != active_step_epoch) {
                clear_process_state();
                state.observed_step_epoch = active_step_epoch;
                if (trace_origin_ns() <= 0LL) {
                    trace_origin_ns() = now_ns;
                }
            }

            const long long dispatch_ns = std::max(now_ns, state.last_emitted_ns);
            state.last_emitted_ns = dispatch_ns;
            if (scheduled_event) {
                state.previous_semantic_dispatch_ns = dispatch_ns;
                state.previous_semantic_api = canonical_api ? canonical_api : "";
                state.previous_semantic_type = type ? type : "";
                state.previous_semantic_mod_name = mod_name ? mod_name : "";
            }
            return dispatch_ns / 1000LL;
        }

        ThreadTimingState& thread_state = current_thread_state();
        const std::size_t active_step_epoch = step_epoch().load(std::memory_order_acquire);
        if (thread_state.observed_step_epoch != active_step_epoch) {
            clear_thread_state();
            thread_state.observed_step_epoch = active_step_epoch;
        }
        {
            std::lock_guard<std::mutex> lock(global_schedule_mutex());
            if (trace_origin_ns() <= 0LL) {
                trace_origin_ns() = now_ns;
            }
        }

        const long long dispatch_ns = std::max(now_ns, thread_state.last_emitted_ns);
        thread_state.last_emitted_ns = dispatch_ns;
        if (scheduled_event) {
            thread_state.previous_semantic_dispatch_ns = dispatch_ns;
            thread_state.previous_semantic_api = canonical_api ? canonical_api : "";
            thread_state.previous_semantic_type = type ? type : "";
            thread_state.previous_semantic_mod_name = mod_name ? mod_name : "";
        }
        return dispatch_ns / 1000LL;
    }

    if (uses_shared_schedule_state(cfg.dispatch_scope())) {
        const std::size_t active_step_epoch = step_epoch().load(std::memory_order_acquire);
        std::lock_guard<std::mutex> lock(global_schedule_mutex());
        if (trace_origin_ns() <= 0LL) {
            trace_origin_ns() = now_ns;
        }
        ProcessTimingState& state = current_process_state();
        if (state.observed_step_epoch != active_step_epoch) {
            clear_process_state();
            state.observed_step_epoch = active_step_epoch;
            if (trace_origin_ns() <= 0LL) {
                trace_origin_ns() = now_ns;
            }
        }

        if (!scheduled_event) {
            long long dispatch_ns = state.last_emitted_ns > 0LL ? state.last_emitted_ns : trace_origin_ns();
            if (cfg.mode() == Mode::kSleep) {
                dispatch_ns = std::max(dispatch_ns, now_ns);
            }
            state.last_emitted_ns = dispatch_ns;
            return dispatch_ns / 1000LL;
        }

        const bool has_previous_dispatch = state.previous_semantic_dispatch_ns > 0LL;
        const std::string pair_key = has_previous_dispatch && canonical_api != nullptr
            ? state.previous_semantic_api + "->" + canonical_api
            : std::string();
        const std::size_t pair_occurrence_index = has_previous_dispatch
            ? state.pair_occurrence_counts[pair_key]
            : 0;
        const long long delay_ns = has_previous_dispatch
            ? std::max(
                  0LL,
                  cfg.lookup_delay_ns(
                      state.previous_semantic_api.empty() ? nullptr : state.previous_semantic_api.c_str(),
                      state.previous_semantic_type.empty() ? nullptr : state.previous_semantic_type.c_str(),
                      state.previous_semantic_mod_name.empty() ? nullptr : state.previous_semantic_mod_name.c_str(),
                      canonical_api,
                      pair_occurrence_index
                  )
              )
            : 0LL;
        long long scheduled_dispatch_ns = now_ns;
        if (has_previous_dispatch) {
            scheduled_dispatch_ns = state.previous_semantic_dispatch_ns + delay_ns;
        } else {
            const std::string thread_start_key = canonical_api ? std::string(canonical_api) : std::string();
            auto& start_counts = thread_start_occurrence_counts();
            const std::size_t thread_start_occurrence_index = start_counts[thread_start_key];
            start_counts[thread_start_key] = thread_start_occurrence_index + 1;
            const long long thread_start_delay_ns = std::max(
                0LL,
                cfg.lookup_thread_start_delay_ns(canonical_api, thread_start_occurrence_index)
            );
            scheduled_dispatch_ns = trace_origin_ns() + thread_start_delay_ns;
        }

        long long dispatch_ns = std::max(scheduled_dispatch_ns, state.last_emitted_ns);
        if (cfg.mode() == Mode::kSleep) {
            dispatch_ns = std::max(now_ns, scheduled_dispatch_ns);
            dispatch_ns = std::max(dispatch_ns, state.last_emitted_ns);
            if (dispatch_ns > now_ns) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(dispatch_ns - now_ns));
                dispatch_ns = real_now_ns();
            }
        }

        state.last_emitted_ns = dispatch_ns;
        state.previous_semantic_dispatch_ns = dispatch_ns;
        state.previous_semantic_api = canonical_api ? canonical_api : "";
        state.previous_semantic_type = type ? type : "";
        state.previous_semantic_mod_name = mod_name ? mod_name : "";
        if (has_previous_dispatch) {
            state.pair_occurrence_counts[pair_key] = pair_occurrence_index + 1;
        }

        return dispatch_ns / 1000LL;
    }

    ThreadTimingState& thread_state = current_thread_state();
    const std::size_t active_step_epoch = step_epoch().load(std::memory_order_acquire);
    if (thread_state.observed_step_epoch != active_step_epoch) {
        clear_thread_state();
        thread_state.observed_step_epoch = active_step_epoch;
    }
    {
        std::lock_guard<std::mutex> lock(global_schedule_mutex());
        if (trace_origin_ns() <= 0LL) {
            trace_origin_ns() = now_ns;
        }
    }

    if (!scheduled_event) {
        long long dispatch_ns = thread_state.last_emitted_ns > 0LL ? thread_state.last_emitted_ns : trace_origin_ns();
        if (cfg.mode() == Mode::kSleep) {
            dispatch_ns = std::max(dispatch_ns, now_ns);
        }
        thread_state.last_emitted_ns = dispatch_ns;
        return dispatch_ns / 1000LL;
    }

    const bool has_previous_dispatch = thread_state.previous_semantic_dispatch_ns > 0LL;
    const std::string pair_key = has_previous_dispatch && canonical_api != nullptr
        ? thread_state.previous_semantic_api + "->" + canonical_api
        : std::string();
    const std::size_t pair_occurrence_index = has_previous_dispatch
        ? thread_state.pair_occurrence_counts[pair_key]
        : 0;
    const long long delay_ns = has_previous_dispatch
        ? std::max(
              0LL,
              cfg.lookup_delay_ns(
                  thread_state.previous_semantic_api.empty() ? nullptr : thread_state.previous_semantic_api.c_str(),
                  thread_state.previous_semantic_type.empty() ? nullptr : thread_state.previous_semantic_type.c_str(),
                  thread_state.previous_semantic_mod_name.empty() ? nullptr : thread_state.previous_semantic_mod_name.c_str(),
                  canonical_api,
                  pair_occurrence_index
              )
          )
        : 0LL;
    long long scheduled_dispatch_ns = now_ns;
    if (has_previous_dispatch) {
        scheduled_dispatch_ns = thread_state.previous_semantic_dispatch_ns + delay_ns;
    } else {
        std::size_t thread_start_occurrence_index = 0;
        long long trace_origin_snapshot = now_ns;
        {
            std::lock_guard<std::mutex> lock(global_schedule_mutex());
            trace_origin_snapshot = trace_origin_ns();
            const std::string thread_start_key = canonical_api ? std::string(canonical_api) : std::string();
            auto& start_counts = thread_start_occurrence_counts();
            thread_start_occurrence_index = start_counts[thread_start_key];
            start_counts[thread_start_key] = thread_start_occurrence_index + 1;
        }
        const long long thread_start_delay_ns = std::max(
            0LL,
            cfg.lookup_thread_start_delay_ns(canonical_api, thread_start_occurrence_index)
        );
        scheduled_dispatch_ns = trace_origin_snapshot + thread_start_delay_ns;
    }

    long long dispatch_ns = std::max(scheduled_dispatch_ns, thread_state.last_emitted_ns);
    if (cfg.mode() == Mode::kSleep) {
        dispatch_ns = std::max(now_ns, scheduled_dispatch_ns);
        dispatch_ns = std::max(dispatch_ns, thread_state.last_emitted_ns);
        if (dispatch_ns > now_ns) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(dispatch_ns - now_ns));
            dispatch_ns = real_now_ns();
        }
    }

    thread_state.last_emitted_ns = dispatch_ns;
    thread_state.previous_semantic_dispatch_ns = dispatch_ns;
    thread_state.previous_semantic_api = canonical_api ? canonical_api : "";
    thread_state.previous_semantic_type = type ? type : "";
    thread_state.previous_semantic_mod_name = mod_name ? mod_name : "";
    if (has_previous_dispatch) {
        thread_state.pair_occurrence_counts[pair_key] = pair_occurrence_index + 1;
    }

    return dispatch_ns / 1000LL;
}

}  // namespace fakecuda::host_timing
