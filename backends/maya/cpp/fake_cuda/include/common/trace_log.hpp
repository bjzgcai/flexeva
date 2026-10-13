#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>
#include <string>
#include "host_timing.hpp"
#include "capture_mode.hpp"
#include "../device/json.hpp"

inline bool trace_binary_enabled() {
    static const bool enabled = [] {
        const char* format = std::getenv("FAKECUDA_TRACE_FORMAT");
        return format && std::strcmp(format, "msgpack") == 0;
    }();
    return enabled;
}

// Thread-safe JSONL/MessagePack trace logger for fake-cuda
// Output format matches Frida hook traces with optional extra fields:
// {ts, pid, tid, mod, api, type, ...extras}
// Enable: FAKECUDA_TRACE=1
// Output: inherited FAKECUDA_TRACE_FD, or FAKECUDA_TRACE_PATH (diagnostic files).

inline const char* canonical_trace_api_name(const char* api) {
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

inline std::string trace_json_escape(const char* value) {
    if (!value) {
        return "";
    }
    std::string escaped;
    escaped.reserve(strlen(value) + 8);
    for (const unsigned char ch : std::string(value)) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(ch));
                    escaped += buf;
                } else {
                    escaped.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return escaped;
}

struct WrapperTimingEnrichment {
    long long emitted_ts_us;
    std::string payload_json;
    bool timing_added = false;
};

inline const char* wrapper_runtime_contract(const char* api, const char* type) {
    api = canonical_trace_api_name(api);
    if (type != nullptr) {
        if (strcmp(type, "kernel_launch") == 0 ||
            strcmp(type, "blas_compute") == 0 ||
            strcmp(type, "nccl_collective") == 0) {
            return "dispatch_only";
        }
    }
    if (api == nullptr) {
        return nullptr;
    }
    if (strcmp(api, "cudaMemcpyAsync") == 0 ||
        strcmp(api, "cudaMallocAsync") == 0 ||
        strcmp(api, "cudaFreeAsync") == 0 ||
        strcmp(api, "cudaEventRecord") == 0 ||
        strcmp(api, "cudaStreamWaitEvent") == 0 ||
        strcmp(api, "cublasSetStream_v2") == 0) {
        return "dispatch_only";
    }
    if (strcmp(api, "cudaDeviceSynchronize") == 0 ||
        strcmp(api, "cudaStreamSynchronize") == 0 ||
        strcmp(api, "cudaEventSynchronize") == 0 ||
        strcmp(api, "cudaMemcpy") == 0 ||
        strcmp(api, "cudaMalloc") == 0 ||
        strcmp(api, "cudaFree") == 0 ||
        strcmp(api, "cudaMemGetInfo") == 0 ||
        strcmp(api, "ncclCommCount") == 0 ||
        strcmp(api, "ncclCommInitRank") == 0 ||
        strcmp(api, "ncclCommInitRankConfig") == 0 ||
        strcmp(api, "ncclCommUserRank") == 0) {
        return "direct_runtime";
    }
    return nullptr;
}

inline bool should_emit_boundary_origin_provenance(const char* api) {
    api = canonical_trace_api_name(api);
    return api != nullptr &&
           (strcmp(api, "__cudaPopCallConfiguration") == 0 ||
            strcmp(api, "__cudaPushCallConfiguration") == 0 ||
            strcmp(api, "cudaGetDevice") == 0 ||
            strcmp(api, "cudaGetLastError") == 0 ||
            strcmp(api, "cudaLaunchKernel") == 0 ||
            strcmp(api, "cublasSetStream_v2") == 0 ||
            strcmp(api, "cublasGemmEx") == 0 ||
            strcmp(api, "cublasGemmStridedBatchedEx") == 0 ||
            strcmp(api, "cudaEventRecord") == 0 ||
            strcmp(api, "cudaStreamWaitEvent") == 0);
}

inline bool should_emit_prepop_launch_neighborhood_provenance(const char* api) {
    api = canonical_trace_api_name(api);
    return api != nullptr &&
           (strcmp(api, "cudaGetLastError") == 0 ||
            strcmp(api, "cublasGemmEx") == 0 ||
            strcmp(api, "cublasGemmStridedBatchedEx") == 0 ||
            strcmp(api, "__cudaPushCallConfiguration") == 0 ||
            strcmp(api, "__cudaPopCallConfiguration") == 0);
}

inline bool launch_boundary_visibility_env_value_enabled(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return strcmp(value, "1") == 0 ||
           strcmp(value, "true") == 0 ||
           strcmp(value, "TRUE") == 0 ||
           strcmp(value, "yes") == 0 ||
           strcmp(value, "YES") == 0 ||
           strcmp(value, "on") == 0 ||
           strcmp(value, "ON") == 0;
}

inline bool launch_boundary_visibility_diagnostics_enabled() {
    static const bool enabled = [] {
        const char* raw = getenv("MAYA_ENABLE_LAUNCH_BOUNDARY_VISIBILITY_DIAGNOSTICS");
        if (raw == nullptr || raw[0] == '\0') {
            raw = getenv("FLEXSIM_MAYA_ENABLE_LAUNCH_BOUNDARY_VISIBILITY_DIAGNOSTICS");
        }
        return launch_boundary_visibility_env_value_enabled(raw);
    }();
    return enabled;
}

inline std::string format_duration_us(double duration_us) {
    char duration_buf[64];
    snprintf(duration_buf, sizeof(duration_buf), "%.3f", std::max(duration_us, 0.0));
    return std::string(duration_buf);
}

inline std::string append_json_fragment(
    const std::string& payload_json,
    const std::string& fragment_with_optional_leading_comma
) {
    if (fragment_with_optional_leading_comma.empty()) {
        return payload_json;
    }
    std::string fragment = fragment_with_optional_leading_comma;
    if (!fragment.empty() && fragment[0] == ',') {
        fragment.erase(0, 1);
    }
    if (payload_json.empty()) {
        return fragment;
    }
    return payload_json + "," + fragment;
}

inline const char* prepop_launch_neighborhood_row_role(const char* api) {
    api = canonical_trace_api_name(api);
    if (api == nullptr) {
        return "unresolved";
    }
    if (strcmp(api, "__cudaPushCallConfiguration") == 0) {
        return "launch_config_push_context";
    }
    if (strcmp(api, "__cudaPopCallConfiguration") == 0) {
        return "launch_config_pop_prepop_endpoint";
    }
    if (strcmp(api, "cudaGetLastError") == 0 ||
        strcmp(api, "cublasGemmEx") == 0 ||
        strcmp(api, "cublasGemmStridedBatchedEx") == 0) {
        return "prepop_predecessor_candidate";
    }
    return "unresolved";
}

inline std::string prepop_launch_neighborhood_family(const char* api) {
    api = canonical_trace_api_name(api);
    if (api == nullptr) {
        return "unresolved";
    }
    if (strcmp(api, "__cudaPopCallConfiguration") == 0) {
        return "__cudaPopCallConfiguration -> cudaLaunchKernel";
    }
    if (strcmp(api, "__cudaPushCallConfiguration") == 0) {
        return "__cudaPushCallConfiguration -> __cudaPopCallConfiguration";
    }
    if (strcmp(api, "cudaGetLastError") == 0 ||
        strcmp(api, "cublasGemmEx") == 0 ||
        strcmp(api, "cublasGemmStridedBatchedEx") == 0) {
        return std::string(api) + " -> __cudaPopCallConfiguration";
    }
    return "unresolved";
}

inline std::string prepop_launch_neighborhood_provenance_fragment(const char* api) {
    api = canonical_trace_api_name(api);
    if (!should_emit_prepop_launch_neighborhood_provenance(api)) {
        return "";
    }
    const bool is_pop = api != nullptr && strcmp(api, "__cudaPopCallConfiguration") == 0;
    const bool is_push = api != nullptr && strcmp(api, "__cudaPushCallConfiguration") == 0;
    const bool is_predecessor = api != nullptr &&
        (strcmp(api, "cudaGetLastError") == 0 ||
         strcmp(api, "cublasGemmEx") == 0 ||
         strcmp(api, "cublasGemmStridedBatchedEx") == 0);

    std::string fragment =
        ",\"prepop_launch_neighborhood_schema_version\":\"prepop_launch_gemm_visibility_row_evidence_v1\"" +
        std::string(",\"prepop_launch_neighborhood_opt_in_flag\":true") +
        ",\"prepop_launch_neighborhood_api\":\"" + trace_json_escape(api) + "\"" +
        ",\"prepop_launch_neighborhood_row_role\":\"" +
            trace_json_escape(prepop_launch_neighborhood_row_role(api)) + "\"" +
        ",\"prepop_launch_neighborhood_boundary_family\":\"" +
            trace_json_escape(prepop_launch_neighborhood_family(api).c_str()) + "\"" +
        ",\"prepop_launch_neighborhood_visibility_kind\":\"mixed_or_unresolved\"" +
        ",\"prepop_launch_neighborhood_visibility_status\":\"structural_row_label_only_unresolved\"" +
        ",\"prepop_launch_neighborhood_classification_basis\":\"" +
            "fakecuda_raw_row_structural_label_no_adjacent_gap_split\"" +
        ",\"prepop_launch_neighborhood_mechanical_split_status\":\"unavailable\"" +
        ",\"prepop_launch_neighborhood_mechanical_split_unavailable_reason\":\"" +
            "producer_does_not_measure_or_split_prepop_hostdelay_gap\"" +
        ",\"prepop_launch_neighborhood_count_once_status\":\"unavailable\"" +
        ",\"prepop_launch_neighborhood_count_once_unavailable_reason\":\"" +
            "requires_collated_interval_and_figure6_count_once_ledger_not_available_in_raw_producer\"" +
        ",\"prepop_launch_neighborhood_wait_map_safety_status\":\"unavailable\"" +
        ",\"prepop_launch_neighborhood_wait_map_safety_unavailable_reason\":\"" +
            "requires_replay_wait_map_and_stream_dependency_ledger_not_available_in_raw_producer\"" +
        ",\"prepop_launch_neighborhood_double_counting_overlap_status\":\"unavailable\"" +
        ",\"prepop_launch_neighborhood_runtime_substitution_status\":\"forbidden\"" +
        ",\"prepop_launch_neighborhood_endpoint_timestamp_substitution_status\":\"forbidden\"" +
        ",\"prepop_launch_neighborhood_repair_ready\":false" +
        ",\"prepop_launch_neighborhood_safe_to_use_as_repair_evidence\":false" +
        ",\"prepop_launch_neighborhood_safe_to_use_as_subtraction_delta\":false";
    if (is_predecessor) {
        fragment +=
            ",\"prepop_launch_neighborhood_candidate_predecessor_api\":\"" +
            trace_json_escape(api) + "\"" +
            ",\"prepop_launch_neighborhood_expected_successor_api\":\"__cudaPopCallConfiguration\"";
    } else if (is_push) {
        fragment +=
            ",\"prepop_launch_neighborhood_expected_successor_api\":\"__cudaPopCallConfiguration\"";
    } else if (is_pop) {
        fragment +=
            ",\"prepop_launch_neighborhood_expected_successor_api\":\"cudaLaunchKernel\"";
    }
    return fragment;
}

struct BoundaryVisibilitySegmentLabel {
    const char* name;
    const char* visibility_kind;
    const char* source_file_function;
    const char* classification_basis;
    bool included_in_paper_visible_host_duration;
    bool included_in_instrumentation_only_duration;
};

inline std::string boundary_visibility_segment_label_json(
    const BoundaryVisibilitySegmentLabel& segment
) {
    std::string json = "{";
    json += "\"name\":\"" + trace_json_escape(segment.name) + "\"";
    json += ",\"visibility_kind\":\"" + trace_json_escape(segment.visibility_kind) + "\"";
    json += ",\"start_offset_us\":null";
    json += ",\"end_offset_us\":null";
    json += ",\"duration_us\":null";
    json += ",\"clock\":\"unmeasured\"";
    json += ",\"source_file_function\":\"" +
            trace_json_escape(segment.source_file_function) + "\"";
    json += ",\"classification_basis\":\"" +
            trace_json_escape(segment.classification_basis) + "\"";
    json += ",\"included_in_paper_visible_host_duration\":";
    json += segment.included_in_paper_visible_host_duration ? "true" : "false";
    json += ",\"included_in_instrumentation_only_duration\":";
    json += segment.included_in_instrumentation_only_duration ? "true" : "false";
    json += "}";
    return json;
}

inline std::vector<BoundaryVisibilitySegmentLabel> boundary_visibility_segment_labels_for_api(
    const char* api
) {
    api = canonical_trace_api_name(api);
    if (api != nullptr && strcmp(api, "__cudaPushCallConfiguration") == 0) {
        return {{
            "fake_launch_config_push_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/runtime/common.cpp::__cudaPushCallConfiguration",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    if (api != nullptr && strcmp(api, "__cudaPopCallConfiguration") == 0) {
        return {{
            "fake_launch_config_pop_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/runtime/common.cpp::__cudaPopCallConfiguration",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    if (api != nullptr && strcmp(api, "cudaLaunchKernel") == 0) {
        return {
            {
                "stream_enqueue_bookkeeping",
                "mixed_or_unresolved",
                "cpp/fake_cuda/src/runtime/cudaLaunch.cpp::cudaLaunchKernel",
                "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
                false,
                false,
            },
            {
                "fake_kernel_metadata_lookup",
                "mixed_or_unresolved",
                "cpp/fake_cuda/src/runtime/cudaLaunch.cpp::cudaLaunchKernel",
                "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
                false,
                false,
            },
            {
                "fake_dispatch_kernel_args",
                "mixed_or_unresolved",
                "cpp/fake_cuda/src/runtime/cudaLaunch.cpp::cudaLaunchKernel",
                "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
                false,
                false,
            },
        };
    }
    if (api != nullptr && strcmp(api, "cudaGetDevice") == 0) {
        return {{
            "fake_cuda_get_device_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/runtime/cudaGet.cpp::cudaGetDevice",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    if (api != nullptr && strcmp(api, "cudaGetLastError") == 0) {
        return {{
            "fake_cuda_get_last_error_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/runtime/cudaGet.cpp::cudaGetLastError",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    if (api != nullptr && strcmp(api, "cublasSetStream_v2") == 0) {
        return {{
            "fake_cublas_set_stream_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/cublas/cublasSet.cpp::cublasSetStream_v2",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    if (api != nullptr && strcmp(api, "cublasGemmEx") == 0) {
        return {{
            "fake_cublas_gemm_ex_dispatch_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/cublas/common.cpp::cublasGemmEx",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    if (api != nullptr && strcmp(api, "cublasGemmStridedBatchedEx") == 0) {
        return {{
            "fake_cublas_gemm_strided_batched_ex_dispatch_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/cublas/common.cpp::cublasGemmStridedBatchedEx",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    if (api != nullptr && strcmp(api, "cudaEventRecord") == 0) {
        return {{
            "fake_cuda_event_record_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/runtime/cudaEvent.cpp::cudaEventRecord",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    if (api != nullptr && strcmp(api, "cudaStreamWaitEvent") == 0) {
        return {{
            "fake_cuda_stream_wait_event_body",
            "mixed_or_unresolved",
            "cpp/fake_cuda/src/runtime/cudaStream.cpp::cudaStreamWaitEvent",
            "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
            false,
            false,
        }};
    }
    return {};
}

inline std::string boundary_visibility_segments_json_for_api(const char* api) {
    const std::vector<BoundaryVisibilitySegmentLabel> segments =
        boundary_visibility_segment_labels_for_api(api);
    std::string json = "[";
    bool first = true;
    for (const auto& segment : segments) {
        if (!first) {
            json += ",";
        }
        json += boundary_visibility_segment_label_json(segment);
        first = false;
    }
    json += "]";
    return json;
}

inline const char* trace_host_machine_id() {
    const char* value = getenv("FLEXSIM_HOST_MACHINE_ID");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return nullptr;
}

inline const char* trace_host_dispatch_queue_id() {
    const char* value = getenv("FLEXSIM_HOST_DISPATCH_QUEUE_ID");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return trace_host_machine_id();
}

inline WrapperTimingEnrichment enrich_payload_with_wrapper_timing(
    const char* api,
    const char* type,
    long long ts_us,
    const std::string& payload_json,
    long long wrapper_exit_ns = 0LL,
    nlohmann::json* typed = nullptr
) {
    WrapperTimingEnrichment enrichment{ts_us, payload_json};
    std::string prepop_launch_neighborhood_fragment;
    if (launch_boundary_visibility_diagnostics_enabled() &&
        should_emit_prepop_launch_neighborhood_provenance(api)) {
        prepop_launch_neighborhood_fragment =
            prepop_launch_neighborhood_provenance_fragment(api);
    }
    long long entry_start_ns = 0LL;
    if (!fakecuda::trace::take_wrapper_entry_start_ns(api, &entry_start_ns)) {
        enrichment.payload_json = append_json_fragment(
            payload_json,
            prepop_launch_neighborhood_fragment
        );
        return enrichment;
    }

    const long long end_wall_ns = wrapper_exit_ns > 0LL
        ? wrapper_exit_ns
        : fakecuda::host_timing::real_now_ns();
    long long timing_entry_start_ns = entry_start_ns;
    long long timing_end_wall_ns = end_wall_ns;
    const fakecuda::host_timing::Mode mode = fakecuda::host_timing::HostTimingConfig::instance().mode();
    if (mode == fakecuda::host_timing::Mode::kMeasure) {
        timing_entry_start_ns = fakecuda::trace::logical_host_time_ns(entry_start_ns);
        timing_end_wall_ns = fakecuda::trace::logical_host_time_ns(end_wall_ns);
    }
    double host_duration_us = std::max(
        static_cast<double>(timing_end_wall_ns - timing_entry_start_ns) / 1000.0,
        0.0
    );
    long long emitted_ts_us = ts_us;
    long long end_ts_us = ts_us;
    const long long observed_wrapper_start_ts_us = timing_entry_start_ns / 1000LL;
    const long long observed_wrapper_end_ts_us = std::max(
        timing_end_wall_ns / 1000LL,
        observed_wrapper_start_ts_us
    );

    const fakecuda::host_timing::DispatchScope dispatch_scope =
        fakecuda::host_timing::HostTimingConfig::instance().dispatch_scope();
    const char* host_timing_mode_name = fakecuda::host_timing::mode_name(mode);
    const char* host_timing_dispatch_scope_name =
        fakecuda::host_timing::dispatch_scope_name(dispatch_scope);
    const char* host_timing_source = nullptr;
    if (mode == fakecuda::host_timing::Mode::kMeasure ||
        mode == fakecuda::host_timing::Mode::kSleep) {
        // Wrapper timing is an annotation, not the trace ordering clock.
        // Keep ts/end_ts on the trace emission timeline so marker windows,
        // API ordering, and Figure 13 compare semantics remain stable. The
        // measured wrapper envelope is exposed separately via observed_wrapper_*
        // fields and host_duration_us for replay.
        end_ts_us = emitted_ts_us;
    }
    if (mode == fakecuda::host_timing::Mode::kMeasure) {
        host_timing_source = "direct_wallclock";
    } else if (mode == fakecuda::host_timing::Mode::kTrace) {
        host_timing_source = "profile_scheduled_trace";
    } else if (mode == fakecuda::host_timing::Mode::kSleep) {
        host_timing_source = "profile_scheduled_sleep";
    }

    enrichment.emitted_ts_us = emitted_ts_us;
    enrichment.timing_added = true;

    char duration_buf[64];
    snprintf(duration_buf, sizeof(duration_buf), "%.3f", host_duration_us);
    const char* contract = wrapper_runtime_contract(api, type);
    if (typed != nullptr) {
        // The ordinary binary path never formats/parses a JSON record. Keep
        // the legacy decimal rounding for the measured duration field.
        (*typed)["end_ts"] = end_ts_us;
        (*typed)["host_duration_us"] = std::strtod(duration_buf, nullptr);
        if (contract != nullptr) {
            (*typed)["wrapper_runtime_contract"] = contract;
            if (std::strcmp(contract, "direct_runtime") == 0 && !typed->contains("direct_runtime_us"))
                (*typed)["direct_runtime_us"] = std::strtod(duration_buf, nullptr);
        }
        if (mode != fakecuda::host_timing::Mode::kNone) {
            (*typed)["host_timing_mode"] = host_timing_mode_name;
            (*typed)["host_timing_dispatch_scope"] = host_timing_dispatch_scope_name;
            (*typed)["observed_wrapper_start_ts_us"] = observed_wrapper_start_ts_us;
            (*typed)["observed_wrapper_end_ts_us"] = observed_wrapper_end_ts_us;
        }
        if (host_timing_source) (*typed)["host_timing_source"] = host_timing_source;
        if (const char* machine = trace_host_machine_id()) (*typed)["host_machine_id"] = machine;
        if (const char* queue = trace_host_dispatch_queue_id()) (*typed)["host_dispatch_queue_id"] = queue;
        return enrichment;
    }
    std::string contract_fragment;
    std::string direct_runtime_fragment;
    if (contract != nullptr) {
        contract_fragment = std::string(",\"wrapper_runtime_contract\":\"") +
                            trace_json_escape(contract) + "\"";
        if (strcmp(contract, "direct_runtime") == 0 &&
            payload_json.find("\"direct_runtime_us\"") == std::string::npos) {
            direct_runtime_fragment = std::string(",\"direct_runtime_us\":") + duration_buf;
        }
    }
    std::string host_timing_mode_fragment;
    std::string host_timing_dispatch_scope_fragment;
    std::string host_timing_source_fragment;
    std::string observed_wrapper_start_fragment;
    std::string observed_wrapper_end_fragment;
    std::string boundary_origin_fragment;
    std::string host_machine_fragment;
    std::string host_dispatch_queue_fragment;
    if (mode != fakecuda::host_timing::Mode::kNone) {
        host_timing_mode_fragment = std::string(",\"host_timing_mode\":\"") +
                                    trace_json_escape(host_timing_mode_name) + "\"";
        host_timing_dispatch_scope_fragment =
            std::string(",\"host_timing_dispatch_scope\":\"") +
            trace_json_escape(host_timing_dispatch_scope_name) + "\"";
    }
    const char* host_machine_id = trace_host_machine_id();
    if (host_machine_id != nullptr) {
        host_machine_fragment = std::string(",\"host_machine_id\":\"") +
                                trace_json_escape(host_machine_id) + "\"";
    }
    const char* host_dispatch_queue_id = trace_host_dispatch_queue_id();
    if (host_dispatch_queue_id != nullptr) {
        host_dispatch_queue_fragment =
            std::string(",\"host_dispatch_queue_id\":\"") +
            trace_json_escape(host_dispatch_queue_id) + "\"";
    }
    if (host_timing_source != nullptr) {
        host_timing_source_fragment = std::string(",\"host_timing_source\":\"") +
                                      trace_json_escape(host_timing_source) + "\"";
    }
    if (mode != fakecuda::host_timing::Mode::kNone) {
        observed_wrapper_start_fragment = std::string(",\"observed_wrapper_start_ts_us\":") +
                                          std::to_string(observed_wrapper_start_ts_us);
        observed_wrapper_end_fragment = std::string(",\"observed_wrapper_end_ts_us\":") +
                                        std::to_string(observed_wrapper_end_ts_us);
    }
    if (launch_boundary_visibility_diagnostics_enabled() &&
        should_emit_boundary_origin_provenance(api)) {
        std::string launch_boundary_fragment;
        if (api != nullptr &&
            (strcmp(api, "__cudaPopCallConfiguration") == 0 ||
             strcmp(api, "cudaLaunchKernel") == 0)) {
            launch_boundary_fragment = std::string(",\"launch_boundary_id_unavailable_reason\":\"") +
                "fakecuda_launch_pair_id_disabled_to_preserve_host_duration\"";
        }
        const std::string segment_json = boundary_visibility_segments_json_for_api(api);
        boundary_origin_fragment = std::string(",\"boundary_origin_kind\":\"mixed_or_unresolved\"") +
            ",\"boundary_segment_schema_version\":\"launch_boundary_visibility_v1\"" +
            launch_boundary_fragment +
            ",\"wrapper_segment_coverage\":\"structural_labels_only_unmeasured\"" +
            ",\"wrapper_segment_sum_us\":0.000" +
            ",\"wrapper_segment_unattributed_us\":" + format_duration_us(host_duration_us) +
            ",\"paper_visible_host_duration_us\":null" +
            ",\"boundary_origin_classification_basis\":\"" +
                "producer_segment_timing_disabled_to_preserve_host_duration\"" +
            ",\"boundary_visibility_segments\":" + segment_json +
            ",\"caller_visible_elapsed_us\":" + duration_buf +
            ",\"fake_api_body_duration_us\":null" +
            ",\"actual_host_dispatch_duration_us\":null" +
            ",\"actual_counterpart_component_id\":\"host_dispatch_overhead\"" +
            ",\"actual_counterpart_visibility_kind\":\"mixed_or_unresolved\"" +
            ",\"wrapper_internal_duration_us\":null" +
            ",\"instrumentation_only_duration_us\":null" +
            ",\"unresolved_mixed_duration_us\":" + format_duration_us(host_duration_us);
    }
    boundary_origin_fragment += prepop_launch_neighborhood_fragment;
    if (payload_json.empty()) {
        enrichment.payload_json = std::string("\"end_ts\":") + std::to_string(end_ts_us) +
                                  ",\"host_duration_us\":" + duration_buf +
                                  observed_wrapper_start_fragment +
                                  observed_wrapper_end_fragment +
                                  boundary_origin_fragment +
                                  direct_runtime_fragment +
                                  contract_fragment +
                                  host_timing_mode_fragment +
                                  host_timing_dispatch_scope_fragment +
                                  host_machine_fragment +
                                  host_dispatch_queue_fragment +
                                  host_timing_source_fragment;
        return enrichment;
    }
    enrichment.payload_json = payload_json + ",\"end_ts\":" + std::to_string(end_ts_us) +
                              ",\"host_duration_us\":" + duration_buf +
                              observed_wrapper_start_fragment +
                              observed_wrapper_end_fragment +
                              boundary_origin_fragment +
                              direct_runtime_fragment +
                              contract_fragment +
                              host_timing_mode_fragment +
                              host_timing_dispatch_scope_fragment +
                              host_machine_fragment +
                              host_dispatch_queue_fragment +
                              host_timing_source_fragment;
    return enrichment;
}

inline std::string enrich_payload_with_trace_context(const std::string& payload_json) {
    std::string host_timing_mode_fragment;
    std::string host_timing_dispatch_scope_fragment;
    std::string host_timing_source_fragment;
    std::string host_machine_fragment;
    std::string host_dispatch_queue_fragment;

    const fakecuda::host_timing::Mode mode =
        fakecuda::host_timing::HostTimingConfig::instance().mode();
    const fakecuda::host_timing::DispatchScope dispatch_scope =
        fakecuda::host_timing::HostTimingConfig::instance().dispatch_scope();
    if (mode != fakecuda::host_timing::Mode::kNone) {
        host_timing_mode_fragment = std::string(",\"host_timing_mode\":\"") +
                                    trace_json_escape(fakecuda::host_timing::mode_name(mode)) + "\"";
        host_timing_dispatch_scope_fragment =
            std::string(",\"host_timing_dispatch_scope\":\"") +
            trace_json_escape(fakecuda::host_timing::dispatch_scope_name(dispatch_scope)) + "\"";
    }

    const char* host_machine_id = trace_host_machine_id();
    if (host_machine_id != nullptr) {
        host_machine_fragment = std::string(",\"host_machine_id\":\"") +
                                trace_json_escape(host_machine_id) + "\"";
    }
    const char* host_dispatch_queue_id = trace_host_dispatch_queue_id();
    if (host_dispatch_queue_id != nullptr) {
        host_dispatch_queue_fragment =
            std::string(",\"host_dispatch_queue_id\":\"") +
            trace_json_escape(host_dispatch_queue_id) + "\"";
    }
    if (mode == fakecuda::host_timing::Mode::kMeasure) {
        host_timing_source_fragment = ",\"host_timing_source\":\"direct_wallclock\"";
    } else if (mode == fakecuda::host_timing::Mode::kTrace) {
        host_timing_source_fragment = ",\"host_timing_source\":\"profile_scheduled_trace\"";
    } else if (mode == fakecuda::host_timing::Mode::kSleep) {
        host_timing_source_fragment = ",\"host_timing_source\":\"profile_scheduled_sleep\"";
    }

    if (payload_json.empty()) {
        return host_timing_mode_fragment +
               host_timing_dispatch_scope_fragment +
               host_machine_fragment +
               host_dispatch_queue_fragment +
               host_timing_source_fragment;
    }
    return payload_json +
           host_timing_mode_fragment +
           host_timing_dispatch_scope_fragment +
           host_machine_fragment +
           host_dispatch_queue_fragment +
           host_timing_source_fragment;
}

class TracePayloadBuilder {
private:
    std::vector<std::string> items_;
    std::vector<std::string> keys_;
    std::vector<std::uint8_t> binary_fields_;

    template <typename T>
    void add_item(const char* key, const T& value) {
        fakecuda::capture::count(fakecuda::capture::Counter::PayloadFields);
        if (trace_binary_enabled()) {
            if (contains(key)) throw std::runtime_error("duplicate binary trace field");
            keys_.emplace_back(key);
            nlohmann::json::to_msgpack(nlohmann::json(key), binary_fields_);
            nlohmann::json::to_msgpack(nlohmann::json(value), binary_fields_);
        } else {
            items_.push_back("\"" + trace_json_escape(key) + "\":" + nlohmann::json(value).dump());
        }
    }

public:
    TracePayloadBuilder() {
        fakecuda::capture::count(fakecuda::capture::Counter::PayloadBuilders);
    }

    bool empty() const {
        return trace_binary_enabled() ? keys_.empty() : items_.empty();
    }

    void add_value(const char* key, const nlohmann::json& value) {
        add_item(key, value);
    }

    void add_string(const char* key, const char* value) {
        add_item(key, value ? value : "");
    }

    void add_string(const char* key, const std::string& value) {
        add_item(key, value);
    }

    void add_int(const char* key, int value) {
        add_item(key, value);
    }

    void add_int64(const char* key, std::int64_t value) {
        add_item(key, value);
    }

    void add_uint(const char* key, unsigned int value) {
        add_item(key, value);
    }

    void add_uint64(const char* key, std::uint64_t value) {
        add_item(key, value);
    }

    void add_size(const char* key, size_t value) {
        add_item(key, value);
    }

    void add_bool(const char* key, bool value) {
        add_item(key, value);
    }

    void add_pointer(const char* key, const void* value) {
        add_uint64(key, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value)));
    }

    std::string to_json_fragment() const {
        fakecuda::capture::count(fakecuda::capture::Counter::JsonFragments);
        if (trace_binary_enabled()) {
            const auto json = values().dump();
            return json.substr(1, json.size() - 2);
        }
        std::string fragment;
        for (size_t i = 0; i < items_.size(); ++i) {
            if (i > 0) {
                fragment += ",";
            }
            fragment += items_[i];
        }
        return fragment;
    }

    bool contains(const char* key) const {
        return std::find(keys_.begin(), keys_.end(), key) != keys_.end();
    }

    std::vector<std::uint8_t> binary_record(const nlohmann::json& extras) const {
        for (auto item = extras.begin(); item != extras.end(); ++item)
            if (contains(item.key().c_str())) throw std::runtime_error("duplicate binary trace metadata");
        const auto tail = nlohmann::json::to_msgpack(extras);
        const auto count = keys_.size() + extras.size();
        std::vector<std::uint8_t> result;
        result.reserve(binary_fields_.size() + tail.size() + 5);
        if (count < 16) result.push_back(static_cast<std::uint8_t>(0x80 | count));
        else {
            const int width = count <= 65535 ? 2 : 4;
            result.push_back(width == 2 ? 0xde : 0xdf);
            for (int shift = (width - 1) * 8; shift >= 0; shift -= 8)
                result.push_back(static_cast<std::uint8_t>((count >> shift) & 255));
        }
        result.insert(result.end(), binary_fields_.begin(), binary_fields_.end());
        const auto header = extras.size() < 16 ? 1 : extras.size() <= 65535 ? 3 : 5;
        result.insert(result.end(), tail.begin() + header, tail.end());
        return result;
    }

    nlohmann::json values() const {
        return nlohmann::json::from_msgpack(binary_record(nlohmann::json::object()));
    }
};

// Do not construct a builder in off mode. Call sites also guard each add_* call
// so its arguments (including string formatting) are not eagerly evaluated.
inline std::optional<TracePayloadBuilder> make_trace_payload() {
    if (!fakecuda::capture::generates_trace()) return std::nullopt;
    return std::optional<TracePayloadBuilder>(std::in_place);
}

using FlexMayaHookRecordApi = std::uint64_t (*) (
    const char*, const char*, int, std::uint64_t, int, std::uint64_t,
    std::uint64_t, std::uint64_t, double, std::uint64_t, std::uint64_t,
    int, std::uint64_t, std::uint64_t, const char*, const char*, int);

inline std::size_t trace_payload_value_offset(const std::string& payload, const char* key) {
    const std::string token = std::string("\"") + key + "\":";
    const std::size_t found = payload.find(token);
    if (found == std::string::npos) {
        return std::string::npos;
    }
    std::size_t offset = found + token.size();
    while (offset < payload.size() &&
           (payload[offset] == ' ' || payload[offset] == '\t' ||
            payload[offset] == '\n' || payload[offset] == '\r')) {
        ++offset;
    }
    return offset;
}

inline double trace_payload_number(const std::string& payload, const char* key, double fallback) {
    const std::size_t offset = trace_payload_value_offset(payload, key);
    if (offset == std::string::npos || offset >= payload.size()) {
        return fallback;
    }
    char* end = nullptr;
    const double value = std::strtod(payload.c_str() + offset, &end);
    return end == payload.c_str() + offset ? fallback : value;
}

inline std::uint64_t trace_payload_uint64(
    const std::string& payload,
    const char* key,
    std::uint64_t fallback = 0) {
    const std::size_t offset = trace_payload_value_offset(payload, key);
    if (offset == std::string::npos || offset >= payload.size()) {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long long value = std::strtoull(payload.c_str() + offset, &end, 10);
    return end == payload.c_str() + offset ? fallback : static_cast<std::uint64_t>(value);
}

inline int trace_payload_int(const std::string& payload, const char* key, int fallback = 0) {
    const std::size_t offset = trace_payload_value_offset(payload, key);
    if (offset == std::string::npos || offset >= payload.size()) {
        return fallback;
    }
    char* end = nullptr;
    const long value = std::strtol(payload.c_str() + offset, &end, 10);
    return end == payload.c_str() + offset ? fallback : static_cast<int>(value);
}

inline FlexMayaHookRecordApi resolve_flexmaya_hook() {
    static FlexMayaHookRecordApi hook = []() -> FlexMayaHookRecordApi {
        const char* library = std::getenv("PLAIN_MAYA_HOOK_LIBRARY");
        void* handle = RTLD_DEFAULT;
        if (library != nullptr && library[0] != '\0') {
            handle = dlopen(library, RTLD_NOW | RTLD_GLOBAL);
            if (handle == nullptr) {
                return nullptr;
            }
        }
        return reinterpret_cast<FlexMayaHookRecordApi>(dlsym(handle, "plain_maya_hook_record_api_v2"));
    }();
    return hook;
}

inline void emit_flexmaya_hook_record(
    const char* api,
    const char* type,
    long long timestamp_us,
    const std::string& payload) {
    if (!fakecuda::capture::writes_trace()) return;
    const char* model_window = std::getenv("FLEXMAYA_TRACE_MODEL_WINDOW");
    const char* shared_arena = std::getenv("FLEXMAYA_SHM_NAME");
    if (api == nullptr || type == nullptr ||
        std::strcmp(type, "marker") == 0 || std::strcmp(type, "other") == 0 ||
        std::strcmp(api, "cudaGetDevice") == 0 ||
        model_window == nullptr || std::strcmp(model_window, "1") != 0) {
        return;
    }
    if (shared_arena == nullptr || shared_arena[0] == '\0') {
        return;
    }
    const FlexMayaHookRecordApi hook = resolve_flexmaya_hook();
    if (hook == nullptr) {
        return;
    }

    const std::uint64_t count_from_payload = trace_payload_uint64(payload, "count");
    std::uint64_t count = count_from_payload;
    if (count == 0 && std::strncmp(api, "cublas", 6) == 0) {
        count = trace_payload_uint64(payload, "m", 1) *
                trace_payload_uint64(payload, "n", 1) *
                trace_payload_uint64(payload, "k", 1) *
                trace_payload_uint64(payload, "batch_count", 1);
    }
    if (count == 0 && std::strcmp(type, "kernel_launch") == 0) {
        count = trace_payload_uint64(payload, "grid_x", 1) *
                trace_payload_uint64(payload, "grid_y", 1) *
                trace_payload_uint64(payload, "grid_z", 1) *
                trace_payload_uint64(payload, "block_x", 1) *
                trace_payload_uint64(payload, "block_y", 1) *
                trace_payload_uint64(payload, "block_z", 1);
    }
    std::uint64_t bytes = trace_payload_uint64(payload, "bytes");
    if (bytes == 0 && std::strcmp(type, "nccl_collective") == 0) {
        bytes = count * 2ULL;
    }

    std::string collective_group;
    if (std::strcmp(type, "nccl_collective") == 0) {
        collective_group = std::string(api) + ":call=" +
            std::to_string(trace_payload_uint64(payload, "call_idx"));
    }
    const char* code_partition = std::getenv("FLEXMAYA_CODE_PARTITION");
    if (code_partition == nullptr) {
        code_partition = "";
    }
    // The legacy hook's label slot also carries opaque scoped ownership. Its
    // consumer maps the token; FakeCUDA does not interpret the source region.
    const auto owner_token = fakecuda::capture::scoped_token();
    const std::string scoped_owner = owner_token == 0 ? "" :
        "capture:" + std::to_string(owner_token);
    if (owner_token != 0) code_partition = scoped_owner.c_str();
    const char* rank_text = std::getenv("RANK");
    const int rank = rank_text == nullptr ? 0 : std::atoi(rank_text);
    const std::uint64_t event_id = trace_payload_uint64(payload, "event_id");
    const bool is_event_record = std::strstr(api, "EventRecord") != nullptr;
    const bool is_event_wait = std::strstr(api, "WaitEvent") != nullptr;
    fakecuda::capture::count(fakecuda::capture::Counter::Hooks);
    const std::uint64_t recorded_id = hook(
        api,
        type,
        rank,
        static_cast<std::uint64_t>(::syscall(SYS_gettid)),
        trace_payload_int(payload, "device"),
        trace_payload_uint64(payload, "stream_id"),
        static_cast<std::uint64_t>(timestamp_us),
        static_cast<std::uint64_t>(timestamp_us) * 1000ULL,
        trace_payload_number(payload, "host_duration_us", 0.0),
        bytes,
        count,
        trace_payload_int(payload, "peer", -1),
        is_event_record ? event_id : 0,
        is_event_wait ? event_id : 0,
        collective_group.c_str(),
        code_partition,
        (std::strstr(api, "Synchronize") != nullptr) ? 1 : 0);
    if (recorded_id != 0) {
        fakecuda::capture::count(fakecuda::capture::Counter::ArenaAppends);
    }
}

class TraceLogger {
private:
    enum class TraceSurface {
        kAll,
        kSemantic,
    };

    FILE* file_ = nullptr;
    std::string raw_audit_path_;
    std::uint64_t cuda_get_device_raw_count_ = 0;
    std::mutex mutex_;
    bool enabled_ = false;
    bool initialized_ = false;
    bool flush_per_event_ = true;
    size_t flush_every_ = 1;
    size_t pending_events_ = 0;
    size_t stdio_buffer_bytes_ = 0;
    TraceSurface trace_surface_ = TraceSurface::kAll;

    static const char* modNames_[];

    static size_t parse_size_env(const char* name, size_t fallback) {
        const char* raw = getenv(name);
        if (!raw || raw[0] == '\0') {
            return fallback;
        }
        char* end = nullptr;
        unsigned long long parsed = strtoull(raw, &end, 10);
        if (end == raw || (end && *end != '\0') || parsed == 0ULL) {
            return fallback;
        }
        return static_cast<size_t>(parsed);
    }

    void init_raw_audit_() {
        if (!fakecuda::capture::writes_trace()) return;
        const char* directory = getenv("FLEXMAYA_RAW_AUDIT_DIR");
        if (directory == nullptr || directory[0] == '\0') {
            return;
        }
        const char* tag = getenv("FLEXMAYA_RAW_AUDIT_TAG");
        if (tag == nullptr || tag[0] == '\0') {
            tag = "node";
        }
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.cudaGetDevice.%d.jsonl", directory, tag, (int)getpid());
        raw_audit_path_ = path;
    }

    void init() {
        if (initialized_) return;
        initialized_ = true;

        const char* env = getenv("FAKECUDA_TRACE");
        const char* shared_only = getenv("FLEXMAYA_SHARED_ARENA_ONLY");
        const bool arena_only = shared_only != nullptr && strcmp(shared_only, "1") == 0;
        if ((!env || strcmp(env, "1") != 0) && !arena_only) {
            enabled_ = false;
            return;
        }
        enabled_ = true;

        if (!fakecuda::capture::writes_trace()) {
            // Preserve filtering/enrichment without opening any output sink.
            const char* surface = getenv("FAKECUDA_TRACE_SURFACE");
            if (!arena_only && surface && strcmp(surface, "semantic") == 0) {
                trace_surface_ = TraceSurface::kSemantic;
            }
            return;
        }

        if (arena_only) {
            init_raw_audit_();
            return;
        }

        const char* flush_mode = getenv("FAKECUDA_TRACE_FLUSH_MODE");
        if (flush_mode && strcmp(flush_mode, "buffered") == 0) {
            flush_per_event_ = false;
            flush_every_ = parse_size_env("FAKECUDA_TRACE_FLUSH_EVERY", 4096);
            stdio_buffer_bytes_ = parse_size_env("FAKECUDA_TRACE_STDIO_BUFFER_BYTES", 4 * 1024 * 1024);
        } else {
            flush_per_event_ = true;
            flush_every_ = 1;
            stdio_buffer_bytes_ = parse_size_env("FAKECUDA_TRACE_STDIO_BUFFER_BYTES", 0);
        }

        const char* trace_surface = getenv("FAKECUDA_TRACE_SURFACE");
        if (trace_surface && strcmp(trace_surface, "semantic") == 0) {
            trace_surface_ = TraceSurface::kSemantic;
        } else {
            trace_surface_ = TraceSurface::kAll;
        }

        const char* path = getenv("FAKECUDA_TRACE_PATH");
        char buf[256];
        if (!path || strlen(path) == 0) {
            snprintf(buf, sizeof(buf), "/tmp/fakecuda_trace_%d.jsonl", getpid());
            path = buf;
        }
        const char* inherited_fd = getenv("FAKECUDA_TRACE_FD");
        if (inherited_fd) {
            char* end = nullptr;
            errno = 0;
            const long parsed = strtol(inherited_fd, &end, 10);
            if (errno || end == inherited_fd || *end || parsed < 0 || parsed > INT_MAX)
                throw std::runtime_error("invalid FAKECUDA_TRACE_FD");
            const int fd = dup(static_cast<int>(parsed));
            if (fd < 0) throw std::runtime_error("cannot duplicate FAKECUDA_TRACE_FD");
            file_ = fdopen(fd, "a");
            if (!file_) {
                close(fd);
                throw std::runtime_error("cannot open FAKECUDA_TRACE_FD");
            }
        } else {
            file_ = fopen(path, "a");
        }
        if (!file_) {
            fprintf(stderr, "[fakecuda-trace] ERROR: cannot open %s\n", path);
            enabled_ = false;
            return;
        }
        if (stdio_buffer_bytes_ > 0) {
            setvbuf(file_, nullptr, _IOFBF, stdio_buffer_bytes_);
        }
    }

    static pid_t gettid_() {
        return (pid_t)syscall(SYS_gettid);
    }

    bool should_log_event_(const char* api, const char* type) const {
        if (trace_surface_ != TraceSurface::kSemantic) {
            return true;
        }
        if (type != nullptr && std::strcmp(type, "marker") == 0) {
            return true;
        }
        return fakecuda::host_timing::is_semantic_schedule_event(api, type);
    }

public:
    static TraceLogger& instance() {
        // Keep diagnostic aggregation alive until after this logger's final
        // buffered flush, including callers that use the logger directly.
        if (!fakecuda_ras_active()) (void)fakecuda::capture::state();
        static TraceLogger inst;
        return inst;
    }

    long long log(int lib, const char* api, const char* type) {
        return log_with_payload_impl(lib, api, type, nullptr, nullptr);
    }

    long long log_with_payload(int lib, const char* api, const char* type, const std::string& payload_json) {
        return log_with_payload_impl(lib, api, type, &payload_json, nullptr);
    }

    long long log_with_payload(int lib, const char* api, const char* type, const TracePayloadBuilder& payload) {
        return log_with_payload_impl(lib, api, type, nullptr, &payload);
    }

    long long log_with_payload(int lib, const char* api, const char* type,
                               const std::optional<TracePayloadBuilder>& payload) {
        if (fakecuda_ras_active()) { fakecuda_ras_fail(api); return -1LL; }
        if (!payload) return -1LL;
        return log_with_payload_impl(lib, api, type, nullptr, &*payload);
    }

    long long log_lightweight_with_payload(int lib, const char* api, const char* type, const std::string& payload_json) {
        return log_lightweight_with_payload_impl(lib, api, type, &payload_json, nullptr);
    }

    long long log_lightweight_with_payload(int lib, const char* api, const char* type, const TracePayloadBuilder& payload) {
        return log_lightweight_with_payload_impl(lib, api, type, nullptr, &payload);
    }

    long long log_lightweight_with_payload(int lib, const char* api, const char* type,
                                           const std::optional<TracePayloadBuilder>& payload) {
        if (fakecuda_ras_active()) { fakecuda_ras_fail(api); return -1LL; }
        if (!payload) return -1LL;
        return log_lightweight_with_payload_impl(lib, api, type, nullptr, &*payload);
    }

private:
    long long log_binary_payload_(const char* mod, const char* api, const char* type,
        long long ts, const std::string* prebuilt, const TracePayloadBuilder* builder,
        long long overhead_start_ns, bool lightweight) {
        const bool diagnostics = launch_boundary_visibility_diagnostics_enabled();
        const bool direct = builder && !prebuilt && !diagnostics && !std::getenv("FLEXMAYA_SHM_NAME");
        auto payload = builder && !direct ? builder->values() : nlohmann::json::object();
        const bool existing_runtime = direct && builder->contains("direct_runtime_us");
        if (existing_runtime) payload["direct_runtime_us"] = nullptr;
        if (prebuilt && !prebuilt->empty())
            payload = nlohmann::json::parse("{" + *prebuilt + "}");
        const auto owner = fakecuda::capture::scoped_token();
        if (owner) payload["capture_scope"] = owner;
        fakecuda::capture::count(fakecuda::capture::Counter::Enrichments);
        // Opt-in legacy provenance diagnostics retain their complete nested
        // schema through the old formatter. Normal capture stays typed.
        std::string fragment;
        if (diagnostics) {
            const auto encoded = payload.dump();
            fragment = encoded.substr(1, encoded.size() - 2);
        }
        const auto enrichment = enrich_payload_with_wrapper_timing(
            api, type, ts, fragment, overhead_start_ns, diagnostics ? nullptr : &payload);
        ts = enrichment.emitted_ts_us;
        if (existing_runtime) payload.erase("direct_runtime_us");
        if (diagnostics) payload = nlohmann::json::parse("{" + enrichment.payload_json + "}");
        if (lightweight && !enrichment.timing_added) {
            const auto mode = fakecuda::host_timing::HostTimingConfig::instance().mode();
            if (mode != fakecuda::host_timing::Mode::kNone) {
                payload["host_timing_mode"] = fakecuda::host_timing::mode_name(mode);
                payload["host_timing_dispatch_scope"] = fakecuda::host_timing::dispatch_scope_name(
                    fakecuda::host_timing::HostTimingConfig::instance().dispatch_scope());
            }
            const char* source = mode == fakecuda::host_timing::Mode::kMeasure ? "direct_wallclock" :
                mode == fakecuda::host_timing::Mode::kTrace ? "profile_scheduled_trace" :
                mode == fakecuda::host_timing::Mode::kSleep ? "profile_scheduled_sleep" : nullptr;
            if (source) payload["host_timing_source"] = source;
            if (const char* machine = trace_host_machine_id()) payload["host_machine_id"] = machine;
            if (const char* queue = trace_host_dispatch_queue_id()) payload["host_dispatch_queue_id"] = queue;
        }
        if (!fakecuda::capture::writes_trace()) return ts;
        if (std::getenv("FLEXMAYA_SHM_NAME")) {
            const auto json = payload.dump();
            emit_flexmaya_hook_record(api, type, ts, json.substr(1, json.size() - 2));
        }
        payload["ts"] = ts;
        payload["pid"] = static_cast<int>(getpid());
        payload["tid"] = static_cast<int>(gettid_());
        payload["mod"] = mod;
        payload["api"] = api;
        payload["type"] = type;
        const auto bytes = direct ? builder->binary_record(payload) : nlohmann::json::to_msgpack(payload);
        if (bytes.size() > 64 * 1024 * 1024) throw std::runtime_error("binary trace row too large");
        std::string frame = "MMP1";
        const auto size = static_cast<std::uint32_t>(bytes.size());
        for (int shift = 24; shift >= 0; shift -= 8) frame += static_cast<char>((size >> shift) & 255);
        frame.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_) {
            if (fwrite(frame.data(), 1, frame.size(), file_) != frame.size())
                throw std::runtime_error("binary trace write failed");
            fakecuda::capture::count(fakecuda::capture::Counter::Records);
            fakecuda::capture::count(fakecuda::capture::Counter::TraceBytes, frame.size());
            if (++pending_events_ >= flush_every_ || flush_per_event_) {
                fakecuda::capture::count(fakecuda::capture::Counter::Flushes);
                if (fflush(file_) != 0) throw std::runtime_error("binary trace flush failed");
                pending_events_ = 0;
            }
        }
        return ts;
    }

    long long log_with_payload_impl(
        int lib,
        const char* api,
        const char* type,
        const std::string* prebuilt_payload_json,
        const TracePayloadBuilder* payload_builder
    ) {
        if (fakecuda_ras_active()) { fakecuda_ras_fail(api); return -1LL; }
        if (!fakecuda::capture::generates_trace()) return -1LL;
        const bool measure_mode =
            fakecuda::host_timing::HostTimingConfig::instance().mode() ==
            fakecuda::host_timing::Mode::kMeasure;
        const long long trace_overhead_start_ns = measure_mode
            ? fakecuda::host_timing::real_now_ns()
            : 0LL;
        auto finish_trace_overhead = [&]() {
            if (!measure_mode || trace_overhead_start_ns <= 0LL) {
                return;
            }
            fakecuda::trace::add_trace_writer_overhead_ns(
                fakecuda::host_timing::real_now_ns() - trace_overhead_start_ns
            );
        };
        if (!initialized_) {
            std::lock_guard<std::mutex> lock(mutex_);
            init();
        }
        if (!enabled_) {
            finish_trace_overhead();
            return -1LL;
        }

        api = canonical_trace_api_name(api);
        if (std::strcmp(api, "cudaGetDevice") == 0) {
            ++cuda_get_device_raw_count_;
        }
        const char* mod = (lib >= 0 && lib < 6) ? modNames_[lib] : "unknown";
        if (!should_log_event_(api, type)) {
            long long discarded_start_ns = 0LL;
            (void)fakecuda::trace::take_wrapper_entry_start_ns(api, &discarded_start_ns);
            finish_trace_overhead();
            return -1LL;
        }
        fakecuda::host_timing::maybe_block_measure_before_trace(api, type, mod);
        long long ts = fakecuda::host_timing::reserve_trace_timestamp_us(api, type, mod);
        if (trace_binary_enabled()) {
            ts = log_binary_payload_(mod, api, type, ts, prebuilt_payload_json, payload_builder,
                                     trace_overhead_start_ns, false);
            finish_trace_overhead();
            return ts;
        }
        std::string built_payload_json;
        const std::string* payload_json_ptr = prebuilt_payload_json;
        if (payload_json_ptr == nullptr) {
            if (payload_builder != nullptr) {
                built_payload_json = payload_builder->to_json_fragment();
            }
            payload_json_ptr = &built_payload_json;
        }
        const auto owner_token = fakecuda::capture::scoped_token();
        const std::string scoped_payload = owner_token == 0 ? std::string{} :
            append_json_fragment(*payload_json_ptr, "\"capture_scope\":" + std::to_string(owner_token));
        const std::string& payload_json = owner_token == 0 ? *payload_json_ptr : scoped_payload;
        fakecuda::capture::count(fakecuda::capture::Counter::Enrichments);
        const WrapperTimingEnrichment enrichment = enrich_payload_with_wrapper_timing(
            api,
            type,
            ts,
            payload_json,
            trace_overhead_start_ns
        );
        ts = enrichment.emitted_ts_us;
        const std::string& enriched_payload = enrichment.payload_json;
        if (!fakecuda::capture::writes_trace()) {
            finish_trace_overhead();
            return ts;
        }
        emit_flexmaya_hook_record(api, type, ts, enriched_payload);
        pid_t pid = getpid();
        pid_t tid = gettid_();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (file_) {
                int written = 0;
                if (enriched_payload.empty()) {
                    written = fprintf(file_,
                        "{\"ts\":%lld,\"pid\":%d,\"tid\":%d,\"mod\":\"%s\",\"api\":\"%s\",\"type\":\"%s\"}\n",
                        ts, (int)pid, (int)tid, mod, api, type);
                } else {
                    written = fprintf(file_,
                        "{\"ts\":%lld,\"pid\":%d,\"tid\":%d,\"mod\":\"%s\",\"api\":\"%s\",\"type\":\"%s\",%s}\n",
                        ts, (int)pid, (int)tid, mod, api, type, enriched_payload.c_str());
                }
                if (written > 0) {
                    fakecuda::capture::count(fakecuda::capture::Counter::Records);
                    fakecuda::capture::count(fakecuda::capture::Counter::TraceBytes, written);
                }
                pending_events_ += 1;
                if (flush_per_event_ || pending_events_ >= flush_every_) {
                    fakecuda::capture::count(fakecuda::capture::Counter::Flushes);
                    fflush(file_);
                    pending_events_ = 0;
                }
            }
        }
        finish_trace_overhead();
        return ts;
    }

    long long log_lightweight_with_payload_impl(
        int lib,
        const char* api,
        const char* type,
        const std::string* prebuilt_payload_json,
        const TracePayloadBuilder* payload_builder
    ) {
        if (fakecuda_ras_active()) { fakecuda_ras_fail(api); return -1LL; }
        if (!fakecuda::capture::generates_trace()) return -1LL;
        const bool measure_mode =
            fakecuda::host_timing::HostTimingConfig::instance().mode() ==
            fakecuda::host_timing::Mode::kMeasure;
        const long long trace_overhead_start_ns = measure_mode
            ? fakecuda::host_timing::real_now_ns()
            : 0LL;
        auto finish_trace_overhead = [&]() {
            if (!measure_mode || trace_overhead_start_ns <= 0LL) {
                return;
            }
            fakecuda::trace::add_trace_writer_overhead_ns(
                fakecuda::host_timing::real_now_ns() - trace_overhead_start_ns
            );
        };
        if (!initialized_) {
            std::lock_guard<std::mutex> lock(mutex_);
            init();
        }
        if (!enabled_) {
            finish_trace_overhead();
            return -1LL;
        }

        api = canonical_trace_api_name(api);
        if (std::strcmp(api, "cudaGetDevice") == 0) {
            ++cuda_get_device_raw_count_;
        }
        const char* mod = (lib >= 0 && lib < 6) ? modNames_[lib] : "unknown";
        if (!should_log_event_(api, type)) {
            finish_trace_overhead();
            return -1LL;
        }
        fakecuda::host_timing::maybe_block_measure_before_trace(api, type, mod);
        long long ts = fakecuda::host_timing::reserve_trace_timestamp_us(api, type, mod);

        if (trace_binary_enabled()) {
            ts = log_binary_payload_(mod, api, type, ts, prebuilt_payload_json, payload_builder,
                                     trace_overhead_start_ns, true);
            finish_trace_overhead();
            return ts;
        }
        std::string built_payload_json;
        const std::string* payload_json_ptr = prebuilt_payload_json;
        if (payload_json_ptr == nullptr) {
            if (payload_builder != nullptr) {
                built_payload_json = payload_builder->to_json_fragment();
            }
            payload_json_ptr = &built_payload_json;
        }
        const auto owner_token = fakecuda::capture::scoped_token();
        const std::string scoped_payload = owner_token == 0 ? std::string{} :
            append_json_fragment(*payload_json_ptr, "\"capture_scope\":" + std::to_string(owner_token));
        const std::string& payload_json = owner_token == 0 ? *payload_json_ptr : scoped_payload;
        fakecuda::capture::count(fakecuda::capture::Counter::Enrichments);
        const WrapperTimingEnrichment enrichment = enrich_payload_with_wrapper_timing(
            api,
            type,
            ts,
            payload_json,
            trace_overhead_start_ns
        );
        ts = enrichment.emitted_ts_us;
        const std::string enriched_payload = (
            enrichment.payload_json == payload_json ||
            enrichment.payload_json.find("\"host_duration_us\"") == std::string::npos
        )
            ? enrich_payload_with_trace_context(enrichment.payload_json)
            : enrichment.payload_json;
        if (!fakecuda::capture::writes_trace()) {
            finish_trace_overhead();
            return ts;
        }
        emit_flexmaya_hook_record(api, type, ts, enriched_payload);
        pid_t pid = getpid();
        pid_t tid = gettid_();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (file_) {
                int written = 0;
                if (enriched_payload.empty()) {
                    written = fprintf(file_,
                        "{\"ts\":%lld,\"pid\":%d,\"tid\":%d,\"mod\":\"%s\",\"api\":\"%s\",\"type\":\"%s\"}\n",
                        ts, (int)pid, (int)tid, mod, api, type);
                } else {
                    written = fprintf(file_,
                        "{\"ts\":%lld,\"pid\":%d,\"tid\":%d,\"mod\":\"%s\",\"api\":\"%s\",\"type\":\"%s\",%s}\n",
                        ts, (int)pid, (int)tid, mod, api, type, enriched_payload.c_str());
                }
                if (written > 0) {
                    fakecuda::capture::count(fakecuda::capture::Counter::Records);
                    fakecuda::capture::count(fakecuda::capture::Counter::TraceBytes, written);
                }
                pending_events_ += 1;
                if (flush_per_event_ || pending_events_ >= flush_every_) {
                    fakecuda::capture::count(fakecuda::capture::Counter::Flushes);
                    fflush(file_);
                    pending_events_ = 0;
                }
            }
        }
        finish_trace_overhead();
        return ts;
    }

public:
    ~TraceLogger() {
        if (file_) {
            if (pending_events_ > 0) {
                fakecuda::capture::count(fakecuda::capture::Counter::Flushes);
                fflush(file_);
                pending_events_ = 0;
            }
            fclose(file_);
            file_ = nullptr;
        }
        if (!raw_audit_path_.empty()) {
            FILE* audit = fopen(raw_audit_path_.c_str(), "w");
            if (audit != nullptr) {
                fprintf(audit, "{\"api\":\"cudaGetDevice\",\"count\":%llu}\n",
                    static_cast<unsigned long long>(cuda_get_device_raw_count_));
                fclose(audit);
            }
        }
    }
};

// Module name mapping (matches Frida trace "mod" field)
inline const char* TraceLogger::modNames_[] = {
    "libcuda.so.1",       // CUDA    = 0
    "libcudart.so.12",    // CUDART  = 1
    "libcublas.so.12",    // CUBLAS  = 2
    "libcublasLt.so.12",  // CUBLASLT= 3
    "libnvidia-ml.so.1",  // NVML    = 4
    "libnccl.so.2"        // NCCL    = 5
};

// Main tracing macro — add at the top of each stub function
// lib: LibType enum (CUDA, CUDART, CUBLAS, CUBLASLT, NVML, NCCL)
// name: API function name string
// type: category string (kernel_launch, blas_compute, mem_copy, mem_alloc,
//       stream_op, context_op, nccl_collective, other)
#define TRACE_API(lib, name, type) \
    (fakecuda_ras_active() ? (fakecuda_ras_fail(name), -1LL) : \
     TraceLogger::instance().log(lib, name, type))

#define TRACE_API_EX(lib, name, type, payload) \
    (fakecuda_ras_active() ? (fakecuda_ras_fail(name), -1LL) : \
     TraceLogger::instance().log_with_payload(lib, name, type, payload))

#define TRACE_API_LIGHT_EX(lib, name, type, payload) \
    (fakecuda_ras_active() ? (fakecuda_ras_fail(name), -1LL) : \
     TraceLogger::instance().log_lightweight_with_payload(lib, name, type, payload))

inline void TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR(const char* source_api) {
    // Maya's raw emulator trace is the application/API stream plus explicit
    // hostDelay gaps. Do not inject fake cudaGetLastError calls after launch
    // or BLAS wrappers; real cudaGetLastError() stubs are traced normally.
    (void)source_api;
}
