#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include "capture_scope.h"

namespace fakecuda::capture {

// A process-wide startup ceiling, independent of the legacy trace sink settings.
// Opaque thread-bound scopes may suppress capture but cannot enable startup Off.
enum class Mode { Full, Discard, Off };
enum class Counter : std::size_t {
    ApiCalls, PayloadBuilders, PayloadFields, WrapperEntries, KernelArgParsers,
    JsonFragments, Enrichments, Hooks, ArenaAppends, Records, Flushes, TraceBytes,
    NcclSequenceAdvances,
    Count
};
static_assert(static_cast<std::size_t>(Counter::Count) == 13,
              "capture receipt ABI v1 counter layout must remain stable");

class State {
public:
    State() {
        const char* raw = std::getenv("FAKECUDA_CAPTURE_MODE");
        if (raw == nullptr || std::strcmp(raw, "full") == 0) {
            mode_ = Mode::Full;
        } else if (std::strcmp(raw, "discard") == 0) {
            mode_ = Mode::Discard;
        } else if (std::strcmp(raw, "off") == 0) {
            mode_ = Mode::Off;
        } else {
            fail("FAKECUDA_CAPTURE_MODE must be full, discard, or off");
        }
        const char* timing = std::getenv("FAKECUDA_HOST_TIMING_MODE");
        if (mode_ != Mode::Full && timing != nullptr && std::strcmp(timing, "sleep") == 0) {
            fail("capture ablation does not support FAKECUDA_HOST_TIMING_MODE=sleep");
        }
        const char* path = std::getenv("FAKECUDA_CAPTURE_STATS_PATH");
        if (path != nullptr && path[0] != '\0') {
            stats_path_ = path;
            const std::string pid = std::to_string(::getpid());
            std::size_t offset = 0;
            while ((offset = stats_path_.find("%p", offset)) != std::string::npos) {
                stats_path_.replace(offset, 2, pid);
                offset += pid.size();
            }
            stats_enabled_ = true;
        }
    }

    ~State() {
        if (!stats_enabled_) return;
        // Diagnostic-only output, once at normal process exit. No per-event I/O.
        FILE* file = std::fopen(stats_path_.c_str(), "w");
        if (file == nullptr) fail("cannot write FAKECUDA_CAPTURE_STATS_PATH");
        std::fprintf(file,
            "{\"schema_version\":1,\"mode\":\"%s\",\"pid\":%d,"
            "\"api_calls_scope\":\"trace_producing_native_entries\"",
            name(), static_cast<int>(::getpid()));
        static constexpr const char* names[] = {
            "api_calls", "payload_builders", "payload_fields", "wrapper_entries",
            "kernel_arg_parsers", "json_fragments", "enrichments", "hooks",
            "arena_appends", "records", "flushes", "trace_bytes", "nccl_sequence_advances"
        };
        for (std::size_t i = 0; i < counters_.size(); ++i) {
            std::fprintf(file, ",\"%s\":%llu", names[i],
                static_cast<unsigned long long>(counters_[i].load(std::memory_order_relaxed)));
        }
        std::fputs("}\n", file);
        if (std::fclose(file) != 0) fail("cannot flush FAKECUDA_CAPTURE_STATS_PATH");
    }

    Mode mode() const { return mode_; }
    const char* name() const {
        return mode_ == Mode::Full ? "full" : mode_ == Mode::Discard ? "discard" : "off";
    }
    void count(Counter counter, std::uint64_t amount = 1) {
        scope_count(static_cast<std::size_t>(counter), amount);
        if (stats_enabled_) {
            counters_[static_cast<std::size_t>(counter)].fetch_add(amount, std::memory_order_relaxed);
        }
    }

private:
    [[noreturn]] static void fail(const char* message) {
        std::fprintf(stderr, "[fakecuda-capture] ERROR: %s\n", message);
        std::abort();
    }
    Mode mode_ = Mode::Full;
    bool stats_enabled_ = false;
    std::string stats_path_;
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Counter::Count)> counters_{};
};

// C++ inline static storage is shared by the FakeCUDA DSOs, as are the existing
// header-owned trace/runtime singletons. Initialization is thread-safe.
inline State& state() {
    static State instance;
    return instance;
}
inline Mode mode() {
    const auto startup_mode = state().mode();
    return scoped_decision() == FAKECUDA_RETAIN ? Mode::Off : startup_mode;
}
inline bool generates_trace() { return !fakecuda_ras_active() && mode() != Mode::Off; }
inline bool writes_trace() { return !fakecuda_ras_active() && mode() == Mode::Full; }
inline void count(Counter counter, std::uint64_t amount = 1) {
    if (!fakecuda_ras_active()) state().count(counter, amount);
}

}  // namespace fakecuda::capture
