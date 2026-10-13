#pragma once

#include <flexeva/ras/ras.hpp>

#include <cstdint>
#include <string_view>

struct FakeCudaRasRecordV2;

namespace flexeva::maya {

enum class PaperCostPolicy { ArtifactCompatible, DeterministicV1 };

struct EventCostInput {
    std::string_view api;
    std::string_view kind;
    // ArtifactCompatible consumes the original stream number. DeterministicV1
    // requires a stable evaluator-assigned logical stream, never a pointer.
    std::uint64_t stream = 0;
    std::uint64_t bytes = 0;
    std::uint64_t count = 0;
    std::int64_t rank = 0;
    bool blocking = false;
    double duration_hint_us = 0;
};

// Decode a complete ABI-2 record without allocating or reading borrowed names.
// stream and rank must already be logical IDs. The returned strings are static.
// Retained records have no cost payload: reuse their prior Execution receipt.
EventCostInput native_event_cost_input(const FakeCudaRasRecordV2&);

struct PaperCostCounters {
    std::uint64_t events = 0;
    std::uint64_t hint_uses = 0;
    // Matches the old predictor's calls: includes cache hits, excludes hints.
    std::uint64_t prediction_calls = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t cache_probes = 0;
};

// Pure deterministic-v1 operation for Core callbacks: no transaction, cache,
// or host-duration hints. Store the returned cost in the Execution RAS payload.
double predict_event_us(const EventCostInput&);

// The artifact's local synthetic forest formula, not a trained GPU estimator.
// All durable cache entries are immutable RAS buffers in cache_directory().
// The caller owns the transaction and must keep it unmoved and open while this
// object is used. A fresh cache reproduces replay_trace_once's cache lifetime;
// reopening an inherited cache explicitly requests continuation semantics.
class PaperCostModel {
public:
    PaperCostModel(ras::Transaction&, PaperCostPolicy, ras::DirectoryId cache = {});
    double predict_us(const EventCostInput&);
    std::string_view identity() const noexcept;
    const PaperCostCounters& counters() const noexcept { return counters_; }
    ras::DirectoryId cache_directory() const noexcept { return cache_; }
private:
    ras::Transaction* transaction_;
    PaperCostPolicy policy_;
    ras::DirectoryId cache_;
    PaperCostCounters counters_;
};

} // namespace flexeva::maya
