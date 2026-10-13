#pragma once

#include <flexeva/ras/ras.hpp>
#include <ras_capture.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace flexeva::maya {

using Tick = std::uint64_t; // Explicit model nanoseconds, never capture wall time.
inline constexpr ras::SchemaId source_schema = 0x4d4159410001ULL;
inline constexpr ras::SchemaId trace_schema = 0x4d4159410002ULL;
inline constexpr ras::SchemaId execution_schema = 0x4d4159410003ULL;
inline constexpr ras::SchemaId timing_schema = 0x4d4159410004ULL;
inline constexpr ras::SchemaId state_schema = 0x4d4159410005ULL;

struct Layers {
    ras::LayerId source, trace, execution, timing;
};

struct PartitionPolicy {
    std::size_t max_payload_bytes = 4096;
    auto operator<=>(const PartitionPolicy&) const = default;
};
struct Policies {
    PartitionPolicy trace, execution, timing;
    auto operator<=>(const Policies&) const = default;
};

// Native pointers are resolved to logical resource-creation EventId.value.
// The communicator UID is likewise normalized by its UID-producing event.
// Event identity is in RAS metadata; its physical partition is not in payload.
struct Operation {
    std::uint64_t boundary = 0;
    std::uint64_t sequence = 0;
    std::uint32_t rank = 0;
    std::uint32_t reserved = 0;
    FakeCudaRasRecord call{};
    std::uint64_t src_offset = 0, dst_offset = 0;
    std::uint64_t a_offset = 0, b_offset = 0, c_offset = 0;
    std::uint64_t producers[16]{};
    std::uint32_t producer_count = 0, padding = 0;
};

struct Cost {
    Tick host = 0, device = 0;
    auto operator<=>(const Cost&) const = default;
};
using CostProvider = std::function<Cost(ras::EventId, const Operation&)>;

struct Statistics {
    std::uint64_t hooks = 0, retained_hooks = 0;
    std::uint64_t raw_records_generated = 0, raw_payload_bytes_written = 0;
    std::uint64_t repacked_bytes = 0;
    // Bytes copied into fresh immutable Execution/Timing blocks during refresh.
    // Repartition copies are counted separately above; stack POD descriptors
    // and compact index updates are not whole-payload copies.
    std::uint64_t refresh_copy_bytes = 0;
    std::uint64_t trace_partitions_read = 0, trace_partitions_written = 0;
    std::uint64_t execution_partitions_read = 0, execution_partitions_written = 0;
    std::uint64_t timing_partitions_read = 0, timing_partitions_written = 0;
    std::uint64_t cost_calls = 0, scheduled_events = 0, index_visits = 0;
    bool structural_rebuild = false;
};

struct Feedback {
    Tick makespan = 0;
    // Small terminal summary; event-level times live only in Timing RAS.
    std::vector<std::pair<std::uint32_t, Tick>> rank_end;
    const char* cost_basis = "controlled";
};

struct Candidate {
    ras::Version version;
    Layers layers;
    Policies policies;
    std::uint64_t context = 0;
    Feedback feedback;
};

struct TraceDelta {
    std::vector<ras::PartitionId> changed, removed;
    std::vector<ras::EventId> changed_events;
    bool structural = false;
    bool layout_only = false;
};

} // namespace flexeva::maya
