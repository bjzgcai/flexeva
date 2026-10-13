#pragma once

#include <flexeva/core/runtime.hpp>
#include <flexeva/maya/model.hpp>

#include <array>
#include <functional>
#include <span>
#include <vector>

namespace flexeva::maya::paper {

inline constexpr ras::SchemaId replay_timing_schema = 0x4d4159411008ULL;

// A borrowed input cursor reconstructs one small scheduling descriptor from
// immutable RAS storage. It must support repeated random reads. Input ordinals
// may interleave ranks; each rank's sequence must be strictly increasing.
// An all-zero COMM_INIT UID reserves a handle whose first transfer must carry
// its normalized logical UID. Nonzero transfer nranks/rank are validated.
struct ReplayOperation {
    FakeCudaRasRecord call{};
    Cost cost{};
    std::uint64_t boundary = 0, sequence = 0;
    std::uint32_t rank = 0;
};
using ReplayUid = std::array<std::uint8_t, 128>;
struct ReplayMembership { ReplayUid uid{}; std::vector<std::uint32_t> members; };
struct ReplayStream { std::uint32_t rank = 0; std::uint64_t id = 0; bool blocking = true; };
struct ReplayEvent { std::uint32_t rank = 0; std::uint64_t id = 0; };
struct ReplayCommunicator {
    std::uint32_t rank = 0, local_rank = 0;
    std::uint64_t id = 0;
    ReplayUid uid{};
    std::vector<std::uint32_t> members;
};
struct ReplayInput {
    std::uint64_t count = 0;
    std::function<ReplayOperation(std::uint64_t)> read;
    std::span<const ReplayMembership> memberships;
    std::span<const ReplayStream> streams;
    std::span<const ReplayEvent> events;
    std::span<const ReplayCommunicator> communicators;
    // Scratch and output blocks have fixed capacity and never relocate.
    std::size_t block_bytes = 64 * 1024;
};
enum class ReplayNodeKind : std::uint32_t { Host = 1, Device = 2, Communication = 3 };
struct ReplayTiming {
    std::uint64_t origin = 0, boundary = 0, sequence = 0;
    std::uint32_t rank = 0;
    ReplayNodeKind kind = ReplayNodeKind::Host;
    Tick start = 0, end = 0;
    std::uint64_t topological_order = 0;
};
struct ReplayResult {
    // Full replay returns a contiguous backing. Cached replay returns immutable
    // pages covering the complete graph, including timings reused from parent.
    std::vector<ras::Buffer> timing;
    Feedback feedback;
    std::uint64_t scheduled_events = 0, dependency_edges = 0, input_reads = 0;
    std::uint64_t total_events = 0, reused_events = 0, updated_events = 0;
    bool graph_rebuilt = false;
    std::uint64_t scratch_capacity_bytes = 0, timing_capacity_bytes = 0;
};

struct ReplayRange { std::uint64_t first = 0, count = 0; };

// This schedules the complete supplied evaluation window. It makes no claim
// of incremental scheduler restoration. Every large scratch/output allocation
// uses this callback's RAS Context; no Store is created or published here.
ReplayResult replay(core::runtime::Context&, const ReplayInput&);

// Backend-private graph/timing annotations live in the caller's RAS transaction.
// Identity must describe the ordered input layout and all entry resources, but
// may exclude costs and native body identity. The caller declares every range
// whose operation or cost may differ. Only those native inputs are read; an
// exact structural mismatch conservatively rebuilds the complete graph. Stable
// graphs propagate timing changes through successors until their ends agree.
// include_timing=false avoids gathering unchanged timing page references when
// only aggregate feedback is needed. It does not discard cached timing pages.
ReplayResult replay_cached(core::runtime::Context&, const ReplayInput&,
    std::span<const ReplayRange> changed, const core::runtime::Evidence& identity,
    bool include_timing = true);

} // namespace flexeva::maya::paper
