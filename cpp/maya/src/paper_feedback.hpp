#pragma once

#include <flexeva/maya/paper_runtime.hpp>
#include "paper_scheduler.hpp"

namespace flexeva::maya::paper {

struct TerminalCreatedResource {
    // 1: stream, 2: event, 3: communicator; id is already rank-projected.
    std::uint32_t rank = 0, kind = 0;
    std::uint64_t id = 0;
};
// Borrowed descriptions of canonical normalized output. These are computation
// inputs, not another RAS layer or another set of Core partitions.
struct TerminalOutput {
    Id id = 0;
    // Producer-issued immutable body identity. Fresh capture allocates a new
    // version; adoption retains it. A digest is not an equality certificate.
    Id body_version = 0;
    Evidence semantic, projection;
    std::uint64_t count = 0, first = 0;
    std::vector<ras::Buffer> blocks;
    std::vector<std::uint32_t> ranks;
    // Producer-owned (uint64 block, uint64 offset) records, already in RAS.
    // A sliced metadata buffer is adopted directly without schema conversion,
    // iteration, or another index allocation. block is receipt-relative.
    ras::Buffer position_backing;
    std::uint64_t position_block_base = 0;
    // Optional producer-built (physical block, byte offset) index. When absent,
    // feedback builds and caches it once; publication does not read the body.
    // Legacy/import path only; normal capture supplies position_backing.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> positions;
    std::vector<TerminalCreatedResource> created_resources;
    bool creations_complete = false;
};
struct TerminalFeedbackInput {
    std::span<const TerminalOutput> outputs;
    std::span<const Id> removed;
    Evidence replay_identity;
    // count/read are supplied by terminal_feedback. Entry resources declared
    // here are filtered against creation operations in the current window.
    ReplayInput seeds;
    // Projects sequence, rank, call, d and aux without copying output evidence.
    std::function<NativeRecord(const TerminalOutput&, std::uint32_t,
                              const NativeRecord&)> project;
};
struct TerminalFeedbackResult {
    core::runtime::State state;
    Feedback feedback;
    // These count terminal evaluation work, never normalization/publication.
    std::uint64_t index_body_reads = 0, feedback_body_reads = 0;
    bool reused = false;
};

// Call from Trace build_state. Cost annotations and the scheduler's dependency
// graph/timing annotations share this layer's transaction-scoped directories.
// NativeRecord bodies remain solely in canonical normalized storage; the graph
// stores only derived scheduling evidence and never another full native trace.
TerminalFeedbackResult terminal_feedback(core::runtime::Context&,
    const TerminalFeedbackInput&, const Options&, Metrics&);
Feedback decode_terminal_feedback(const ras::Buffer&);

} // namespace flexeva::maya::paper
