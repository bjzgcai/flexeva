#pragma once

#include <flexeva/maya/model.hpp>
#include <span>

namespace flexeva::maya {

enum class NodeKind : std::uint32_t { Host = 1, Device = 2, Communication = 3 };

// All references are stable RAS event identities. Cost contains only this
// execution node's explicit model duration, never a capture wall timestamp.
struct Execution {
    ras::EventId origin;
    std::uint64_t boundary = 0, stream = 0, ordinal = 0;
    std::uint32_t rank = 0, opcode = 0;
    NodeKind kind = NodeKind::Host;
    std::uint32_t reserved = 0;
    Tick cost = 0;
};
struct Timing {
    ras::EventId execution;
    Tick start = 0, end = 0;
    std::uint64_t topological_order = 0;
};
struct EngineResult {
    ras::Version version;
    Feedback feedback;
};

// Input already contains this candidate's Source and Trace. Internal versions
// are returned only after graph validation and scheduling succeed. The parent
// is immutable, including when a communication mismatch or cycle is rejected.
EngineResult evaluate_engine(ras::Store&, const Candidate* parent,
    const ras::Version& trace_version, const Layers&, const Policies&,
    const TraceDelta&, std::span<const std::uint64_t> cost_dirty_boundaries,
    const CostProvider&, Statistics&);

} // namespace flexeva::maya
