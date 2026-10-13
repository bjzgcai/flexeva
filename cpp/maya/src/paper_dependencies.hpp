#pragma once

#include <flexeva/core/runtime.hpp>
#include <flexeva/maya/model.hpp>

#include <functional>

namespace flexeva::maya::paper {

struct DependencyOperation {
    std::uint32_t rank = 0;
    std::uint64_t sequence = 0;
    FakeCudaRasRecord call{}; // Native-normalized resources and logical membership UID.
};
using DependencyVisitor = std::function<void(const DependencyOperation&)>;
// The caller opens only this requested partition and expands each active rank
// once. Host/device lane projections must not duplicate the raw call.
// Each rank's records form one contiguous interval, as guaranteed by capture
// sealing and active-lane compaction's at-most-one-member-per-rank rule.
using DependencyPartitionReader = std::function<void(core::runtime::Id, const DependencyVisitor&)>;

// Evaluator-local resource/communication indexes belong to the current RAS
// transaction. Unchanged partition payloads are never opened by this helper.
// These are reuse dependencies, not a second copy of the scheduler's event DAG.
// Ordinary host/stream order stays in lane indexes; full-window replay resolves
// timing projections, default-stream joins and communicator readiness exactly.
core::runtime::DependencyDelta native_dependencies(
    const core::runtime::LayerView&, const core::runtime::LayerDelta&,
    core::runtime::Context&, const DependencyPartitionReader&);

} // namespace flexeva::maya::paper
