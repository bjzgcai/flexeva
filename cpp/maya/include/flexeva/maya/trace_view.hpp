#pragma once

#include <flexeva/maya/paper_runtime.hpp>

namespace flexeva::maya::paper {

// Read-only access to the normalized outputs owned by one published snapshot.
// Opening a view does not enumerate outputs, materialize partitions, or decode
// native records. The Core views keep the snapshot and its capabilities alive.
class TraceView {
public:
    TraceView(const Session& session, const Snapshot& snapshot)
        : outputs_(session.trace_outputs(snapshot)), state_(session.layer(snapshot, 1)) {}

    core::runtime::Receipt output(Id boundary) const { return outputs_.output(boundary); }
    ras::Buffer block(Id boundary, std::size_t ordinal) const {
        return outputs_.output_chunk(boundary, ordinal);
    }
    std::vector<ras::Buffer> range(
            Id boundary, std::span<const core::runtime::OutputSlice> slices) const {
        return outputs_.output_range(boundary, slices);
    }
    void downstream(Id producer,
                    const std::function<void(const core::runtime::OutputTarget&)>& visit) const {
        outputs_.downstream(producer, visit);
    }
    // Enumeration is explicit; targeted readers use output() or downstream().
    void outputs(const std::function<void(const core::runtime::Receipt&)>& visit) const {
        outputs_.outputs(visit);
    }
    core::runtime::State state() const { return state_.state(); }

private:
    core::runtime::BoundaryView outputs_;
    core::runtime::LayerView state_;
};

} // namespace flexeva::maya::paper
