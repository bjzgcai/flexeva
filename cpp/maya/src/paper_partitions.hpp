#pragma once
#include <flexeva/core/runtime.hpp>
#include "paper_lanes.hpp"
#include <array>

namespace flexeva::maya::paper {
// Logical body identity is independent of its physical block packing. The
// public payload describes the body held in Partition::bundles.front().
struct BodyDescriptor {
    std::uint64_t bytes=0, blocks=0, events=0;
};
// Only regenerated members are materialized in this scratch structure. All
// durable membership, equivalence buckets and ordering live in RAS directories.
struct PartitionCandidate {
    core::runtime::Id origin=0;
    std::uint64_t ordinal=0, rank=0, first=0, last=0, events=0;
    std::array<std::uint64_t,4> digest{};
    core::runtime::Evidence state;
    ras::Buffer payload;
    LaneContext lane;
    std::uint64_t receipt_chunk=0; // Physical block index, independent of the stable occurrence ordinal.
    std::vector<ras::Buffer> blocks; // Nonempty: exact ordered body bytes; payload is BodyDescriptor.
};
core::runtime::PartitionDelta merge_partitions(
    const core::runtime::PartitionRequest&, core::runtime::Context&,
    std::span<const PartitionCandidate>, bool compact);
std::vector<std::uint32_t> partition_ranks(const core::runtime::Partition&,
                                         const std::function<std::uint32_t(core::runtime::Id)>&);
}
