#pragma once

#include <flexeva/core/runtime.hpp>
#include <array>

namespace flexeva::maya::paper {

using DependencyResourceKey = std::array<std::uint64_t, 4>;
enum class FootprintMode : std::uint64_t { Define, UsePrevious, UseNext };
struct Footprint {
    DependencyResourceKey resource{};
    std::uint64_t sequence = 0;
    FootprintMode mode = FootprintMode::Define;
    auto operator<=>(const Footprint&) const = default;
};
struct IndexUpdate {
    core::runtime::Id partition = 0;
    std::vector<Footprint> footprints;
};
inline constexpr core::runtime::Id dependency_index_slot_prefix = 0x4d440100ULL;

// All persistent state belongs to evaluator directories in the active RAS
// transaction. Only changed footprints and affected consumer intervals are read.
// Definitions are unique by exact resource and sequence. Previous/next uses
// choose strictly earlier/later definitions; self-partition edges are omitted.
core::runtime::DependencyDelta update_dependencies(core::runtime::Context&,
    std::span<const IndexUpdate> updates, std::span<const core::runtime::Id> removed = {});
std::vector<Footprint> read_dependency_footprints(core::runtime::Context&,
    core::runtime::Id partition);

} // namespace flexeva::maya::paper
