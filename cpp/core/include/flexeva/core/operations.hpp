#pragma once

#include <flexeva/ras/ras.hpp>

#include <functional>
#include <ranges>
#include <set>
#include <utility>
#include <vector>

namespace flexeva::core {

// Evaluators distinguish changed values/dependencies from physical relocation.
// These are selected IDs, not a duplicate event or payload store. Relation
// changes may require propagation even when no event payload changed.
struct LayerDelta {
    std::vector<ras::EventId> events;
    bool relations_changed = false;
    bool layout_changed = false;
    bool semantic_empty() const noexcept { return events.empty() && !relations_changed; }
};
struct LayerUpdate {
    ras::Version version;
    LayerDelta delta;
};

// Lane/Payload/Partition produce RAS records through layer.hpp; the evaluator
// resolves DataDep through the indexed relation writer. Once these operations
// yield a LayerUpdate, this driver enforces the inter-layer refresh ordering:
//
// Link(version, delta) -> Version repairs incoming provenance, including when
// values are unchanged. Boundary(linked_version, delta) -> selected locates the
// next layer's affected region. Apply(linked_version, selected) -> Version runs
// the evaluator transform only for that selection. All callbacks are ordinary
// C++ callables and retain their domain-specific semantics and selection types.
template<class Link, class Boundary, class Apply>
ras::Version refresh_layer(LayerUpdate update, Link&& link, Boundary&& boundary, Apply&& apply) {
    update.version = std::invoke(std::forward<Link>(link), update.version, update.delta);
    if (update.delta.semantic_empty()) return std::move(update.version);
    decltype(auto) selected = std::invoke(std::forward<Boundary>(boundary), update.version, update.delta);
    return std::invoke(std::forward<Apply>(apply), update.version, selected);
}

// Link supplies producers; Core repairs only the selected targets' incoming
// lineage index. No event payload is read here. A source can be a boundary or
// any other producer required by the evaluator's dependency semantics.
// Enumerate is callable as enumerate(version, target, emit_producer), where
// emit_producer takes a PartitionId. Duplicate targets/producers are harmless.
template<std::ranges::input_range Targets, class Enumerate>
ras::Version rebind_lineage(ras::Store& store, const ras::Version& version,
                            const Targets& targets, Enumerate&& enumerate) {
    if (std::ranges::begin(targets) == std::ranges::end(targets)) return version;
    auto tx = store.begin(version);
    std::set<ras::PartitionId> visited;
    for (const auto target : targets) {
        if (!visited.insert(target).second) continue;
        version.incoming(target, [&](ras::RelationId id, const ras::Relation& relation) {
            if (relation.kind == ras::RelationKind::Lineage) tx.erase_relation(id);
        });
        std::set<ras::PartitionId> producers;
        std::invoke(enumerate, version, target, [&](ras::PartitionId source) {
            if (producers.insert(source).second)
                tx.add_relation({{source}, {target}, ras::RelationKind::Lineage});
        });
    }
    return tx.publish();
}

} // namespace flexeva::core
