#include <flexeva/ras/ras.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <tuple>
#include <vector>

using namespace flexeva::ras;

static void check(bool condition, const char* message) {
    if (!condition) throw Error(message);
}

struct Input {
    std::array<unsigned, 3> value{3, 7, 3};
    std::array<std::size_t, 3> upper_bytes{64, 128, 64};
    std::array<std::size_t, 6> lower_bytes{131072, 48, 1024, 8192, 256, 192};
    std::array<std::size_t, 6> producer{0, 0, 1, 1, 2, 2};
};

struct Graph {
    Store store;
    Version version;
    std::array<LayerId, 2> layers;
    std::array<LaneId, 3> lanes;
    std::array<PartitionId, 9> parts;
    std::array<RelationId, 6> links;
    std::array<const std::byte*, 9> written{};
};

static void emit(Graph& graph, Transaction& tx, const Input& input, std::size_t part, bool replace) {
    const auto upper = part < 3;
    const auto local = upper ? part : part - 3;
    const auto size = upper ? input.upper_bytes[local] : input.lower_bytes[local];
    const auto value = input.value[upper ? local : input.producer[local]];
    auto writer = tx.partition(graph.layers[upper ? 0 : 1], size, {101, 1, 1},
                               replace ? graph.parts[part] : PartitionId{});
    auto borrow = writer.payload().reserve(size);
    auto bytes = borrow.bytes();
    graph.written[part] = bytes.data();
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::byte((value + part * 13 + i * 17) % 251);
    borrow.commit();
    if (upper) {
        LanePosition position{graph.lanes[0], local};
        writer.event(1, {0, size}, {&position, 1});
    } else {
        std::array<LanePosition, 2> positions{{{graph.lanes[1], local * 2}, {graph.lanes[2], local * 2}}};
        writer.event(2, {0, size / 2}, positions);
        positions[0].order++;
        writer.event(3, {size / 2, size - size / 2}, {positions.data(), 1});
    }
    graph.parts[part] = writer.finish();
}

static void build(Graph& graph, const Input& input) {
    auto tx = graph.store.begin();
    graph.layers = {tx.add_layer(10), tx.add_layer(20)};
    graph.lanes = {tx.add_lane(graph.layers[0], 0), tx.add_lane(graph.layers[1], 0), tx.add_lane(graph.layers[1], 1)};
    for (std::size_t part = 0; part < graph.parts.size(); ++part) emit(graph, tx, input, part, false);
    for (std::size_t i = 0; i < 6; ++i)
        graph.links[i] = tx.add_relation({{graph.parts[input.producer[i]]}, {graph.parts[i + 3]}, RelationKind::Lineage});
    tx.add_relation({{graph.parts[4], 1}, {graph.parts[5], 0}, RelationKind::Order, 9});
    graph.version = tx.publish();
    for (std::size_t i = 0; i < graph.parts.size(); ++i)
        check(graph.version.partition(graph.parts[i]).payload().bytes().span().data() == graph.written[i],
              "seal moved payload away from its write address");
}

template<class T, std::size_t N> static std::size_t ordinal(const std::array<T, N>& ids, T id) {
    auto found = std::find(ids.begin(), ids.end(), id);
    check(found != ids.end(), "unexpected semantic identity");
    return std::size_t(found - ids.begin());
}

using Edge = std::tuple<std::size_t, std::uint64_t, std::size_t, std::uint64_t, RelationKind, std::uint64_t>;
static std::vector<Edge> edges(const Graph& graph, const Version& version) {
    std::vector<Edge> result;
    for (auto part : graph.parts) version.outgoing(part, [&](RelationId, const Relation& relation) {
        result.emplace_back(ordinal(graph.parts, relation.source.partition), relation.source.event,
                            ordinal(graph.parts, relation.target.partition), relation.target.event,
                            relation.kind, relation.tag);
    });
    std::sort(result.begin(), result.end());
    return result;
}

// The oracle is a fresh Store rebuilt from semantic inputs. This whole-state
// comparison runs outside refresh counters and timing, never inside publication.
static void compare_full(const Graph& actual, const Version& version, const Input& input) {
    Graph full;
    build(full, input);
    for (std::size_t part = 0; part < actual.parts.size(); ++part) {
        auto a = version.partition(actual.parts[part]);
        auto b = full.version.partition(full.parts[part]);
        auto ap = a.payload().bytes(), bp = b.payload().bytes();
        check(std::ranges::equal(ap.span(), bp.span()), "selective/full payload mismatch");
        check(a.event_count() == b.event_count(), "selective/full event count mismatch");
        for (std::size_t i = 0; i < a.event_count(); ++i) {
            auto ae = a.events()[i], be = b.events()[i];
            check(std::tie(ae.kind, ae.payload.offset, ae.payload.size, ae.positions.offset, ae.positions.size) ==
                  std::tie(be.kind, be.payload.offset, be.payload.size, be.positions.offset, be.positions.size),
                  "selective/full event metadata mismatch");
        }
        auto apos = a.positions(), bpos = b.positions();
        check(apos.size() == bpos.size(), "selective/full lane count mismatch");
        for (std::size_t i = 0; i < apos.size(); ++i)
            check(ordinal(actual.lanes, apos[i].lane) == ordinal(full.lanes, bpos[i].lane) && apos[i].order == bpos[i].order,
                  "selective/full lane order mismatch");
    }
    check(edges(actual, version) == edges(full, full.version), "selective/full relation mismatch");
}

static Version refresh(Graph& graph, const Version& base, const Input& input, std::size_t changed, const char* name) {
    const auto before = graph.store.stats();
    auto tx = graph.store.begin(base);
    std::array<bool, 9> generated{};
    emit(graph, tx, input, changed, true); generated[changed] = true;
    base.outgoing(graph.parts[changed], [&](RelationId, const Relation& relation) {
        if (relation.kind != RelationKind::Lineage) return;
        auto part = ordinal(graph.parts, relation.target.partition);
        check(!generated[part], "partition generated more than once");
        emit(graph, tx, input, part, true); generated[part] = true;
    });
    auto next = tx.publish();
    const auto after = graph.store.stats();
    check(after.view_requests == before.view_requests, "refresh inspected existing payload");
    check(after.copied_payload_bytes == before.copied_payload_bytes, "refresh copied existing payload");
    std::size_t count = 0, bytes = 0;
    for (std::size_t i = 0; i < generated.size(); ++i) {
        auto old = base.partition(graph.parts[i]).payload();
        auto fresh = next.partition(graph.parts[i]).payload();
        if (generated[i]) {
            check(!fresh.same_storage(old), "changed partition reused stale payload");
            check(fresh.bytes().span().data() != old.bytes().span().data(), "live historical payload was overwritten");
            check(fresh.bytes().span().data() == graph.written[i], "publish moved newly written payload");
            ++count; bytes += fresh.size();
        } else check(fresh.same_storage(old), "unselected partition lost its storage");
    }
    check(count == 3 && after.allocated_payload_bytes - before.allocated_payload_bytes == bytes,
          "unexpected generation or allocation outside affected partitions");
    compare_full(graph, next, input);
    std::cout << "PASS " << name << " rebuilt=3 reused=6 generated_bytes=" << bytes << " payload_copy_bytes=0 full_equal=yes\n";
    return next;
}

static void run(Graph& graph) {
    Input input;
    const auto original = input;
    build(graph, input);
    const auto v0 = graph.version;
    const auto before = graph.store.stats();
    auto noop = graph.store.begin(v0).publish();
    const auto after = graph.store.stats();
    check(after.created_index_pages == before.created_index_pages && after.index_page_visits == before.index_page_visits &&
          after.allocated_payload_bytes == before.allocated_payload_bytes && after.view_requests == before.view_requests,
          "no-op publication scanned or rebuilt storage");
    std::cout << "PASS no-op rebuilt=0 reused=9 payload_copy_bytes=0 index_pages_created=0\n";

    input.value[1] = 8;
    auto v1 = refresh(graph, noop, input, 1, "same-size change");
    input.upper_bytes[1] *= 2; input.lower_bytes[2] *= 2; input.lower_bytes[3] *= 2;
    auto v2 = refresh(graph, v1, input, 1, "resized change");

    auto before_al = graph.store.stats();
    auto tx = graph.store.begin(v2);
    tx.erase_relation(graph.links[4]);
    graph.links[4] = tx.add_relation({{graph.parts[0]}, {graph.parts[7]}, RelationKind::Lineage});
    auto v3 = tx.publish();
    auto after_al = graph.store.stats();
    check(after_al.allocated_payload_bytes == before_al.allocated_payload_bytes &&
          after_al.view_requests == before_al.view_requests && after_al.copied_payload_bytes == before_al.copied_payload_bytes,
          "AL-only change allocated, read or copied payload");
    for (auto part : graph.parts)
        check(v3.partition(part).payload().same_storage(v2.partition(part).payload()), "AL update replaced a payload");
    input.producer[4] = 0; // Producers 0 and 2 have equal values; provenance alone changes.
    compare_full(graph, v3, input);
    std::cout << "PASS AL-only rebuilt=0 reused=9 generated_bytes=0 payload_copy_bytes=0 full_equal=yes\n";

    auto branch_input = original;
    branch_input.value[2] = 19;
    auto branch = refresh(graph, v0, branch_input, 2, "historical branch");
    compare_full(graph, v0, original);
    compare_full(graph, v3, input);
    compare_full(graph, branch, branch_input);
    std::cout << "PASS old version and independent branches preserved\n";
    graph.version = {};
}

int main() {
    try {
        Graph graph;
        run(graph);
        const auto stats = graph.store.stats();
        check(stats.capacity_bytes == 0 && stats.metadata_bytes == 0 && stats.index_bytes == 0 &&
              stats.live_payload_allocations == 0 && stats.versions == 0, "released graph retains storage");
    } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
