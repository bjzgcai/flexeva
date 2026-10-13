#include <flexeva/maya/engine.hpp>
#include <flexeva/core/layer.hpp>
#include <flexeva/core/operations.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <tuple>
#include <vector>

namespace flexeva::maya {
namespace {
using namespace ras;
constexpr auto terminal_key = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t graph_tag = 1;
struct Origin { EventId host, device; };
struct Terminal { std::uint32_t rank = 0, reserved = 0; EventId event; };
struct RawOrder { std::uint32_t rank = 0, reserved = 0; std::uint64_t sequence = 0; EventId event; };
struct Scratch { EventId event; std::uint64_t indegree = 0; Tick ready = 0; };

[[noreturn]] void fail(const char* what) { throw Error(what); }
Tick plus(Tick a, Tick b) {
    if (a > std::numeric_limits<Tick>::max() - b) fail("Maya modeled time overflow");
    return a + b;
}
template<class T> void state(Transaction& tx, LayerId layer, std::uint64_t key, std::span<const T> data) {
    core::put_state<T>(tx, layer, key, data, state_schema);
}
template<class T> void state(Transaction& tx, LayerId layer, std::uint64_t key, const T& value) {
    state<T>(tx, layer, key, std::span<const T>(&value, 1));
}
template<class T> ArrayView<T> state(const Version& v, LayerId layer, std::uint64_t key) {
    return core::state_view<T>(v, layer, key, state_schema);
}
Origin origin(const Version& v, const Layers& l, EventId id) {
    auto a = state<Origin>(v, l.execution, id.value);
    if (a.size() != 1) fail("Maya origin index cardinality mismatch");
    return a[0];
}
EventId timing_id(const Version& v, const Layers& l, EventId id) {
    auto a = state<EventId>(v, l.timing, id.value);
    if (a.size() != 1) fail("Maya timing index cardinality mismatch");
    return a[0];
}
bool has_device(std::uint32_t op) {
    switch (op) {
    case FAKECUDA_RAS_EVENT_RECORD: case FAKECUDA_RAS_STREAM_WAIT_EVENT:
    case FAKECUDA_RAS_MEMCPY_ASYNC: case FAKECUDA_RAS_MEMSET_ASYNC: case FAKECUDA_RAS_GEMM:
    case FAKECUDA_RAS_SEND: case FAKECUDA_RAS_RECV: case FAKECUDA_RAS_ALLREDUCE: return true;
    default: return false;
    }
}
bool communication(std::uint32_t op) {
    return op == FAKECUDA_RAS_SEND || op == FAKECUDA_RAS_RECV || op == FAKECUDA_RAS_ALLREDUCE;
}
Operation raw(const Version& v, EventId id) { return core::event_view<Operation>(v, id, trace_schema).get(); }
Execution execution(const Version& v, EventId id) { return core::event_view<Execution>(v, id, execution_schema).get(); }
Timing timing(const Version& v, EventId id) { return core::event_view<Timing>(v, id, timing_schema).get(); }

void erase_incident(Transaction& tx, const Version& v, const std::set<PartitionId>& parts) {
    std::set<RelationId> erase;
    for (auto p : parts) {
        v.incoming(p, [&](RelationId id, const Relation&) { erase.insert(id); });
        v.outgoing(p, [&](RelationId id, const Relation&) { erase.insert(id); });
    }
    for (auto id : erase) tx.erase_relation(id);
}
Version clear_downstream(Store& store, const Version& v, const Layers& l) {
    auto tx = store.begin(v);
    std::set<PartitionId> parts;
    std::set<StateId> states;
    v.partitions(l.execution, [&](PartitionId p) {
        parts.insert(p);
        auto part = v.partition(p);
        for (auto e : part.events().span()) {
            const auto x = execution(v, e.id);
            if (auto s = v.find_state(l.execution, x.origin.value)) states.insert(s);
            if (auto s = v.find_state(l.execution, e.id.value)) states.insert(s);
            if (auto s = v.find_state(l.timing, e.id.value)) states.insert(s);
        }
    });
    v.partitions(l.timing, [&](PartitionId p) {
        parts.insert(p);
        for (auto e : v.partition(p).events().span())
            if (auto found = v.find_state(l.timing, e.id.value)) states.insert(found);
    });
    if (auto s = v.find_state(l.execution, terminal_key)) states.insert(s);
    erase_incident(tx, v, parts);
    for (auto p : parts) tx.erase_partition(p);
    for (auto s : states) tx.erase_state(s);
    return tx.publish();
}

struct RawDirectory {
    Buffer data;
    ArrayView<RawOrder> view() const { return data.array<RawOrder>(state_schema); }
};
RawDirectory raw_directory(Store& store, const Version& v, const Layers& l, Statistics& stats) {
    std::size_t n = 0;
    v.partitions(l.trace, [&](PartitionId p) { n += v.partition(p).event_count(); });
    auto tx = store.begin(v);
    auto writer = tx.array<RawOrder>(state_schema, n);
    if (n) {
        auto borrow = writer.reserve(n * sizeof(RawOrder));
        auto values = borrow.array<RawOrder>();
        std::size_t i = 0;
        v.partitions(l.trace, [&](PartitionId p) {
            ++stats.trace_partitions_read;
            auto part = v.partition(p);
            for (auto e : part.events().span()) {
                const auto op = raw(v, e.id);
                if (op.call.abi_version != 1 || op.call.op < FAKECUDA_RAS_ALLOC || op.call.op > FAKECUDA_RAS_ALLREDUCE)
                    fail("unsupported Maya raw operation");
                values[i++] = {op.rank, 0, op.sequence, e.id};
            }
        });
        std::sort(values.begin(), values.end(), [](const RawOrder& a, const RawOrder& b) {
            return std::tie(a.rank, a.sequence) < std::tie(b.rank, b.sequence);
        });
        for (std::size_t j = 1; j < n; ++j)
            if (values[j - 1].rank == values[j].rank && values[j - 1].sequence == values[j].sequence)
                fail("duplicate per-rank Maya sequence");
        borrow.commit();
    }
    auto buffer = writer.finish();
    tx.publish();
    return {std::move(buffer)};
}

Version create_execution(Store& store, const Version& v, const Layers& l, const Policies& policy,
                         const RawDirectory& order, const CostProvider& costs, Statistics& stats) {
    auto tx = store.begin(v);
    core::PartitionBuilder<Execution> builder(tx, l.execution, execution_schema, policy.execution.max_payload_bytes);
    std::uint64_t ordinal = 0;
    for (const auto& entry : order.view().span()) {
        const auto op = raw(v, entry.event);
        const Cost cost = costs(entry.event, op); ++stats.cost_calls;
        Origin map{};
        Execution node{entry.event, op.boundary, op.call.stream, ordinal++, op.rank, op.call.op, NodeKind::Host, 0, cost.host};
        const LanePosition host_lane{tx.add_lane(l.execution, 3 * std::uint64_t(op.rank)), op.sequence};
        map.host = builder.append(node, std::uint64_t(node.kind), {&host_lane, 1});
        if (has_device(op.call.op)) {
            node.ordinal = ordinal++; node.cost = cost.device;
            node.kind = communication(op.call.op) ? NodeKind::Communication : NodeKind::Device;
            // Modulo-3 namespaces separate host, default-stream and user
            // stream labels without depending on the allocator's ID range.
            if (op.call.stream > (std::numeric_limits<std::uint64_t>::max() - 2) / 3)
                fail("Maya stream identity exceeds lane-label range");
            const auto label = op.call.stream ? 3 * op.call.stream + 2 : 3 * std::uint64_t(op.rank) + 1;
            const LanePosition device_lane{tx.add_lane(l.execution, label), op.sequence};
            map.device = builder.append(node, std::uint64_t(node.kind), {&device_lane, 1});
        } else if (cost.device) fail("device cost supplied for host-only Maya operation");
        state(tx, l.execution, entry.event.value, map);
        state(tx, l.execution, map.host.value, entry.event);
        if (map.device) state(tx, l.execution, map.device.value, entry.event);
    }
    builder.finish(); stats.execution_partitions_written += builder.partitions().size();
    return tx.publish();
}

// Temporary state contains only open stream/event/group frontiers and unmatched
// communication. The complete graph is written directly into RAS relations.
struct Stream { bool blocking = true; std::vector<EventId> frontier; };
struct Pending { EventId origin, node; std::vector<EventId> ready; std::uint64_t stream = 0; std::uint32_t participants = 0; };
struct Rank {
    EventId host;
    std::map<std::uint64_t, Stream> streams{{0, Stream{}}};
    std::map<std::uint64_t, EventId> events;
    unsigned group_depth = 0;
    std::vector<Pending> group;
};
using Uid = std::array<std::uint8_t, 128>;
struct Comm { Uid uid{}; std::uint32_t rank = 0, size = 0; };
struct P2PKey {
    Uid uid{}; std::uint32_t source = 0, target = 0; std::uint64_t ordinal = 0;
    auto operator<=>(const P2PKey&) const = default;
};
struct CollectiveKey { Uid uid{}; std::uint64_t ordinal = 0; auto operator<=>(const CollectiveKey&) const = default; };
struct Collective { std::map<std::uint32_t, Pending> members; std::uint32_t size = 0; };

class Graph {
public:
    Graph(Store& store, const Version& v, const Layers& layers)
        : v_(v), l_(layers), tx_(store.begin(v)) {}
    void edge(EventId from, EventId to) {
        if (!from || !to) return;
        if (from == to) fail("Maya self dependency");
        // Equal edges are harmless semantically but unnecessary scheduling work.
        // The small per-target set is discarded at the next target.
        if (last_target_ != to) { last_target_ = to; last_sources_.clear(); }
        if (!last_sources_.insert(from).second) return;
        tx_.add_relation({v_.locate(from), v_.locate(to), RelationKind::Order, graph_tag});
    }
    void consume(const RawOrder& entry) {
        const auto op = raw(v_, entry.event);
        const auto m = origin(v_, l_, entry.event);
        Rank& rank = ranks_[op.rank];
        edge(rank.host, m.host); rank.host = m.host;
        const auto& call = op.call;
        switch (call.op) {
        case FAKECUDA_RAS_STREAM_CREATE:
            if (!call.stream || !rank.streams.emplace(call.stream, Stream{(call.flags & 1) == 0, {}}).second)
                fail("invalid Maya stream creation");
            break;
        case FAKECUDA_RAS_STREAM_DESTROY:
            // Destroy is nonblocking; leave the frontier for device sync and
            // terminal completion even though further use is invalid.
            stream(rank, call.stream); destroyed_.insert({op.rank, call.stream}); break;
        case FAKECUDA_RAS_EVENT_CREATE:
            if (!call.event || !rank.events.emplace(call.event, EventId{}).second) fail("invalid Maya event creation");
            break;
        case FAKECUDA_RAS_EVENT_DESTROY:
            if (!rank.events.erase(call.event)) fail("unknown Maya event destruction");
            break;
        case FAKECUDA_RAS_EVENT_RECORD:
            event(rank, call.event); device(rank, op, m); rank.events[call.event] = m.device; break;
        case FAKECUDA_RAS_STREAM_WAIT_EVENT:
            edge(event(rank, call.event), m.device); device(rank, op, m); break;
        case FAKECUDA_RAS_EVENT_SYNC: edge(event(rank, call.event), m.host); break;
        case FAKECUDA_RAS_STREAM_SYNC:
            for (auto pred : stream(rank, call.stream).frontier) edge(pred, m.host);
            break;
        case FAKECUDA_RAS_ALLOC: case FAKECUDA_RAS_FREE: // Legacy, synchronous allocation API subset.
        case FAKECUDA_RAS_DEVICE_SYNC:
            for (auto& [id, s] : rank.streams) { (void)id; for (auto pred : s.frontier) edge(pred, m.host); } break;
        case FAKECUDA_RAS_MEMCPY_ASYNC: case FAKECUDA_RAS_MEMSET_ASYNC: case FAKECUDA_RAS_GEMM:
            device(rank, op, m); break;
        case FAKECUDA_RAS_COMM_INIT: {
            if (call.nranks <= 0 || call.rank < 0 || call.rank >= call.nranks || !call.comm) fail("invalid Maya communicator");
            Comm c; std::copy_n(call.comm_uid, c.uid.size(), c.uid.begin()); c.rank = call.rank; c.size = call.nranks;
            if (!comms_.emplace(call.comm, c).second) fail("duplicate Maya communicator");
            break;
        }
        case FAKECUDA_RAS_COMM_DESTROY:
            if (!comms_.erase(call.comm)) fail("unknown Maya communicator destruction");
            break;
        case FAKECUDA_RAS_GROUP_START:
            if (rank.group_depth == std::numeric_limits<unsigned>::max()) fail("NCCL group nesting overflow");
            ++rank.group_depth; break;
        case FAKECUDA_RAS_GROUP_END:
            if (!rank.group_depth) fail("unmatched NCCL GroupEnd");
            if (!--rank.group_depth) close_group(rank, m.host);
            break;
        case FAKECUDA_RAS_SEND: case FAKECUDA_RAS_RECV: case FAKECUDA_RAS_ALLREDUCE: {
            if (destroyed_.contains({op.rank, call.stream})) fail("use after Maya stream destruction");
            Pending p{entry.event, m.device, readiness(rank, call.stream), call.stream};
            if (rank.group_depth) rank.group.push_back(std::move(p));
            else { p.ready.push_back(m.host); submit(p); stream(rank, call.stream).frontier = {m.device}; }
            break;
        }
        default: break;
        }
    }
    Version finish() {
        if (!p2p_.empty() || !collectives_.empty()) fail("incomplete NCCL communication");
        std::vector<Terminal> terminals;
        for (auto& [rankid, rank] : ranks_) {
            if (rank.group_depth || !rank.group.empty()) fail("incomplete NCCL group");
            if (rank.host) terminals.push_back({rankid, 0, rank.host});
            for (const auto& [id, s] : rank.streams) { (void)id; for (auto e : s.frontier) terminals.push_back({rankid, 0, e}); }
        }
        state<Terminal>(tx_, l_.execution, terminal_key, terminals);
        return tx_.publish();
    }
private:
    Stream& stream(Rank& r, std::uint64_t id) {
        auto it = r.streams.find(id); if (it == r.streams.end()) fail("unknown Maya stream"); return it->second;
    }
    EventId event(Rank& r, std::uint64_t id) {
        auto it = r.events.find(id); if (it == r.events.end()) fail("unknown Maya event"); return it->second;
    }
    std::vector<EventId> readiness(Rank& r, std::uint64_t id) {
        auto ready = stream(r, id).frontier;
        if (!id) {
            for (const auto& [other, s] : r.streams) if (other && s.blocking) ready.insert(ready.end(), s.frontier.begin(), s.frontier.end());
        } else if (stream(r, id).blocking) {
            const auto& f = stream(r, 0).frontier; ready.insert(ready.end(), f.begin(), f.end());
        }
        return ready;
    }
    void device(Rank& r, const Operation& op, Origin m) {
        if (destroyed_.contains({op.rank, op.call.stream})) fail("use after Maya stream destruction");
        edge(m.host, m.device);
        for (auto e : readiness(r, op.call.stream)) edge(e, m.device);
        stream(r, op.call.stream).frontier = {m.device};
    }
    void close_group(Rank& r, EventId commit) {
        // NCCL groups submit together across their participating streams. CUDA
        // calls made while a group is open contribute their current frontiers.
        // Capture every input before installing any member completion.
        std::set<EventId> group_ready{commit};
        std::set<std::uint64_t> streams;
        std::vector<EventId> completions;
        for (const auto& p : r.group) {
            group_ready.insert(p.ready.begin(), p.ready.end());
            const auto f = readiness(r, p.stream); group_ready.insert(f.begin(), f.end());
            streams.insert(p.stream); completions.push_back(p.node);
        }
        for (auto& p : r.group) {
            p.ready.assign(group_ready.begin(), group_ready.end()); submit(p);
        }
        // A subsequent operation on any participating stream waits for the
        // whole group, including members submitted to its other streams.
        for (auto id : streams) stream(r, id).frontier = completions;
        r.group.clear();
    }
    void match(const std::vector<Pending>& members) {
        if (members.empty()) fail("empty communication match");
        const auto first = raw(v_, members[0].origin);
        const auto duration = execution(v_, members[0].node).cost;
        std::vector<EventId> origins;
        for (const auto& p : members) {
            const auto op = raw(v_, p.origin);
            if (p.participants != members[0].participants) fail("NCCL communicator size mismatch");
            if (op.call.count != first.call.count || op.call.datatype != first.call.datatype ||
                op.call.bytes != first.call.bytes || (first.call.op == FAKECUDA_RAS_ALLREDUCE && op.call.reduction != first.call.reduction))
                fail("NCCL signature mismatch");
            if (execution(v_, p.node).cost != duration) fail("NCCL participants have different modeled costs");
            origins.push_back(p.origin);
        }
        for (const auto& target : members) {
            for (const auto& source : members) for (auto e : source.ready) edge(e, target.node);
            state<EventId>(tx_, l_.execution, target.node.value, origins);
        }
    }
    void submit(const Pending& pending) {
        Pending p = pending;
        const auto op = raw(v_, p.origin);
        const auto it = comms_.find(op.call.comm); if (it == comms_.end()) fail("unknown Maya communicator");
        const auto& c = it->second;
        p.participants = c.size;
        if (op.call.op == FAKECUDA_RAS_ALLREDUCE) {
            const auto serial = collective_ordinals_[{c.uid, c.rank}]++;
            const CollectiveKey key{c.uid, serial};
            auto& pending = collectives_[key];
            if (pending.size && pending.size != c.size) fail("NCCL communicator size mismatch");
            pending.size = c.size;
            if (!pending.members.emplace(c.rank, p).second) fail("duplicate collective rank");
            if (pending.members.size() == c.size) {
                std::vector<Pending> members; for (auto& [rank, member] : pending.members) { (void)rank; members.push_back(std::move(member)); }
                match(members); collectives_.erase(key);
            }
        } else {
            if (op.call.peer < 0 || std::uint32_t(op.call.peer) >= c.size) fail("NCCL peer outside communicator");
            const bool send = op.call.op == FAKECUDA_RAS_SEND;
            const auto source = send ? c.rank : std::uint32_t(op.call.peer);
            const auto target = send ? std::uint32_t(op.call.peer) : c.rank;
            const auto ordinal = p2p_ordinals_[{c.uid, source, target, send}]++;
            P2PKey key{c.uid, source, target, ordinal};
            auto old = p2p_.find(key);
            if (old == p2p_.end()) p2p_.emplace(std::move(key), p);
            else {
                const auto peer = raw(v_, old->second.origin);
                if (peer.call.op == op.call.op) fail("duplicate NCCL P2P direction");
                const auto peer_comm = comms_.find(peer.call.comm);
                // A communicator may be destroyed after its last queued call;
                // the signature and logical direction were captured already.
                if (peer_comm != comms_.end() && peer_comm->second.size != c.size) fail("NCCL communicator size mismatch");
                match({old->second, p}); p2p_.erase(old);
            }
        }
    }
    const Version& v_; const Layers& l_; Transaction tx_;
    std::map<std::uint32_t, Rank> ranks_;
    std::set<std::pair<std::uint32_t, std::uint64_t>> destroyed_;
    std::map<std::uint64_t, Comm> comms_;
    std::map<P2PKey, Pending> p2p_;
    std::map<CollectiveKey, Collective> collectives_;
    std::map<std::tuple<Uid, std::uint32_t, std::uint32_t, bool>, std::uint64_t> p2p_ordinals_;
    std::map<std::pair<Uid, std::uint32_t>, std::uint64_t> collective_ordinals_;
    EventId last_target_; std::set<EventId> last_sources_;
};

// AL is a partition index; event provenance remains stable inside payloads.
Version link_trace_execution(Store& store, const Version& v, const Layers& l,
                             const std::set<PartitionId>& targets) {
    return core::rebind_lineage(store, v, targets, [&](const Version& current, PartitionId p, auto emit) {
        for (auto e : current.partition(p).events().span()) {
            const auto sources = state<EventId>(current, l.execution, e.id.value);
            for (auto source : sources.span()) emit(current.locate(source).partition);
        }
    });
}
Version link_execution_timing(Store& store, const Version& v, const Layers& l,
                             const std::set<PartitionId>& targets) {
    return core::rebind_lineage(store, v, targets, [&](const Version& current, PartitionId p, auto emit) {
        for (auto e : current.partition(p).events().span()) {
            const auto source = state<EventId>(current, l.timing, e.id.value);
            if (source.size() != 1) fail("invalid timing provenance");
            emit(current.locate(source[0]).partition);
        }
    });
}
std::set<PartitionId> all_parts(const Version& v, LayerId layer) {
    std::set<PartitionId> result; v.partitions(layer, [&](PartitionId p) { result.insert(p); }); return result;
}

template<class Fn> void predecessors(const Version& v, EventId id, Fn&& fn) {
    const auto ep = v.locate(id);
    v.incoming(ep.partition, [&](RelationId, const Relation& r) {
        if (r.kind == RelationKind::Order && r.tag == graph_tag && r.target.event == ep.event) {
            const auto part = v.partition(r.source.partition); fn(part.events()[r.source.event].id);
        }
    });
}
template<class Fn> void successors(const Version& v, EventId id, Fn&& fn) {
    const auto ep = v.locate(id);
    v.outgoing(ep.partition, [&](RelationId, const Relation& r) {
        if (r.kind == RelationKind::Order && r.tag == graph_tag && r.source.event == ep.event) {
            const auto part = v.partition(r.target.partition); fn(part.events()[r.target.event].id);
        }
    });
}
Feedback feedback(const Version& v, const Layers& l, Statistics& stats,
                  const Feedback* parent = nullptr, const std::set<std::uint32_t>* dirty_ranks = nullptr) {
    Feedback result = parent ? *parent : Feedback{};
    std::set<PartitionId> read;
    const auto terminals = state<Terminal>(v, l.execution, terminal_key);
    const auto values = terminals.span();
    auto completion = [&](const Terminal& t) {
        const auto tid = timing_id(v, l, t.event);
        read.insert(v.locate(tid).partition);
        return timing(v, tid).end;
    };
    if (parent) {
        if (!dirty_ranks) fail("missing dirty feedback ranks");
        for (auto rank : *dirty_ranks) {
            // Graph::finish emits the terminal directory in rank order. This
            // searches the immutable RAS view without copying or scanning the
            // unrelated ranks' terminal records or Timing payloads.
            auto first = std::lower_bound(values.begin(), values.end(), rank,
                [](const Terminal& t, std::uint32_t r) { return t.rank < r; });
            Tick end = 0;
            for (auto it = first; it != values.end() && it->rank == rank; ++it) end = std::max(end, completion(*it));
            auto summary = std::lower_bound(result.rank_end.begin(), result.rank_end.end(), rank,
                [](const auto& entry, std::uint32_t r) { return entry.first < r; });
            if (summary == result.rank_end.end() || summary->first != rank) fail("fixed topology changed feedback ranks");
            summary->second = end;
        }
    } else {
        for (const auto& t : values) {
            if (result.rank_end.empty() || result.rank_end.back().first != t.rank) result.rank_end.emplace_back(t.rank, 0);
            result.rank_end.back().second = std::max(result.rank_end.back().second, completion(t));
        }
    }
    result.makespan = 0;
    for (const auto& [rank, end] : result.rank_end) { (void)rank; result.makespan = std::max(result.makespan, end); }
    stats.timing_partitions_read += read.size();
    return result;
}

Version schedule_full(Store& store, const Version& v, const Layers& l, const Policies& policy, Statistics& stats) {
    std::size_t count = 0;
    v.partitions(l.execution, [&](PartitionId p) { count += v.partition(p).event_count(); });
    if (!count) return v;
    auto tx = store.begin(v);
    auto scratch = tx.array<Scratch>(state_schema, count);
    auto borrow = scratch.reserve(count * sizeof(Scratch));
    auto work = borrow.array<Scratch>();
    std::fill(work.begin(), work.end(), Scratch{});
    std::queue<EventId> ready;
    v.partitions(l.execution, [&](PartitionId p) {
        ++stats.execution_partitions_read;
        for (auto e : v.partition(p).events().span()) {
            const auto x = execution(v, e.id);
            if (x.ordinal >= count || work[x.ordinal].event) fail("invalid execution ordinal");
            auto& s = work[x.ordinal]; s.event = e.id;
            predecessors(v, e.id, [&](EventId) { ++s.indegree; });
            if (!s.indegree) ready.push(e.id);
        }
    });
    core::PartitionBuilder<Timing> builder(tx, l.timing, timing_schema, policy.timing.max_payload_bytes);
    std::uint64_t visited = 0;
    while (!ready.empty()) {
        const auto id = ready.front(); ready.pop(); const auto x = execution(v, id);
        auto& s = work[x.ordinal];
        const Tick end = plus(s.ready, x.cost);
        const LanePosition lane{tx.add_lane(l.timing, std::uint64_t(x.rank) + 1), visited};
        const auto tid = builder.append({id, s.ready, end, visited++}, 1, {&lane, 1});
        state(tx, l.timing, id.value, tid); state(tx, l.timing, tid.value, id); ++stats.scheduled_events;
        successors(v, id, [&](EventId next) {
            const auto y = execution(v, next); auto& target = work[y.ordinal];
            if (!target.indegree) fail("invalid execution dependency count");
            target.ready = std::max(target.ready, end);
            if (!--target.indegree) ready.push(next);
        });
    }
    if (visited != count) fail("Maya execution dependency cycle");
    builder.finish(); stats.timing_partitions_written += builder.partitions().size();
    borrow.commit(); auto owned_scratch = scratch.finish();
    auto result = tx.publish();
    return link_execution_timing(store, result, l, all_parts(result, l.timing));
}

// Repartition materializes only this layer. Stable identity rewrites graph
// endpoints and adjacent AL; consumers keep their payloads and time results.
template<class T> Version repartition(Store& store, const Version& v, LayerId layer, SchemaId schema,
                                      std::size_t capacity, Statistics& stats) {
    const auto parts = all_parts(v, layer);
    if (parts.empty()) return v;
    struct SavedRelation { EventId source, target; PartitionId upper, lower; RelationKind kind; std::uint64_t tag; };
    // Relation metadata is transient; payloads remain solely RAS-owned. The
    // saved relation table itself is placed in a RAS allocation below.
    std::set<RelationId> incident;
    for (auto p : parts) {
        v.incoming(p, [&](RelationId id, const Relation&) { incident.insert(id); });
        v.outgoing(p, [&](RelationId id, const Relation&) { incident.insert(id); });
    }
    auto tx = store.begin(v);
    auto relations = tx.array<SavedRelation>(state_schema, incident.size());
    if (!incident.empty()) {
        auto edit = relations.reserve(incident.size() * sizeof(SavedRelation)); auto saved = edit.template array<SavedRelation>();
        std::size_t i = 0;
        for (auto rid : incident) {
            const auto r = v.relation(rid); SavedRelation s{}; s.kind = r.kind; s.tag = r.tag;
            if (r.kind != RelationKind::Lineage) {
                s.source = v.partition(r.source.partition).events()[r.source.event].id;
                s.target = v.partition(r.target.partition).events()[r.target.event].id;
            }
            saved[i++] = s;
            tx.erase_relation(rid);
        }
        edit.commit();
    }
    auto saved = relations.finish();
    for (auto p : parts) tx.erase_partition(p);
    core::PartitionBuilder<T> builder(tx, layer, schema, capacity);
    for (auto p : parts) {
        auto part = v.partition(p);
        stats.repacked_bytes += part.event_count() * sizeof(T);
        for (std::size_t i = 0; i < part.event_count(); ++i) {
            const auto meta = part.events()[i];
            const auto value = core::event_view<T>(v, meta.id, schema);
            const auto positions = part.event_positions(i);
            builder.append(value.get(), meta.kind, positions.span(), meta.id);
        }
    }
    builder.finish();
    if (schema == execution_schema) { stats.execution_partitions_read += parts.size(); stats.execution_partitions_written += builder.partitions().size(); }
    else { stats.timing_partitions_read += parts.size(); stats.timing_partitions_written += builder.partitions().size(); }
    auto moved = tx.publish(); auto reltx = store.begin(moved);
    for (const auto& r : saved.template array<SavedRelation>(state_schema).span())
        if (r.kind != RelationKind::Lineage) reltx.add_relation({moved.locate(r.source), moved.locate(r.target), r.kind, r.tag});
    return reltx.publish();
}

// Mutable output exists only in new RAS allocations. This map holds writer
// capabilities for touched blocks, never a second copy of the whole layer.
template<class T> class Editors {
    struct Edit {
        BufferWriter writer;
        WriteSpan borrow;
        std::span<T> values;
        Edit(Transaction& tx, const Buffer& original) : writer(tx.buffer(original.size(), original.layout())), borrow(writer.reserve(original.size())) {
            auto dst = borrow.bytes(); const auto src = original.bytes();
            std::memcpy(dst.data(), src.span().data(), dst.size());
            values = borrow.template array<T>();
        }
    };
public:
    Editors(Transaction& tx, const Version& v, SchemaId schema) : tx_(tx), v_(v), schema_(schema) {}
    T read(EventId id) const {
        auto ep = v_.locate(id); auto found = edits_.find(ep.partition);
        if (found == edits_.end()) return core::event_view<T>(v_, id, schema_).get();
        const auto meta = v_.partition(ep.partition).events()[ep.event]; return found->second.values[meta.payload.offset / sizeof(T)];
    }
    void put(EventId id, const T& value) {
        auto ep = v_.locate(id); auto part = v_.partition(ep.partition);
        auto [it, fresh] = edits_.try_emplace(ep.partition, tx_, part.payload());
        if (fresh) copied_bytes_ += part.payload().size();
        it->second.values[part.events()[ep.event].payload.offset / sizeof(T)] = value;
    }
    std::size_t size() const { return edits_.size(); }
    std::uint64_t copied_bytes() const { return copied_bytes_; }
    void finish() {
        for (auto& [p, edit] : edits_) { edit.borrow.commit(); tx_.replace_payload(p, edit.writer.finish()); }
    }
private:
    Transaction& tx_; const Version& v_; SchemaId schema_; std::map<PartitionId, Edit> edits_;
    std::uint64_t copied_bytes_ = 0;
};

bool same_topology(const Operation& a, const Operation& b) {
    const auto& x = a.call; const auto& y = b.call;
    return a.rank == b.rank && a.sequence == b.sequence && x.op == y.op && x.stream == y.stream &&
           x.event == y.event && x.comm == y.comm && x.peer == y.peer && x.rank == y.rank && x.nranks == y.nranks &&
           x.flags == y.flags && std::memcmp(x.comm_uid, y.comm_uid, sizeof(x.comm_uid)) == 0;
}

std::pair<Version, std::vector<EventId>> update_costs(Store& store, const Version& v, const Candidate& parent,
        const Layers& l, const TraceDelta& delta, std::span<const std::uint64_t> boundaries,
        const CostProvider& costs, Statistics& stats) {
    std::set<EventId> dirty(delta.changed_events.begin(), delta.changed_events.end());
    for (auto key : boundaries) {
        const auto ids = state<EventId>(v, l.trace, key);
        dirty.insert(ids.span().begin(), ids.span().end());
    }
    std::set<PartitionId> read_trace, read_exec;
    auto tx = store.begin(v); Editors<Execution> edits(tx, v, execution_schema);
    std::vector<EventId> changed;
    for (auto id : dirty) {
        const auto op = raw(v, id); read_trace.insert(v.locate(id).partition);
        if (!same_topology(raw(parent.version, id), op)) fail("Maya topology change requires structural rebuild");
        const auto map = origin(v, l, id);
        const auto cost = costs(id, op); ++stats.cost_calls;
        for (auto [node, duration] : {std::pair(map.host, cost.host), std::pair(map.device, cost.device)}) {
            if (!node) { if (duration) fail("device cost supplied for host-only Maya operation"); continue; }
            auto x = edits.read(node); read_exec.insert(v.locate(node).partition);
            if (x.cost != duration) { x.cost = duration; edits.put(node, x); changed.push_back(node); }
        }
    }
    // Validate all affected communication signatures and durations before any
    // result is published. Participant lookup is a RAS named-state index.
    for (auto id : dirty) {
        const auto op = raw(v, id); if (!communication(op.call.op)) continue;
        const auto map = origin(v, l, id); const auto duration = edits.read(map.device).cost;
        const auto peers = state<EventId>(v, l.execution, map.device.value);
        for (auto peer : peers.span()) {
            const auto other = raw(v, peer); read_trace.insert(v.locate(peer).partition);
            if (other.call.count != op.call.count || other.call.datatype != op.call.datatype || other.call.bytes != op.call.bytes ||
                (op.call.op == FAKECUDA_RAS_ALLREDUCE && other.call.reduction != op.call.reduction)) fail("NCCL signature mismatch");
            const auto peer_node = origin(v, l, peer).device; read_exec.insert(v.locate(peer_node).partition);
            if (edits.read(peer_node).cost != duration) fail("NCCL participants have different modeled costs");
        }
    }
    stats.trace_partitions_read += read_trace.size(); stats.execution_partitions_read += read_exec.size();
    stats.execution_partitions_written += edits.size(); stats.refresh_copy_bytes += edits.copied_bytes(); edits.finish();
    return {tx.publish(), std::move(changed)};
}

Version schedule_incremental(Store& store, const Version& v, const Layers& l,
                             const std::vector<EventId>& changed, Statistics& stats,
                             std::set<std::uint32_t>& dirty_ranks) {
    if (changed.empty()) return v;
    auto tx = store.begin(v); Editors<Timing> edits(tx, v, timing_schema);
    using QueueItem = std::pair<std::uint64_t, EventId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> ready;
    std::set<EventId> queued;
    std::set<PartitionId> read_exec, read_timing;
    auto enqueue = [&](EventId id) {
        if (!queued.insert(id).second) return;
        const auto tid = timing_id(v, l, id); read_timing.insert(v.locate(tid).partition);
        ready.push({edits.read(tid).topological_order, id});
    };
    for (auto id : changed) enqueue(id);
    while (!ready.empty()) {
        const auto id = ready.top().second; ready.pop(); queued.erase(id);
        const auto x = execution(v, id); read_exec.insert(v.locate(id).partition);
        const auto tid = timing_id(v, l, id); const auto old = edits.read(tid);
        Tick start = 0;
        predecessors(v, id, [&](EventId pred) {
            const auto pt = timing_id(v, l, pred); read_timing.insert(v.locate(pt).partition);
            start = std::max(start, edits.read(pt).end);
        });
        const Tick end = plus(start, x.cost); ++stats.scheduled_events;
        if (start != old.start || end != old.end) {
            edits.put(tid, Timing{id, start, end, old.topological_order});
            // Successors depend on completion, not this node's start alone.
            if (end != old.end) { dirty_ranks.insert(x.rank); successors(v, id, enqueue); }
        }
    }
    stats.execution_partitions_read += read_exec.size(); stats.timing_partitions_read += read_timing.size();
    stats.timing_partitions_written += edits.size(); stats.refresh_copy_bytes += edits.copied_bytes(); edits.finish(); return tx.publish();
}

} // namespace

EngineResult evaluate_engine(Store& store, const Candidate* parent, const Version& trace_version,
    const Layers& layers, const Policies& policies, const TraceDelta& delta,
    std::span<const std::uint64_t> cost_dirty_boundaries, const CostProvider& costs, Statistics& stats) {
    if (!costs) fail("Maya requires an explicit CostProvider");
    if (!policies.execution.max_payload_bytes || !policies.timing.max_payload_bytes) fail("Maya partition capacity must be positive");
    const auto before = store.stats().index_page_visits;
    Version v = trace_version;
    bool feedback_changed = true;
    std::set<std::uint32_t> dirty_ranks;
    if (!parent || delta.structural) {
        stats.structural_rebuild = parent != nullptr;
        if (parent) v = clear_downstream(store, v, layers);
        const auto order = raw_directory(store, v, layers, stats);
        v = create_execution(store, v, layers, policies, order, costs, stats);
        Graph graph(store, v, layers);
        for (const auto& entry : order.view().span()) graph.consume(entry);
        v = graph.finish();
        v = link_trace_execution(store, v, layers, all_parts(v, layers.execution));
        v = schedule_full(store, v, layers, policies, stats);
    } else {
        auto [cost_version, changed] = update_costs(store, v, *parent, layers, delta, cost_dirty_boundaries, costs, stats);
        feedback_changed = !changed.empty();
        core::LayerUpdate update{std::move(cost_version), {std::move(changed), false,
            delta.layout_only || !delta.removed.empty()}};
        v = core::refresh_layer(std::move(update),
            [&](const Version& current, const core::LayerDelta& change) {
                if (!change.layout_changed) return current;
                // Metadata/state provenance repairs Link even when modeled
                // values have not changed and Apply will be skipped.
                std::set<PartitionId> affected;
                for (auto old : delta.removed) parent->version.outgoing(old, [&](RelationId, const Relation& r) {
                    if (r.kind == RelationKind::Lineage && parent->version.partition(r.target.partition).layer() == layers.execution)
                        affected.insert(r.target.partition);
                });
                for (auto id : delta.changed_events) {
                    const auto map = origin(current, layers, id);
                    affected.insert(current.locate(map.host).partition);
                    if (map.device) affected.insert(current.locate(map.device).partition);
                }
                return link_trace_execution(store, current, layers, affected);
            },
            [](const Version&, const core::LayerDelta& change) -> const std::vector<EventId>& { return change.events; },
            [&](const Version& current, const std::vector<EventId>& selected) {
                return schedule_incremental(store, current, layers, selected, stats, dirty_ranks);
            });
        const bool repack_execution = parent->policies.execution != policies.execution;
        const bool repack_timing = parent->policies.timing != policies.timing;
        if (repack_execution) {
            v = repartition<Execution>(store, v, layers.execution, execution_schema, policies.execution.max_payload_bytes, stats);
            v = link_trace_execution(store, v, layers, all_parts(v, layers.execution));
        }
        if (repack_timing) v = repartition<Timing>(store, v, layers.timing, timing_schema, policies.timing.max_payload_bytes, stats);
        if (repack_execution || repack_timing) v = link_execution_timing(store, v, layers, all_parts(v, layers.timing));
    }
    const bool full = !parent || delta.structural;
    auto result = full ? feedback(v, layers, stats) : (feedback_changed
        ? feedback(v, layers, stats, &parent->feedback, &dirty_ranks) : parent->feedback);
    stats.index_visits += store.stats().index_page_visits - before;
    return {std::move(v), std::move(result)};
}

} // namespace flexeva::maya
