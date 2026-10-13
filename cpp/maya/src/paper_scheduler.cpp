#include "paper_scheduler.hpp"
#include "paper_lanes.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace flexeva::maya::paper {
namespace {
using NodeId = std::uint64_t;
constexpr NodeId none = std::numeric_limits<NodeId>::max();
constexpr ras::SchemaId scratch_schema = 0x4d41594110ffULL;

[[noreturn]] void fail(const char* message) { throw ras::Error(message); }
Tick add(Tick a, Tick b) {
    if (b > std::numeric_limits<Tick>::max() - a) fail("paper replay modeled time overflow");
    return a + b;
}
std::size_t timing_capacity(const ReplayInput& input) {
    if (input.count > std::numeric_limits<std::size_t>::max() / (2 * sizeof(ReplayTiming)))
        fail("paper replay timing capacity overflow");
    return std::max<std::size_t>(1, input.count) * 2 * sizeof(ReplayTiming);
}

// Pages retain their writable borrow until scheduling completes. Only the
// pointer directory lives on the C++ heap; event-sized arrays live in RAS.
template<class T> class Arena {
    struct Page {
        core::runtime::OutputWriter writer;
        ras::WriteSpan borrow;
        std::span<T> values;
        Page(core::runtime::Context& context, std::size_t count, ras::SchemaId schema)
            : writer(context.buffer(count * sizeof(T), {schema, sizeof(T), alignof(T)})),
              borrow(writer.reserve(count * sizeof(T))), values(borrow.array<T>()) {
            std::fill(values.begin(), values.end(), T{});
        }
    };
public:
    Arena(core::runtime::Context& context, std::size_t bytes, ras::SchemaId schema = scratch_schema)
        : context_(context), per_page_(std::max<std::size_t>(1, bytes / sizeof(T))), schema_(schema) {}
    std::uint64_t append(const T& value) {
        if (size_ == none) fail("paper replay arena overflow");
        if (size_ % per_page_ == 0) pages_.push_back(std::make_unique<Page>(context_, per_page_, schema_));
        const auto index = size_++;
        at(index) = value;
        return index;
    }
    T& at(std::uint64_t index) {
        if (index >= size_) fail("paper replay scratch index outside arena");
        return pages_[index / per_page_]->values[index % per_page_];
    }
    const T& at(std::uint64_t index) const { return const_cast<Arena*>(this)->at(index); }
    std::uint64_t size() const { return size_; }
    std::uint64_t capacity_bytes() const { return pages_.size() * per_page_ * sizeof(T); }
    std::vector<ras::Buffer> finish() {
        std::vector<ras::Buffer> result;
        result.reserve(pages_.size());
        std::uint64_t remaining = size_;
        for (auto& page : pages_) {
            page->borrow.commit();
            auto buffer = page->writer.finish();
            const auto count = std::min<std::uint64_t>(remaining, per_page_);
            result.push_back(buffer.slice({0, count * sizeof(T)}));
            remaining -= count;
        }
        pages_.clear();
        return result;
    }
private:
    core::runtime::Context& context_;
    std::size_t per_page_;
    ras::SchemaId schema_;
    std::uint64_t size_ = 0;
    std::vector<std::unique_ptr<Page>> pages_;
};

struct Node {
    NodeId head = none;
    std::uint64_t indegree = 0;
    Tick ready = 0, duration = 0;
    NodeId predecessors = none, match = none;
};
struct Edge { NodeId target = none; std::uint64_t next = none; NodeId source = none, previous = none; };
struct Origin { NodeId host = none, device = none; };
// Exact scheduling evidence, not a second native trace: tensor pointers,
// shapes, kernel names and model-cost inputs remain in normalized storage.
struct ScheduleKey {
    std::uint64_t boundary = 0, sequence = 0, stream = 0, event = 0, comm = 0;
    std::uint64_t count = 0, bytes = 0;
    std::uint32_t rank = 0, op = 0, flags = 0, abi = 0;
    std::int32_t peer = 0, nranks = 0, local = 0, datatype = 0, reduction = 0;
    ReplayUid uid{};
    bool operator==(const ScheduleKey&) const = default;
};
bool collective_operation(std::uint32_t op) {
    return op == FAKECUDA_RAS_ALLREDUCE || op == FAKECUDA_RAS_BROADCAST ||
        op == FAKECUDA_RAS_ALLGATHER || op == FAKECUDA_RAS_REDUCESCATTER;
}
ScheduleKey schedule_key(const ReplayOperation& op) {
    ScheduleKey key{};key.boundary=op.boundary;key.sequence=op.sequence;key.rank=op.rank;key.op=op.call.op;key.abi=op.call.abi_version;
    const auto& call=op.call;
    // Include every descriptor field consulted by graph construction and
    // validation; omitted numerical fields only affect the supplied costs.
    key.stream=call.stream;key.event=call.event;
    if(call.op==FAKECUDA_RAS_STREAM_CREATE)key.flags=call.flags&1;
    if(call.op==FAKECUDA_RAS_COMM_INIT||call.op==FAKECUDA_RAS_COMM_DESTROY||communication(call.op))key.comm=call.comm;
    if(call.op==FAKECUDA_RAS_COMM_INIT||communication(call.op)){
        key.nranks=call.nranks;key.local=call.rank;
        std::copy_n(call.comm_uid,key.uid.size(),key.uid.begin());
    }
    if(communication(call.op)){
        key.count=call.count;key.bytes=call.bytes;key.datatype=call.datatype;
        if(call.op==FAKECUDA_RAS_SEND||call.op==FAKECUDA_RAS_RECV||call.op==FAKECUDA_RAS_BROADCAST)key.peer=call.peer;
        if(call.op==FAKECUDA_RAS_ALLREDUCE||call.op==FAKECUDA_RAS_REDUCESCATTER)key.reduction=call.reduction;
    }
    return key;
}
struct CachedInput { ScheduleKey key; Origin origin; };
struct Endpoint { std::uint32_t rank = 0, reserved = 0; NodeId node = none; };
struct GraphSnapshot {
    std::vector<ras::Buffer> nodes, edges, inputs, endpoints;
    std::uint64_t endpoint_count = 0;
};
struct Stream { bool blocking = true, destroyed = false; std::vector<NodeId> frontier; };
struct Comm { ReplayUid uid{}; std::uint32_t local = 0, size = 0; };
struct Pending {
    std::uint64_t input = 0, stream = 0;
    NodeId node = none;
    Comm comm;
    std::vector<NodeId> ready;
};
struct Rank {
    NodeId host = none;
    std::uint64_t sequence = 0;
    bool seen = false;
    std::map<std::uint64_t, Stream> streams{{0, Stream{}}};
    std::map<std::uint64_t, NodeId> events;
    std::map<std::uint64_t, Comm> communicators;
    unsigned group_depth = 0;
    std::vector<Pending> group;
};
struct Membership {
    std::uint32_t size = 0;
    std::map<std::uint32_t, std::uint32_t> local_to_global;
};
struct P2PKey {
    ReplayUid uid{};
    std::uint32_t source = 0, target = 0;
    std::uint64_t ordinal = 0;
    auto operator<=>(const P2PKey&) const = default;
};
struct CollectiveKey {
    ReplayUid uid{};
    std::uint64_t ordinal = 0;
    auto operator<=>(const CollectiveKey&) const = default;
};
struct Collective { std::uint32_t size = 0; std::map<std::uint32_t, Pending> members; };

bool uid_empty(const ReplayUid& uid) {
    return std::all_of(uid.begin(), uid.end(), [](auto byte) { return byte == 0; });
}

class Scheduler {
public:
    Scheduler(core::runtime::Context& context, const ReplayInput& input, bool cache = false)
        : input_(input), nodes_(context, input.block_bytes), edges_(context, input.block_bytes),
          queue_(context, input.block_bytes), timings_(context, cache?input.block_bytes:timing_capacity(input), replay_timing_schema),
          inputs_(cache?std::make_unique<Arena<CachedInput>>(context,input.block_bytes):nullptr),
          endpoints_(cache?std::make_unique<Arena<Endpoint>>(context,input.block_bytes):nullptr) {
        if (!input.block_bytes) fail("paper replay block capacity must be positive");
        if (input.count && !input.read) fail("paper replay missing input cursor");
        for (const auto& membership : input.memberships) register_membership(membership.uid, membership.members);
        for (const auto& seed : input.streams) {
            if (!seed.id || !ranks_[seed.rank].streams.emplace(seed.id, Stream{seed.blocking, false, {}}).second)
                fail("paper replay duplicate or default stream seed");
        }
        for (const auto& seed : input.events) {
            if (!seed.id || !ranks_[seed.rank].events.emplace(seed.id, none).second)
                fail("paper replay duplicate or zero event seed");
        }
        for (const auto& seed : input.communicators) {
            register_membership(seed.uid, seed.members);
            if (!seed.id || seed.local_rank >= seed.members.size() || seed.members[seed.local_rank] != seed.rank)
                fail("paper replay invalid communicator seed membership");
            Comm comm{seed.uid, seed.local_rank, static_cast<std::uint32_t>(seed.members.size())};
            if (!ranks_[seed.rank].communicators.emplace(seed.id, comm).second)
                fail("paper replay duplicate communicator seed");
        }
    }

    ReplayResult run(GraphSnapshot* snapshot = nullptr) {
        for (std::uint64_t i = 0; i < input_.count; ++i) consume(i);
        if (!p2p_.empty() || !collectives_.empty()) fail("paper replay incomplete NCCL communication");
        for (const auto& [id, rank] : ranks_) {
            (void)id;
            if (rank.group_depth || !rank.group.empty()) fail("paper replay incomplete NCCL group");
        }
        for (const auto& [uid, membership] : memberships_) {
            (void)uid;
            if (membership.local_to_global.size() != membership.size)
                fail("paper replay incomplete communicator membership");
        }
        for (NodeId i = 0; i < nodes_.size(); ++i) if (!nodes_.at(i).indegree) queue_.append(i);
        std::map<std::uint32_t, Tick> rank_end;
        for (std::uint64_t cursor = 0; cursor < queue_.size(); ++cursor) {
            const auto id = queue_.at(cursor);
            auto& node = nodes_.at(id);
            auto& timing = timings_.at(id);
            timing.start = node.ready;
            timing.end = add(timing.start, node.duration);
            timing.topological_order = result_.scheduled_events++;
            rank_end[timing.rank] = std::max(rank_end[timing.rank], timing.end);
            for (auto edge_id = node.head; edge_id != none; edge_id = edges_.at(edge_id).next) {
                auto& target = nodes_.at(edges_.at(edge_id).target);
                if (!target.indegree) fail("paper replay inconsistent dependency count");
                target.ready = std::max(target.ready, timing.end);
                if (!--target.indegree) queue_.append(edges_.at(edge_id).target);
            }
        }
        if (result_.scheduled_events != nodes_.size()) {
            std::ostringstream message;
            message << "paper replay execution dependency cycle: completed=" << result_.scheduled_events
                    << "/" << nodes_.size();
            std::vector<NodeId> blocked;
            for (NodeId id = 0; id < nodes_.size() && blocked.size() < 8; ++id)
                if (nodes_.at(id).indegree) blocked.push_back(id);
            auto describe = [&](NodeId id) {
                const auto& timing = timings_.at(id);
                const auto op = read(timing.origin);
                message << id << "(rank=" << timing.rank << ",seq=" << timing.sequence
                        << ",op=" << op.call.op << ",kind=" << static_cast<unsigned>(timing.kind)
                        << ",stream=" << op.call.stream << ",event=" << op.call.event
                        << ",peer=" << op.call.peer << ")";
            };
            for (auto target : blocked) {
                message << "; blocked="; describe(target); message << " waits=";
                unsigned shown = 0;
                for (NodeId source = 0; source < nodes_.size() && shown < 4; ++source) {
                    if (!nodes_.at(source).indegree) continue;
                    for (auto eid = nodes_.at(source).head; eid != none && shown < 4; eid = edges_.at(eid).next)
                        if (edges_.at(eid).target == target) { if (shown++) message << ','; describe(source); }
                }
            }
            throw ras::Error(message.str());
        }
        for (const auto& [rank, end] : rank_end) {
            result_.feedback.rank_end.emplace_back(rank, end);
            result_.feedback.makespan = std::max(result_.feedback.makespan, end);
        }
        result_.dependency_edges = edges_.size();
        result_.total_events = result_.scheduled_events;
        result_.updated_events = result_.scheduled_events;
        result_.graph_rebuilt = true;
        result_.scratch_capacity_bytes = nodes_.capacity_bytes() + edges_.capacity_bytes() + queue_.capacity_bytes();
        result_.timing_capacity_bytes = timings_.capacity_bytes();
        result_.timing = timings_.finish();
        if(snapshot){
            for(const auto& [id,rank]:ranks_){
                std::set<NodeId> tails;
                if(rank.host!=none)tails.insert(rank.host);
                for(const auto& [stream_id,value]:rank.streams){(void)stream_id;tails.insert(value.frontier.begin(),value.frontier.end());}
                for(auto tail:tails)if(tail!=none)endpoints_->append({id,0,tail});
            }
            snapshot->endpoint_count=endpoints_->size();
            snapshot->nodes=nodes_.finish();snapshot->edges=edges_.finish();snapshot->inputs=inputs_->finish();snapshot->endpoints=endpoints_->finish();
        }
        return std::move(result_);
    }

private:
    ReplayOperation read(std::uint64_t i) { ++result_.input_reads; return input_.read(i); }
    void register_membership(const ReplayUid& uid, std::span<const std::uint32_t> members) {
        if (members.empty() || members.size() > std::numeric_limits<std::uint32_t>::max())
            fail("paper replay invalid communicator membership size");
        std::set<std::uint32_t> seen;
        auto& membership = memberships_[uid];
        if (membership.size && membership.size != members.size()) fail("paper replay communicator size mismatch");
        membership.size = static_cast<std::uint32_t>(members.size());
        for (std::uint32_t local = 0; local < members.size(); ++local) {
            if (!seen.insert(members[local]).second) fail("paper replay duplicate communicator member");
            auto [it, inserted] = membership.local_to_global.emplace(local, members[local]);
            if (!inserted && it->second != members[local]) fail("paper replay communicator membership mismatch");
        }
    }
    void bind_member(const Comm& comm, std::uint32_t global) {
        auto& membership = memberships_[comm.uid];
        if (membership.size && membership.size != comm.size) fail("paper replay communicator size mismatch");
        membership.size = comm.size;
        for (const auto& [local, old_global] : membership.local_to_global)
            if (old_global == global && local != comm.local) fail("paper replay duplicate global communicator rank");
        auto [it, inserted] = membership.local_to_global.emplace(comm.local, global);
        if (!inserted && it->second != global) fail("paper replay communicator local/global rank mismatch");
    }
    Stream& stream(Rank& rank, std::uint64_t id, bool allow_destroyed = false) {
        auto it = rank.streams.find(id);
        if (it == rank.streams.end()) fail("paper replay unknown stream");
        if (!allow_destroyed && it->second.destroyed) fail("paper replay use after stream destruction");
        return it->second;
    }
    NodeId event(Rank& rank, std::uint64_t id) {
        auto it = rank.events.find(id);
        if (it == rank.events.end()) fail("paper replay unknown event");
        return it->second;
    }
    void edge(NodeId from, NodeId to) {
        if (from == none || to == none) return;
        if (from == to) fail("paper replay self dependency");
        // Duplicate edges are valid: indegree and traversal count each edge.
        // Deduplicate the current target's small frontier to limit group work.
        if (last_target_ != to) { last_target_ = to; last_sources_.clear(); }
        if (!last_sources_.insert(from).second) return;
        auto& source = nodes_.at(from);
        auto& target = nodes_.at(to);
        const auto edge_id=edges_.append({to, source.head, from, target.predecessors});
        source.head=edge_id;target.predecessors=edge_id;
        if (target.indegree == none) fail("paper replay dependency count overflow");
        ++target.indegree;
    }
    std::vector<NodeId> readiness(Rank& rank, std::uint64_t id) {
        auto& current = stream(rank, id);
        auto ready = current.frontier;
        if (!id) {
            for (const auto& [other, s] : rank.streams)
                if (other && s.blocking) ready.insert(ready.end(), s.frontier.begin(), s.frontier.end());
        } else if (current.blocking) {
            const auto& defaults = stream(rank, 0).frontier;
            ready.insert(ready.end(), defaults.begin(), defaults.end());
        }
        return ready;
    }
    NodeId node(std::uint64_t input, const ReplayOperation& op, ReplayNodeKind kind, Tick duration) {
        auto id = nodes_.append({none, 0, 0, duration});
        const auto timing_id = timings_.append({input, op.boundary, op.sequence, op.rank, kind, 0, 0, 0});
        if (id != timing_id) fail("paper replay node/timing index mismatch");
        return id;
    }
    void device(Rank& rank, const ReplayOperation& op, Origin origin) {
        edge(origin.host, origin.device);
        for (auto pred : readiness(rank, op.call.stream)) edge(pred, origin.device);
        stream(rank, op.call.stream).frontier = {origin.device};
    }
    void consume(std::uint64_t input) {
        const auto op = read(input);
        if ((op.call.abi_version != 1 && op.call.abi_version != 2) ||
            op.call.op < FAKECUDA_RAS_ALLOC || op.call.op > FAKECUDA_RAS_REDUCESCATTER)
            fail("paper replay unsupported native operation");
        auto& rank = ranks_[op.rank];
        if (rank.seen && op.sequence <= rank.sequence) fail("paper replay rank sequence is not strictly increasing");
        rank.seen = true; rank.sequence = op.sequence;
        Origin origin{node(input, op, ReplayNodeKind::Host, op.cost.host), none};
        if (device_operation(op.call.op))
            origin.device = node(input, op, communication(op.call.op) ? ReplayNodeKind::Communication : ReplayNodeKind::Device, op.cost.device);
        else if (op.cost.device) fail("paper replay device cost on host-only operation");
        if(inputs_&&inputs_->append({schedule_key(op),origin})!=input)fail("paper replay cache input order mismatch");
        edge(rank.host, origin.host); rank.host = origin.host;
        const auto& call = op.call;
        switch (call.op) {
        case FAKECUDA_RAS_STREAM_CREATE:
            if (!call.stream || !rank.streams.emplace(call.stream, Stream{(call.flags & 1) == 0, false, {}}).second)
                fail("paper replay invalid stream creation");
            break;
        case FAKECUDA_RAS_STREAM_DESTROY:
            if (!call.stream) fail("paper replay cannot destroy default stream");
            stream(rank, call.stream).destroyed = true; break;
        case FAKECUDA_RAS_EVENT_CREATE:
            if (!call.event || !rank.events.emplace(call.event, none).second) fail("paper replay invalid event creation");
            break;
        case FAKECUDA_RAS_EVENT_DESTROY:
            if (!rank.events.erase(call.event)) fail("paper replay unknown event destruction");
            break;
        case FAKECUDA_RAS_EVENT_RECORD:
            event(rank, call.event); device(rank, op, origin); rank.events[call.event] = origin.device; break;
        case FAKECUDA_RAS_STREAM_WAIT_EVENT:
            edge(event(rank, call.event), origin.device); device(rank, op, origin); break;
        case FAKECUDA_RAS_EVENT_SYNC:
            edge(event(rank, call.event), origin.host); break;
        case FAKECUDA_RAS_STREAM_SYNC:
            for (auto pred : stream(rank, call.stream).frontier) edge(pred, origin.host);
            break;
        case FAKECUDA_RAS_ALLOC: case FAKECUDA_RAS_FREE: case FAKECUDA_RAS_DEVICE_SYNC:
            for (const auto& [id, s] : rank.streams) { (void)id; for (auto pred : s.frontier) edge(pred, origin.host); }
            break;
        // Native capture emits explicit STREAM_SYNC records for operations
        // that complete before host return. API-name flags alone do not imply
        // host synchronization (notably D2D memcpy and device memset).
        case FAKECUDA_RAS_MEMCPY_ASYNC: case FAKECUDA_RAS_MEMSET_ASYNC:
        case FAKECUDA_RAS_GEMM: case FAKECUDA_RAS_KERNEL:
            device(rank, op, origin); break;
        case FAKECUDA_RAS_COMM_INIT: {
            if (!call.comm || call.nranks <= 0 || call.rank < 0 || call.rank >= call.nranks)
                fail("paper replay invalid communicator initialization");
            Comm comm{};
            std::copy_n(call.comm_uid, comm.uid.size(), comm.uid.begin());
            comm.local = static_cast<std::uint32_t>(call.rank); comm.size = static_cast<std::uint32_t>(call.nranks);
            // A native handle can be created before the Python wrapper binds
            // it to the frozen logical group. Its first transfer carries that
            // group's normalized UID; unused handles need no group binding.
            if (!uid_empty(comm.uid)) bind_member(comm, op.rank);
            if (!rank.communicators.emplace(call.comm, comm).second) fail("paper replay duplicate communicator handle");
            break;
        }
        case FAKECUDA_RAS_COMM_DESTROY:
            if (!rank.communicators.erase(call.comm)) fail("paper replay unknown communicator destruction");
            break;
        case FAKECUDA_RAS_GROUP_START:
            if (rank.group_depth == std::numeric_limits<unsigned>::max()) fail("paper replay NCCL group nesting overflow");
            ++rank.group_depth; break;
        case FAKECUDA_RAS_GROUP_END:
            if (!rank.group_depth) fail("paper replay unmatched NCCL GroupEnd");
            if (!--rank.group_depth) close_group(rank, origin.host);
            break;
        case FAKECUDA_RAS_SEND: case FAKECUDA_RAS_RECV: case FAKECUDA_RAS_ALLREDUCE:
        case FAKECUDA_RAS_BROADCAST: case FAKECUDA_RAS_ALLGATHER: case FAKECUDA_RAS_REDUCESCATTER: {
            auto comm = rank.communicators.find(call.comm);
            if (comm == rank.communicators.end()) fail("paper replay unknown communicator");
            ReplayUid call_uid{};
            std::copy_n(call.comm_uid, call_uid.size(), call_uid.begin());
            if (uid_empty(comm->second.uid)) {
                if (uid_empty(call_uid)) fail("paper replay communicator has no logical group binding");
                comm->second.uid = call_uid;
                bind_member(comm->second, op.rank);
            } else if (!uid_empty(call_uid) && comm->second.uid != call_uid) {
                fail("paper replay communication changed logical communicator group");
            }
            if (call.nranks > 0 && (static_cast<std::uint32_t>(call.nranks) != comm->second.size ||
                call.rank < 0 || static_cast<std::uint32_t>(call.rank) != comm->second.local))
                fail("paper replay communication communicator rank/size mismatch");
            if(call.op==FAKECUDA_RAS_BROADCAST&&(call.peer<0||static_cast<std::uint32_t>(call.peer)>=comm->second.size))
                fail("paper replay broadcast root is outside communicator");
            Pending pending{input, call.stream, origin.device, comm->second, readiness(rank, call.stream)};
            if (rank.group_depth) rank.group.push_back(std::move(pending));
            else {
                pending.ready.push_back(origin.host); submit(std::move(pending));
                stream(rank, call.stream).frontier = {origin.device};
            }
            break;
        }
        default: break;
        }
    }
    void close_group(Rank& rank, NodeId commit) {
        std::set<NodeId> ready{commit};
        std::set<std::uint64_t> streams;
        std::vector<NodeId> completions;
        for (const auto& pending : rank.group) {
            ready.insert(pending.ready.begin(), pending.ready.end());
            const auto latest = readiness(rank, pending.stream);
            ready.insert(latest.begin(), latest.end());
            streams.insert(pending.stream); completions.push_back(pending.node);
        }
        for (auto& pending : rank.group) {
            pending.ready.assign(ready.begin(), ready.end()); submit(std::move(pending));
        }
        for (const auto id : streams) stream(rank, id).frontier = completions;
        rank.group.clear();
    }
    void match(std::span<const Pending> members) {
        if (members.empty()) fail("paper replay empty communication match");
        const auto first = read(members[0].input);
        const auto duration = nodes_.at(members[0].node).duration;
        for (const auto& member : members) {
            const auto other = read(member.input);
            if (member.comm.size != members[0].comm.size) fail("paper replay NCCL communicator size mismatch");
            if (other.call.count != first.call.count || other.call.datatype != first.call.datatype ||
                other.call.bytes != first.call.bytes ||
                (collective_operation(first.call.op)&&other.call.op!=first.call.op) ||
                ((first.call.op == FAKECUDA_RAS_ALLREDUCE||first.call.op==FAKECUDA_RAS_REDUCESCATTER) && other.call.reduction != first.call.reduction) ||
                (first.call.op==FAKECUDA_RAS_BROADCAST&&other.call.peer!=first.call.peer))
                fail("paper replay NCCL signature mismatch");
            if (nodes_.at(member.node).duration != duration) fail("paper replay NCCL participants have unequal modeled costs");
        }
        for(std::size_t i=0;i<members.size();++i)nodes_.at(members[i].node).match=members[(i+1)%members.size()].node;
        for (const auto& target : members)
            for (const auto& source : members)
                for (auto ready : source.ready) edge(ready, target.node);
    }
    void submit(Pending pending) {
        const auto op = read(pending.input);
        const auto& comm = pending.comm;
        if (collective_operation(op.call.op)) {
            const auto ordinal = collective_ordinals_[{comm.uid, comm.local}]++;
            const CollectiveKey key{comm.uid, ordinal};
            auto& collective = collectives_[key];
            if (collective.size && collective.size != comm.size) fail("paper replay NCCL communicator size mismatch");
            collective.size = comm.size;
            if (!collective.members.emplace(comm.local, std::move(pending)).second) fail("paper replay duplicate collective rank");
            if (collective.members.size() == collective.size) {
                std::vector<Pending> members;
                members.reserve(collective.members.size());
                for (auto& [local, member] : collective.members) { (void)local; members.push_back(std::move(member)); }
                match(members); collectives_.erase(key);
            }
        } else {
            if (op.call.peer < 0 || static_cast<std::uint32_t>(op.call.peer) >= comm.size)
                fail("paper replay peer is outside communicator");
            const bool send = op.call.op == FAKECUDA_RAS_SEND;
            const auto source = send ? comm.local : static_cast<std::uint32_t>(op.call.peer);
            const auto target = send ? static_cast<std::uint32_t>(op.call.peer) : comm.local;
            const auto ordinal = p2p_ordinals_[{comm.uid, source, target, send}]++;
            const P2PKey key{comm.uid, source, target, ordinal};
            auto old = p2p_.find(key);
            if (old == p2p_.end()) p2p_.emplace(key, std::move(pending));
            else {
                if (read(old->second.input).call.op == op.call.op) fail("paper replay duplicate P2P direction");
                std::array<Pending, 2> members{std::move(old->second), std::move(pending)};
                match(members); p2p_.erase(old);
            }
        }
    }

    const ReplayInput& input_;
    Arena<Node> nodes_;
    Arena<Edge> edges_;
    Arena<NodeId> queue_;
    Arena<ReplayTiming> timings_;
    std::unique_ptr<Arena<CachedInput>> inputs_;
    std::unique_ptr<Arena<Endpoint>> endpoints_;
    ReplayResult result_;
    std::map<std::uint32_t, Rank> ranks_;
    std::map<ReplayUid, Membership> memberships_;
    std::map<P2PKey, Pending> p2p_;
    std::map<CollectiveKey, Collective> collectives_;
    std::map<std::tuple<ReplayUid, std::uint32_t, std::uint32_t, bool>, std::uint64_t> p2p_ordinals_;
    std::map<std::pair<ReplayUid, std::uint32_t>, std::uint64_t> collective_ordinals_;
    NodeId last_target_ = none;
    std::set<NodeId> last_sources_;
};

namespace rt=core::runtime;
constexpr rt::Id cache_header_slot=310,cache_nodes_slot=311,cache_edges_slot=312;
constexpr rt::Id cache_timings_slot=313,cache_inputs_slot=314,cache_endpoints_slot=315;
constexpr ras::SchemaId cache_schema=0x4d4159411010ULL;
struct CacheHeader {
    std::uint64_t version=1,inputs=0,nodes=0,edges=0,endpoints=0,block_bytes=0;
};
struct CachedRankEnd { std::uint32_t rank=0,reserved=0;Tick end=0; };
template<class T> ras::Buffer cache_write(rt::Context& context,std::span<const T> values) {
    auto writer=context.buffer(values.size_bytes(),{cache_schema,sizeof(T),alignof(T)});
    if(!values.empty()){
        auto bytes=writer.reserve(values.size_bytes());std::memcpy(bytes.bytes().data(),values.data(),values.size_bytes());bytes.commit();
    }
    return writer.finish();
}
template<class T> ras::Buffer cache_write(rt::Context& context,const T& value) {
    return cache_write<T>(context,std::span<const T>(&value,1));
}
template<class T> T cache_scalar(const ras::Buffer& buffer) {
    if(buffer.size()!=sizeof(T))fail("paper replay invalid cache header");
    T result{};const auto bytes=buffer.bytes();std::memcpy(&result,bytes.span().data(),sizeof(T));return result;
}
bool cache_equal(const ras::Buffer& buffer,const rt::Evidence& evidence) {
    if(!buffer||buffer.size()!=evidence.size())return false;
    const auto bytes=buffer.bytes();return !std::memcmp(bytes.span().data(),evidence.data(),evidence.size());
}
std::uint64_t pages(std::uint64_t count,std::size_t bytes,std::size_t element) {
    const auto per_page=std::max<std::size_t>(1,bytes/element);
    return count/per_page+(count%per_page!=0);
}
void cache_pages(rt::Context& context,ras::DirectoryId directory,const std::vector<ras::Buffer>& buffers,
                 std::uint64_t previous_pages) {
    for(std::size_t i=0;i<buffers.size();++i)context.metadata_put(directory,{0,i},{ {},buffers[i],{} });
    for(auto i=buffers.size();i<previous_pages;++i)context.metadata_erase(directory,{0,i});
}
// Borrow unchanged RAS pages and copy only pages containing modified annotation
// rows. Event-sized graph/timing arrays never acquire another heap-owned store.
template<class T> class CachedPages {
    struct Writable {
        rt::OutputWriter writer;
        ras::WriteSpan borrow;
        std::span<T> values;
        Writable(rt::Context& context,const ras::ArrayView<T>& source,ras::SchemaId schema)
            :writer(context.buffer(source.size()*sizeof(T),{schema,sizeof(T),alignof(T)})),
             borrow(writer.reserve(source.size()*sizeof(T))),values(borrow.array<T>()) {
            std::memcpy(values.data(),source.span().data(),values.size_bytes());
        }
    };
public:
    CachedPages(rt::Context& context,ras::DirectoryId directory,std::uint64_t count,std::size_t block_bytes,ras::SchemaId schema=scratch_schema)
        :context_(context),directory_(directory),count_(count),per_page_(std::max<std::size_t>(1,block_bytes/sizeof(T))),schema_(schema){}
    const T& at(std::uint64_t index) {
        if(index>=count_)fail("paper replay cached index outside graph");
        const auto page=index/per_page_;
        if(const auto found=dirty_.find(page);found!=dirty_.end())return found->second->values[index%per_page_];
        return load(page)[index%per_page_];
    }
    T& edit(std::uint64_t index) {
        if(index>=count_)fail("paper replay cached edit outside graph");
        const auto page=index/per_page_;auto found=dirty_.find(page);
        if(found==dirty_.end())found=dirty_.emplace(page,std::make_unique<Writable>(context_,load(page),schema_)).first;
        return found->second->values[index%per_page_];
    }
    void flush() {
        for(auto& [page,output]:dirty_){output->borrow.commit();context_.metadata_put(directory_,{0,page},{ {},output->writer.finish(),{} });}
        dirty_.clear();views_.clear();
    }
    std::vector<ras::Buffer> buffers() {
        std::vector<ras::Buffer> result;const auto count=count_/per_page_+(count_%per_page_!=0);result.reserve(count);
        for(std::uint64_t page=0;page<count;++page){const auto row=context_.metadata_find(directory_,{0,page});if(!row)fail("paper replay missing cached page");result.push_back(row->payload);}
        return result;
    }
private:
    const ras::ArrayView<T>& load(std::uint64_t page) {
        auto found=views_.find(page);if(found!=views_.end())return found->second;
        const auto row=context_.metadata_find(directory_,{0,page});if(!row)fail("paper replay missing cached graph page");
        auto view=row->payload.array<T>(schema_);
        if(view.size()!=std::min<std::uint64_t>(per_page_,count_-page*per_page_))fail("paper replay cached page extent mismatch");
        return views_.emplace(page,std::move(view)).first->second;
    }
    rt::Context& context_;ras::DirectoryId directory_;std::uint64_t count_;std::size_t per_page_;
    ras::SchemaId schema_;
    std::map<std::uint64_t,ras::ArrayView<T>> views_;
    std::map<std::uint64_t,std::unique_ptr<Writable>> dirty_;
};
Feedback cache_feedback(const ras::Buffer& buffer) {
    Feedback feedback;
    const auto rows=buffer.array<CachedRankEnd>(cache_schema);
    for(const auto& row:rows.span()){
        feedback.rank_end.emplace_back(row.rank,row.end);feedback.makespan=std::max(feedback.makespan,row.end);
    }
    return feedback;
}
ras::Buffer save_feedback(rt::Context& context,const Feedback& feedback) {
    std::vector<CachedRankEnd> ranks;ranks.reserve(feedback.rank_end.size());
    for(const auto& [rank,end]:feedback.rank_end)ranks.push_back({rank,0,end});
    return cache_write<CachedRankEnd>(context,ranks);
}
// Entry declarations are part of graph identity even for direct callers. An
// omitted caller key must never permit reuse across a changed membership.
rt::Evidence cache_identity(const ReplayInput& input,const rt::Evidence& supplied) {
    rt::Evidence result;
    auto append=[&]<class T>(const T& value){const auto bytes=std::as_bytes(std::span<const T>(&value,1));result.insert(result.end(),bytes.begin(),bytes.end());};
    append(std::uint64_t(supplied.size()));result.insert(result.end(),supplied.begin(),supplied.end());
    append(std::uint64_t(input.memberships.size()));
    for(const auto& value:input.memberships){append(value.uid);append(std::uint64_t(value.members.size()));for(auto rank:value.members)append(rank);}
    append(std::uint64_t(input.streams.size()));
    for(const auto& value:input.streams){append(value.rank);append(value.id);append(std::uint32_t(value.blocking));}
    append(std::uint64_t(input.events.size()));for(const auto& value:input.events){append(value.rank);append(value.id);}
    append(std::uint64_t(input.communicators.size()));
    for(const auto& value:input.communicators){append(value.rank);append(value.id);append(value.local_rank);append(value.uid);append(std::uint64_t(value.members.size()));for(auto rank:value.members)append(rank);}
    return result;
}
} // namespace

ReplayResult replay(core::runtime::Context& context, const ReplayInput& input) {
    return Scheduler(context, input).run();
}

ReplayResult replay_cached(rt::Context& context,const ReplayInput& input,
                          std::span<const ReplayRange> changed,const rt::Evidence& identity,bool include_timing) {
    if(!input.block_bytes)fail("paper replay block capacity must be positive");
    if(input.count&&!input.read)fail("paper replay missing input cursor");
    const auto header_dir=context.evaluator_directory(cache_header_slot);
    const auto nodes_dir=context.evaluator_directory(cache_nodes_slot),edges_dir=context.evaluator_directory(cache_edges_slot);
    const auto timing_dir=context.evaluator_directory(cache_timings_slot),inputs_dir=context.evaluator_directory(cache_inputs_slot);
    const auto endpoints_dir=context.evaluator_directory(cache_endpoints_slot);
    const auto exact=cache_identity(input,identity);const auto previous=context.metadata_find(header_dir,{0,0});
    const auto old=previous?cache_scalar<CacheHeader>(previous->payload):CacheHeader{};
    auto rebuild=[&](std::uint64_t validation_reads=0){
        GraphSnapshot graph;auto result=Scheduler(context,input,true).run(&graph);
        result.input_reads+=validation_reads;
        cache_pages(context,nodes_dir,graph.nodes,previous?pages(old.nodes,old.block_bytes,sizeof(Node)):0);
        cache_pages(context,edges_dir,graph.edges,previous?pages(old.edges,old.block_bytes,sizeof(Edge)):0);
        cache_pages(context,timing_dir,result.timing,previous?pages(old.nodes,old.block_bytes,sizeof(ReplayTiming)):0);
        cache_pages(context,inputs_dir,graph.inputs,previous?pages(old.inputs,old.block_bytes,sizeof(CachedInput)):0);
        cache_pages(context,endpoints_dir,graph.endpoints,previous?pages(old.endpoints,old.block_bytes,sizeof(Endpoint)):0);
        const CacheHeader header{1,input.count,result.total_events,result.dependency_edges,graph.endpoint_count,input.block_bytes};
        context.metadata_put(header_dir,{0,0},{context.copy(exact,cache_schema),cache_write(context,header),save_feedback(context,result.feedback)});
        if(!include_timing)result.timing.clear();
        return result;
    };
    if(!previous||old.version!=1||old.inputs!=input.count||old.block_bytes!=input.block_bytes||!cache_equal(previous->metadata,exact))return rebuild();
    ReplayResult result;result.total_events=old.nodes;result.dependency_edges=old.edges;
    CachedPages<Node> nodes(context,nodes_dir,old.nodes,input.block_bytes);
    CachedPages<Edge> edges(context,edges_dir,old.edges,input.block_bytes);
    CachedPages<ReplayTiming> timings(context,timing_dir,old.nodes,input.block_bytes,replay_timing_schema);
    CachedPages<CachedInput> inputs(context,inputs_dir,old.inputs,input.block_bytes);
    CachedPages<Endpoint> endpoints(context,endpoints_dir,old.endpoints,input.block_bytes);
    std::vector<ReplayRange> ranges(changed.begin(),changed.end());
    for(const auto& range:ranges)if(range.first>input.count||range.count>input.count-range.first)fail("paper replay changed range outside input");
    std::sort(ranges.begin(),ranges.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    std::set<std::pair<std::uint64_t,NodeId>> queue;
    std::set<NodeId> communication_changes;
    auto update_cost=[&](NodeId id,Tick value){
        if(nodes.at(id).duration==value)return;
        nodes.edit(id).duration=value;
        const auto& timing=timings.at(id);queue.emplace(timing.topological_order,id);
        if(timing.kind==ReplayNodeKind::Communication)communication_changes.insert(id);
    };
    std::uint64_t consumed=0;
    for(const auto& range:ranges){
        const auto end=range.first+range.count;
        for(auto ordinal=std::max(consumed,range.first);ordinal<end;++ordinal){
            const auto op=input.read(ordinal);++result.input_reads;const auto& before=inputs.at(ordinal);
            if(!(before.key==schedule_key(op)))return rebuild(result.input_reads);
            update_cost(before.origin.host,op.cost.host);
            if(before.origin.device!=none)update_cost(before.origin.device,op.cost.device);
            else if(op.cost.device)fail("paper replay device cost on host-only operation");
        }
        consumed=std::max(consumed,end);
    }
    // A stable topology still requires the full scheduler's equal modeled
    // duration invariant for every communication match, including unchanged peers.
    std::set<NodeId> checked;
    for(auto id:communication_changes)if(!checked.contains(id)){
        const auto duration=nodes.at(id).duration;auto peer=id;
        do{
            if(peer==none||peer>=old.nodes)fail("paper replay invalid cached communication match");
            if(nodes.at(peer).duration!=duration)fail("paper replay NCCL participants have unequal modeled costs");
            if(!checked.insert(peer).second)fail("paper replay invalid cached communication cycle");
            peer=nodes.at(peer).match;
        }while(peer!=id);
    }
    while(!queue.empty()){
        const auto id=queue.begin()->second;queue.erase(queue.begin());++result.scheduled_events;
        const auto& node=nodes.at(id);Tick ready=0;
        for(auto eid=node.predecessors;eid!=none;eid=edges.at(eid).previous)ready=std::max(ready,timings.at(edges.at(eid).source).end);
        const auto finish=add(ready,node.duration);const auto prior=timings.at(id);
        if(prior.start!=ready||prior.end!=finish){auto& timing=timings.edit(id);timing.start=ready;timing.end=finish;++result.updated_events;}
        if(prior.end==finish)continue;
        for(auto eid=node.head;eid!=none;eid=edges.at(eid).next){const auto target=edges.at(eid).target;const auto& timing=timings.at(target);
            if(timing.topological_order<=prior.topological_order)fail("paper replay invalid cached topological order");
            queue.emplace(timing.topological_order,target);
        }
    }
    result.reused_events=old.nodes-result.scheduled_events;
    if(!result.scheduled_events)result.feedback=cache_feedback(previous->auxiliary);
    else{
        std::map<std::uint32_t,Tick> rank_end;
        for(std::uint64_t i=0;i<old.endpoints;++i){const auto& endpoint=endpoints.at(i);rank_end[endpoint.rank]=std::max(rank_end[endpoint.rank],timings.at(endpoint.node).end);}
        for(const auto& [rank,end]:rank_end){result.feedback.rank_end.emplace_back(rank,end);result.feedback.makespan=std::max(result.feedback.makespan,end);}
        nodes.flush();timings.flush();
        context.metadata_put(header_dir,{0,0},{previous->metadata,previous->payload,save_feedback(context,result.feedback)});
    }
    if(include_timing)result.timing=timings.buffers();
    return result;
}

} // namespace flexeva::maya::paper
