#include "paper_dependencies.hpp"
#include "paper_dependencies_index.hpp"
#include "paper_lanes.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace flexeva::maya::paper {
namespace {
namespace rt = core::runtime;
using Id = rt::Id;
constexpr Id prefix = 0x4d440300, maximum = std::numeric_limits<Id>::max();
constexpr ras::SchemaId index_schema = 0x4d4159411010ULL;
enum Resource : Id {
    Allocation = 1, StreamLife, EventLife, BlasLife, CommLife, EventValue, StreamValue,
    BlasBinding, GroupSubmit = 100, GroupMember, GroupCompletion, ChannelPeer = 110
};
struct IndexedCall {
    Id partition=0, sequence=0, stream=0, channel=0, side=0, side_index=0;
    std::uint32_t rank=0, op=0, participants=0, padding=0;
};
struct Group {
    Id begin=0, end=0, commit=0;
    auto operator<=>(const Group&) const = default;
};
struct Range { Id first=maximum, last=0; void add(Id a,Id b){first=std::min(first,a);last=std::max(last,b);} };
struct Pending { std::vector<Footprint> footprints; std::vector<IndexedCall> calls; };

// Capture chunks are contiguous intervals on each participating rank. A
// rank-local resource cannot have an outside definition/use between two calls
// inside one such interval. Persist its last definition and external frontier
// uses, rather than an index row for every raw operation. Compaction admits at
// most one interval per rank into a partition; rank is part of each key here.
void coalesce(std::vector<Footprint>& values) {
    const bool group_commit=std::any_of(values.begin(),values.end(),[](const auto& f){
        return f.resource[0]==GroupSubmit&&f.mode==FootprintMode::Define;
    });
    std::sort(values.begin(),values.end());
    values.erase(std::unique(values.begin(),values.end()),values.end());
    std::vector<Footprint> result;result.reserve(values.size());
    for(std::size_t first=0;first<values.size();) {
        auto last=first+1;while(last<values.size()&&values[last].resource==values[first].resource)++last;
        const auto kind=values[first].resource[0];
        // Group submit/peer references use synthetic positions. The common
        // commit also reads stream readiness at both ends of its whole group.
        if((kind>=GroupSubmit&&kind!=GroupCompletion)||(group_commit&&kind==StreamValue)) {
            result.insert(result.end(),values.begin()+first,values.begin()+last);first=last;continue;
        }
        const Footprint *first_definition=nullptr,*last_definition=nullptr,*first_use=nullptr,*last_use=nullptr;
        for(auto i=first;i<last;++i) {
            const auto& f=values[i];
            if(f.mode==FootprintMode::Define) {if(!first_definition)first_definition=&f;last_definition=&f;}
            else if(f.mode==FootprintMode::UsePrevious) {if(!first_use)first_use=&f;}
            else if(f.mode==FootprintMode::UseNext)last_use=&f;
        }
        if(first_use&&(!first_definition||first_use->sequence<=first_definition->sequence))result.push_back(*first_use);
        if(last_definition)result.push_back(*last_definition);
        if(last_use&&(!last_definition||last_use->sequence>=last_definition->sequence))result.push_back(*last_use);
        first=last;
    }
    values=std::move(result);
}

template<class T> ras::Buffer save(rt::Context& ctx,std::span<const T> values) {
    return ctx.copy(std::as_bytes(values),index_schema);
}
template<class T> ras::Buffer save(rt::Context& ctx,const T& value) {return save<T>(ctx,{&value,1});}
template<class T> T scalar(const ras::DirectoryRecord& record) {
    auto view=record.metadata.bytes();if(view.size()!=sizeof(T))throw ras::Error("Maya dependency index scalar size mismatch");
    T value{};std::memcpy(&value,view.span().data(),sizeof(T));return value;
}
template<class T> std::vector<T> rows(const ras::DirectoryRecord& record) {
    auto view=record.metadata.bytes();if(view.size()%sizeof(T))throw ras::Error("Maya dependency index row size mismatch");
    std::vector<T> result(view.size()/sizeof(T));if(!result.empty())std::memcpy(result.data(),view.span().data(),view.size());return result;
}
bool marker(std::uint32_t op) {return op==FAKECUDA_RAS_GROUP_START||op==FAKECUDA_RAS_GROUP_END;}
bool global_wait(std::uint32_t op) {return op==FAKECUDA_RAS_ALLOC||op==FAKECUDA_RAS_FREE||op==FAKECUDA_RAS_DEVICE_SYNC;}
bool indexed(std::uint32_t op) {
    return marker(op)||communication(op)||global_wait(op)||op==FAKECUDA_RAS_STREAM_WAIT_EVENT||op==FAKECUDA_RAS_EVENT_RECORD;
}
DependencyResourceKey resource(Id kind,Id rank,Id id=0,Id extra=0){return {kind,rank,id,extra};}

class Dependencies {
public:
    Dependencies(rt::Context& context,const DependencyPartitionReader& reader)
        : ctx_(context),reader_(reader),calls_(ctx_.evaluator_directory(prefix)),
          markers_(ctx_.evaluator_directory(prefix+1)),groups_(ctx_.evaluator_directory(prefix+2)),
          partition_calls_(ctx_.evaluator_directory(prefix+3)),channel_calls_(ctx_.evaluator_directory(prefix+4)),
          channel_sides_(ctx_.evaluator_directory(prefix+5)),streams_(ctx_.evaluator_directory(prefix+6)) {}

    rt::DependencyDelta update(std::span<const Id> changed,std::span<const Id> removed) {
        removed_.insert(removed.begin(),removed.end());
        for(auto id:changed)remove_calls(id);
        for(auto id:removed)remove_calls(id);
        for(auto id:changed) {
            auto& pending=pending_[id];pending.footprints.clear();pending.calls.clear();
            reader_(id,[&](const DependencyOperation& operation){map(id,operation,pending);});
        }
        // Publish all new call indexes before resolving neighboring definitions,
        // groups or matching peers: the result must not depend on delta order.
        for(auto id:changed)insert_calls(id,pending_.at(id).calls);
        for(auto id:changed)for(const auto& call:pending_.at(id).calls)if(global_wait(call.op))wait(call);
        for(auto [rank,range]:marker_changes_)repair_groups(rank,range);
        for(const auto& [rank,range]:group_changes_)repair_group_members(rank,range);
        for(auto channel:channels_)repair_channel(channel);
        for(auto [rank,first]:new_streams_) {
            ctx_.metadata_range(calls_,{rank,first},{rank,maximum},[&](auto,const auto& record){
                const auto call=scalar<IndexedCall>(record);if(global_wait(call.op))wait(call);
            });
        }
        std::vector<IndexUpdate> updates;updates.reserve(pending_.size());
        for(auto& [id,pending]:pending_)if(!removed_.contains(id)) {
            coalesce(pending.footprints);
            updates.push_back({id,std::move(pending.footprints)});
        }
        return update_dependencies(ctx_,updates,removed);
    }
private:
    Pending& pending(Id partition) {
        auto [it,inserted]=pending_.try_emplace(partition);
        if(inserted)it->second.footprints=read_dependency_footprints(ctx_,partition);
        return it->second;
    }
    void add(Id partition,DependencyResourceKey key,Id sequence,FootprintMode mode) {
        if(!removed_.contains(partition))pending(partition).footprints.push_back({key,sequence,mode});
    }
    void use(Pending& p,Id kind,std::uint32_t rank,Id id,Id seq) {
        if(id)p.footprints.push_back({resource(kind,rank,id),seq,FootprintMode::UsePrevious});
    }
    void define(Pending& p,Id kind,std::uint32_t rank,Id id,Id seq) {
        if(id)p.footprints.push_back({resource(kind,rank,id),seq,FootprintMode::Define});
    }
    std::optional<Group> containing(std::uint32_t rank,Id sequence) {
        auto found=sequence==maximum?ctx_.metadata_prev(groups_,{Id(rank)+1,0}):ctx_.metadata_prev(groups_,{rank,sequence+1});
        if(!found||found->first.first!=rank)return {};
        const auto group=scalar<Group>(found->second);return group.end>=sequence?std::optional(group):std::nullopt;
    }
    void affect_group(const IndexedCall& call) {
        if(auto group=containing(call.rank,call.sequence))group_changes_[call.rank].add(group->begin,group->end);
        if(marker(call.op))marker_changes_[call.rank].add(call.sequence,call.sequence);
    }
    void remove_calls(Id partition) {
        const auto old=ctx_.metadata_find(partition_calls_,{partition,0});if(!old)return;
        for(const auto& call:rows<IndexedCall>(*old)) {
            affect_group(call);
            ctx_.metadata_erase(calls_,{call.rank,call.sequence});
            if(marker(call.op))ctx_.metadata_erase(markers_,{call.rank,call.sequence});
            if(call.channel) {
                channels_.insert(call.channel);channel_old_parts_[call.channel].insert(partition);
                ctx_.metadata_erase(channel_calls_,{call.side_index,call.sequence});
            }
        }
        ctx_.metadata_erase(partition_calls_,{partition,0});
    }
    void stream(std::uint32_t rank,Id stream_id,Id sequence) {
        if(!ctx_.metadata_find(streams_,{rank,stream_id})) {
            ctx_.metadata_put(streams_,{rank,stream_id},{});
            auto [it,inserted]=new_streams_.try_emplace(rank,sequence);
            if(!inserted)it->second=std::min(it->second,sequence);
        }
    }
    void map(Id partition,const DependencyOperation& operation,Pending& p) {
        const auto& c=operation.call;const auto rank=operation.rank;const auto sequence=operation.sequence;
        stream(rank,0,sequence);
        if(c.stream||device_operation(c.op)||c.op==FAKECUDA_RAS_STREAM_SYNC)stream(rank,c.stream,sequence);
        use(p,EventLife,rank,c.event,sequence);use(p,BlasLife,rank,c.handle,sequence);use(p,CommLife,rank,c.comm,sequence);
        use(p,StreamLife,rank,c.stream,sequence);
        // Pointer alias identity is the allocation identity produced by native
        // normalization; Tensor values themselves are not cost-model inputs.
        for(auto address:{c.src,c.dst,c.a,c.b,c.c})use(p,Allocation,rank,address,sequence);
        if(c.op==FAKECUDA_RAS_ALLOC)define(p,Allocation,rank,c.dst,sequence);
        if(c.op==FAKECUDA_RAS_STREAM_CREATE)define(p,StreamLife,rank,c.stream,sequence);
        if(c.op==FAKECUDA_RAS_EVENT_CREATE)define(p,EventLife,rank,c.event,sequence);
        if(c.op==FAKECUDA_RAS_BLAS_CREATE)define(p,BlasLife,rank,c.handle,sequence);
        if(c.op==FAKECUDA_RAS_COMM_INIT)define(p,CommLife,rank,c.comm,sequence);
        if(c.op==FAKECUDA_RAS_BLAS_SET_STREAM)define(p,BlasBinding,rank,c.handle,sequence);
        if(c.op==FAKECUDA_RAS_GEMM)use(p,BlasBinding,rank,c.handle,sequence);
        if(device_operation(c.op)) {
            p.footprints.push_back({resource(StreamValue,rank,c.stream),sequence,FootprintMode::Define});
            p.footprints.push_back({resource(GroupCompletion,rank,c.stream),sequence,FootprintMode::UsePrevious});
        }
        if(c.op==FAKECUDA_RAS_EVENT_RECORD) {
            define(p,EventValue,rank,c.event,sequence);
            p.footprints.push_back({resource(StreamValue,rank,c.stream),sequence,FootprintMode::UsePrevious});
        }
        if(c.op==FAKECUDA_RAS_STREAM_WAIT_EVENT||c.op==FAKECUDA_RAS_EVENT_SYNC)
            use(p,EventValue,rank,c.event,sequence);
        if(c.op==FAKECUDA_RAS_STREAM_WAIT_EVENT||c.op==FAKECUDA_RAS_STREAM_SYNC) {
            p.footprints.push_back({resource(StreamValue,rank,c.stream),sequence,FootprintMode::UsePrevious});
            p.footprints.push_back({resource(GroupCompletion,rank,c.stream),sequence,FootprintMode::UsePrevious});
        }
        if(!indexed(c.op))return;
        IndexedCall indexed_call{partition,sequence,c.stream,0,0,0,rank,c.op,0,0};
        if(communication(c.op)) {
            Id uid=0;std::memcpy(&uid,c.comm_uid,sizeof(uid));
            if(!uid||c.rank<0||c.nranks<=0||c.rank>=c.nranks)throw ras::Error("Maya dependency communication lacks normalized membership");
            Id source=c.rank,target=c.rank,type=0;
            if(c.op==FAKECUDA_RAS_ALLREDUCE) {source=target=0;type=1;indexed_call.side=c.rank;indexed_call.participants=c.nranks;}
            else {
                if(c.peer<0||c.peer>=c.nranks)throw ras::Error("Maya dependency communication peer outside membership");
                const bool send=c.op==FAKECUDA_RAS_SEND;source=send?c.rank:c.peer;target=send?c.peer:c.rank;
                indexed_call.side=send?0:1;indexed_call.participants=2;
            }
            const std::array<Id,4> coordinate{uid,source,target,type};
            indexed_call.channel=ctx_.identity(prefix+16,coordinate);
            const std::array<Id,2> side{indexed_call.channel,indexed_call.side};
            indexed_call.side_index=ctx_.identity(prefix+17,side);
        }
        p.calls.push_back(indexed_call);
    }
    void insert_calls(Id partition,const std::vector<IndexedCall>& values) {
        if(values.empty())return;
        const auto backing=save<IndexedCall>(ctx_,values);ctx_.metadata_put(partition_calls_,{partition,0},{backing,{},{}});
        for(std::size_t i=0;i<values.size();++i) {
            const auto& call=values[i];const ras::DirectoryRecord record{backing.slice({i*sizeof(IndexedCall),sizeof(IndexedCall)}),{}, {}};
            ctx_.metadata_put(calls_,{call.rank,call.sequence},record);
            if(marker(call.op))ctx_.metadata_put(markers_,{call.rank,call.sequence},record);
            affect_group(call);
            if(call.channel) {
                channels_.insert(call.channel);
                ctx_.metadata_put(channel_calls_,{call.side_index,call.sequence},record);
                // Reusing a side preserves its index; no complete trace exists
                // here, only an ordered membership index over communication.
                if(!ctx_.metadata_find(channel_sides_,{call.channel,call.side}))
                    ctx_.metadata_put(channel_sides_,{call.channel,call.side},record);
            }
        }
    }
    void repair_groups(std::uint32_t rank,Range range) {
        if(auto group=containing(rank,range.first))range.add(group->begin,group->end);
        if(auto group=containing(rank,range.last))range.add(group->begin,group->end);
        std::vector<Group> replacement;
        unsigned depth=0;Id begin=0,cursor=range.first,end=range.last;
        for(;;) {
            const auto next=ctx_.metadata_next(markers_,{rank,cursor});
            if(!next||next->first.first!=rank) {
                if(depth)throw ras::Error("Maya dependency incomplete NCCL group");
                break;
            }
            const auto call=scalar<IndexedCall>(next->second);
            if(!depth&&call.sequence>range.last)break;
            if(call.op==FAKECUDA_RAS_GROUP_START) {
                if(depth==std::numeric_limits<unsigned>::max())throw ras::Error("Maya dependency NCCL nesting overflow");
                if(!depth++)begin=call.sequence;
            } else {
                if(!depth)throw ras::Error("Maya dependency unmatched NCCL GroupEnd");
                if(!--depth)replacement.push_back({begin,call.sequence,call.partition});
            }
            end=std::max(end,call.sequence);
            if(call.sequence==maximum)break;
            cursor=call.sequence+1;
            if(!depth&&call.sequence>=range.last)break;
        }
        std::vector<ras::DirectoryKey> old;
        ctx_.metadata_range(groups_,{rank,range.first},{rank,end},[&](auto key,const auto& record){
            old.push_back(key);const auto group=scalar<Group>(record);clear_group(rank,group);end=std::max(end,group.end);
        });
        for(auto key:old)ctx_.metadata_erase(groups_,key);
        for(const auto& group:replacement)ctx_.metadata_put(groups_,{rank,group.begin},{save(ctx_,group),{}, {}});
        group_changes_[rank].add(range.first,end);
    }
    void clear_group(std::uint32_t rank,const Group& group) {
        auto clear=[&](Id partition) {
            if(removed_.contains(partition))return;
            auto& values=pending(partition).footprints;
            std::erase_if(values,[&](const Footprint& f){
                return ((f.resource[0]==GroupSubmit||f.resource[0]==GroupMember)&&f.resource[1]==rank&&f.resource[2]==group.begin)||
                    (f.resource[0]==GroupCompletion&&f.resource[1]==rank&&f.mode==FootprintMode::Define&&f.sequence==group.end);
            });
        };
        clear(group.commit);
        ctx_.metadata_range(calls_,{rank,group.begin},{rank,group.end},[&](auto,const auto& record){clear(scalar<IndexedCall>(record).partition);});
    }
    void repair_group_members(std::uint32_t rank,Range range) {
        // Restore ordinary stream definitions for calls that left a group.
        std::vector<IndexedCall> affected;
        ctx_.metadata_range(calls_,{rank,range.first},{rank,range.last},[&](auto,const auto& record){
            const auto call=scalar<IndexedCall>(record);if(communication(call.op))affected.push_back(call);
        });
        for(const auto& call:affected) {
            auto& values=pending(call.partition).footprints;
            std::erase_if(values,[&](const Footprint& f){return f.resource==resource(StreamValue,rank,call.stream)&&f.sequence==call.sequence&&f.mode==FootprintMode::Define;});
            if(!containing(rank,call.sequence))add(call.partition,resource(StreamValue,rank,call.stream),call.sequence,FootprintMode::Define);
        }
        ctx_.metadata_range(groups_,{rank,range.first},{rank,range.last},[&](auto,const auto& record){
            const auto group=scalar<Group>(record);clear_group(rank,group);
            std::vector<IndexedCall> members;std::set<Id> participating_streams;
            ctx_.metadata_range(calls_,{rank,group.begin},{rank,group.end},[&](auto,const auto& rec){
                const auto call=scalar<IndexedCall>(rec);
                if(communication(call.op)) {members.push_back(call);participating_streams.insert(call.stream);}
                else if(call.op==FAKECUDA_RAS_STREAM_WAIT_EVENT||call.op==FAKECUDA_RAS_EVENT_RECORD)members.push_back(call);
            });
            const auto submit=resource(GroupSubmit,rank,group.begin);
            add(group.commit,submit,0,FootprintMode::Define);
            for(const auto& call:members) {
                const auto member=resource(GroupMember,rank,group.begin,call.sequence);
                add(call.partition,member,0,FootprintMode::Define);add(group.commit,member,1,FootprintMode::UsePrevious);
                if(communication(call.op))add(call.partition,submit,call.partition,FootprintMode::UsePrevious);
            }
            for(auto stream_id:participating_streams) {
                add(group.commit,resource(GroupCompletion,rank,stream_id),group.end,FootprintMode::Define);
                add(group.commit,resource(StreamValue,rank,stream_id),group.begin,FootprintMode::UsePrevious);
                add(group.commit,resource(StreamValue,rank,stream_id),group.end,FootprintMode::UsePrevious);
            }
        });
    }
    void repair_channel(Id channel) {
        // Directed ordinal matching can shift after insertion/removal. Visit
        // only this channel's compact membership index, never trace payloads
        // or unrelated communicators. Equal partition footprints are retained
        // by update_dependencies even when their channel was inspected here.
        std::vector<std::vector<IndexedCall>> sides;
        std::set<Id> partitions=channel_old_parts_[channel];
        std::uint32_t participants=0;
        ctx_.metadata_range(channel_sides_,{channel,0},{channel,maximum},[&](auto,const auto& record){
            const auto entry=scalar<IndexedCall>(record);
            std::vector<IndexedCall> calls;
            ctx_.metadata_range(channel_calls_,{entry.side_index,0},{entry.side_index,maximum},[&](auto,const auto& rec){
                auto call=scalar<IndexedCall>(rec);partitions.insert(call.partition);calls.push_back(call);
            });
            if(!calls.empty()) {
                const auto size=calls.front().participants;
                if(participants&&participants!=size)throw ras::Error("Maya dependency inconsistent communication membership");
                for(const auto& call:calls)if(call.participants!=size)throw ras::Error("Maya dependency changed membership within channel");
                participants=size;sides.push_back(std::move(calls));
            }
        });
        for(auto partition:partitions)if(!removed_.contains(partition)) {
            auto& values=pending(partition).footprints;
            std::erase_if(values,[&](const Footprint& f){return f.resource[0]==ChannelPeer&&f.resource[1]==channel;});
        }
        if(sides.empty())return;
        if(sides.size()!=participants)throw ras::Error("Maya dependency incomplete communication membership");
        const auto count=sides.front().size();
        for(const auto& side:sides)if(side.size()!=count)throw ras::Error("Maya dependency incomplete directed communication channel");
        for(std::size_t ordinal=0;ordinal<count;++ordinal)for(std::size_t side=0;side<sides.size();++side) {
            const auto& producer=sides[side][ordinal];const auto key=resource(ChannelPeer,channel,ordinal+1,producer.side);
            add(producer.partition,key,0,FootprintMode::Define);
            for(std::size_t other=0;other<sides.size();++other)if(other!=side)
                add(sides[other][ordinal].partition,key,sides[other][ordinal].partition,FootprintMode::UsePrevious);
        }
    }
    void wait(const IndexedCall& call) {
        if(removed_.contains(call.partition))return;
        ctx_.metadata_range(streams_,{call.rank,0},{call.rank,maximum},[&](auto key,const auto&){
            add(call.partition,resource(StreamValue,call.rank,key.second),call.sequence,FootprintMode::UsePrevious);
            add(call.partition,resource(GroupCompletion,call.rank,key.second),call.sequence,FootprintMode::UsePrevious);
        });
    }
    rt::Context& ctx_;
    const DependencyPartitionReader& reader_;
    ras::DirectoryId calls_,markers_,groups_,partition_calls_,channel_calls_,channel_sides_,streams_;
    std::map<Id,Pending> pending_;
    std::set<Id> removed_,channels_;
    std::map<Id,std::set<Id>> channel_old_parts_;
    std::map<std::uint32_t,Range> marker_changes_,group_changes_;
    std::map<std::uint32_t,Id> new_streams_;
};
} // namespace

rt::DependencyDelta native_dependencies(const rt::LayerView&,const rt::LayerDelta& delta,
                                        rt::Context& context,const DependencyPartitionReader& read) {
    std::vector<Id> changed=delta.added;
    changed.insert(changed.end(),delta.changed.begin(),delta.changed.end());
    changed.insert(changed.end(),delta.state_changed.begin(),delta.state_changed.end());
    changed.insert(changed.end(),delta.topology_changed.begin(),delta.topology_changed.end());
    std::sort(changed.begin(),changed.end());changed.erase(std::unique(changed.begin(),changed.end()),changed.end());
    if(changed.empty()&&delta.removed.empty())return {};
    return Dependencies(context,read).update(changed,delta.removed);
}
} // namespace flexeva::maya::paper
