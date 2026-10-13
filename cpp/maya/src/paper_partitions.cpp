#include "paper_partitions.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace flexeva::maya::paper {
namespace {
namespace rt=core::runtime;
using Id=rt::Id;
constexpr auto maxid=std::numeric_limits<Id>::max();
constexpr ras::SchemaId schema=0x4d4159411010ULL;
struct Member {Id id=0,origin=0,ordinal=0,group=0,rank=0;};
struct Group {Id id=0,content=0,rank=0,first=0,last=0,events=0;std::array<Id,4> digest{};Id stream=0,device=0,bucket=0;};
template<class T>T read(const ras::Buffer& b){auto v=b.bytes();if(v.size()!=sizeof(T))throw ras::Error("Maya partition index extent mismatch");T x{};std::memcpy(&x,v.span().data(),sizeof x);return x;}
template<class T>ras::Buffer write(rt::Context& c,const T& x){return c.copy(std::as_bytes(std::span(&x,1)),schema);}
rt::Evidence evidence(Id id){auto b=std::as_bytes(std::span(&id,1));return {b.begin(),b.end()};}
bool equal(const ras::Buffer& a,const ras::Buffer& b){if(a.same_storage(b))return true;if(a.size()!=b.size())return false;auto x=a.bytes(),y=b.bytes();return !std::memcmp(x.span().data(),y.span().data(),x.size());}
bool equal_state(const ras::Buffer& a,const rt::Evidence& b){if(a.size()!=b.size())return false;auto x=a.bytes();return !std::memcmp(x.span().data(),b.data(),b.size());}
std::uint64_t body_bytes(std::span<const ras::Buffer> blocks){
    std::uint64_t bytes=0;
    for(const auto& block:blocks){
        if(!block||!block.size()||block.size()>std::numeric_limits<std::uint64_t>::max()-bytes)
            throw ras::Error("invalid Maya body block extent");
        if(block.layout()!=blocks.front().layout())throw ras::Error("inconsistent Maya body block layouts");
        bytes+=block.size();
    }
    return bytes;
}
bool equal_body(std::span<const ras::Buffer> a,std::span<const ras::Buffer> b){
    if(body_bytes(a)!=body_bytes(b))return false;
    std::size_t ai=0,bi=0,ao=0,bo=0;
    while(ai<a.size()&&bi<b.size()){
        const auto count=std::min(a[ai].size()-ao,b[bi].size()-bo);
        if(ao!=bo||!a[ai].same_storage(b[bi])){
            const auto x=a[ai].bytes(),y=b[bi].bytes();
            if(std::memcmp(x.span().data()+ao,y.span().data()+bo,count))return false;
        }
        ao+=count;bo+=count;
        if(ao==a[ai].size()){++ai;ao=0;}
        if(bo==b[bi].size()){++bi;bo=0;}
    }
    return ai==a.size()&&bi==b.size();
}
bool same_packing(std::span<const ras::Buffer> a,std::span<const ras::Buffer> b){
    if(a.size()!=b.size())return false;
    for(std::size_t i=0;i<a.size();++i)if(a[i].size()!=b[i].size()||a[i].layout()!=b[i].layout())return false;
    return true;
}
Id candidate_bucket(const PartitionCandidate& p){
    // This is only a lookup accelerator. matches() still compares the complete
    // evidence and body bytes. Placement belongs here because equal bodies at
    // different occurrences cannot share a logical partition.
    Id hash=14695981039346656037ULL;
    const auto add=[&](std::span<const std::byte> bytes){for(auto byte:bytes){hash^=std::to_integer<unsigned char>(byte);hash*=1099511628211ULL;}};
    const std::array<Id,10> fields{p.digest[0],p.digest[1],p.digest[2],p.digest[3],p.first,p.last,p.events,Id(p.lane.device),p.lane.stream,Id(p.state.size())};
    add(std::as_bytes(std::span(fields)));add(p.state);return hash;
}
struct WorkingGroup {
    Group data;
    ras::Buffer payload,state;
    rt::BlockBundle body;
    std::vector<ras::Buffer> blocks;
    bool loaded=false,modified=false;
};
}
std::vector<std::uint32_t> partition_ranks(const rt::Partition& p,const std::function<std::uint32_t(Id)>& rank){std::vector<std::uint32_t> out;for(auto id:p.origins)out.push_back(rank(id));std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;}
rt::PartitionDelta merge_partitions(const rt::PartitionRequest& request,rt::Context& context,
    std::span<const PartitionCandidate> candidates,bool compact){
    if(!request.initial&&request.changed_receipts.empty()&&request.removed_receipts.empty())return {};
    const auto members=context.evaluator_directory(101),groups=context.evaluator_directory(102),
        buckets=context.evaluator_directory(103),bygroup=context.evaluator_directory(104),
        ordered=context.evaluator_directory(105);
    const auto previous=context.previous();
    std::map<Id,WorkingGroup> working;
    auto group=[&](Id id)->WorkingGroup&{auto& g=working[id];if(!g.loaded){
        auto row=context.metadata_find(groups,{id,0});if(!row)throw ras::Error("Maya member has no group");
        g.data=read<Group>(row->metadata);g.payload=row->payload;g.state=row->auxiliary;g.loaded=true;
        if(previous&&previous.contains(id)){
            const auto part=previous.partition(id);
            if(!part.bundles.empty()){
                // Admission can retain an equal physical descriptor from the
                // preceding version. Reuse its public backing as well as the
                // public bundle, so a later state-only update stays physical-
                // layout neutral.
                g.payload=part.payload;
                g.body=part.bundles.front();
                for(std::size_t i=0;i<g.body.count();++i)g.blocks.push_back(context.block(g.body,i));
                const auto descriptor=read<BodyDescriptor>(g.payload);
                if(descriptor.bytes!=body_bytes(g.blocks)||descriptor.blocks!=g.body.count()||descriptor.events!=g.data.events)
                    throw ras::Error("Maya stored body descriptor mismatch");
            }
        }
    }return g;};
    auto group_members=[&](Id id){std::vector<Member> out;context.metadata_range(bygroup,{id,0},{id,maxid},[&](auto,const auto& row){out.push_back(read<Member>(row.metadata));});return out;};
    auto save_member=[&](const Member& m){ras::DirectoryRecord r{write(context,m),{}, {}};context.metadata_put(members,{m.origin,m.ordinal},r);context.metadata_put(bygroup,{m.group,m.id},r);};
    auto same_body=[&](const WorkingGroup& g,const PartitionCandidate& p){
        if(bool(g.body)!=!p.blocks.empty())return false;
        return p.blocks.empty()?equal(g.payload,p.payload):equal_body(g.blocks,p.blocks);
    };
    auto replace_body=[&](WorkingGroup& g,const PartitionCandidate& p){
        g.payload=p.payload;g.body={};g.blocks=p.blocks;
        if(!p.blocks.empty()){
            auto writer=context.blocks(p.blocks.front().size(),p.blocks.front().layout());
            for(const auto& block:p.blocks)writer.append(block);
            g.body=writer.finish();
        }
    };
    auto repack_body=[&](WorkingGroup& g,const PartitionCandidate& p){
        if(!p.blocks.empty()&&!same_packing(g.blocks,p.blocks)){replace_body(g,p);g.modified=true;}
    };
    auto matches=[&](WorkingGroup& g,const PartitionCandidate& p){return g.data.digest==p.digest&&g.data.events==p.events&&g.data.first==p.first&&g.data.last==p.last&&g.data.device==p.lane.device&&g.data.stream==p.lane.stream&&equal_state(g.state,p.state)&&same_body(g,p);};
    struct Generated {const PartitionCandidate* candidate;Id bucket;};
    std::map<std::pair<Id,Id>,Generated> generated;
    for(const auto& p:candidates){
        if(!p.blocks.empty()){
            const auto descriptor=read<BodyDescriptor>(p.payload);
            if(descriptor.bytes!=body_bytes(p.blocks)||descriptor.blocks!=p.blocks.size()||descriptor.events!=p.events)
                throw ras::Error("Maya candidate body descriptor mismatch");
        }
        if(!generated.emplace(std::pair{p.origin,p.ordinal},Generated{&p,candidate_bucket(p)}).second)throw ras::Error("duplicate Maya generated member");
    }
    std::set<Id> origins(request.changed_receipts.begin(),request.changed_receipts.end());origins.insert(request.removed_receipts.begin(),request.removed_receipts.end());
    std::map<std::pair<Id,Id>,Member> detached;
    for(auto origin:origins){std::vector<Member> old;context.metadata_range(members,{origin,0},{origin,maxid},[&](auto,const auto&r){old.push_back(read<Member>(r.metadata));});
        for(const auto& member:old){auto& g=group(member.group);auto it=generated.find({origin,member.ordinal});if(it!=generated.end()&&matches(g,*it->second.candidate)){
                if(it->second.candidate->blocks.empty())context.retain_receipt_chunk(origin,it->second.candidate->receipt_chunk,g.payload);
                else repack_body(g,*it->second.candidate);
                generated.erase(it);continue;
            }
            g.modified=true;detached[{origin,member.ordinal}]=member;context.metadata_erase(members,{origin,member.ordinal});context.metadata_erase(bygroup,{member.group,member.id});
        }
    }
    // Empty old groups remain candidates until merge completes. A changed
    // origin can replace the last member without changing the group's identity.
    for(const auto&[coordinate,generated_member]:generated){const auto& p=*generated_member.candidate;Id chosen=0;
        if(compact){context.metadata_range(buckets,{generated_member.bucket,0},{generated_member.bucket,maxid},[&](auto key,const auto&){if(chosen)return;auto& g=group(key.second);if(!matches(g,p))return;const auto ms=group_members(key.second);if(std::none_of(ms.begin(),ms.end(),[&](const auto&m){return m.rank==p.rank;}))chosen=key.second;});}
        const auto prior=detached.find(coordinate);
        if(!chosen){if(prior!=detached.end()&&group_members(prior->second.group).empty())chosen=prior->second.group;else chosen=context.new_identity();
            auto& g=working[chosen];if(g.loaded)context.metadata_erase(buckets,{g.data.bucket,chosen});
            const bool retain_body=g.loaded&&g.data.digest==p.digest&&g.data.events==p.events&&same_body(g,p);
            const auto content=retain_body?g.data.content:context.new_identity();
            if(retain_body){if(p.blocks.empty())context.retain_receipt_chunk(p.origin,p.receipt_chunk,g.payload);else repack_body(g,p);}
            else replace_body(g,p);
            g.data={chosen,content,p.rank,p.first,p.last,p.events,p.digest,p.lane.stream,p.lane.device,generated_member.bucket};g.state=context.copy(p.state,schema);g.loaded=g.modified=true;
            context.metadata_put(groups,{chosen,0},{write(context,g.data),g.payload,g.state});context.metadata_put(buckets,{generated_member.bucket,chosen},{});
        }else if(p.blocks.empty()){context.retain_receipt_chunk(p.origin,p.receipt_chunk,group(chosen).payload);}
        else repack_body(group(chosen),p);
        auto& g=group(chosen);g.modified=true;
        const auto member_id=prior==detached.end()?context.new_identity():prior->second.id;
        save_member({member_id,p.origin,p.ordinal,chosen,p.rank});
    }
    rt::PartitionDelta result;
    std::map<Id,ras::DirectoryKey> new_keys;
    std::set<Id> moved;
    std::map<std::size_t,rt::OrderSplice> splices;
    // Each touched group is at most an active-lane set, never the whole layer.
    for(auto&[id,g]:working){if(!g.modified)continue;const auto ms=group_members(id);
        const bool existed=previous&&previous.contains(id);ras::DirectoryKey oldkey{};
        if(existed){const auto old=context.metadata_find(groups,{id,0});if(!old)throw ras::Error("missing Maya old group index");
            // The old partition records rank and sequence through an immutable
            // order-key row, even when replacement changed group metadata.
            auto keyrow=context.metadata_find(context.evaluator_directory(106),{id,0});if(!keyrow)throw ras::Error("missing Maya group order key");oldkey=read<ras::DirectoryKey>(keyrow->metadata);
        }
        if(ms.empty()){context.metadata_erase(buckets,{g.data.bucket,id});context.metadata_erase(groups,{id,0});if(existed){result.removed.push_back(id);moved.insert(id);context.metadata_erase(ordered,oldkey);}continue;}
        rt::Partition part;part.id=id;part.payload=g.payload;part.event_count=g.data.events;part.semantic=evidence(g.data.content);auto state=g.state.bytes();part.state={state.span().begin(),state.span().end()};
        if(g.body){
            part.bundles={g.body};
            // Core may retain equal-sized payload storage when semantics and
            // state are unchanged. A changed physical descriptor must survive
            // that optimization without becoming semantic evidence.
            const auto descriptor=g.payload.bytes();part.layout={descriptor.span().begin(),descriptor.span().end()};
        }
        g.data.rank=maxid;
        for(const auto& m:ms){part.origins.push_back(m.origin);g.data.rank=std::min(g.data.rank,m.rank);const auto projection=lane_projection(static_cast<std::uint32_t>(m.rank),{bool(g.data.device),g.data.stream});for(std::size_t n=0;n<projection.size;++n){const auto& lane=projection.keys[n];const std::array<Id,3> key{lane.rank,Id(lane.kind),lane.stream};part.lanes.push_back(context.identity(0x4d41594c414e45ULL,key));}}
        std::sort(part.origins.begin(),part.origins.end());part.origins.erase(std::unique(part.origins.begin(),part.origins.end()),part.origins.end());std::sort(part.lanes.begin(),part.lanes.end());part.lanes.erase(std::unique(part.lanes.begin(),part.lanes.end()),part.lanes.end());
        result.upserts.push_back(std::move(part));context.metadata_put(groups,{id,0},{write(context,g.data),g.payload,g.state});
        const ras::DirectoryKey key{g.data.rank,g.data.first};if(existed&&key==oldkey)continue;
        if(existed)context.metadata_erase(ordered,oldkey);
        moved.insert(id);new_keys[id]=key;
    }
    for(auto id:moved)if(previous&&previous.contains(id)){const auto pos=previous.rank(id);splices[pos]={pos,1,{}};}
    // Remove all moving keys before locating insertion positions. Unchanged
    // successors provide parent-order coordinates; inserts sharing one gap are
    // sorted locally. No old-order enumeration or suffix rewrite is needed.
    std::map<std::size_t,std::vector<std::pair<ras::DirectoryKey,Id>>> insertions;
    for(auto[id,key]:new_keys){auto next=context.metadata_next(ordered,key);const auto pos=next?previous.rank(read<Id>(next->second.metadata)):(previous?previous.size():0);insertions[pos].push_back({key,id});}
    for(auto&[pos,values]:insertions){std::sort(values.begin(),values.end());auto& splice=splices[pos];splice.start=pos;for(const auto&[key,id]:values){splice.partitions.push_back(id);if(context.metadata_find(ordered,key))throw ras::Error("Maya partition order coordinate collision");context.metadata_put(ordered,key,{write(context,id),{}, {}});context.metadata_put(context.evaluator_directory(106),{id,0},{write(context,key),{}, {}});}}
    for(auto id:result.removed)context.metadata_erase(context.evaluator_directory(106),{id,0});
    for(auto&[pos,splice]:splices){(void)pos;result.order.push_back(std::move(splice));}
    return result;
}
}
