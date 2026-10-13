#include "paper_feedback.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace flexeva::maya::paper {
namespace {
namespace rt = core::runtime;
using Clock = std::chrono::steady_clock;
constexpr ras::SchemaId metadata_schema = 0x4d4159411101ULL;
constexpr ras::SchemaId position_schema = 0x4d4159411102ULL;
constexpr ras::SchemaId cost_schema = 0x4d4159411103ULL;
constexpr Id annotation_domain = 0x4d4159411104ULL;
constexpr Id index_slot = 301, header_slot = 302, cost_slot = 303, feedback_slot = 304;

[[noreturn]] void fail(const char* message) { throw ras::Error(message); }
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
template<class T> void append(Evidence& out,const T& value) {
    const auto bytes=std::as_bytes(std::span<const T>(&value,1));out.insert(out.end(),bytes.begin(),bytes.end());
}
void append(Evidence& out,const Evidence& value) {
    append(out,std::uint64_t(value.size()));out.insert(out.end(),value.begin(),value.end());
}
template<class T> ras::Buffer write(rt::Context& context,std::span<const T> rows,ras::SchemaId schema=metadata_schema) {
    auto writer=context.buffer(rows.size_bytes(),{schema,sizeof(T),alignof(T)});
    if(!rows.empty()){auto bytes=writer.reserve(rows.size_bytes());std::memcpy(bytes.bytes().data(),rows.data(),rows.size_bytes());bytes.commit();}
    return writer.finish();
}
template<class T> ras::Buffer write(rt::Context& context,const T& value,ras::SchemaId schema=metadata_schema) {
    return write<T>(context,std::span<const T>(&value,1),schema);
}
template<class T> T scalar(const ras::Buffer& buffer) {
    if(buffer.size()!=sizeof(T))fail("invalid Maya terminal metadata extent");
    T result{};const auto bytes=buffer.bytes();std::memcpy(&result,bytes.span().data(),sizeof(T));return result;
}
bool equal(const ras::Buffer& buffer,const Evidence& evidence) {
    if(!buffer||buffer.size()!=evidence.size())return false;
    const auto bytes=buffer.bytes();return !std::memcmp(bytes.span().data(),evidence.data(),evidence.size());
}
std::size_t extent(std::uint64_t count,std::size_t element) {
    if(count>std::numeric_limits<std::size_t>::max()/element)fail("Maya terminal buffer extent overflow");
    return static_cast<std::size_t>(count)*element;
}
struct Position { std::uint64_t block=0,offset=0; };
struct RankCache { Id space=0;std::uint64_t count=0,batches=0; };
struct FeedbackHeader { Tick makespan=0;std::uint64_t ranks=0,events=0,edges=0; };
struct RankEnd { std::uint32_t rank=0,reserved=0;Tick end=0; };
struct Body {
    const TerminalOutput* output=nullptr;
    ras::ByteView positions;
    std::vector<ras::ByteView> blocks;
    explicit Body(const TerminalOutput& value,ras::ByteView index)
        :output(&value),positions(std::move(index)) {
        blocks.reserve(value.blocks.size());for(const auto& block:value.blocks)blocks.push_back(block.bytes());
    }
    std::pair<const NativeRecord&,std::string_view> read(std::uint64_t ordinal) const {
        if(ordinal>=positions.size()/sizeof(Position))fail("Maya terminal native ordinal outside output");
        Position at{};std::memcpy(&at,positions.span().data()+ordinal*sizeof(Position),sizeof(at));
        if(output->position_backing){if(at.block<output->position_block_base)fail("Maya terminal index block precedes output");at.block-=output->position_block_base;}
        if(at.block>=blocks.size())fail("Maya terminal index block outside output");
        const auto bytes=blocks[at.block].span();
        if(at.offset>bytes.size()||bytes.size()-at.offset<sizeof(NativeRecord)||at.offset%alignof(NativeRecord))
            fail("Maya terminal native offset outside block");
        const auto& row=*reinterpret_cast<const NativeRecord*>(bytes.data()+at.offset);
        if(row.kernel_name_size>bytes.size()-at.offset-sizeof(row))fail("Maya terminal kernel name outside block");
        return {row,{reinterpret_cast<const char*>(&row+1),static_cast<std::size_t>(row.kernel_name_size)}};
    }
};
FakeCudaRasRecordV2 raw_input(const NativeRecord& row,std::string_view name) {
    FakeCudaRasRecordV2 raw{};raw.base=row.operation.call;raw.base.rank=static_cast<std::int32_t>(row.operation.rank);
    raw.struct_size=sizeof(raw);raw.kernel_name=name.data();raw.kernel_name_size=name.size();raw.flags_v2=row.flags_v2;
    std::copy_n(row.grid,3,raw.grid);std::copy_n(row.block,3,raw.block);raw.shared_mem=row.shared_mem;raw.batch_count=row.batch_count;
    raw.stride_a=row.stride_a;raw.stride_b=row.stride_b;raw.stride_c=row.stride_c;raw.stride_d=row.stride_d;raw.d=row.d;raw.aux=row.aux;
    raw.ldd=row.ldd;raw.d_type=row.d_type;raw.epilogue=row.epilogue;raw.pointer_mode=row.pointer_mode;raw.math_mode=row.math_mode;raw.api_kind=row.api_kind;
    return raw;
}
NativeRecord projected(const TerminalFeedbackInput& input,const TerminalOutput& output,std::uint32_t rank,const NativeRecord& row) {
    auto result=input.project?input.project(output,rank,row):row;
    result.operation.rank=rank;return result;
}
Evidence output_key(const TerminalOutput& output,const Options& options,std::uint32_t rank) {
    Evidence key;append(key,std::uint64_t{2});append(key,options.cost_identity);append(key,output.body_version);append(key,output.semantic);
    append(key,output.projection);append(key,output.count);append(key,output.first);append(key,rank);append(key,options.chunk_bytes);
    append(key,std::uint64_t(bool(options.native_cost_key)));append(key,std::uint64_t(bool(options.native_costs)));
    return key;
}
Evidence feedback_key(const TerminalFeedbackInput& input,const Options& options) {
    Evidence key;append(key,std::uint64_t{2});append(key,options.cost_identity);append(key,input.replay_identity);
    append(key,std::uint64_t(bool(options.native_cost_key)));append(key,std::uint64_t(bool(options.native_costs)));
    append(key,std::uint64_t(input.outputs.size()));
    for(const auto& output:input.outputs){append(key,output.id);append(key,output.body_version);append(key,output.semantic);append(key,output.projection);append(key,output.count);append(key,output.first);append(key,std::uint64_t(output.ranks.size()));for(auto rank:output.ranks)append(key,rank);}
    append(key,std::uint64_t(input.seeds.memberships.size()));for(const auto& value:input.seeds.memberships){append(key,value.uid);append(key,std::uint64_t(value.members.size()));for(auto rank:value.members)append(key,rank);}
    append(key,std::uint64_t(input.seeds.streams.size()));for(const auto& value:input.seeds.streams){append(key,value.rank);append(key,value.id);append(key,std::uint64_t(value.blocking));}
    append(key,std::uint64_t(input.seeds.events.size()));for(const auto& value:input.seeds.events){append(key,value.rank);append(key,value.id);}
    append(key,std::uint64_t(input.seeds.communicators.size()));for(const auto& value:input.seeds.communicators){append(key,value.rank);append(key,value.id);append(key,value.local_rank);append(key,value.uid);append(key,std::uint64_t(value.members.size()));for(auto rank:value.members)append(key,rank);}
    return key;
}
ras::Buffer native_index(rt::Context& context,ras::DirectoryId directory,const TerminalOutput& output,TerminalFeedbackResult& result) {
    if(output.position_backing){
        if(output.position_backing.size()!=extent(output.count,sizeof(Position)))fail("Maya producer position backing count mismatch");
        return output.position_backing;
    }
    Evidence key;append(key,output.count);append(key,std::uint64_t(output.blocks.size()));
    for(const auto& block:output.blocks){const auto identity=block.identity();append(key,identity.store);append(key,identity.allocation);append(key,identity.bytes.offset);append(key,identity.bytes.size);}
    if(const auto old=context.metadata_find(directory,{output.id,0});old&&equal(old->metadata,key))return old->payload;
    const auto bytes_count=extent(output.count,sizeof(Position));
    auto writer=context.buffer(bytes_count,{position_schema,sizeof(Position),alignof(Position)});
    auto bytes=writer.reserve(bytes_count);auto positions=bytes.array<Position>();std::size_t ordinal=0;
    if(!output.positions.empty()) {
        if(output.positions.size()!=output.count)fail("Maya producer index count mismatch");
        for(const auto& [block,offset]:output.positions){if(block>=output.blocks.size()||offset>output.blocks[block].size()||output.blocks[block].size()-offset<sizeof(NativeRecord)||offset%alignof(NativeRecord))fail("Maya producer index extent mismatch");positions[ordinal++]={block,offset};}
    } else {
        for(std::size_t block=0;block<output.blocks.size();++block){const auto view=output.blocks[block].bytes();std::size_t offset=0;
            while(offset<view.size()){
                if(ordinal>=positions.size()||view.size()-offset<sizeof(NativeRecord))fail("Maya terminal index native count mismatch");
                const auto& row=*reinterpret_cast<const NativeRecord*>(view.span().data()+offset);
                if(row.kernel_name_size>view.size()-offset-sizeof(row))fail("Maya terminal index truncated native record");
                const auto raw=sizeof(row)+static_cast<std::size_t>(row.kernel_name_size);
                if(raw>std::numeric_limits<std::size_t>::max()-7)fail("Maya terminal index padding overflow");
                const auto size=(raw+7)&~std::size_t{7};if(size>view.size()-offset)fail("Maya terminal index truncated padding");
                positions[ordinal++]={block,offset};offset+=size;++result.index_body_reads;
            }
        }
        if(ordinal!=positions.size())fail("Maya terminal index native count mismatch");
    }
    bytes.commit();auto buffer=writer.finish();context.metadata_put(directory,{output.id,0},{context.copy(key,metadata_schema),buffer,{}});return buffer;
}
void remove_annotations(rt::Context& context,ras::DirectoryId headers,ras::DirectoryId costs,Id output,
                        const std::set<std::uint32_t>& retained={}) {
    std::vector<std::pair<ras::DirectoryKey,RankCache>> removed;
    context.metadata_range(headers,{output,0},{output,std::numeric_limits<Id>::max()},[&](auto key,const auto& row){
        if(key.second>std::numeric_limits<std::uint32_t>::max()||!retained.contains(static_cast<std::uint32_t>(key.second)))removed.emplace_back(key,scalar<RankCache>(row.auxiliary));
    });
    for(const auto& [key,cache]:removed){for(std::uint64_t batch=0;batch<cache.batches;++batch)context.metadata_erase(costs,{cache.space,batch});context.metadata_erase(headers,key);}
}
struct Annotated {
    std::size_t body=0;
    std::uint32_t rank=0;
    std::uint64_t end=0;
    bool changed=false;
    std::vector<ras::ArrayView<Cost>> costs;
};
} // namespace

Feedback decode_terminal_feedback(const ras::Buffer& buffer) {
    Feedback result;if(!buffer)return result;const auto bytes=buffer.bytes();
    if(bytes.size()<sizeof(FeedbackHeader))fail("invalid Maya terminal feedback header");
    FeedbackHeader header{};std::memcpy(&header,bytes.span().data(),sizeof(header));
    if(header.ranks>(bytes.size()-sizeof(header))/sizeof(RankEnd)||bytes.size()!=sizeof(header)+header.ranks*sizeof(RankEnd))fail("invalid Maya terminal rank feedback extent");
    result.makespan=header.makespan;
    for(std::size_t i=0;i<header.ranks;++i){RankEnd rank{};std::memcpy(&rank,bytes.span().data()+sizeof(header)+i*sizeof(rank),sizeof(rank));result.rank_end.emplace_back(rank.rank,rank.end);}
    return result;
}

TerminalFeedbackResult terminal_feedback(rt::Context& context,const TerminalFeedbackInput& input,const Options& options,Metrics& metrics) {
    TerminalFeedbackResult result;const auto directory=context.evaluator_directory(feedback_slot);
    const auto key=feedback_key(input,options);
    if(const auto old=context.metadata_find(directory,{1,0});old&&equal(old->metadata,key)){
        result.state.payload=old->payload;const auto bytes=old->payload.bytes();result.state.semantic.assign(bytes.span().begin(),bytes.span().end());
        result.feedback=decode_terminal_feedback(old->payload);result.reused=true;return result;
    }
    const auto started=Clock::now();
    const auto indexes=context.evaluator_directory(index_slot),headers=context.evaluator_directory(header_slot),costs=context.evaluator_directory(cost_slot);
    for(auto id:input.removed){remove_annotations(context,headers,costs,id);context.metadata_erase(indexes,{id,0});}
    const auto per_batch=std::max<std::size_t>(1,options.chunk_bytes/sizeof(Cost));
    std::vector<Body> bodies;std::vector<Annotated> annotations;
    // Complete native keys describe all consumed cost inputs, so aliases and
    // independent bodies may share annotations within this transaction. The
    // conservative custom-cost path remains scoped to its exact output/rank.
    std::map<Evidence,ras::Buffer> shared_native_costs;
    bodies.reserve(input.outputs.size());std::set<Id> ids;
    for(const auto& output:input.outputs){
        if(!output.id||(output.count&&!output.body_version)||output.semantic.empty()||!ids.insert(output.id).second)fail("invalid Maya terminal output identity");
        const std::set<std::uint32_t> ranks(output.ranks.begin(),output.ranks.end());
        if(ranks.size()!=output.ranks.size()||(output.count&&ranks.empty()))fail("invalid Maya terminal logical ranks");
        if(!output.count){remove_annotations(context,headers,costs,output.id);context.metadata_erase(indexes,{output.id,0});continue;}
        remove_annotations(context,headers,costs,output.id,ranks);
        const auto body_index=bodies.size();const auto index=native_index(context,indexes,output,result);
        bodies.emplace_back(output,index.bytes());const auto& body=bodies.back();
        if(body.positions.size()!=extent(output.count,sizeof(Position)))fail("Maya terminal cached index count mismatch");
        const auto batches=output.count/per_batch+(output.count%per_batch!=0);
        for(auto rank:output.ranks){
            Annotated annotated;annotated.body=body_index;annotated.rank=rank;annotated.costs.reserve(batches);
            const auto exact=output_key(output,options,rank);const auto old=context.metadata_find(headers,{output.id,rank});
            const auto old_cache=old?scalar<RankCache>(old->auxiliary):RankCache{};
            const bool unchanged=old&&equal(old->metadata,exact)&&old_cache.count==output.count&&old_cache.batches==batches;
            annotated.changed=!unchanged;
            const std::array<Id,2> coordinate{output.id,rank};
            const Id space=old?old_cache.space:context.identity(annotation_domain,coordinate);
            for(std::uint64_t batch=0;batch<batches;++batch){
                const auto first=batch*per_batch,count=std::min<std::uint64_t>(per_batch,output.count-first);
                const auto previous=context.metadata_find(costs,{space,batch});ras::Buffer buffer;
                if(unchanged){if(!previous)fail("missing Maya terminal cached cost batch");buffer=previous->payload;
                    if(options.native_cost_key){const auto bytes=previous->metadata.bytes();shared_native_costs.emplace(Evidence(bytes.span().begin(),bytes.span().end()),buffer);}
                }
                else {
                    Evidence batch_key;append(batch_key,options.cost_identity);append(batch_key,first);append(batch_key,count);
                    if(options.native_cost_key){
                        for(std::uint64_t i=0;i<count;++i){const auto [native,name]=body.read(first+i);++result.feedback_body_reads;
                            const auto row=projected(input,output,rank,native);append(batch_key,options.native_cost_key(raw_input(row,name)));}
                    }else append(batch_key,rank);
                    if(const auto found=options.native_cost_key?shared_native_costs.find(batch_key):shared_native_costs.end();found!=shared_native_costs.end())buffer=found->second;
                    else if(options.native_cost_key&&previous&&equal(previous->metadata,batch_key))buffer=previous->payload;
                    else {
                        auto writer=context.buffer(extent(count,sizeof(Cost)),{cost_schema,sizeof(Cost),alignof(Cost)});
                        auto bytes=writer.reserve(extent(count,sizeof(Cost)));auto values=bytes.array<Cost>();
                        for(std::uint64_t i=0;i<count;++i){const auto [native,name]=body.read(first+i);++result.feedback_body_reads;
                            auto row=projected(input,output,rank,native);row.operation.boundary=0;
                            if(!options.native_costs&&!options.costs)fail("Maya terminal missing cost provider");
                            values[i]=options.native_costs?options.native_costs(raw_input(row,name)):options.costs({},row.operation);++metrics.cost_calls;
                        }
                        bytes.commit();buffer=writer.finish();
                    }
                    if(options.native_cost_key)shared_native_costs.emplace(batch_key,buffer);
                    context.metadata_put(costs,{space,batch},{context.copy(batch_key,metadata_schema),buffer,{}});
                }
                auto values=buffer.array<Cost>(cost_schema);if(values.size()!=count)fail("Maya terminal cached cost count mismatch");
                annotated.costs.push_back(std::move(values));
            }
            if(!unchanged){for(std::uint64_t batch=batches;batch<old_cache.batches;++batch)context.metadata_erase(costs,{space,batch});
                context.metadata_put(headers,{output.id,rank},{context.copy(exact,metadata_schema),{},write(context,RankCache{space,output.count,batches})});}
            annotations.push_back(std::move(annotated));
        }
    }
    metrics.cost_seconds+=elapsed(started);const auto replay_started=Clock::now();
    std::sort(annotations.begin(),annotations.end(),[&](const auto& a,const auto& b){return std::tie(a.rank,bodies[a.body].output->first,bodies[a.body].output->id)<std::tie(b.rank,bodies[b.body].output->first,bodies[b.body].output->id);});
    std::uint64_t total=0;std::vector<ReplayRange> changed;
    Evidence replay_key;append(replay_key,std::uint64_t{1});append(replay_key,input.replay_identity);append(replay_key,std::uint64_t(annotations.size()));
    for(auto& annotation:annotations){const auto& output=*bodies[annotation.body].output;const auto count=output.count;
        if(count>std::numeric_limits<std::uint64_t>::max()-total)fail("Maya terminal logical count overflow");
        if(annotation.changed)changed.push_back({total,count});
        append(replay_key,output.id);append(replay_key,annotation.rank);append(replay_key,output.first);append(replay_key,count);
        total+=count;annotation.end=total;
    }
    // Entry resource declarations can overlap creates in this window. Discover
    // those declarations as terminal replay input consumption, not publication.
    enum ResourceKind { Stream=1,Event=2,Comm=3 };
    std::set<std::tuple<std::uint32_t,int,std::uint64_t>> created;
    for(const auto& output:input.outputs)if(output.creations_complete){
        for(const auto& value:output.created_resources){if(value.kind<Stream||value.kind>Comm)fail("invalid Maya terminal created resource kind");created.emplace(value.rank,static_cast<int>(value.kind),value.id);}
    }
    for(const auto& annotation:annotations){const auto& body=bodies[annotation.body];
        if(body.output->creations_complete)continue;
        for(std::uint64_t i=0;i<body.output->count;++i){const auto [native,name]=body.read(i);(void)name;++result.feedback_body_reads;
            const auto row=projected(input,*body.output,annotation.rank,native);const auto& call=row.operation.call;
            switch(call.op){case FAKECUDA_RAS_STREAM_CREATE:created.emplace(annotation.rank,Stream,call.stream);break;case FAKECUDA_RAS_EVENT_CREATE:created.emplace(annotation.rank,Event,call.event);break;case FAKECUDA_RAS_COMM_INIT:created.emplace(annotation.rank,Comm,call.comm);break;default:break;}
        }
    }
    std::vector<ReplayStream> streams;std::vector<ReplayEvent> events;std::vector<ReplayCommunicator> communicators;
    for(const auto& seed:input.seeds.streams)if(!created.contains({seed.rank,Stream,seed.id}))streams.push_back(seed);
    for(const auto& seed:input.seeds.events)if(!created.contains({seed.rank,Event,seed.id}))events.push_back(seed);
    for(const auto& seed:input.seeds.communicators)if(!created.contains({seed.rank,Comm,seed.id}))communicators.push_back(seed);
    auto replay_input=input.seeds;replay_input.count=total;replay_input.streams=streams;replay_input.events=events;replay_input.communicators=communicators;replay_input.block_bytes=options.chunk_bytes;
    replay_input.read=[&](std::uint64_t ordinal){
        const auto found=std::upper_bound(annotations.begin(),annotations.end(),ordinal,[](auto value,const auto& annotation){return value<annotation.end;});
        if(found==annotations.end())fail("Maya terminal replay ordinal outside window");
        const auto& body=bodies[found->body];const auto local=ordinal-(found->end-body.output->count);
        const auto [native,name]=body.read(local);(void)name;++result.feedback_body_reads;const auto row=projected(input,*body.output,found->rank,native);
        const auto cost=found->costs.at(local/per_batch)[local%per_batch];
        return ReplayOperation{row.operation.call,cost,body.output->id,row.operation.sequence,found->rank};
    };
    auto replayed=replay_cached(context,replay_input,changed,replay_key,false);metrics.scheduled_events+=replayed.scheduled_events;
    metrics.replay_input_reads+=replayed.input_reads;metrics.replay_graph_nodes+=replayed.total_events;
    metrics.replay_updated_nodes+=replayed.updated_events;metrics.replay_reused_nodes+=replayed.reused_events;
    metrics.replay_graph_rebuilt+=replayed.graph_rebuilt;
    result.feedback=replayed.feedback;const FeedbackHeader header{result.feedback.makespan,result.feedback.rank_end.size(),replayed.total_events,replayed.dependency_edges};
    Evidence feedback;append(feedback,header);for(const auto& [rank,end]:result.feedback.rank_end)append(feedback,RankEnd{rank,0,end});
    result.state.payload=context.copy(feedback,metadata_schema);result.state.semantic=feedback;
    context.metadata_put(directory,{1,0},{context.copy(key,metadata_schema),result.state.payload,{}});
    metrics.replay_seconds+=elapsed(replay_started);return result;
}
} // namespace flexeva::maya::paper
