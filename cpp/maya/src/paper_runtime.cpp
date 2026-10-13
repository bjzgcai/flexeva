#include <flexeva/maya/paper_runtime.hpp>
#include "paper_scheduler.hpp"
#include "paper_partitions.hpp"
#include "paper_feedback.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace flexeva::maya::paper {
namespace {
namespace rt = core::runtime;
using Clock = std::chrono::steady_clock;
constexpr ras::SchemaId native_schema = trace_schema;
constexpr ras::SchemaId meta_schema = 0x4d4159411009ULL;

void count_native_borrows(void* counter,const ras::BufferIdentity&,
                          const ras::Layout& layout,std::size_t) {
    if(layout.schema==native_schema)++*static_cast<std::uint64_t*>(counter);
}

[[noreturn]] void fail(const char* message) { throw ras::Error(message); }
double seconds(Clock::time_point since) { return std::chrono::duration<double>(Clock::now()-since).count(); }
std::size_t padded(std::size_t size) {
    if (size > std::numeric_limits<std::size_t>::max()-7) fail("Maya native extent overflow");
    return (size+7)&~std::size_t(7);
}
template<class T> Evidence evidence(const T& value) {
    auto bytes=std::as_bytes(std::span<const T>(&value,1));return {bytes.begin(),bytes.end()};
}
void append(Evidence& destination,std::span<const std::byte> value) { destination.insert(destination.end(),value.begin(),value.end()); }
void append(Evidence& destination,const Evidence& value) { const std::uint64_t length=value.size();append(destination,std::as_bytes(std::span(&length,1)));append(destination,std::span<const std::byte>(value)); }
template<class T> void append(Evidence& destination,const T& value) { append(destination,std::as_bytes(std::span<const T>(&value,1))); }
template<class T> ras::Buffer write(rt::Context& context,std::span<const T> rows,ras::SchemaId schema=meta_schema) {
    auto writer=context.buffer(rows.size_bytes(),{schema,sizeof(T),alignof(T)});
    if(!rows.empty()) { auto output=writer.reserve(rows.size_bytes());std::memcpy(output.bytes().data(),rows.data(),rows.size_bytes());output.commit(); }
    return writer.finish();
}
template<class T> ras::Buffer write(rt::Context& context,const T& value,ras::SchemaId schema=meta_schema) { return write<T>(context,{&value,1},schema); }
template<class T> T scalar(const ras::Buffer& buffer) {
    auto bytes=buffer.bytes();if(bytes.size()!=sizeof(T))fail("Maya metadata extent mismatch");T value{};std::memcpy(&value,bytes.span().data(),sizeof(T));return value;
}
struct Digest {
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> context{EVP_MD_CTX_new(),&EVP_MD_CTX_free};
    Digest() { if(!context||EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)!=1)fail("cannot initialize Maya SHA256"); }
    Digest(Digest&&)=default;
    Digest& operator=(Digest&&)=default;
    void add(std::span<const std::byte> bytes) {if(EVP_DigestUpdate(context.get(),bytes.data(),bytes.size())!=1)fail("cannot update Maya SHA256");}
    template<class T> void add(const T& input) { add(std::as_bytes(std::span<const T>(&input,1))); }
    std::array<std::uint64_t,4> value() const {
        std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> copy{EVP_MD_CTX_new(),&EVP_MD_CTX_free};
        std::array<std::uint64_t,4> result{};unsigned size=0;
        if(!copy||EVP_MD_CTX_copy_ex(copy.get(),context.get())!=1||EVP_DigestFinal_ex(copy.get(),reinterpret_cast<unsigned char*>(result.data()),&size)!=1||size!=32)fail("cannot finish Maya SHA256");
        return result;
    }
    std::string hex() const {const auto result=value();std::ostringstream output;output<<std::hex<<std::setfill('0');for(auto b:std::as_bytes(std::span(result)))output<<std::setw(2)<<std::to_integer<unsigned>(b);return output.str();}
};
using CodeEvidence = code::EvidencePool;
auto code_header(const ras::Buffer& buffer) { return code::occurrence_header(buffer); }
auto code_region(const ras::Buffer& buffer) { return code::occurrence(buffer); }
auto code_slices(const ras::Buffer& buffer) { return code::evidence_slices(buffer); }
struct ChunkMeta { std::uint64_t events=0,first_sequence=0,last_sequence=0;std::array<std::uint64_t,4> digest{};std::uint64_t stream=0,device=0,relative_base=0,first_block=0,blocks=0; };
struct ReceiptMeta {
    Id region=0; std::uint64_t chunks=0,events=0;
    std::uint32_t rank=0,reserved=0;
    std::uint64_t slots=0,positions=0,created=0,body_version=0,resources=0,creations=0;
    std::array<std::uint64_t,4> entry{},exit{};
};
struct RecordPosition { std::uint64_t block=0,offset=0; };
struct CreatedResource { std::uint64_t chunk=0,kind=0,slot=0; };
struct ResourceSlot {std::uint64_t chunk=0,slot=0,id=0;};
constexpr std::uint64_t independent_rank=std::numeric_limits<std::uint64_t>::max();
struct NativeExtent { std::uint64_t count=0,first=0,last=0,stream=0,device=0,representative=independent_rank,relative_base=0,resources=0; };
struct ProjectionBinding { std::uint64_t rank=0,source_membership=0,target_membership=0,comm=0,local_rank=0,members=0; };
NativeExtent native_extent(const Evidence& state) {
    if(state.size()<sizeof(NativeExtent)||(state.size()-sizeof(NativeExtent))%sizeof(ProjectionBinding))fail("invalid Maya native projection state");
    NativeExtent result{};std::memcpy(&result,state.data(),sizeof(result));if(result.resources>(state.size()-sizeof(result))/sizeof(ProjectionBinding))fail("invalid Maya resource-slot extent");return result;
}
std::uint64_t project_resource(std::uint64_t slot,const Evidence& state) {
    if(!slot)return 0;
    const auto index=(slot>>1)-1;if(index<(state.size()-sizeof(NativeExtent))/sizeof(ProjectionBinding)){ProjectionBinding binding{};std::memcpy(&binding,state.data()+sizeof(NativeExtent)+index*sizeof(binding),sizeof(binding));if(binding.rank==independent_rank&&binding.source_membership==slot)return binding.target_membership;}
    fail("Maya body resource slot lacks a projection");
}
FakeCudaRasRecord project_call(FakeCudaRasRecord call,std::uint32_t rank,const Evidence& state) {
    const auto extent=native_extent(state);
    for(auto field:{&call.stream,&call.event,&call.handle,&call.comm,&call.src,&call.dst,&call.a,&call.b,&call.c})*field=project_resource(*field,state);
    Id membership=0;std::memcpy(&membership,call.comm_uid,sizeof(membership));if(membership){membership=project_resource(membership,state);std::memcpy(call.comm_uid,&membership,sizeof(membership));}
    if(extent.representative==independent_rank)return call;
    if(call.op!=FAKECUDA_RAS_COMM_INIT&&call.op!=FAKECUDA_RAS_COMM_DESTROY&&call.op!=FAKECUDA_RAS_SEND&&call.op!=FAKECUDA_RAS_RECV&&call.op!=FAKECUDA_RAS_ALLREDUCE&&call.op!=FAKECUDA_RAS_BROADCAST&&call.op!=FAKECUDA_RAS_ALLGATHER&&call.op!=FAKECUDA_RAS_REDUCESCATTER)return call;
    for(std::size_t offset=sizeof(NativeExtent)+extent.resources*sizeof(ProjectionBinding);offset<state.size();offset+=sizeof(ProjectionBinding)) {
        ProjectionBinding binding{};std::memcpy(&binding,state.data()+offset,sizeof(binding));
        if(binding.rank!=rank||((call.op==FAKECUDA_RAS_COMM_INIT||call.op==FAKECUDA_RAS_COMM_DESTROY)?binding.comm!=call.comm:binding.source_membership!=membership))continue;
        call.rank=static_cast<std::int32_t>(binding.local_rank);call.nranks=static_cast<std::int32_t>(binding.members);
        if(call.op!=FAKECUDA_RAS_COMM_INIT&&call.op!=FAKECUDA_RAS_COMM_DESTROY){std::memset(call.comm_uid,0,sizeof(call.comm_uid));std::memcpy(call.comm_uid,&binding.target_membership,sizeof(binding.target_membership));}
        return call;
    }
    if(extent.representative==rank)return call;
    fail("Maya rank projection lacks communicator binding");
}
struct Resource {
    std::uint64_t address=0,id=0,extent=0,extra=0;
    std::uint32_t rank=0,kind=0,flags=0,local_rank=0;
};
enum ResourceKind : std::uint32_t { Allocation=1,Stream=2,Event=3,Handle=4,Comm=5,Uid=6,HostAllocation=7 };
using ResourceKey=std::tuple<std::uint32_t,std::uint32_t,std::uint64_t>;
struct ResourceSummary { Resource value{}; std::uint64_t live=1,external=0; };
struct ResourceCreation {std::uint64_t kind=0,id=0;};
std::vector<ras::Buffer> body_blocks(const rt::LayerView& view,const rt::Partition& p) {
    if(p.bundles.empty())fail("Maya logical body lacks its storage bundle");
    const auto descriptor=scalar<BodyDescriptor>(p.payload);if(descriptor.blocks!=p.bundles.front().count()||descriptor.events!=p.event_count)fail("Maya logical body descriptor disagrees with its bundle");std::vector<ras::Buffer> result;std::uint64_t bytes=0;for(std::size_t n=0;n<descriptor.blocks;++n){auto block=view.block(p.bundles.front(),n);bytes+=block.size();result.push_back(std::move(block));}if(bytes!=descriptor.bytes)fail("Maya logical body byte extent mismatch");return result;
}

}

void for_each_native(const ras::Buffer& buffer,const std::function<void(const NativeRecord&,std::string_view)>& visit) {
    const auto bytes=buffer.bytes();std::size_t cursor=0;
    while(cursor<bytes.size()) {
        if(bytes.size()-cursor<sizeof(NativeRecord))fail("truncated Maya native record");
        const auto* row=reinterpret_cast<const NativeRecord*>(bytes.span().data()+cursor);
        if(row->kernel_name_size>bytes.size()-cursor-sizeof(NativeRecord))fail("truncated Maya kernel name");
        const auto extent=padded(sizeof(NativeRecord)+row->kernel_name_size);
        if(extent>bytes.size()-cursor)fail("truncated Maya native padding");
        visit(*row,{reinterpret_cast<const char*>(row+1),static_cast<std::size_t>(row->kernel_name_size)});cursor+=extent;
    }
}
void for_each_native(const rt::LayerView& view,const rt::Partition& p,const std::function<void(const NativeRecord&,std::string_view)>& visit){for(const auto& block:body_blocks(view,p))for_each_native(block,visit);}

struct Capture::Impl {
    const Input& input;
    Metrics& metrics;
    std::map<Id,rt::Context> scopes;
    std::map<Id,rt::ChunkWriter> writers;
    std::map<Id,rt::BlockWriter> prepared_writers;
    rt::Context* root_context=nullptr;
    std::map<Id,Evidence> ground_evidence;
    std::map<Id,std::set<Id>> ground_facts;
    std::set<Id> volatile_ground;
    std::map<Id,std::vector<rt::BlockBundle>> ground_values;
    std::optional<code::EvidenceWriter> code_writer;
    Id code_version=0;
    std::map<Id,std::vector<rt::OutputSlice>> code_slices;
    std::map<Id,std::vector<ChunkMeta>> metas;
    std::map<Id,std::vector<ResourceSlot>> resource_slots;
    std::map<Id,std::map<std::pair<Id,bool>,Id>> current_slots;
    std::map<Id,ChunkMeta> current_meta;
    std::map<Id,Digest> current_digest;
    std::map<ResourceKey,Resource>& resources;
    std::map<ResourceKey,Resource> history;
    std::map<std::tuple<std::uint32_t,std::uint32_t,std::uint64_t>,std::uint64_t> next_resource;
    std::map<Id,const Region*> regions;
    std::map<std::uint32_t,std::uint64_t> segments,offsets;
    std::map<std::pair<std::uint32_t,Id>,std::uint64_t> site_segments;
    std::map<Id,std::vector<ChunkMeta>> retained_metas;
    std::map<Id,std::vector<RecordPosition>> positions;
    std::map<Id,std::vector<CreatedResource>> created_resources;
    std::map<Id,std::array<std::uint64_t,4>> entry_keys,exit_keys;
    std::map<std::uint32_t,std::map<std::pair<std::uint32_t,Id>,Resource>> live_carry;
    std::map<std::uint32_t,Id> carry_versions;
    std::map<Id,std::map<std::pair<std::uint32_t,std::uint64_t>,ResourceSummary>> resource_summaries;
    std::map<Id,std::map<std::pair<std::uint32_t,std::uint64_t>,ResourceSummary>> retained_resources;
    std::map<Id,std::vector<ResourceCreation>> resource_creations,retained_creations;
    std::map<std::pair<Id,std::uint32_t>,std::size_t> creation_cursor;
    std::map<Id,Evidence>& resource_coordinates;
    std::uint32_t raw_operation=0;
    std::map<std::uint32_t,std::map<Id,Id>> blas_streams;
    std::function<void(Region,std::span<const Id>)> declare;
    std::function<void(Id)> decide;
    std::function<void(Id,std::span<const Id>)> dependencies;
    std::map<std::uint32_t,LaneClassifier> lane_classifiers;
    std::vector<Id> stack;
    std::set<Id> seen;
    std::exception_ptr failure;
    std::size_t capacity;
    bool live=true;
    Impl(const Input& i,Metrics& m,std::map<ResourceKey,Resource>& r,std::map<Id,Evidence>& coordinates,std::size_t bytes):input(i),metrics(m),resources(r),resource_coordinates(coordinates),capacity(bytes) {
        for(const auto& region:i.regions)regions.emplace(region.id,&region);
        for(auto& [key,value]:resources) {
            (void)key;value.id=0;
            // The new normalization window has no established BLAS binding
            // observations. Match its empty folding cache; imported physical
            // handles must not carry a previous window's known-binding flag.
            if(value.kind==Handle){value.flags=0;value.extra=0;}
        }
    }
    rt::OutputSlice append_code(std::uint64_t kind,std::span<const std::byte> packet,std::span<const Id> facts={}) {
        if(!root_context)fail("Code evidence requires a joint capture");
        if(!code_writer){code_writer.emplace(*root_context);code_version=code_writer->version();}
        return code_writer->append(kind,packet,facts);
    }
    const Region& active() const { if(!live||stack.empty())fail("Maya native operation outside registered region");return *regions.at(stack.back()); }
    bool record() const { return live&&!stack.empty()&&(writers.contains(stack.back())||prepared_writers.contains(stack.back()))&&active().produces_trace; }
    ras::WriteSpan reserve(Id id,std::size_t bytes,bool exact) {
        if(auto it=prepared_writers.find(id);it!=prepared_writers.end())return exact?it->second.reserve_exact(bytes):it->second.reserve(bytes);
        return exact?writers.at(id).reserve_exact(bytes):writers.at(id).reserve(bytes);
    }
    Resource& resource(std::uint32_t kind,std::uint64_t address,bool create=false,std::uint64_t extent=0) {
        const auto rank=active().rank;auto key=ResourceKey{rank,kind,address};auto it=resources.find(key);
        if(it!=resources.end()) { if(create && kind!=Allocation)fail("Maya resource created while already live");if(extent)it->second.extent=extent;if(!it->second.id)it->second.id=resource_id(rank,kind);remember(it->second);return it->second; }
        Id identity=0;
        if(!creates(raw_operation,kind))if(const auto scope=retained_resources.find(active().id);scope!=retained_resources.end())if(const auto old=scope->second.find({kind,address});old!=scope->second.end())identity=old->second.value.id;
        if(!identity)identity=resource_id(rank,kind);
        auto& result=resources.emplace(key,Resource{address,identity,extent,0,rank,kind,0,0}).first->second;remember(result);return result;
    }
    static bool sparse_kind(std::uint32_t kind) {return kind==Stream||kind==Event||kind==Handle||kind==Comm;}
    static bool creates(std::uint32_t op,std::uint32_t kind) {
        return (op==FAKECUDA_RAS_ALLOC&&kind==Allocation)||(op==FAKECUDA_RAS_STREAM_CREATE&&kind==Stream)||
            (op==FAKECUDA_RAS_EVENT_CREATE&&kind==Event)||(op==FAKECUDA_RAS_BLAS_CREATE&&kind==Handle)||
            (op==FAKECUDA_RAS_COMM_INIT&&kind==Comm);
    }
    void update_carry(std::uint32_t rank) {
        // Only sparse live normalization state participates. An exact interned
        // map is recomputed on resource mutations, never at every boundary.
        // Deletion and restored BLAS bindings therefore reconverge exactly.
        std::vector<Id> coordinate;
        for(const auto& [key,r]:live_carry[rank]){(void)key;coordinate.insert(coordinate.end(),{r.kind,r.id,r.extent,r.extra,r.flags,r.local_rank});}
        carry_versions[rank]=coordinate.empty()?0:root_context->identity(0x4d41594143415252ULL,coordinate);
    }
    void remember(const Resource& r) {
        if(!r.id)return;
        history[{r.rank,r.kind,r.id}]=r;
        if(!stack.empty()&&record()){
            auto& summaries=resource_summaries[active().id];const auto key=std::pair{r.kind,r.address};
            auto [found,fresh]=summaries.emplace(key,ResourceSummary{r,1,std::uint64_t(!creates(raw_operation,r.kind))});
            if(!fresh){found->second.value=r;found->second.live=1;}
        }
        if(!sparse_kind(r.kind))return;
        auto stable=r;stable.address=0;stable.rank=0;
        auto& rows=live_carry[r.rank];const auto key=std::pair{r.kind,r.id};auto found=rows.find(key);
        if(found!=rows.end()&&std::memcmp(&found->second,&stable,sizeof(stable))==0)return;
        rows[key]=stable;update_carry(r.rank);
    }
    std::array<std::uint64_t,4> carry(std::uint32_t rank) {
        return {carry_versions[rank],lane_classifiers[rank].group_depth(),0,0};
    }
    std::uint64_t resource_id(std::uint32_t rank,std::uint32_t kind) {
        if((kind==Allocation||kind==HostAllocation)&&!active().produces_trace)return 0;
        const bool creation=creates(raw_operation,kind);
        if(creation&&active().produces_trace)if(const auto old=retained_creations.find(active().id);old!=retained_creations.end()){
            auto& cursor=creation_cursor[{active().id,kind}];
            while(cursor<old->second.size()){const auto row=old->second[cursor++];if(row.kind==kind)return row.id;}
            fail("retained resource creation exceeds complete occurrence evidence");
        }
        const auto site=active().site_id?active().site_id:active().id;
        const auto occurrence=site_segments[{rank,site}];
        auto& next=next_resource[{rank,kind,site}];
        if(next==std::numeric_limits<std::uint64_t>::max())fail("Maya resource ordinal overflow");
        Evidence coordinate=active().site_identity.empty()?evidence(site):active().site_identity;
        append(coordinate,occurrence);append(coordinate,kind);append(coordinate,++next);
        Digest key;key.add(std::span<const std::byte>(coordinate));auto id=key.value()[0];if(!id)id=1;
        const auto [found,fresh]=resource_coordinates.emplace(id,coordinate);
        if(!fresh&&found->second!=coordinate)fail("Maya canonical resource identity collision");
        if(creation&&record())resource_creations[active().id].push_back({kind,id});
        return id;
    }
    std::uint64_t logical(std::uint32_t kind,std::uint64_t address) { return address?resource(kind,address).id:0; }
    void erase(std::uint32_t kind,std::uint64_t address) {
        const auto rank=active().rank;const auto key=ResourceKey{rank,kind,address};auto it=resources.find(key);
        if(it!=resources.end()){
            remember(it->second);
            if(record())resource_summaries[active().id][{kind,address}].live=0;
            if(sparse_kind(kind)&&live_carry[rank].erase({kind,it->second.id}))update_carry(rank);
            if(kind==Handle)blas_streams[rank].erase(it->second.id);
        }
        resources.erase(key);
    }
    void memory(std::uint64_t address,std::uint64_t& id,std::uint64_t& offset) {
        if(!address){id=offset=0;return;}const auto rank=active().rank;
        Resource* exact=nullptr;
        auto it=resources.upper_bound(ResourceKey{rank,Allocation,address});
        while(it!=resources.begin()) {
            --it;auto& r=it->second;if(r.rank!=rank||r.kind!=Allocation)break;
            if(r.extent&&address-r.address<r.extent) {
                if(!r.id)r.id=resource_id(rank,Allocation);
                remember(r);id=r.id;offset=address-r.address;return;
            }
            if(address==r.address&&!r.extent)exact=&r;
            // An exact imported suballocation must not hide its containing
            // native allocation. Positive native allocation extents do not overlap.
            if(r.extent)break;
        }
        // Torch may retain a live native pool across candidate captures. Import
        // its actual range so a fresh full run sees the same aliases and offsets
        // as a cold run that observed the pool's original allocation.
        std::uint64_t base=0,extent=0;
        if(fakecuda_ras_allocation_range(address,&base,&extent)) {
            if(!extent||address<base||address-base>=extent)fail("Maya native allocation query returned an invalid range");
            auto& r=resource(Allocation,base,false,extent);id=r.id;offset=address-base;return;
        }
        if(exact){if(!exact->id)exact->id=resource_id(rank,Allocation);remember(*exact);id=exact->id;offset=0;return;}
        // Other imported device addresses retain exact-pointer identity when
        // their provider cannot expose an allocation extent.
        id=resource(Allocation,address).id;offset=0;
    }
    void physical_flush(Id id) {
        const auto found=writers.find(id);const auto prepared=prepared_writers.find(id);if(found==writers.end()&&prepared==prepared_writers.end())return;
        if(!current_bytes_[id])return;
        auto& meta=current_meta[id];if(!meta.blocks)meta.first_block=physical_blocks[id];++meta.blocks;++physical_blocks[id];current_bytes_[id]=0;
        if(prepared!=prepared_writers.end())prepared->second.seal();else found->second.seal();
    }
    void seal(Id id) {
        if(!prepared_writers.contains(id)&&!writers.contains(id))return;
        physical_flush(id);auto& meta=current_meta[id];if(!meta.events)return;
        meta.digest=current_digest[id].value();metas[id].push_back(meta);meta={};current_digest[id]={};current_slots[id].clear();
    }
    bool can_retain_resources(std::span<const ResourceSummary> summaries,std::uint32_t rank) const {
        for(const auto& row:summaries)if(row.external) {
            const auto& old=row.value;const auto found=resources.find({rank,old.kind,old.address});
            if(found==resources.end())return false;
            const auto& now=found->second;
            if(const auto used=history.find({rank,old.kind,old.id});used!=history.end()&&used->second.address!=old.address)return false;
            if((now.id&&now.id!=old.id)||(now.extent&&old.extent&&now.extent!=old.extent))return false;
        }
        return true;
    }
    void bind_retained_resources(Id id) {
        for(const auto&[key,row]:retained_resources[id])if(row.external){(void)key;
            auto& r=resources.at({active().rank,row.value.kind,row.value.address});r.id=row.value.id;remember(r);
        }
    }
    void begin(Id id) {
        if(!live||!regions.contains(id))fail("unknown or expired Maya region");
        if(!stack.empty())seal(stack.back());
        const auto rank=regions.at(id)->rank;
        if(decide)decide(id); // Admission happens before the interceptor can elide raw payloads.
        if(!entry_keys.contains(id))entry_keys[id]=carry(rank);
        stack.push_back(id);++segments[rank];offsets[rank]=0;seen.insert(id);
        if(retained_metas.contains(id)){
            metas[id]=retained_metas.at(id);
            if(!metas[id].empty()){const auto before=metas[id].front().first_sequence>>32;
                for(auto& m:metas[id]){m.first_sequence=((segments[rank]+(m.first_sequence>>32)-before)<<32)|(m.first_sequence&0xffffffffULL);m.last_sequence=((segments[rank]+(m.last_sequence>>32)-before)<<32)|(m.last_sequence&0xffffffffULL);}}
            bind_retained_resources(id);
        }
        const auto site=active().site_id?active().site_id:active().id;
        ++site_segments[{rank,site}];
        for(std::uint32_t kind=Allocation;kind<=HostAllocation;++kind)next_resource[{rank,kind,site}]=0;
        ++metrics.executed_regions;if(!active().produces_trace)++metrics.preparation_regions;
    }
    void end() {
        const auto id=active().id;const auto rank=active().rank;seal(id);exit_keys[id]=carry(rank);
        stack.pop_back();++segments[rank];offsets[rank]=0;
        if(!stack.empty()){const auto site=active().site_id?active().site_id:active().id;++site_segments[{active().rank,site}];for(std::uint32_t kind=Allocation;kind<=HostAllocation;++kind)next_resource[{active().rank,kind,site}]=0;}
        if(!stack.empty()&&active().rank!=rank){++segments[active().rank];offsets[active().rank]=0;}
    }
    void communicator(std::uint64_t handle,Id membership,std::uint32_t local) {
        auto found=std::find_if(input.communicators.begin(),input.communicators.end(),[&](const auto& c){return c.id==membership;});
        if(found==input.communicators.end()||local>=found->members.size()||found->members[local]!=active().rank)fail("Maya communicator membership disagrees with global rank");
        auto& r=resource(Comm,handle);r.extra=membership;r.local_rank=local;r.extent=found->members.size();remember(r);
    }
    NativeRecord normalize(const FakeCudaRasRecordV2& raw) {
        ++metrics.normalization_raw_records;
        NativeRecord result{};auto& op=result.operation;op.call=raw.base;op.call.abi_version=2;
        op.rank=0;op.boundary=0;const auto rank=active().rank;
        if(segments[rank]>=1ULL<<32||offsets[rank]>=1ULL<<32)fail("Maya region occurrence sequence overflow");
        op.sequence=(segments[rank]<<32)|offsets[rank]++;
        // State-only allocator reset can free cached blocks from a prior
        // logical rank. Retiring these process-local entry allocations must
        // not consume identities in the new modeled window's namespace.
        if(raw.base.op==FAKECUDA_RAS_FREE&&!active().produces_trace) {
            for(auto it=resources.begin();it!=resources.end();) {
                if(it->second.kind==Allocation&&it->second.address==raw.base.src) {remember(it->second);it=resources.erase(it);}else ++it;
            }
            op.call.src=0;return result;
        }
        auto& c=op.call;c.stream=logical(Stream,raw.base.stream);
        c.event=logical(Event,raw.base.event);c.handle=logical(Handle,raw.base.handle);c.comm=logical(Comm,raw.base.comm);
        if(c.op==FAKECUDA_RAS_MEMCPY_ASYNC) {
            // Host pointer reuse must not consume or alias device allocation
            // identities. Explicit copy kinds identify each endpoint's space.
            if(c.copy_kind<0||c.copy_kind>3)fail("Maya memcpy requires an explicit supported copy kind");
            const bool host_source=c.copy_kind==0||c.copy_kind==1;
            const bool host_destination=c.copy_kind==0||c.copy_kind==2;
            if(host_source)c.src=logical(HostAllocation,raw.base.src);else memory(raw.base.src,c.src,op.src_offset);
            if(host_destination)c.dst=logical(HostAllocation,raw.base.dst);else memory(raw.base.dst,c.dst,op.dst_offset);
        } else {
            memory(raw.base.src,c.src,op.src_offset);memory(raw.base.dst,c.dst,op.dst_offset);
        }
        memory(raw.base.a,c.a,op.a_offset);memory(raw.base.b,c.b,op.b_offset);memory(raw.base.c,c.c,op.c_offset);
        switch(c.op) {
        case FAKECUDA_RAS_ALLOC: {auto& r=resource(Allocation,raw.base.dst,false,raw.base.bytes);c.dst=r.id;break;}
        case FAKECUDA_RAS_FREE: erase(Allocation,raw.base.src);break;
        case FAKECUDA_RAS_STREAM_CREATE: {auto& stream=resource(Stream,raw.base.stream);stream.flags=raw.base.flags;remember(stream);break;}
        case FAKECUDA_RAS_STREAM_DESTROY: erase(Stream,raw.base.stream);break;
        case FAKECUDA_RAS_EVENT_DESTROY: erase(Event,raw.base.event);break;
        case FAKECUDA_RAS_BLAS_CREATE: {auto& h=resource(Handle,raw.base.handle);h.extra=0;h.flags=0;blas_streams[rank].erase(h.id);remember(h);break;}
        case FAKECUDA_RAS_BLAS_DESTROY: erase(Handle,raw.base.handle);break;
        case FAKECUDA_RAS_BLAS_SET_STREAM: {auto& handle=resource(Handle,raw.base.handle);handle.extra=c.stream;handle.flags|=1;remember(handle);break;}
        case FAKECUDA_RAS_COMM_DESTROY: erase(Comm,raw.base.comm);break;
        case FAKECUDA_RAS_COMM_INIT: {
            auto& comm=resource(Comm,raw.base.comm);comm.local_rank=raw.base.rank;comm.extent=raw.base.nranks;remember(comm);
            // Membership is bound immediately after init; matching is done on
            // transfer records, whose normalized UID uses the registered ID.
            c.rank=raw.base.rank;c.nranks=raw.base.nranks;std::memset(c.comm_uid,0,sizeof(c.comm_uid));break;
        }
        case FAKECUDA_RAS_SEND:case FAKECUDA_RAS_RECV:case FAKECUDA_RAS_ALLREDUCE:case FAKECUDA_RAS_BROADCAST:case FAKECUDA_RAS_ALLGATHER:case FAKECUDA_RAS_REDUCESCATTER: {
            const auto& comm=resource(Comm,raw.base.comm);if(!comm.extra)fail("Maya transfer uses unregistered communicator membership");
            c.rank=static_cast<std::int32_t>(comm.local_rank);c.nranks=static_cast<std::int32_t>(comm.extent);
            std::memset(c.comm_uid,0,sizeof(c.comm_uid));std::memcpy(c.comm_uid,&comm.extra,sizeof(comm.extra));break;
        }
        case FAKECUDA_RAS_NCCL_UID: std::memset(c.comm_uid,0,sizeof(c.comm_uid));break;
        default:break;
        }
        result.flags_v2=raw.flags_v2;std::copy_n(raw.grid,3,result.grid);std::copy_n(raw.block,3,result.block);
        result.shared_mem=raw.shared_mem;result.batch_count=raw.batch_count;result.stride_a=raw.stride_a;result.stride_b=raw.stride_b;
        result.stride_c=raw.stride_c;result.stride_d=raw.stride_d;memory(raw.d,result.d,result.d_offset);memory(raw.aux,result.aux,result.aux_offset);
        result.ldd=raw.ldd;result.d_type=raw.d_type;result.epilogue=raw.epilogue;result.pointer_mode=raw.pointer_mode;result.math_mode=raw.math_mode;result.api_kind=raw.api_kind;
        result.kernel_name_size=raw.kernel_name_size;return result;
    }
    void advance_state(const FakeCudaRasRecordV2& raw) {
        const auto& c=raw.base;
        switch(c.op) {
        case FAKECUDA_RAS_ALLOC: resource(Allocation,c.dst,false,c.bytes);break;
        case FAKECUDA_RAS_FREE:
            if(!active().produces_trace){for(auto it=resources.begin();it!=resources.end();){if(it->second.kind==Allocation&&it->second.address==c.src)it=resources.erase(it);else ++it;}}
            else erase(Allocation,c.src);
            break;
        case FAKECUDA_RAS_STREAM_CREATE: {auto& r=resource(Stream,c.stream);r.flags=c.flags;remember(r);break;}
        case FAKECUDA_RAS_STREAM_DESTROY: erase(Stream,c.stream);break;
        case FAKECUDA_RAS_EVENT_CREATE: resource(Event,c.event);break;
        case FAKECUDA_RAS_EVENT_DESTROY: erase(Event,c.event);break;
        case FAKECUDA_RAS_BLAS_CREATE: {auto& h=resource(Handle,c.handle);h.extra=0;h.flags=0;blas_streams[active().rank].erase(h.id);remember(h);break;}
        case FAKECUDA_RAS_BLAS_DESTROY: erase(Handle,c.handle);break;
        case FAKECUDA_RAS_BLAS_SET_STREAM: {const auto stream=logical(Stream,c.stream);auto& h=resource(Handle,c.handle);h.extra=stream;h.flags|=1;remember(h);blas_streams[active().rank][h.id]=stream;break;}
        case FAKECUDA_RAS_COMM_INIT: {auto& comm=resource(Comm,c.comm);comm.local_rank=c.rank;comm.extent=c.nranks;remember(comm);break;}
        case FAKECUDA_RAS_COMM_DESTROY: erase(Comm,c.comm);break;
        case FAKECUDA_RAS_GROUP_START:case FAKECUDA_RAS_GROUP_END:lane_classifiers[active().rank].consume(c);break;
        default:break;
        }
    }
    void emit(const FakeCudaRasRecordV2& raw) {
        ++metrics.native_calls;raw_operation=raw.base.op;
        const bool generating=record();
        struct PhaseTimer {
            double& elapsed;
            bool enabled;
            Clock::time_point started;
            PhaseTimer(double& target,bool active):elapsed(target),enabled(active),started(active?Clock::now():Clock::time_point{}){}
            ~PhaseTimer(){if(enabled)elapsed+=seconds(started);}
        } timer{generating?metrics.normalization_seconds:metrics.retained_state_seconds,metrics.diagnostics_enabled};
        if(!generating) {
            // Payload reuse bypasses normalization, event indexes, local slots,
            // lane classification, folding and record hashing. Only native
            // operations that change sparse continuation state are interpreted.
            advance_state(raw);
            ++metrics.retained_calls;if(active().produces_trace)++metrics.retained_trace_calls;return;
        }
        auto normalized=normalize(raw);
        const auto rank=active().rank;lane_classifiers[rank].consume(normalized.operation.call);
        auto& bindings=blas_streams[rank];const auto& call=normalized.operation.call;
        bool folded=false;
        if(call.op==FAKECUDA_RAS_BLAS_SET_STREAM){auto found=bindings.find(call.handle);folded=found!=bindings.end()&&found->second==call.stream;bindings[call.handle]=call.stream;}
        if(call.op==FAKECUDA_RAS_BLAS_DESTROY)bindings.erase(call.handle);
        if(folded)return;
        const auto id=active().id;auto& meta=current_meta[id];
        const auto sequence=normalized.operation.sequence;normalized.operation.sequence=meta.events;
        local_slots(normalized,id,metas[id].size());
        if(!meta.events)meta.first_sequence=sequence;
        meta.last_sequence=sequence;++meta.events;
        if(raw.flags_v2&FAKECUDA_RAS_PAYLOAD_ELIDED)fail("Maya selected record was elided by native interceptor");
        if(raw.kernel_name_size&&!raw.kernel_name)fail("Maya native kernel name is null");
        const auto bytes=padded(sizeof(NativeRecord)+raw.kernel_name_size);
        if(current_bytes_[id]&&current_bytes_[id]+bytes>capacity)physical_flush(id);
        positions[id].push_back({physical_blocks[id],current_bytes_[id]});
        auto output=reserve(id,bytes,false);std::memset(output.bytes().data(),0,bytes);
        std::memcpy(output.bytes().data(),&normalized,sizeof(normalized));
        if(raw.kernel_name_size)std::memcpy(output.bytes().data()+sizeof(normalized),raw.kernel_name,raw.kernel_name_size);
        current_digest[id].add(std::span<const std::byte>(output.bytes().data(),bytes));output.commit();current_bytes_[id]+=bytes;
        if(call.op==FAKECUDA_RAS_STREAM_CREATE)created_resources[id].push_back({metas[id].size(),Stream,normalized.operation.call.stream});
        if(call.op==FAKECUDA_RAS_EVENT_CREATE)created_resources[id].push_back({metas[id].size(),Event,normalized.operation.call.event});
        if(call.op==FAKECUDA_RAS_COMM_INIT)created_resources[id].push_back({metas[id].size(),Comm,normalized.operation.call.comm});
        ++metrics.raw_records;metrics.raw_bytes+=bytes;
        if(bytes>=capacity)physical_flush(id);
    }
    static int sink(void* opaque,const FakeCudaRasRecordV2* record) noexcept {
        auto& self=*static_cast<Impl*>(opaque);try{self.emit(*record);return 0;}catch(...){if(!self.failure)self.failure=std::current_exception();return -1;}
    }
    static int policy(void* opaque,std::uint32_t) noexcept {
        auto& self=*static_cast<Impl*>(opaque);try{(void)self.active();return self.record()?1:0;}catch(...){if(!self.failure)self.failure=std::current_exception();return -1;}
    }
    void local_slots(NativeRecord& record,Id region,std::uint64_t chunk) {
        auto slot=[&](Id& value,bool membership=false){if(!value)return;auto& map=current_slots[region];const auto key=std::pair{value,membership};auto found=map.find(key);if(found==map.end()){const Id identity=((map.size()+1)<<1)|Id(membership);resource_slots[region].push_back({chunk,identity,value});found=map.emplace(key,identity).first;}value=found->second;};
        auto& call=record.operation.call;for(auto field:{&call.stream,&call.event,&call.handle,&call.comm,&call.src,&call.dst,&call.a,&call.b,&call.c,&record.d,&record.aux})slot(*field);
        Id membership=0;std::memcpy(&membership,call.comm_uid,sizeof(membership));slot(membership,true);std::memcpy(call.comm_uid,&membership,sizeof(membership));
    }
    std::map<Id,std::size_t> current_bytes_;
    std::map<Id,std::size_t> physical_blocks;
};
Capture::Capture(Impl& impl):impl_(&impl){}
Id Capture::identity(Id domain,std::span<const std::byte> coordinate){
    if(!impl_->root_context)fail("Maya dynamic identity requires joint root capture");
    std::vector<Id> words(1+(coordinate.size()+sizeof(Id)-1)/sizeof(Id),0);words[0]=coordinate.size();
    if(!coordinate.empty())std::memcpy(words.data()+1,coordinate.data(),coordinate.size());
    return impl_->root_context->identity(domain,words);
}
void Capture::declare_region(Region region,std::span<const Id> refs){
    if(!impl_->declare)fail("Maya capture does not admit dynamic Code declarations");
    impl_->declare(std::move(region),refs);
}
void Capture::begin_region(Id id){impl_->begin(id);}
void Capture::end_region(){impl_->end();}
bool Capture::recording()const{return impl_->record();}
void Capture::check_native_failure()const{if(impl_->failure)std::rethrow_exception(impl_->failure);}
void Capture::region(Id id,const std::function<void()>& callback){
    begin_region(id);
    try{callback();}
    catch(...){try{end_region();}catch(...){}throw;}
    end_region();
}
void Capture::communicator(std::uint64_t handle,Id membership,std::uint32_t local){impl_->communicator(handle,membership,local);}
rt::GroundResolution Capture::ground(rt::GroundRequest request,std::span<const std::byte> bytes,ras::SchemaId schema) {
    if(!impl_->root_context)fail("Maya Ground requires joint root capture");
    const auto id=impl_->active().id;
    request.inputs.payload=impl_->root_context->copy(bytes,schema);
    const auto result=impl_->root_context->ground(request);
    auto& proof=impl_->ground_evidence[id];append(proof,request.point_id);append(proof,request.query);append(proof,request.inputs.semantic);
    append(proof,result.fact_id);append(proof,result.value.semantic);
    impl_->ground_facts[id].insert(result.fact_id);
    if(!result.reusable)impl_->volatile_ground.insert(id);
    if(result.value.payload){auto writer=impl_->root_context->blocks(std::max<std::size_t>(1,result.value.payload.size()),result.value.payload.layout());writer.append(result.value.payload);impl_->ground_values[id].push_back(writer.finish());}
    impl_->ground_values[id].insert(impl_->ground_values[id].end(),result.value.bundles.begin(),result.value.bundles.end());
    return result;
}
void Capture::code_occurrence(Id region,std::span<const std::byte> packet,std::span<const Id> facts,std::span<const Id> refs) {
    if(impl_->dependencies)impl_->dependencies(region,refs);
    if(!impl_->root_context||!impl_->regions.contains(region)||!impl_->seen.contains(region))fail("Code occurrence requires an executed joint-capture region");
    for(auto fact:facts)if(!impl_->ground_facts[region].contains(fact))fail("Code occurrence references an unconsumed Ground fact");
    if(packet.empty())fail("empty Code occurrence evidence");
    impl_->code_slices[region].push_back(impl_->append_code(1,packet,facts));
}
void Capture::code_dictionary(std::span<const std::byte> packet) {impl_->append_code(2,packet);}


struct Session::Impl {
    Options options;
    ras::Store store;
    rt::Coordinator coordinator;
    Input current_input;
    Input* input=nullptr;
    const Snapshot* parent=nullptr;
    Metrics metrics;
    code::SourceGraph source_graph;
    const std::map<Id,const Region*>& regions=source_graph.regions();
    std::map<ResourceKey,Resource> resources;
    std::map<Id,Evidence> resource_coordinates; // Exact canonical bytes detect deterministic-ID collisions across independent roots.
    std::vector<Resource> entry_resources,replay_resources;
    using CommunicatorIndex=std::map<std::pair<std::uint32_t,Id>,std::vector<Id>>;
    CommunicatorIndex entry_communicators,replay_communicators;
    static void index_communicator(CommunicatorIndex& index,const Resource& resource) {
        if(resource.kind==Comm)index[{resource.rank,resource.extra}].push_back(resource.id);
    }
    std::optional<Capture::Impl> capture;
    std::uint64_t round=0;
    struct Prepared {rt::BlockBundle native;ras::Buffer metadata;Evidence projection;};
    struct Directory {
        ras::ByteView bytes;ReceiptMeta header{};
        std::span<const ChunkMeta> metas;
        std::span<const ResourceSlot> slots;
        std::span<const RecordPosition> positions;
        std::span<const CreatedResource> created;
        std::span<const ResourceSummary> resources;
        std::span<const ResourceCreation> creations;
    };
    struct ProjectionMeta {std::uint64_t chunks=0,slots=0;std::uint32_t rank=0,representative=0;std::uint64_t key_bytes=0;};
    std::map<Id,Prepared> prepared;
    std::map<Id,Evidence> ground_evidence;
    std::map<Id,CodeEvidence> code_evidence;
    std::map<Id,std::vector<rt::BlockBundle>> ground_values;
    std::set<Id> volatile_ground,changed_sources;
    std::optional<rt::BlockBundle> frontend_evidence;
    std::vector<Id> changed_outputs,removed_outputs;
    bool exact_window_reused=false;
    explicit Impl(Options o):options(std::move(o)),coordinator(store,{options.diagnostics}) {
        if(options.chunk_bytes<sizeof(NativeRecord)||!options.identity||options.cost_identity.empty())fail("invalid Maya paper session options");
        if(!options.costs&&!options.native_costs)fail("Maya paper session requires explicit cost model");
        register_pipeline();
    }
    std::size_t capture_capacity()const{return input->capture_block_bytes?input->capture_block_bytes:options.chunk_bytes;}
    static Directory directory(const ras::Buffer& payload) {
        Directory result;result.bytes=payload.bytes();const auto bytes=result.bytes.span();
        if(bytes.size()<sizeof(ReceiptMeta))fail("truncated normalized output descriptor");
        std::memcpy(&result.header,bytes.data(),sizeof(result.header));const auto& h=result.header;
        const auto expected=sizeof(h)+h.chunks*sizeof(ChunkMeta)+h.slots*sizeof(ResourceSlot)+h.positions*sizeof(RecordPosition)+h.created*sizeof(CreatedResource)+h.resources*sizeof(ResourceSummary)+h.creations*sizeof(ResourceCreation);
        if(expected!=bytes.size()||h.positions!=h.events)fail("invalid normalized output descriptor extent");
        const auto* cursor=bytes.data()+sizeof(h);
        result.metas={reinterpret_cast<const ChunkMeta*>(cursor),std::size_t(h.chunks)};cursor+=h.chunks*sizeof(ChunkMeta);
        result.slots={reinterpret_cast<const ResourceSlot*>(cursor),std::size_t(h.slots)};cursor+=h.slots*sizeof(ResourceSlot);
        result.positions={reinterpret_cast<const RecordPosition*>(cursor),std::size_t(h.positions)};cursor+=h.positions*sizeof(RecordPosition);
        result.created={reinterpret_cast<const CreatedResource*>(cursor),std::size_t(h.created)};cursor+=h.created*sizeof(CreatedResource);
        result.resources={reinterpret_cast<const ResourceSummary*>(cursor),std::size_t(h.resources)};cursor+=h.resources*sizeof(ResourceSummary);
        result.creations={reinterpret_cast<const ResourceCreation*>(cursor),std::size_t(h.creations)};
        return result;
    }
    Evidence placement(const Region& r,std::span<const ChunkMeta> metas,std::span<const ResourceSlot> slots,std::uint32_t representative) const {
        const auto key=projection_key(r);
        Evidence value=evidence(ProjectionMeta{metas.size(),slots.size(),r.rank,representative,key.size()});
        append(value,std::as_bytes(metas));append(value,std::as_bytes(slots));append(value,std::span<const std::byte>(key));return value;
    }
    static void projection(const Evidence& bytes,ProjectionMeta& h,std::span<const ChunkMeta>& metas,std::span<const ResourceSlot>& slots) {
        if(bytes.size()<sizeof(h))fail("missing normalized output projection");
        std::memcpy(&h,bytes.data(),sizeof(h));
        if(bytes.size()!=sizeof(h)+h.chunks*sizeof(ChunkMeta)+h.slots*sizeof(ResourceSlot)+h.key_bytes)fail("invalid normalized output projection");
        metas={reinterpret_cast<const ChunkMeta*>(bytes.data()+sizeof(h)),std::size_t(h.chunks)};
        slots={reinterpret_cast<const ResourceSlot*>(bytes.data()+sizeof(h)+h.chunks*sizeof(ChunkMeta)),std::size_t(h.slots)};
    }
    bool same_source(const rt::Partition& p,const Region& r,rt::Context& context){return source_graph.same_source(p,r,context);}
    static Evidence code_state(const Region& r){Evidence result;append(result,r.rank);append(result,r.execution_region?r.execution_region:r.id);return result;}
    bool same_projection(const rt::Receipt& old,const Region& r)const {
        ProjectionMeta h{};std::span<const ChunkMeta> metas;std::span<const ResourceSlot> slots;projection(old.state.semantic,h,metas,slots);
        const auto key=projection_key(r);const auto& bytes=old.state.semantic;
        return key.size()==h.key_bytes&&std::equal(key.begin(),key.end(),bytes.end()-h.key_bytes);
    }
    Evidence projection_key(const Region& r)const {
        Evidence result;append(result,r.rank);append(result,r.execution_region?r.execution_region:r.id);
        for(const auto& projection:input->rank_projections)if(projection.rank==r.rank){append(result,projection.representative);for(const auto&[a,b]:projection.communicator_map){append(result,a);append(result,b);for(const auto& comm:input->communicators)if(comm.id==b){append(result,std::uint64_t(comm.members.size()));for(auto member:comm.members)append(result,member);}}}
        Evidence framed;append(framed,std::uint64_t(result.size()));append(framed,result);return framed;
    }

    void add_region(const Region& r) { source_graph.declare(r); }
    void add_dependencies(Id id,std::span<const Id> refs) {
        const auto before=source_graph.dependencies().size();source_graph.dependencies(id,refs);
        const auto& dependencies=source_graph.dependencies();input->dependencies.insert(input->dependencies.end(),dependencies.begin()+before,dependencies.end());
    }
    void validate(const Input& value) {
        if(value.configuration.empty()||!value.execute||(!value.dynamic_regions&&value.regions.empty()))fail("Maya input requires configuration and an executor");
        if(value.capture_block_bytes&&value.capture_block_bytes<sizeof(NativeRecord))fail("Maya physical block hint cannot hold one native record");
        Evidence source_context=value.configuration;append(source_context,std::uint64_t(value.communicators.size()));
        for(const auto& comm:value.communicators){append(source_context,comm.id);append(source_context,std::uint64_t(comm.members.size()));for(auto member:comm.members)append(source_context,member);}
        source_graph.reset(std::move(source_context),round+1,[this](Id current){
            if(capture&&capture->seen.contains(current)&&!capture->prepared_writers.contains(current))fail("Code inputs changed after retained occurrence admission");
        });
        for(const auto& r:value.regions)add_region(r);
        std::set<Id> memberships;for(const auto& c:value.communicators){if(!c.id||!memberships.insert(c.id).second||c.members.empty())fail("invalid Maya communicator declaration");std::set<std::uint32_t> unique(c.members.begin(),c.members.end());if(unique.size()!=c.members.size())fail("duplicate global rank in Maya communicator");}
        for(auto edge:value.dependencies){if(edge.producer==edge.consumer)fail("invalid Code self dependency");const std::array<Id,1> refs{edge.producer};source_graph.dependencies(edge.consumer,refs);}
        std::set<std::uint32_t> projected;for(const auto& p:value.rank_projections){if(!projected.insert(p.rank).second)fail("duplicate Maya rank projection");for(const auto&[from,to]:p.communicator_map)if(!memberships.contains(from)||!memberships.contains(to))fail("rank projection references absent communicator");}
        if(value.window_complete&&value.window_semantic.empty())fail("complete window evidence cannot be empty");
    }
    void restore_resources() {
        resources.clear();entry_resources.clear();replay_resources.clear();entry_communicators.clear();replay_communicators.clear();
        if(!parent)return;
        const auto view=coordinator.layer(parent->anchor,1);const auto state=view.state();
        if(state.bundles.empty())return;
        for(std::size_t n=0;n<state.bundles.front().count();++n){const auto rows=view.block(state.bundles.front(),n).array<Resource>(meta_schema);metrics.resource_records_read+=rows.size();for(auto r:rows.span()){if(r.kind&0x80000000U){r.kind&=0x7fffffffU;entry_resources.push_back(r);index_communicator(entry_communicators,r);}else resources.emplace(ResourceKey{r.rank,r.kind,r.address},r);}}
    }
    Evidence window_key()const {Evidence key;append(key,input->window_complete);append(key,input->window_semantic);return key;}
    bool reuse_window(rt::Context& context,rt::RootCapture& root) {
        const auto previous=context.previous();
        if(options.mode!=Mode::Refresh||!input->dynamic_regions||!input->regions.empty()||!input->window_complete||!options.grounds.empty()||
           !previous||previous.state().semantic!=window_key())return false;
        // The complete window proof includes setup. Local scope reuse rules
        // only govern scopes inside an invocation that actually executes.
        std::map<Id,Region> old;std::vector<Id> order;
        for(std::size_t n=0;n<previous.size();++n){const auto id=previous.at(n);
            if(code_header(previous.payload(id)).reserved)return false;
            old.emplace(id,code_region(previous.payload(id)));order.push_back(id);
        }
        std::map<std::uint32_t,std::vector<std::uint32_t>> members;
        for(const auto& p:input->rank_projections)members[p.representative].push_back(p.rank);
        std::set<std::uint32_t> actual_ranks;
        for(const auto&[id,r]:old)if(!r.execution_region||r.execution_region==id)actual_ranks.insert(r.rank);
        if(!members.empty()){
            std::set<std::uint32_t> requested;for(const auto&[rank,lanes]:members){(void)lanes;requested.insert(rank);}
            if(requested!=actual_ranks)return false;
        }
        const auto projected_id=[&](const Region& r,std::uint32_t rank){
            Evidence bytes=r.occurrence_identity;
            for(unsigned i=0;i<8;++i)bytes.push_back(static_cast<std::byte>((std::uint64_t(rank)>>(8*i))&255));
            std::vector<Id> words(1+(bytes.size()+sizeof(Id)-1)/sizeof(Id),0);words[0]=bytes.size();
            if(!bytes.empty())std::memcpy(words.data()+1,bytes.data(),bytes.size());
            return context.identity(0x4d41594152454701ULL,words);
        };
        bool identical_projection=true;
        std::set<std::uint32_t> old_lanes,new_lanes;
        for(auto id:order){const auto& r=old.at(id);if(r.active_lanes.empty())old_lanes.insert(r.rank);else for(auto lane:r.active_lanes)old_lanes.insert(static_cast<std::uint32_t>(lane));
            auto lanes=r.active_lanes;if(lanes.empty())lanes.push_back(r.rank);
            for(auto lane:lanes){auto logical=r;logical.rank=static_cast<std::uint32_t>(lane);logical.id=lane==r.rank?r.id:projected_id(r,logical.rank);
                if(const auto output=context.parent_output(1,logical.id);output&&!same_projection(*output,logical))identical_projection=false;}
        }
        for(const auto& p:input->rank_projections)new_lanes.insert(p.rank);
        if(!new_lanes.empty()&&old_lanes!=new_lanes)identical_projection=false;
        std::vector<std::pair<Region,Id>> desired;
        std::map<std::pair<Id,std::uint32_t>,Id> remap;
        for(auto id:order){const auto& r=old.at(id);
            if(members.empty()){desired.emplace_back(r,id);remap[{id,r.rank}]=id;continue;}
            if(r.execution_region&&r.execution_region!=id)continue;
            if(!r.execution_region){desired.emplace_back(r,id);remap[{id,r.rank}]=id;continue;}
            if(r.occurrence_identity.empty())return false;
            for(auto rank:members.at(r.rank)){
                auto alias=r;alias.rank=rank;alias.id=projected_id(r,rank);alias.execution_region=id;
                if(rank==r.rank&&alias.id!=id)return false;
                remap[{id,rank}]=alias.id;desired.emplace_back(std::move(alias),id);
            }
        }
        std::map<Id,std::vector<Id>> incoming;
        const auto prior=previous.state();
        if(prior.payload)for(auto edge:prior.payload.array<rt::Dependency>(code::dependency_schema).span())incoming[edge.consumer].push_back(edge.producer);
        std::vector<rt::Dependency> edges;
        for(const auto&[r,original]:desired)for(auto producer:incoming[original]){
            const auto mapped=remap.find({producer,r.rank});
            if(mapped!=remap.end())edges.push_back({mapped->second,r.id});
            else if(old.contains(producer)&&!old.at(producer).execution_region)edges.push_back({producer,r.id});
            else return false;
        }
        bool unchanged=identical_projection;
        for(const auto&[r,original]:desired){(void)original;
            const auto actual=r.execution_region?r.execution_region:r.id;if(!previous.contains(actual)){unchanged=false;continue;}
            if(r.execution_region&&r.execution_region!=r.id)continue;
            if(const auto output=context.parent_output(1,r.id);output&&!same_projection(*output,r))unchanged=false;
        }
        input->regions.clear();input->dependencies.clear();
        for(const auto&[r,original]:desired){(void)original;input->regions.push_back(r);add_region(r);}
        for(auto edge:edges){const std::array<Id,1> refs{edge.producer};add_dependencies(edge.consumer,refs);}
        if(unchanged){exact_window_reused=true;return true;}
        // Projection-only changes borrow both the final body and its index.
        std::set<Id> granted_code_pools;
        for(const auto&[r,original]:desired){const auto p=previous.partition(original);
            if(const auto previous_output=context.parent_output(1,original)){
                const auto info=directory(previous_output->payload);
                prepared.emplace(r.id,Prepared{previous_output->chunks.storage(),previous_output->payload,placement(r,info.metas,info.slots,old.at(original).rank)});
            }
            if(const auto h=code_header(p.payload);h.evidence_version){if(p.bundles.empty())fail("Code evidence lacks its pool owner");code_evidence.emplace(r.id,CodeEvidence{p.bundles[0],h.evidence_version,code_slices(p.payload)});if(granted_code_pools.insert(h.evidence_version).second)root.bundles.push_back(p.bundles[0]);}
            ++metrics.normalized_output_adoptions;
        }
        return true;
    }
    rt::RootCapture capture_root(rt::Context& context) {
        const auto callback_start=Clock::now();
        struct CaptureTimer {Metrics& metrics;Clock::time_point begin,phase;bool executing=false,finalizing=false;
            void host(){const auto now=Clock::now();metrics.capture_setup_seconds+=std::chrono::duration<double>(now-phase).count();phase=now;executing=true;}
            void finish(){const auto now=Clock::now();if(executing)metrics.host_execution_seconds+=std::chrono::duration<double>(now-phase).count();else metrics.capture_setup_seconds+=std::chrono::duration<double>(now-phase).count();phase=now;executing=false;finalizing=true;}
            ~CaptureTimer(){const auto now=Clock::now();const auto tail=std::chrono::duration<double>(now-phase).count();if(finalizing)metrics.capture_finalize_seconds+=tail;else if(executing)metrics.host_execution_seconds+=tail;else metrics.capture_setup_seconds+=tail;metrics.capture_seconds+=std::chrono::duration<double>(now-begin).count();}
        } timer{metrics,callback_start,callback_start};
        std::optional<ras::BufferReadAudit> audit;if(options.diagnostics)audit.emplace(count_native_borrows,&metrics.retained_normalized_body_reads);
        prepared.clear();ground_evidence.clear();code_evidence.clear();ground_values.clear();volatile_ground.clear();changed_sources.clear();exact_window_reused=false;
        rt::RootCapture root{context.copy(input->configuration,meta_schema),input->configuration};
        frontend_evidence.reset();if(!input->frontend_evidence.empty()){auto writer=context.blocks(input->frontend_evidence.size(),{meta_schema,1,1});auto bytes=writer.reserve(input->frontend_evidence.size());std::memcpy(bytes.bytes().data(),input->frontend_evidence.data(),input->frontend_evidence.size());bytes.commit();frontend_evidence=writer.finish();root.bundles.push_back(*frontend_evidence);}
        const auto previous=context.previous();restore_resources();
        if(reuse_window(context,root)){timer.finish();stage_outputs(context);return root;}
        bool execute=input->dynamic_regions||!previous;
        for(const auto& r:input->regions){const auto actual=r.execution_region?r.execution_region:r.id;if(actual!=r.id)continue;if(!r.reuse_allowed||!previous||!previous.contains(actual)||!same_source(previous.partition(actual),r,context)||code_header(previous.payload(actual)).reserved||!context.parent_output(1,actual))execute=true;}
        capture.emplace(*input,metrics,resources,resource_coordinates,capture_capacity());capture->root_context=&context;capture->regions=regions;
        struct PreviousOutput {rt::BlockBundle native;ras::Buffer metadata;};std::map<Id,PreviousOutput> old;
        auto load=[&](Id id)->bool{
            if(old.contains(id))return true;
            if(!previous||!previous.contains(id))return false;
            const auto output=context.parent_output(1,id);if(!output)return false;
            old.emplace(id,PreviousOutput{output->chunks.storage(),output->payload});return true;
        };
        capture->dependencies=[this](Id id,std::span<const Id> refs){add_dependencies(id,refs);};
        capture->declare=[&](Region r,std::span<const Id> refs){const bool fresh=!regions.contains(r.id);add_region(r);if(fresh)input->regions.push_back(r);add_dependencies(r.id,refs);capture->regions[r.id]=regions.at(r.id);};
        capture->decide=[&](Id id){
            if(capture->seen.contains(id))return;
            const auto& r=*regions.at(id);if(r.execution_region&&r.execution_region!=id)fail("workload executed a projected logical rank instead of its representative");
            bool retain=options.mode==Mode::Refresh&&r.reuse_allowed&&load(id)&&same_source(previous.partition(id),r,context)&&!code_header(previous.payload(id)).reserved;
            if(retain){const auto info=directory(old.at(id).metadata);retain=info.header.entry==capture->carry(r.rank)&&capture->can_retain_resources(info.resources,r.rank);
                if(retain){capture->retained_metas[id]={info.metas.begin(),info.metas.end()};capture->retained_creations[id]={info.creations.begin(),info.creations.end()};for(const auto& row:info.resources)capture->retained_resources[id].emplace(std::pair{row.value.kind,row.value.address},row);}}
            if(!retain){capture->prepared_writers.emplace(id,context.prepare(1,id,evidence(std::uint64_t{1}),capture_capacity(),{native_schema,8,alignof(NativeRecord)}));changed_sources.insert(id);++metrics.selected_regions;}
        };
        if(fakecuda_ras_active())fail("Maya capture requires exclusive native sink ownership");
        if(execute&&fakecuda_ras_install_v2(&*capture,&Capture::Impl::sink,&Capture::Impl::policy))fail("cannot install Maya native V2 sink");
        Capture api(*capture);
        try {
            timer.host();if(execute)input->execute(api);timer.finish();
            if(!capture->stack.empty())fail("Maya workload left region scope open");
            for(const auto&[rank,classifier]:capture->lane_classifiers){(void)rank;classifier.finish();}
            if(capture->failure)std::rethrow_exception(capture->failure);
            if(fakecuda_ras_status())throw ras::Error(fakecuda_ras_last_error());
            fakecuda_ras_reset();
            std::map<Id,Prepared> actual;
            for(const auto& r:input->regions){const auto id=r.execution_region?r.execution_region:r.id;if(actual.contains(id))continue;if(!regions.contains(id))fail("logical rank references an undeclared representative");
                if(auto writer=capture->prepared_writers.find(id);writer!=capture->prepared_writers.end()){
                    if(!capture->seen.contains(id))fail("Maya omitted a selected dynamic occurrence");
                    capture->seal(id);auto native=writer->second.finish();const auto& metas=capture->metas[id];const auto& slots=capture->resource_slots[id];const auto& positions=capture->positions[id];const auto& created=capture->created_resources[id];
                    std::vector<ResourceSummary> summary;for(const auto&[key,row]:capture->resource_summaries[id]){(void)key;summary.push_back(row);}
                    ReceiptMeta h{id,metas.size(),positions.size(),regions.at(id)->rank,regions.at(id)->rank,slots.size(),positions.size(),created.size(),context.new_identity(),summary.size(),capture->resource_creations[id].size(),capture->entry_keys[id],capture->exit_keys[id]};
                    Evidence bytes=evidence(h);append(bytes,std::as_bytes(std::span(metas)));append(bytes,std::as_bytes(std::span(slots)));append(bytes,std::as_bytes(std::span(positions)));append(bytes,std::as_bytes(std::span(created)));append(bytes,std::as_bytes(std::span(summary)));append(bytes,std::as_bytes(std::span(capture->resource_creations[id])));
                    actual.emplace(id,Prepared{native,context.copy(bytes,meta_schema),placement(*regions.at(id),metas,slots,regions.at(id)->rank)});
                }else if(load(id)){
                    const auto info=directory(old.at(id).metadata);std::span<const ChunkMeta> metas=info.metas;std::span<const ResourceSlot> slots=info.slots;
                    if(capture->seen.contains(id))metas=capture->metas.at(id);
                    actual.emplace(id,Prepared{old.at(id).native,old.at(id).metadata,placement(*regions.at(id),metas,slots,regions.at(id)->rank)});++metrics.normalized_output_adoptions;
                }else fail("Maya dynamic occurrence has neither generated nor retained output");
            }
            for(const auto& r:input->regions){const auto id=r.execution_region?r.execution_region:r.id;auto output=actual.at(id);ProjectionMeta ph{};std::span<const ChunkMeta> metas;std::span<const ResourceSlot> slots;projection(output.projection,ph,metas,slots);output.projection=placement(r,metas,slots,regions.at(id)->rank);prepared.emplace(r.id,output);}
            ground_evidence=capture->ground_evidence;volatile_ground=capture->volatile_ground;
            if(capture->code_writer){const auto owner=capture->code_writer->finish();root.bundles.push_back(owner);
                for(auto&[id,slices]:capture->code_slices)code_evidence.emplace(id,CodeEvidence{owner,capture->code_version,std::move(slices)});
            }
            ground_values=capture->ground_values;for(const auto&[id,bundles]:ground_values){(void)id;root.bundles.insert(root.bundles.end(),bundles.begin(),bundles.end());}
            replay_resources.clear();replay_communicators.clear();
            for(const auto&[key,r]:capture->history){(void)key;
                if(r.kind==Stream||r.kind==Event||r.kind==Comm){replay_resources.push_back(r);index_communicator(replay_communicators,r);}
            }
            stage_outputs(context);capture->live=false;capture.reset();return root;
        }catch(...){const auto native_failure=capture->failure;fakecuda_ras_reset();capture->live=false;capture.reset();if(native_failure)std::rethrow_exception(native_failure);throw;}
    }
    void stage_outputs(rt::Context& context) {
        for(const auto&[id,output]:prepared){
            rt::Receipt receipt;receipt.boundary=id;receipt.payload=output.metadata;receipt.state.semantic=output.projection;
            const auto info=directory(receipt.payload);receipt.event_count=info.header.events;receipt.semantic=evidence(info.header.body_version);
            ProjectionMeta h{};std::span<const ChunkMeta> metas;std::span<const ResourceSlot> slots;projection(output.projection,h,metas,slots);
            Evidence packet;append(packet,std::uint64_t(metas.size()));const auto& r=*regions.at(id);
            for(std::size_t i=0;i<metas.size();++i)append(packet,native_projection(r,metas[i],i,slots,h.representative));
            receipt.state.payload=context.copy(packet,meta_schema);append(receipt.semantic,packet);
            context.stage_output(1,id,evidence(std::uint64_t{1}),std::move(receipt),output.native);
        }
    }
    std::vector<rt::Dependency> code_dependencies() const {
        std::set<rt::Dependency> unique;
        const auto owner=[&](Id id){const auto& r=*regions.at(id);return r.execution_region?r.execution_region:id;};
        for(const auto& edge:input->dependencies){rt::Dependency mapped{owner(edge.producer),owner(edge.consumer)};if(mapped.producer!=mapped.consumer)unique.insert(mapped);}
        return {unique.begin(),unique.end()};
    }
    rt::PartitionDelta source_partition(rt::Context& context) {
        if(exact_window_reused)return {};
        const auto previous=context.previous();
        std::map<Id,std::vector<Id>> active;
        for(const auto& r:input->regions)active[r.execution_region?r.execution_region:r.id].push_back(r.rank);
        std::vector<Region> occurrences;
        for(const auto& r:input->regions)if(!r.execution_region||r.execution_region==r.id){auto shared=r;shared.active_lanes=std::move(active[r.id]);std::sort(shared.active_lanes.begin(),shared.active_lanes.end());shared.active_lanes.erase(std::unique(shared.active_lanes.begin(),shared.active_lanes.end()),shared.active_lanes.end());occurrences.push_back(std::move(shared));}
        return code::partition(context,source_graph,occurrences,[&](const Region& r){
            code::PartitionEvidence result;const auto actual=r.execution_region?r.execution_region:r.id;
            if(auto found=ground_evidence.find(actual);found!=ground_evidence.end())append(result.semantic_suffix,found->second);
            result.ground_volatile=volatile_ground.contains(actual);if(result.ground_volatile)append(result.semantic_suffix,round);
            if(const auto found=code_evidence.find(r.id);found!=code_evidence.end())result.code=found->second;
            else if(const auto found=code_evidence.find(actual);found!=code_evidence.end())result.code=found->second;
            else if(previous&&previous.contains(actual)){const auto old=previous.partition(actual);const auto h=code_header(old.payload);if(h.evidence_version){if(old.bundles.empty())fail("Code evidence lacks its pool owner");result.code={old.bundles[0],h.evidence_version,code_slices(old.payload)};}}
            result.state=code_state(r);
            if(!ground_values.contains(actual)&&previous&&previous.contains(actual)){const auto old=previous.partition(actual);const auto first=std::size_t(code_header(old.payload).evidence_version!=0);if(old.bundles.size()>first)result.ground_values.assign(old.bundles.begin()+first,old.bundles.end());}
            if(auto found=ground_values.find(actual);found!=ground_values.end())result.ground_values=found->second;
            return result;
        });
    }
    rt::BoundaryDelta boundaries(const rt::LayerView& upper,rt::Context& context) {
        std::vector<Id> changed;changed.reserve(prepared.size());for(const auto&[id,output]:prepared){(void)output;changed.push_back(id);}
        const auto old=context.boundaries();
        // Membership changes retire only routes of affected shared occurrences.
        for(auto owner:context.delta().changed)old.downstream(owner,[&](const rt::OutputTarget& target){if(target.owner&&!regions.contains(target.boundary))changed.push_back(target.boundary);});
        return code::boundaries(upper,context,changed,[&](Id id){
            if(!regions.contains(id))return code::OutputEvidence{};
            if(const auto output=context.prepared_output(1,id)){
                code::OutputEvidence result;result.exists=output->receipt.event_count!=0;result.semantic=evidence(std::uint64_t{1});
                result.state=output->receipt.semantic;append(result.state,output->receipt.state.semantic);const auto& r=*regions.at(id);result.owner=r.execution_region?r.execution_region:id;return result;
            }
            const auto old=context.boundaries();if(!old.contains(id))return code::OutputEvidence{};
            const auto boundary=old.boundary(id);return code::OutputEvidence{true,boundary.semantic,boundary.state,boundary.owner};
        });
    }
    rt::ApplyResult capture_apply(std::span<const rt::Boundary> selected,rt::Context& context) {
        std::optional<ras::BufferReadAudit> audit;if(options.diagnostics)audit.emplace(count_native_borrows,&metrics.publication_body_reads);
        rt::ApplyResult output;output.removed.assign(context.removed_boundaries().begin(),context.removed_boundaries().end());
        for(const auto& b:selected){auto scope=context.for_boundary(b.id);rt::Receipt receipt;
            if(const auto prepared=scope.prepared_output()){receipt=prepared->receipt;receipt.chunks=scope.adopt_output(prepared->body);}
            else {const auto old=scope.previous_output();if(!old)fail("Code occurrence has no normalized output");receipt=*old;receipt.chunks=scope.adopt_previous_output();}
            ++metrics.publication_descriptor_visits;output.generated.push_back(std::move(receipt));
        }
        return output;
    }
    std::map<ResourceKey,Resource> replay_seeds() const {
        std::map<ResourceKey,Resource> seeds;
        std::set<std::uint32_t> ranks;for(const auto& r:input->regions)ranks.insert(r.rank);
        auto add=[&](const Resource& resource){
            if(ranks.contains(resource.rank)&&resource.id&&(resource.kind==Stream||resource.kind==Event||resource.kind==Comm))
                seeds[{resource.rank,resource.kind,resource.id}]=resource;
        };
        for(const auto& resource:entry_resources) {
            // A removed declaration cannot seed an unused historical handle
            // into the new window. Current native uses are recorded below and
            // still require their declared membership when replay is built.
            if(resource.kind==Comm&&std::none_of(input->communicators.begin(),input->communicators.end(),
                [&](const auto& comm){return comm.id==resource.extra;}))continue;
            add(resource);
        }
        for(const auto& resource:replay_resources)add(resource);
        const auto originals=seeds;
        for(const auto& projection:input->rank_projections)
            for(const auto&[key,resource]:originals){(void)key;if(resource.rank!=projection.representative)continue;auto mapped=resource;mapped.rank=projection.rank;
                if(mapped.kind==Comm){const auto mapping=std::find_if(projection.communicator_map.begin(),projection.communicator_map.end(),[&](const auto& item){return item.first==mapped.extra;});if(mapping==projection.communicator_map.end())continue;
                    const auto membership=std::find_if(input->communicators.begin(),input->communicators.end(),[&](const auto& c){return c.id==mapping->second;});if(membership==input->communicators.end())fail("projected seed has no communicator");const auto position=std::find(membership->members.begin(),membership->members.end(),projection.rank);if(position==membership->members.end())fail("projected seed rank is outside communicator");mapped.extra=mapping->second;mapped.extent=membership->members.size();mapped.local_rank=position-membership->members.begin();}
                add(mapped);
            }
        return seeds;
    }

    static Evidence projection_packet(const ras::Buffer& buffer,std::size_t ordinal) {
        const auto view=buffer.bytes();const auto bytes=view.span();std::size_t cursor=0;
        auto number=[&](){if(bytes.size()-cursor<sizeof(std::uint64_t))fail("invalid normalized projection packet");std::uint64_t n=0;std::memcpy(&n,bytes.data()+cursor,sizeof(n));cursor+=sizeof(n);return n;};
        const auto count=number();if(ordinal>=count)fail("normalized projection ordinal out of range");
        for(std::size_t i=0;i<count;++i){const auto size=number();if(size>bytes.size()-cursor)fail("invalid normalized projection extent");if(i==ordinal)return {bytes.begin()+cursor,bytes.begin()+cursor+size};cursor+=size;}
        fail("normalized projection missing");
    }
    Evidence native_projection(const Region& r,const ChunkMeta& meta,std::size_t chunk,std::span<const ResourceSlot> slots,std::uint32_t representative)const {
        Evidence state=evidence(NativeExtent{meta.events,meta.first_sequence,meta.last_sequence,0,0,representative,meta.relative_base});std::vector<ProjectionBinding> projected;
        for(const auto& value:slots)if(value.chunk==chunk){const auto index=(value.slot>>1)-1;if(index>=projected.size())projected.resize(index+1,ProjectionBinding{independent_rank,0,0,0,0,0});projected[index]={independent_rank,value.slot,value.id,0,0,0};}
        auto extent=native_extent(state);extent.resources=projected.size();std::memcpy(state.data(),&extent,sizeof(extent));append(state,std::as_bytes(std::span(projected)));
        const auto projection=std::find_if(input->rank_projections.begin(),input->rank_projections.end(),[&](const auto& p){return p.rank==r.rank;});
        if(projection!=input->rank_projections.end())for(const auto&[from,to]:projection->communicator_map){const auto membership=std::find_if(input->communicators.begin(),input->communicators.end(),[&](const auto& c){return c.id==to;});if(membership==input->communicators.end())fail("projected communicator absent");const auto rank=std::find(membership->members.begin(),membership->members.end(),r.rank);if(rank==membership->members.end())fail("projected rank outside communicator");
            const auto key=std::pair{representative,from};const std::vector<Id>* ids=nullptr;
            if(const auto found=replay_communicators.find(key);found!=replay_communicators.end())ids=&found->second;
            else if(const auto found=entry_communicators.find(key);found!=entry_communicators.end())ids=&found->second;
            if(ids)for(auto id:*ids)append(state,ProjectionBinding{r.rank,from,to,id,std::uint64_t(rank-membership->members.begin()),membership->members.size()});
            else append(state,ProjectionBinding{r.rank,from,to,0,std::uint64_t(rank-membership->members.begin()),membership->members.size()});
        }
        return state;
    }
    rt::State trace_state(rt::Context& context) {
        auto checkpoint=[&](rt::State state,const std::map<ResourceKey,Resource>& seeds){
            std::vector<Resource> rows;for(const auto&[key,r]:resources){(void)key;rows.push_back(r);}for(const auto&[key,r]:seeds){(void)key;auto saved=r;saved.kind|=0x80000000U;rows.push_back(saved);}
            auto writer=context.blocks(std::max<std::size_t>(sizeof(Resource),rows.size()*sizeof(Resource)),{meta_schema,sizeof(Resource),alignof(Resource)});
            if(!rows.empty()){auto bytes=writer.reserve(rows.size()*sizeof(Resource));std::memcpy(bytes.bytes().data(),rows.data(),rows.size()*sizeof(Resource));bytes.commit();}
            state.bundles.insert(state.bundles.begin(),writer.finish());return state;
        };
        if(context.previous()&&changed_outputs.empty()&&removed_outputs.empty()){
            auto state=context.previous().state();if(!metrics.executed_regions)return state;
            // Real host execution can refresh physical resource bindings while
            // all normalized outputs remain unchanged. Preserve the terminal
            // state directly; only its sparse runtime checkpoint changes.
            if(!state.bundles.empty())state.bundles.erase(state.bundles.begin());
            return checkpoint(std::move(state),replay_seeds());
        }
        std::vector<Id> retired_outputs;
        if(parent){const auto old=coordinator.transform(parent->anchor,0);
            auto retire=[&](Id boundary,std::size_t keep){if(!old.contains(boundary))return;const auto info=directory(old.output(boundary).payload);
                for(std::size_t n=keep;n<info.metas.size();++n){const std::array<Id,2> coordinate{boundary,n};retired_outputs.push_back(context.identity(0x4d4159414f5554ULL,coordinate));}};
            for(auto id:removed_outputs)retire(id,0);
            for(auto id:changed_outputs)retire(id,directory(context.receipt(id).payload).metas.size());
        }
        std::vector<TerminalOutput> outputs;std::uint64_t descriptor_count=0;
        context.receipts([&](const rt::Receipt& receipt){const auto info=directory(receipt.payload);const auto& r=*regions.at(receipt.boundary);ProjectionMeta h{};std::span<const ChunkMeta> metas;std::span<const ResourceSlot> slots;projection(receipt.state.semantic,h,metas,slots);if(metas.size()!=info.metas.size())fail("normalized body and projection disagree");
            std::size_t first_position=0;
            for(std::size_t i=0;i<metas.size();++i){const auto& body=info.metas[i];const auto& placed=metas[i];TerminalOutput out;const std::array<Id,2> coordinate{receipt.boundary,i};out.id=context.identity(0x4d4159414f5554ULL,coordinate);out.semantic=evidence(body.digest);out.body_version=info.header.body_version;out.projection=projection_packet(receipt.state.payload,i);out.count=body.events;out.first=placed.first_sequence;out.ranks={r.rank};
                for(std::size_t n=0;n<body.blocks;++n)out.blocks.push_back(context.receipt_chunk(receipt.boundary,body.first_block+n));
                const auto positions_offset=sizeof(ReceiptMeta)+info.header.chunks*sizeof(ChunkMeta)+info.header.slots*sizeof(ResourceSlot)+first_position*sizeof(RecordPosition);
                out.position_backing=receipt.payload.slice({positions_offset,body.events*sizeof(RecordPosition)});
                out.position_block_base=body.first_block;first_position+=body.events;
                out.creations_complete=true;for(const auto& created:info.created)if(created.chunk==i)out.created_resources.push_back({r.rank,created.kind==Stream?1U:created.kind==Event?2U:3U,project_resource(created.slot,out.projection)});
                outputs.push_back(std::move(out));++descriptor_count;
            }
        });
        std::vector<ReplayMembership> memberships;for(const auto& c:input->communicators){ReplayMembership m;std::memcpy(m.uid.data(),&c.id,sizeof(c.id));m.members=c.members;memberships.push_back(std::move(m));}
        std::vector<ReplayStream> streams;std::vector<ReplayEvent> events;std::vector<ReplayCommunicator> comms;
        const auto seeds=replay_seeds();for(const auto&[key,r]:seeds){(void)key;if(r.kind==Stream)streams.push_back({r.rank,r.id,(r.flags&1)==0});else if(r.kind==Event)events.push_back({r.rank,r.id});else if(r.kind==Comm&&r.extra){const auto membership=std::find_if(input->communicators.begin(),input->communicators.end(),[&](const auto& c){return c.id==r.extra;});if(membership==input->communicators.end())fail("imported communicator missing membership");ReplayCommunicator c;c.rank=r.rank;c.local_rank=r.local_rank;c.id=r.id;c.members=membership->members;std::memcpy(c.uid.data(),&r.extra,sizeof(r.extra));comms.push_back(std::move(c));}}
        TerminalFeedbackInput evaluation;evaluation.outputs=outputs;evaluation.removed=retired_outputs;evaluation.replay_identity=evidence(std::uint64_t{0x4d41594152504c03ULL});evaluation.seeds.memberships=memberships;evaluation.seeds.streams=streams;evaluation.seeds.events=events;evaluation.seeds.communicators=comms;evaluation.seeds.block_bytes=options.chunk_bytes;
        evaluation.project=[](const TerminalOutput& out,std::uint32_t rank,const NativeRecord& raw){auto result=raw;result.operation.rank=rank;result.operation.boundary=0;const auto extent=native_extent(out.projection);result.operation.sequence+=extent.first-extent.relative_base;result.operation.call=project_call(result.operation.call,rank,out.projection);result.d=project_resource(result.d,out.projection);result.aux=project_resource(result.aux,out.projection);return result;};
        auto feedback=terminal_feedback(context,evaluation,options,metrics);metrics.feedback_body_reads+=feedback.feedback_body_reads;metrics.trace_partitions=0;metrics.trace_partitions_read=descriptor_count;
        return checkpoint(std::move(feedback.state),seeds);
    }
    void register_pipeline() {
        rt::Pipeline pipeline;pipeline.identity=options.identity;pipeline.grounds=options.grounds;
        rt::LayerRegistration code;code.id=1;code.identity=1;code.payload_schema=code::occurrence_schema;
        code.capture=[this](const rt::RootInput&,rt::Context& c){return capture_root(c);};code.partition=[this](const rt::PartitionRequest&,rt::Context& c){return source_partition(c);};
        code.data_dep=[this](const rt::LayerView&,const rt::LayerDelta&,rt::Context& context){return code::dependency_delta(context,code_dependencies());};
        code.build_state=[this](const rt::LayerView&,rt::Context& c){std::vector<rt::BlockBundle> provenance;if(frontend_evidence)provenance.push_back(*frontend_evidence);return code::state(c,code_dependencies(),window_key(),provenance);};pipeline.layers.push_back(std::move(code));
        rt::LayerRegistration trace;trace.id=2;trace.identity=2;trace.payload_schema=native_schema;
        trace.partition=[this](const rt::PartitionRequest& request,rt::Context&){changed_outputs=request.changed_receipts;removed_outputs=request.removed_receipts;return rt::PartitionDelta{};};
        trace.data_dep=[](const auto&,const auto&,auto&){return rt::DependencyDelta{};};trace.build_state=[this](const rt::LayerView&,rt::Context& c){return trace_state(c);};pipeline.layers.push_back(std::move(trace));
        rt::TransformRegistration transform;transform.id=1;transform.identity=1;transform.upper=1;transform.lower=2;transform.boundary=[this](const rt::LayerView& view,rt::Context& c){return boundaries(view,c);};transform.apply=[this](std::span<const rt::Boundary> selected,rt::Context& c){return capture_apply(selected,c);};pipeline.transforms.push_back(std::move(transform));coordinator.register_pipeline(std::move(pipeline));
    }
    Snapshot run(const Snapshot* prior,const Input& value) {
        current_input=value;input=&current_input;validate(current_input);parent=prior;metrics={};metrics.diagnostics_enabled=options.diagnostics;changed_outputs.clear();removed_outputs.clear();const auto before=store.stats();const auto start=Clock::now();
        try {Snapshot result;if(prior&&options.mode==Mode::Refresh){auto candidate=coordinator.refresh(prior->anchor,{++round,{}});result.anchor=coordinator.commit(candidate);}else result.anchor=coordinator.init({++round,{}});
            result.feedback=decode_terminal_feedback(coordinator.layer(result.anchor,1).state().payload);result.feedback.cost_basis="native-normalized-terminal-v2";metrics.total_seconds=seconds(start);metrics.core=coordinator.last_counters();metrics.memory=store.stats();metrics.published_versions=metrics.memory.publish_count-before.publish_count;result.metrics=metrics;input=nullptr;parent=nullptr;resources.clear();entry_resources.clear();replay_resources.clear();entry_communicators.clear();replay_communicators.clear();return result;
        }catch(...){input=nullptr;parent=nullptr;resources.clear();entry_resources.clear();replay_resources.clear();entry_communicators.clear();replay_communicators.clear();throw;}
    }
};
Session::Session(Options options):impl_(std::make_unique<Impl>(std::move(options))){}
Session::~Session()=default;
Snapshot Session::init(const Input& input){return impl_->run(nullptr,input);}
Snapshot Session::refresh(const Snapshot& parent,const Input& input){return impl_->run(&parent,input);}
Snapshot Session::commit(const Snapshot& snapshot)const{(void)impl_->coordinator.layer(snapshot.anchor,0);return snapshot;}
rt::LayerView Session::layer(const Snapshot& snapshot,std::size_t index)const{return impl_->coordinator.layer(snapshot.anchor,index);}
rt::BoundaryView Session::trace_outputs(const Snapshot& snapshot)const{return impl_->coordinator.transform(snapshot.anchor,0);}
const ras::Store& Session::store()const noexcept{return impl_->store;}
Inspection Session::inspect(const Snapshot& snapshot)const {
    Inspection result;Digest semantic,dependencies;const auto outputs=impl_->coordinator.transform(snapshot.anchor,0);const auto code=layer(snapshot,0);
    struct Ref {rt::Receipt output;ChunkMeta body,placed;Evidence state;std::uint32_t rank=0;};std::vector<Ref> ordered;
    // Inspection is an explicit consumer. It is never part of publication or
    // retention and therefore deliberately reads the complete logical trace.
    outputs.outputs([&](const rt::Receipt& output){const auto info=Impl::directory(output.payload);Impl::ProjectionMeta h{};std::span<const ChunkMeta> metas;std::span<const ResourceSlot> slots;Impl::projection(output.state.semantic,h,metas,slots);auto region=code_region(code.payload(outputs.boundary(output.boundary).owner));region.rank=h.rank;
        for(std::size_t i=0;i<metas.size();++i){Evidence state=Impl::projection_packet(output.state.payload,i);ordered.push_back({output,info.metas[i],metas[i],std::move(state),region.rank});}
    });
    std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return std::tie(a.rank,a.placed.first_sequence)<std::tie(b.rank,b.placed.first_sequence);});
    std::set<std::pair<Id,std::size_t>> physical;
    for(const auto& ref:ordered)for(std::size_t block=0;block<ref.body.blocks;++block){const auto payload=outputs.output_chunk(ref.output.boundary,ref.body.first_block+block);const auto version=Impl::directory(ref.output.payload).header.body_version;const bool fresh=physical.emplace(version,ref.body.first_block+block).second;if(fresh)result.trace_bytes+=payload.size();for_each_native(payload,[&](const NativeRecord& original,std::string_view name){auto row=original;row.operation.sequence+=ref.placed.first_sequence-ref.placed.relative_base;row.operation.call=project_call(row.operation.call,ref.rank,ref.state);row.d=project_resource(row.d,ref.state);row.aux=project_resource(row.aux,ref.state);semantic.add(ref.rank);semantic.add(row);semantic.add(std::as_bytes(std::span(name.data(),name.size())));dependencies.add(ref.rank);dependencies.add(row.operation.sequence);const auto& c=row.operation.call;dependencies.add(c.op);dependencies.add(c.stream);dependencies.add(c.event);dependencies.add(c.comm);dependencies.add(c.rank);dependencies.add(c.peer);dependencies.add(c.nranks);dependencies.add(c.comm_uid);++result.logical_events;if(fresh)++result.physical_records;});}
    result.normalized_digest=semantic.hex();result.dependency_digest=dependencies.hex();return result;
}
} // namespace flexeva::maya::paper
