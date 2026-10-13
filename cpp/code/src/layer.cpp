#include <flexeva/code/layer.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>

namespace flexeva::code {
namespace {
[[noreturn]] void fail(const char* message) { throw ras::Error(message); }
template<class T> Evidence evidence(const T& value) {
    auto bytes=std::as_bytes(std::span<const T>(&value,1));return {bytes.begin(),bytes.end()};
}
void append(Evidence& destination,std::span<const std::byte> value) { destination.insert(destination.end(),value.begin(),value.end()); }
void append(Evidence& destination,const Evidence& value) { const std::uint64_t length=value.size();append(destination,std::as_bytes(std::span(&length,1)));append(destination,std::span<const std::byte>(value)); }
template<class T> void append(Evidence& destination,const T& value) { append(destination,std::as_bytes(std::span<const T>(&value,1))); }
template<class T> ras::Buffer write(rt::Context& context,std::span<const T> rows,ras::SchemaId schema=dependency_schema) {
    auto writer=context.buffer(rows.size_bytes(),{schema,sizeof(T),alignof(T)});
    if(!rows.empty()) { auto output=writer.reserve(rows.size_bytes());std::memcpy(output.bytes().data(),rows.data(),rows.size_bytes());output.commit(); }
    return writer.finish();
}
template<class T> ras::Buffer write(rt::Context& context,const T& value,ras::SchemaId schema=dependency_schema) { return write<T>(context,{&value,1},schema); }
// The caller supplies the current root manifest. Keep its longest unchanged
// subsequence so insertion/removal does not invalidate the entire old order.
std::vector<rt::OrderSplice> manifest_order_delta(const rt::LayerView& previous,const std::vector<Id>& desired) {
    if(!previous)return desired.empty()?std::vector<rt::OrderSplice>{}:std::vector<rt::OrderSplice>{{0,0,desired}};
    struct Entry{std::size_t desired,old,parent;};std::vector<Entry> entries;std::vector<std::size_t> tails;
    constexpr auto none=std::numeric_limits<std::size_t>::max();
    for(std::size_t i=0;i<desired.size();++i)if(previous.contains(desired[i])){const auto old=previous.rank(desired[i]);auto at=std::lower_bound(tails.begin(),tails.end(),old,[&](auto index,auto value){return entries[index].old<value;});const auto position=std::size_t(at-tails.begin());entries.push_back({i,old,position?tails[position-1]:none});if(at==tails.end())tails.push_back(entries.size()-1);else *at=entries.size()-1;}
    std::vector<Entry> anchors;if(!tails.empty())for(auto at=tails.back();at!=none;at=entries[at].parent)anchors.push_back(entries[at]);std::reverse(anchors.begin(),anchors.end());anchors.push_back({desired.size(),previous.size(),none});
    std::vector<rt::OrderSplice> result;std::size_t first_old=0,first_new=0;
    for(const auto& anchor:anchors){if(first_old!=anchor.old||first_new!=anchor.desired)result.push_back({first_old,anchor.old-first_old,{desired.begin()+first_new,desired.begin()+anchor.desired}});first_old=anchor.old+1;first_new=anchor.desired+1;}
    return result;
}
std::vector<Id> touched(const rt::LayerDelta& delta) {
    std::set<Id> ids(delta.added.begin(),delta.added.end());
    ids.insert(delta.changed.begin(),delta.changed.end());
    ids.insert(delta.state_changed.begin(),delta.state_changed.end());ids.insert(delta.invalidated.begin(),delta.invalidated.end());
    return {ids.begin(),ids.end()};
}
}
OccurrenceHeader occurrence_header(const ras::Buffer& buffer) {
    auto bytes=buffer.bytes();if(bytes.size()<sizeof(OccurrenceHeader))fail("invalid Code descriptor");
    OccurrenceHeader value{};std::memcpy(&value,bytes.span().data(),sizeof(value));
    auto remaining=bytes.size()-sizeof(value);
    const auto consume=[&](std::uint64_t count,std::size_t element){if(count>remaining/element)fail("invalid Code semantic extent");remaining-=count*element;};
    consume(value.semantic_bytes,1);consume(value.site_bytes,1);consume(value.occurrence_bytes,1);consume(value.lane_count,sizeof(Id));consume(value.evidence_slices,sizeof(rt::OutputSlice));
    if(remaining)fail("invalid Code semantic extent");
    return value;
}
Occurrence occurrence(const ras::Buffer& buffer) {
    const auto header=occurrence_header(buffer);const auto bytes=buffer.bytes();
    Occurrence r;r.id=header.id;r.revision=header.revision;r.rank=header.rank;r.site_id=header.site;r.execution_region=header.representative;
    r.produces_trace=header.produces;r.semantic_complete=header.complete;r.reuse_allowed=header.reusable;
    auto at=bytes.span().begin()+sizeof(header);r.semantic={at,at+header.semantic_bytes};at+=header.semantic_bytes;
    r.site_identity={at,at+header.site_bytes};at+=header.site_bytes;r.occurrence_identity={at,at+header.occurrence_bytes};at+=header.occurrence_bytes;
    r.active_lanes.resize(header.lane_count);if(header.lane_count)std::memcpy(r.active_lanes.data(),&*at,header.lane_count*sizeof(Id));return r;
}
std::vector<rt::OutputSlice> evidence_slices(const ras::Buffer& buffer) {
    const auto header=occurrence_header(buffer);const auto bytes=buffer.bytes();
    std::vector<rt::OutputSlice> result(header.evidence_slices);
    if(!result.empty())std::memcpy(result.data(),bytes.span().data()+sizeof(header)+header.semantic_bytes+header.site_bytes+header.occurrence_bytes+header.lane_count*sizeof(Id),result.size()*sizeof(rt::OutputSlice));
    return result;
}
ras::Buffer write_code(rt::Context& context,const Occurrence& r,bool ground_volatile,
                       Id evidence_version,std::span<const rt::OutputSlice> slices) {
    OccurrenceHeader header{r.id,r.revision,r.rank,std::uint32_t(ground_volatile),r.site_id,r.execution_region,r.semantic.size(),r.site_identity.size(),r.occurrence_identity.size(),evidence_version,slices.size(),std::uint32_t(r.produces_trace),std::uint32_t(r.semantic_complete),std::uint32_t(r.reuse_allowed),0,r.active_lanes.size()};
    Evidence bytes=evidence(header);append(bytes,std::span<const std::byte>(r.semantic));append(bytes,std::span<const std::byte>(r.site_identity));append(bytes,std::span<const std::byte>(r.occurrence_identity));append(bytes,std::as_bytes(std::span(r.active_lanes)));append(bytes,std::as_bytes(slices));return context.copy(bytes,occurrence_schema);
}

EvidenceWriter::EvidenceWriter(rt::Context& context,std::size_t bytes)
    :writer_(context.blocks(bytes,{evidence_schema,1,1})),version_(context.new_identity()){}
rt::OutputSlice EvidenceWriter::append(std::uint64_t kind,std::span<const std::byte> packet,std::span<const Id> facts) {
    if(packet.empty())fail("empty Code evidence packet");
    const EvidenceRecord header{kind,packet.size(),facts.size()};rt::OutputSlice location;
    auto bytes=writer_.reserve(sizeof(header)+facts.size_bytes()+packet.size(),location);auto* out=bytes.bytes().data();
    std::memcpy(out,&header,sizeof(header));out+=sizeof(header);
    if(!facts.empty()){std::memcpy(out,facts.data(),facts.size_bytes());out+=facts.size_bytes();}
    std::memcpy(out,packet.data(),packet.size());bytes.commit();return location;
}
rt::BlockBundle EvidenceWriter::finish(){return writer_.finish();}
Id EvidenceWriter::version()const noexcept{return version_;}
struct SourceGraph::Impl {
    std::map<Id,Occurrence> declarations;
    std::map<Id,const Occurrence*> regions;
    std::map<std::uint32_t,Evidence> rank_structure;
    std::set<rt::Dependency> source_dependencies;
    std::map<Id,std::set<Id>> source_producers,source_consumers;
    std::map<std::uint32_t,std::set<Id>> rank_structure_users;
    std::map<Id,Id> local_certificates,source_certificates;
    struct RankCertificate {Id id=0;std::size_t bytes=0;};
    std::map<std::uint32_t,RankCertificate> rank_certificates;
    Evidence context_evidence;
    std::uint64_t round=0;
    std::function<void(Id)> before_invalidate;
    std::vector<rt::Dependency> dependency_list;
    Id rank_certificate(std::uint32_t rank,rt::Context& context) {
        auto& cached=rank_certificates[rank];const auto& structure=rank_structure.at(rank);
        while(cached.bytes<structure.size()){
            Id id=0;bool produces=false;std::memcpy(&id,structure.data()+cached.bytes,sizeof(id));cached.bytes+=sizeof(id);
            std::memcpy(&produces,structure.data()+cached.bytes,sizeof(produces));cached.bytes+=sizeof(produces);
            const std::array<Id,3> coordinate{cached.id,id,Id(produces)};
            cached.id=context.identity(0x4d4159415352524bULL,coordinate);
        }
        return cached.id;
    }
    Evidence local_source_semantic(const Occurrence& region,rt::Context& context) {
        Evidence e;append(e,region.semantic);append(e,region.revision);append(e,region.produces_trace);append(e,region.site_id);append(e,region.site_identity);append(e,region.occurrence_identity);append(e,region.execution_region?region.execution_region:region.id);append(e,region.reuse_allowed);if(!region.reuse_allowed)append(e,round);
        if(region.reuse_allowed&&!region.site_id)append(e,rank_certificate(region.rank,context));
        if(region.semantic_complete)return e;
        append(e,region.rank);append(e,context_evidence);
        // Reusable incomplete source evidence keeps a conservative rank-order
        // dependency. Volatile scopes carry this evaluation's round and always
        // Generate; their consumers inherit that round through the source DAG.
        if(region.reuse_allowed)append(e,rank_certificate(region.rank,context));
        return e;
    }
    const std::set<Id>& source_inputs(Id id) const {
        static const std::set<Id> empty;
        const auto found=source_producers.find(id);return found==source_producers.end()?empty:found->second;
    }
    void invalidate_sources(std::span<const Id> ids,bool local=false) {
        if(local)for(auto id:ids)local_certificates.erase(id);
        if(source_certificates.empty()&&!before_invalidate)return;
        std::set<Id> seen(ids.begin(),ids.end());std::vector<Id> pending(seen.begin(),seen.end());
        for(std::size_t next=0;next<pending.size();++next){const auto current=pending[next];
            if(before_invalidate)before_invalidate(current);
            source_certificates.erase(current);
            if(const auto found=source_consumers.find(current);found!=source_consumers.end())
                for(auto consumer:found->second)if(seen.insert(consumer).second)pending.push_back(consumer);}
    }
    void invalidate_source(Id id) {const std::array<Id,1> ids{id};invalidate_sources(ids);}
    Id local_certificate(Id id,rt::Context& context) {
        if(const auto found=local_certificates.find(id);found!=local_certificates.end())return found->second;
        const auto bytes=local_source_semantic(*regions.at(id),context);
        std::vector<Id> coordinate(1+bytes.size()/sizeof(Id)+bool(bytes.size()%sizeof(Id)),0);
        coordinate[0]=bytes.size();if(!bytes.empty())std::memcpy(coordinate.data()+1,bytes.data(),bytes.size());
        // RAS interns the complete coordinate, including byte length and tail;
        // the returned identity is an exact stored certificate, never a hash.
        const auto result=context.identity(0x4d41594153524c43ULL,coordinate);local_certificates.emplace(id,result);return result;
    }
    Id acyclic_certificate(Id id,rt::Context& context) {
        const auto& producers=source_inputs(id);std::vector<Id> coordinate{local_certificate(id,context),producers.size()};
        for(auto producer:producers){coordinate.push_back(producer);coordinate.push_back(source_certificates.at(producer));}
        const auto result=context.identity(0x4d41594153524447ULL,coordinate);source_certificates[id]=result;return result;
    }
    Id cyclic_certificates(Id root,rt::Context& context) {
        // Iterative Kosaraju over the affected reachable subgraph. Cycles share
        // one exact component certificate instead of expanding every closure.
        struct Frame {Id id;std::set<Id>::const_iterator next,end;};
        std::set<Id> visited{root};std::vector<Id> finish;
        const auto frame=[&](Id id){const auto& edges=source_inputs(id);return Frame{id,edges.begin(),edges.end()};};
        std::vector<Frame> stack{frame(root)};
        while(!stack.empty()){auto& top=stack.back();if(top.next==top.end){finish.push_back(top.id);stack.pop_back();continue;}
            const auto producer=*top.next++;if(visited.insert(producer).second)stack.push_back(frame(producer));}
        std::map<Id,std::size_t> component;std::vector<std::vector<Id>> members;
        for(auto next=finish.rbegin();next!=finish.rend();++next){if(component.contains(*next))continue;
            const auto group=members.size();members.emplace_back();std::vector<Id> pending{*next};component.emplace(*next,group);
            for(std::size_t n=0;n<pending.size();++n){const auto id=pending[n];members.back().push_back(id);
                if(const auto found=source_consumers.find(id);found!=source_consumers.end())for(auto consumer:found->second)
                    if(visited.contains(consumer)&&component.emplace(consumer,group).second)pending.push_back(consumer);}
            std::sort(members.back().begin(),members.back().end());}
        for(std::size_t group=members.size();group-- >0;){const auto& ids=members[group];
            if(ids.size()==1){acyclic_certificate(ids.front(),context);continue;}
            std::vector<Id> coordinate{ids.size()};
            for(auto id:ids){coordinate.push_back(id);coordinate.push_back(local_certificate(id,context));}
            for(auto id:ids){const auto& edges=source_inputs(id);coordinate.push_back(id);coordinate.push_back(edges.size());
                for(auto producer:edges){coordinate.push_back(producer);coordinate.push_back(component.at(producer)==group?0:source_certificates.at(producer));}}
            const auto shared=context.identity(0x4d41594153524359ULL,coordinate);
            for(auto id:ids){const std::array<Id,2> node{id,shared};source_certificates[id]=context.identity(0x4d4159415352434eULL,node);}
        }
        return source_certificates.at(root);
    }
    Id source_certificate(Id root,rt::Context& context) {
        if(const auto found=source_certificates.find(root);found!=source_certificates.end())return found->second;
        struct Frame {Id id;std::set<Id>::const_iterator next,end;};
        const auto frame=[&](Id id){const auto& edges=source_inputs(id);return Frame{id,edges.begin(),edges.end()};};
        std::set<Id> active{root};std::vector<Frame> stack{frame(root)};
        while(!stack.empty()){auto& top=stack.back();if(top.next==top.end){acyclic_certificate(top.id,context);active.erase(top.id);stack.pop_back();continue;}
            const auto producer=*top.next;
            if(source_certificates.contains(producer)){++top.next;continue;}
            if(!active.insert(producer).second)return cyclic_certificates(root,context);
            stack.push_back(frame(producer));}
        return source_certificates.at(root);
    }
    Evidence source_semantic(const Occurrence& region,rt::Context& context) {
        Evidence result=evidence(std::uint64_t{0x4d41594143455201ULL});append(result,source_certificate(region.id,context));return result;
    }
    Evidence source_key(const Occurrence& r,rt::Context& context){Evidence key;append(key,source_semantic(r,context));return key;}
    bool same_source(const rt::Partition& p,const Occurrence& r,rt::Context& context){const auto key=source_key(r,context);return p.semantic.size()>=key.size()&&std::equal(key.begin(),key.end(),p.semantic.begin());}
    void add_region(const Occurrence& r) {
        if(!r.id||r.semantic.empty())fail("invalid Code occurrence declaration");
        auto [found,inserted]=declarations.emplace(r.id,r);
        if(!inserted){const auto& old=found->second;if(old.rank!=r.rank||old.revision!=r.revision||old.semantic!=r.semantic||old.site_id!=r.site_id||old.site_identity!=r.site_identity||old.occurrence_identity!=r.occurrence_identity||old.execution_region!=r.execution_region||old.produces_trace!=r.produces_trace||old.semantic_complete!=r.semantic_complete||old.reuse_allowed!=r.reuse_allowed||old.active_lanes!=r.active_lanes)fail("Code declaration changed after entry admission");return;}
        regions.emplace(r.id,&found->second);append(rank_structure[r.rank],r.id);append(rank_structure[r.rank],r.produces_trace);
        if(!local_certificates.empty()||!source_certificates.empty()){
            const auto& users=rank_structure_users[r.rank];const std::vector<Id> ids(users.begin(),users.end());invalidate_sources(ids,true);}
        if(r.reuse_allowed&&(!r.site_id||!r.semantic_complete))rank_structure_users[r.rank].insert(r.id);
    }
    void add_dependencies(Id id,std::span<const Id> refs) {
        if(!regions.contains(id))fail("Code dependency references an undeclared consumer");
        bool changed=false;
        for(auto producer:refs){if(!regions.contains(producer))fail("Code dependency references an undeclared occurrence");if(producer==id)continue;rt::Dependency edge{producer,id};
            if(source_dependencies.insert(edge).second){dependency_list.push_back(edge);source_producers[id].insert(producer);source_consumers[producer].insert(id);changed=true;}}
        if(changed)invalidate_source(id);
    }
};
SourceGraph::SourceGraph():impl_(std::make_unique<Impl>()){}
SourceGraph::~SourceGraph()=default;
void SourceGraph::reset(Evidence context,std::uint64_t round,std::function<void(Id)> invalidate) {
    // Preserve the map object's address for capture adapters holding its view.
    *impl_=Impl{};impl_->context_evidence=std::move(context);impl_->round=round;impl_->before_invalidate=std::move(invalidate);
}
void SourceGraph::declare(const Occurrence& occurrence){impl_->add_region(occurrence);}
void SourceGraph::dependencies(Id id,std::span<const Id> refs){impl_->add_dependencies(id,refs);}
const std::map<Id,const Occurrence*>& SourceGraph::regions()const{return impl_->regions;}
const std::vector<rt::Dependency>& SourceGraph::dependencies()const{return impl_->dependency_list;}
Evidence SourceGraph::key(const Occurrence& value,rt::Context& context){return impl_->source_key(value,context);}
bool SourceGraph::same_source(const rt::Partition& p,const Occurrence& value,rt::Context& context){return impl_->same_source(p,value,context);}
rt::PartitionDelta partition(rt::Context& context,SourceGraph& sources,std::span<const Occurrence> occurrences,
        const std::function<PartitionEvidence(const Occurrence&)>& describe) {
    rt::PartitionDelta delta;const auto previous=context.previous();std::set<Id> current;std::vector<Id> desired;
    for(const auto& occurrence:occurrences){
        current.insert(occurrence.id);desired.push_back(occurrence.id);
        auto metadata=describe(occurrence);auto semantic=sources.key(occurrence,context);
        auto lanes=occurrence.active_lanes;if(lanes.empty())lanes={occurrence.rank};
        if(!std::is_sorted(lanes.begin(),lanes.end())||std::adjacent_find(lanes.begin(),lanes.end())!=lanes.end())fail("Code occurrence has noncanonical active lanes");
        append(semantic,std::span<const std::byte>(metadata.semantic_suffix));const auto layout=evidence(metadata.code.version);
        if(previous&&previous.contains(occurrence.id)){const auto old=previous.partition(occurrence.id);
            if(old.semantic==semantic&&old.state==metadata.state&&old.layout==layout&&old.lanes==lanes)continue;}
        rt::Partition p;p.id=occurrence.id;p.lanes=std::move(lanes);p.semantic=std::move(semantic);p.state=std::move(metadata.state);p.layout=layout;
        p.payload=write_code(context,occurrence,metadata.ground_volatile,metadata.code.version,metadata.code.slices);
        if(metadata.code.owner)p.bundles.push_back(metadata.code.owner);
        p.bundles.insert(p.bundles.end(),metadata.ground_values.begin(),metadata.ground_values.end());delta.upserts.push_back(std::move(p));
    }
    if(previous)for(std::size_t n=0;n<previous.size();++n)if(!current.contains(previous.at(n)))delta.removed.push_back(previous.at(n));
    delta.order=manifest_order_delta(previous,desired);return delta;
}
rt::DependencyDelta dependency_delta(rt::Context& context,std::span<const rt::Dependency> dependencies) {
    std::set<rt::Dependency> old,current(dependencies.begin(),dependencies.end());
    if(context.previous()){const auto state=context.previous().state();if(state.payload)for(auto edge:state.payload.array<rt::Dependency>(dependency_schema).span())old.insert(edge);}
    rt::DependencyDelta result;std::set_difference(current.begin(),current.end(),old.begin(),old.end(),std::back_inserter(result.upserts));
    std::set_difference(old.begin(),old.end(),current.begin(),current.end(),std::back_inserter(result.removed));return result;
}
rt::State state(rt::Context& context,std::span<const rt::Dependency> dependencies,Evidence semantic,std::span<const rt::BlockBundle> provenance) {
    rt::State result{write<rt::Dependency>(context,dependencies),std::move(semantic)};result.bundles.assign(provenance.begin(),provenance.end());return result;
}
rt::BoundaryDelta boundaries(const rt::LayerView& upper,rt::Context& context,std::span<const Id> output_changes,
        const std::function<OutputEvidence(Id)>& describe) {
    rt::BoundaryDelta delta;const auto previous=context.boundaries();auto changed=touched(context.delta());changed.insert(changed.end(),output_changes.begin(),output_changes.end());
    std::sort(changed.begin(),changed.end());changed.erase(std::unique(changed.begin(),changed.end()),changed.end());
    for(auto id:changed){
        auto output=describe(id);if(!output.exists){if(previous.contains(id))delta.removed.push_back(id);continue;}
        const auto owner=output.owner?output.owner:id;if(!upper.contains(owner))fail("Code output owner is absent");
        rt::Boundary b;b.id=id;b.owner=owner;b.inputs={owner};b.semantic=std::move(output.semantic);b.state=std::move(output.state);b.payload=write(context,id);delta.upserts.push_back(std::move(b));
    }
    for(auto id:context.delta().removed)previous.downstream(id,[&](const rt::OutputTarget& target){if(target.owner)delta.removed.push_back(target.boundary);});
    std::sort(delta.removed.begin(),delta.removed.end());delta.removed.erase(std::unique(delta.removed.begin(),delta.removed.end()),delta.removed.end());return delta;
}
} // namespace flexeva::code
