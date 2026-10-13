#include "runtime_internal.hpp"
#include "ground_internal.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <tuple>
#include <chrono>

namespace flexeva::core::runtime {
bool LayerDelta::semantic_empty() const noexcept {
    return added.empty() && changed.empty() && removed.empty() && state_changed.empty() && invalidated.empty() &&
           !order_changed && !dependencies_changed && !layer_state_changed;
}
namespace detail {
namespace {
[[noreturn]] void fail(const char* message) { throw ras::Error(message); }
void require_state_evidence(const State& state) {
    if((state.payload||!state.bundles.empty())&&state.semantic.empty())fail("State payload requires semantic evidence");
}
template<class T> void unique(std::vector<T>& values) {
    std::sort(values.begin(), values.end()); values.erase(std::unique(values.begin(), values.end()), values.end());
}
void canonical(const std::vector<Id>& ids, bool nonempty, const char* message, bool labels=false) {
    if ((nonempty && ids.empty()) || (!labels && std::any_of(ids.begin(), ids.end(), [](Id x) { return !x; })) ||
        !std::is_sorted(ids.begin(), ids.end()) || std::adjacent_find(ids.begin(), ids.end()) != ids.end()) fail(message);
}
std::vector<Id> keys(const ras::ReadView& read, ras::DirectoryId dir, Id first) {
    std::vector<Id> out;
    read.directory_range(dir, {first, 0}, {first, maximum}, [&](ras::DirectoryKey k, const ras::DirectoryRecord&) { out.push_back(k.second); });
    return out;
}
void erase_if(ras::Transaction& tx, ras::DirectoryId d, ras::DirectoryKey k) {
    if (tx.read().directory_find(d, k)) tx.directory_erase(d, k);
}
void pod(Evidence& out, Id value) {
    auto begin=out.size(); out.resize(begin+sizeof(value)); std::memcpy(out.data()+begin,&value,sizeof(value));
}
struct ScopeEnd { std::shared_ptr<Access> access; ~ScopeEnd(){access->scope->live=false;access->scope->chunk_intern.reset();} };
template<class Callback> decltype(auto) invoke_callback(Counters& counters,Callback&& callback){
    struct Timer {
        Counters& counters;
        std::chrono::steady_clock::time_point start;
        ~Timer(){if(counters.diagnostics_enabled)counters.evaluator_callback_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
    } timer{counters,counters.diagnostics_enabled?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{}};
    return callback();
}
}

std::vector<Id> projected_inputs(const ras::ReadView& read,const LayerData& layer,const Boundary& b,Id lane) {
    std::vector<std::pair<std::size_t,Id>> ordered;
    for(auto id:b.inputs){auto p=load_partition(read,layer,id);if(std::binary_search(p.lanes.begin(),p.lanes.end(),lane))ordered.emplace_back(read.order_rank(layer.order,id),id);}
    std::sort(ordered.begin(),ordered.end()); std::vector<Id> result;for(auto [rank,id]:ordered){(void)rank;result.push_back(id);}return result;
}

struct Runtime {
    ras::Store& store;
    Pipeline pipeline;
    CoordinatorOptions options;
    std::shared_ptr<GroundSession> ground_session;
    std::shared_ptr<BundleSession> bundle_session;
    std::shared_ptr<PreparedSession> prepared_session;
    Id operation=0;
    bool registered=false;
    std::shared_ptr<Counters> counters=std::make_shared<Counters>();
    explicit Runtime(ras::Store& s,CoordinatorOptions o):store(s),options(o){counters->diagnostics_enabled=o.diagnostics;}

    void validate(const Manifest& m, const ras::Version& version,const ras::Buffer& manifest,Id window,ras::StateId state) const {
        if(!registered||window!=pipeline.identity||m.window!=window||m.state!=state||m.layers.size()!=pipeline.layers.size()||m.transforms.size()!=pipeline.transforms.size())fail("Core pipeline/window mismatch");
        if(m.layers.empty()||m.control_layer!=m.layers.front().layer||version.read().find_state(m.control_layer,maximum)!=state||
           !version.state(state).data.same_storage(manifest))fail("Core manifest is not bound to this version");
        for(std::size_t i=0;i<m.layers.size();++i){const auto& a=m.layers[i];const auto& b=pipeline.layers[i];if(a.id!=b.id||a.identity!=b.identity||a.schema!=b.payload_schema)fail("Core layer registration mismatch");version.layer(a.layer);}
        for(std::size_t i=0;i<m.transforms.size();++i)if(m.transforms[i].id!=pipeline.transforms[i].id||m.transforms[i].identity!=pipeline.transforms[i].identity)fail("Core transform registration mismatch");
        if(m.ground_policy!=ground_policy_identity(pipeline.grounds))fail("Core Ground registration policy mismatch");
        auto ownership=store.begin(version);ownership.abort();
    }
    Manifest checked(const Anchor& a) const {auto m=load_manifest(a.manifest);validate(m,a.version,a.manifest,a.window,a.manifest_state);return m;}
    Manifest checked(const Candidate& a) const {auto m=load_manifest(a.manifest);validate(m,a.version,a.manifest,a.window,a.manifest_state);return m;}

    std::shared_ptr<Access> context(ras::Transaction& tx,const Manifest& m,const Manifest* old,const ras::ReadView& parent,std::size_t index,Phase phase,
                                    const LayerDelta& delta={},const LayerDelta& upper_delta={},std::optional<std::size_t> transform={}) {
        auto a=std::make_shared<Access>();a->scope=std::make_shared<Scope>();a->scope->tx=&tx;a->scope->counters=counters;
        a->read=tx.read();a->old=parent;a->layer=m.layers[index];a->phase=phase;a->ground_session=ground_session;a->bundle_session=bundle_session;a->prepared_session=prepared_session;a->operation=operation;
        if(old)a->previous_layer=old->layers[index];
        if(index<m.transforms.size()){a->outgoing_transform=m.transforms[index];if(old)a->previous_outgoing_transform=old->transforms[index];}
        if(index)a->upper_layer=m.layers[index-1];
        a->delta=std::make_shared<LayerDelta>(delta);a->upper_delta=std::make_shared<LayerDelta>(upper_delta);
        a->changed_boundaries=std::make_shared<std::vector<Id>>();a->removed_boundaries=std::make_shared<std::vector<Id>>();
        if(transform){a->transform=m.transforms[*transform];a->lower_layer=m.layers[*transform+1];if(old)a->previous_transform=old->transforms[*transform];}
        return a;
    }

    void erase_dependency(ras::Transaction& tx,const LayerData& l,Dependency edge) {
        auto rec=tx.read().directory_find(l.dependencies,{edge.producer,edge.consumer});if(!rec)return;
        tx.erase_relation(load_relation(*rec));tx.directory_erase(l.dependencies,{edge.producer,edge.consumer});tx.directory_erase(l.producers,{edge.consumer,edge.producer});
    }
    void erase_partition(ras::Transaction& tx,const Manifest& m,std::size_t index,Id id) {
        const auto& l=m.layers[index];auto read=tx.read();const auto p=load_partition(read,l,id,counters.get());const auto native=native_partition(read,l,id);replace_bundle_refs(bundle_session,p.bundles,{});
        for(auto consumer:keys(read,l.dependencies,id))erase_dependency(tx,l,{id,consumer});
        for(auto producer:keys(read,l.producers,id))erase_dependency(tx,l,{producer,id});
        for(auto origin:p.origins)tx.directory_erase(l.origins,{origin,id});
        tx.directory_erase(l.parts,{id,0});tx.erase_partition(native);
    }

    LayerDelta admit_partitions(ras::Transaction& tx,const Manifest& m,std::size_t index,const PartitionDelta& proposed,const std::shared_ptr<Access>& access) {
        AdmissionAudit admission(counters.get());
        const auto& layer=m.layers[index];LayerDelta delta;auto initial=tx.read();const auto old_count=initial.order_size(layer.order);
        std::set<Id> touched,removed;
        for(auto id:proposed.removed){if(!id||!removed.insert(id).second||!initial.directory_find(layer.parts,{id,0}))fail("invalid or duplicate removed partition");}
        for(const auto& p:proposed.upserts){
            if(!p.id||!touched.insert(p.id).second||removed.contains(p.id))fail("conflicting partition delta");
            canonical(p.lanes,true,"partition needs a canonical common lane signature",true);canonical(p.origins,false,"invalid partition origins");canonical(p.event_kinds,true,"invalid partition event kinds",true);
            if(!p.event_count||!p.payload||p.semantic.empty()||p.payload.layout().schema!=layer.schema)fail("incomplete partition header or payload schema");
            if(index==0&&!p.origins.empty())fail("root partition cannot claim incoming receipts");
            if(index>0&&p.origins.empty())fail("lower partition must declare receipt origins");
            if(index)for(auto origin:p.origins)if(!initial.directory_find(m.transforms[index-1].receipts,{origin,0}))fail("partition origin is not a current receipt");
            require_granted(*access,p.payload);require_bundles(*access,p.bundles);
        }
        for(auto id:removed){erase_partition(tx,m,index,id);delta.removed.push_back(id);}
        for(const auto& proposed_part:proposed.upserts){
            auto p=proposed_part;std::vector<BlockBundle> old_bundles;auto previous=tx.read().directory_find(layer.parts,{p.id,0});ras::PartitionId native{};
            bool semantic=true,state=true,layout=true,topology=true;
            if(previous){const auto old=load_partition(tx.read(),layer,p.id,counters.get());native=native_partition(tx.read(),layer,p.id);p.versions=old.versions;old_bundles=old.bundles;
                semantic=old.semantic!=p.semantic||old.lanes!=p.lanes||old.event_count!=p.event_count||old.event_kinds!=p.event_kinds;
                state=old.state!=p.state;topology=old.origins!=p.origins;
                layout=!old.payload.same_storage(p.payload)||old.layout!=p.layout||!same_bundles(old.bundles,p.bundles);
                if(semantic)delta.changed.push_back(p.id);
                if(state)delta.state_changed.push_back(p.id);
                if(layout)delta.layout_changed.push_back(p.id);
                if(topology)delta.topology_changed.push_back(p.id);
                // Equal declared content keeps the old immutable backing when layout agrees.
                if(!semantic&&!state&&old.layout==p.layout&&old.payload.layout()==p.payload.layout()&&old.payload.size()==p.payload.size()){
                    p.payload=old.payload;layout=!same_bundles(old.bundles,p.bundles);
                    if(!layout&&!delta.layout_changed.empty()&&delta.layout_changed.back()==p.id)delta.layout_changed.pop_back();
                }
                if(!semantic&&!state&&!layout&&!topology)continue;
                if(topology)for(auto origin:old.origins)if(!std::binary_search(p.origins.begin(),p.origins.end(),origin))tx.directory_erase(layer.origins,{origin,p.id});
            }else delta.added.push_back(p.id);
            p.versions.id=p.id;p.versions.event_count=p.event_count;
            if(semantic)p.versions.semantic_version=tx.new_change_id().value;
            if(state)p.versions.state_version=tx.new_change_id().value;
            if(layout)p.versions.layout_version=tx.new_change_id().value;
            if(topology)p.versions.topology_version=tx.new_change_id().value;
            replace_bundle_refs(bundle_session,old_bundles,p.bundles);
            auto record=save_partition(tx,p,native);native=tx.opaque_partition(layer.layer,p.payload,record.metadata,native);bind_partition(tx,record,native);
            tx.directory_put(layer.parts,{p.id,0},record);
            if(topology)for(auto origin:p.origins)if(!tx.read().directory_find(layer.origins,{origin,p.id})){tx.directory_put(layer.origins,{origin,p.id},{});++counters->materialized_output_bindings;}
        }
        // All splice coordinates belong to the parent order. Remove all ranges
        // first, then insert in ascending adjusted coordinates; moves cannot
        // temporarily collide with a not-yet-deleted value.
        auto splices=proposed.order;std::sort(splices.begin(),splices.end(),[](const auto&a,const auto&b){return a.start<b.start;});
        std::size_t end=0;bool first=true;
        for(const auto& s:splices){if(s.start>old_count||s.delete_count>old_count-s.start||(!first&&s.start<end))fail("overlapping or invalid parent order splice");end=s.start+s.delete_count;first=false;
            for(auto id:s.partitions)if(!id||!tx.read().directory_find(layer.parts,{id,0}))fail("order references an absent partition");
        }
        // Read removed identities before changing the order; no suffix scan.
        for(const auto& s:splices)initial.order_range(layer.order,s.start,s.delete_count,[&](Id id){delta.order_affected.push_back(id);});
        for(auto it=splices.rbegin();it!=splices.rend();++it)if(it->delete_count)tx.order_splice(layer.order,it->start,it->delete_count,{});
        std::size_t deleted=0,inserted=0;
        for(const auto& s:splices){if(!s.partitions.empty())tx.order_splice(layer.order,s.start-deleted+inserted,0,s.partitions);
            deleted+=s.delete_count;inserted+=s.partitions.size();delta.order_affected.insert(delta.order_affected.end(),s.partitions.begin(),s.partitions.end());}
        if(tx.read().order_size(layer.order)!=tx.read().directory_size(layer.parts))fail("partition directory/order coverage mismatch");
        for(auto id:touched)(void)tx.read().order_rank(layer.order,id);
        for(auto id:delta.order_affected)if(tx.read().directory_find(layer.parts,{id,0}))(void)tx.read().order_rank(layer.order,id);
        for(auto id:removed){bool present=true;try{(void)tx.read().order_rank(layer.order,id);}catch(const ras::Error&){present=false;}if(present)fail("removed partition remains in order");}
        delta.order_changed=!splices.empty();unique(delta.order_affected);return delta;
    }

    void admit_dependencies(ras::Transaction& tx,const LayerData& layer,const DependencyDelta& changes,LayerDelta& delta) {
        std::set<Dependency> seen;
        for(auto edge:changes.removed){if(!edge.producer||!edge.consumer||!seen.insert(edge).second)fail("invalid dependency removal");
            if(tx.read().directory_find(layer.dependencies,{edge.producer,edge.consumer})){erase_dependency(tx,layer,edge);delta.invalidated.push_back(edge.consumer);delta.dependencies_changed=true;}}
        for(auto edge:changes.upserts){if(!edge.producer||!edge.consumer||edge.producer==edge.consumer||!seen.insert(edge).second)fail("invalid dependency delta");
            if(!tx.read().directory_find(layer.parts,{edge.producer,0})||!tx.read().directory_find(layer.parts,{edge.consumer,0}))fail("dependency references absent partition");
            if(!tx.read().directory_find(layer.dependencies,{edge.producer,edge.consumer})){
                const auto rid=tx.add_relation({{native_partition(tx.read(),layer,edge.producer)},{native_partition(tx.read(),layer,edge.consumer)},ras::RelationKind::Data});
                const auto record=save_relation(tx,rid);tx.directory_put(layer.dependencies,{edge.producer,edge.consumer},record);tx.directory_put(layer.producers,{edge.consumer,edge.producer},record);
                delta.invalidated.push_back(edge.consumer);delta.dependencies_changed=true;
            }
        }

    }
    void closure(const ras::ReadView& current,const LayerData& layer,const ras::ReadView& old,const LayerData* previous,LayerDelta& delta) {
        std::vector<Id> work=delta.invalidated;work.insert(work.end(),delta.added.begin(),delta.added.end());work.insert(work.end(),delta.changed.begin(),delta.changed.end());work.insert(work.end(),delta.removed.begin(),delta.removed.end());work.insert(work.end(),delta.state_changed.begin(),delta.state_changed.end());
        std::set<Id> visited(work.begin(),work.end());unique(work);
        for(std::size_t n=0;n<work.size();++n){++counters->dependency_visits;auto add=[&](Id id){if(visited.insert(id).second)work.push_back(id);};
            for(auto id:keys(current,layer.dependencies,work[n]))add(id);
            if(previous)for(auto id:keys(old,previous->dependencies,work[n]))add(id);}
        delta.invalidated.assign(visited.begin(),visited.end());
    }

    struct BoundaryChanges { std::vector<Id> generation; LinkContext links; };
    void input_change(ras::Transaction& tx,const TransformData& t,Id boundary,Id input,bool add) {
        change_route(tx,t,input,boundary,route_input,add,counters.get());
        if(add)tx.directory_put(t.members,{boundary,input},{});
        else tx.directory_erase(t.members,{boundary,input});
        ++counters->boundary_input_writes;
    }
    BoundaryChanges admit_boundaries(ras::Transaction& tx,const Manifest& m,std::size_t index,const BoundaryDelta& proposal,const std::shared_ptr<Access>& a) {
        const auto& t=m.transforms[index];const auto& l=m.layers[index];std::set<Id> touched;BoundaryChanges result;
        for(auto id:proposal.removed){
            if(!id||!touched.insert(id).second||!tx.read().directory_find(t.boundaries,{id,0}))fail("invalid boundary removal");
            const auto b=load_boundary(tx.read(),t,id,counters.get(),false);
            for(auto input:keys(tx.read(),t.members,id))input_change(tx,t,id,input,false);
            if(b.consumes_layer_state)tx.directory_erase(t.state_consumers,{0,id});
            change_route(tx,t,b.owner,id,route_owner,false,counters.get());if(!b.parent_receipts.empty())tx.directory_erase(t.parent_consumers,{0,id});
            for(auto receipt:keys(tx.read(),t.consumed_receipts,id))tx.directory_erase(t.consumed_receipts,{id,receipt});
            for(auto receipt:b.parent_receipts)tx.directory_erase(t.parent_inputs,{id,receipt});
            replace_bundle_refs(bundle_session,b.bundles,{});erase_if(tx,t.generations,{id,0});tx.directory_erase(t.boundaries,{id,0});result.generation.push_back(id);result.links.topology.push_back({id,b.owner,0,{}, {},true});
        }
        const auto apply=[&](Id id,const std::optional<Boundary>& replacement,const std::vector<Id>& additions,const std::vector<Id>& removals,bool replace_inputs){
            if(!id||!touched.insert(id).second)fail("invalid or duplicate boundary delta");
            const bool exists=bool(tx.read().directory_find(t.boundaries,{id,0}));
            Boundary old;BoundaryHead head;
            if(exists){old=load_boundary(tx.read(),t,id,counters.get(),false);head=load_boundary_head(tx.read(),t,id,counters.get());}
            else if(!replacement||!replace_inputs)fail("patch references absent Boundary");
            Boundary b=replacement?*replacement:old;
            if(b.id!=id||!b.owner||b.semantic.empty())fail("invalid boundary header");
            canonical(b.parent_receipts,false,"invalid parent receipt capabilities");
            if(!tx.read().directory_find(l.parts,{b.owner,0}))fail("boundary owner is absent");
            if(replacement){require_granted(*a,b.payload);require_bundles(*a,b.bundles);}
            std::vector<Id> add,remove;
            if(replace_inputs){canonical(b.inputs,false,"invalid boundary inputs");
                for(auto input:b.inputs)if(!tx.read().directory_find(t.members,{id,input}))add.push_back(input);
                if(exists)for(auto input:keys(tx.read(),t.members,id))if(!std::binary_search(b.inputs.begin(),b.inputs.end(),input))remove.push_back(input);
            }else{
                if(replacement&&!b.inputs.empty())fail("BoundaryPatch header must omit inputs");
                add=additions;remove=removals;canonical(add,false,"invalid BoundaryPatch additions");canonical(remove,false,"invalid BoundaryPatch removals");
                for(auto input:add)if(std::binary_search(remove.begin(),remove.end(),input)||tx.read().directory_find(t.members,{id,input}))fail("conflicting BoundaryPatch input addition");
                for(auto input:remove)if(!tx.read().directory_find(t.members,{id,input}))fail("BoundaryPatch removes absent input");
            }
            for(auto input:add)if(!tx.read().directory_find(l.parts,{input,0}))fail("boundary semantic input is absent");
            const bool topology=!exists||old.owner!=b.owner||!add.empty()||!remove.empty();
            const bool semantic=!exists||old.semantic!=b.semantic||old.state!=b.state||old.consumes_layer_state!=b.consumes_layer_state||old.parent_receipts!=b.parent_receipts||!add.empty()||!remove.empty();
            const bool physical=!exists||!old.payload.same_storage(b.payload)||!same_bundles(old.bundles,b.bundles);
            if(!topology&&!semantic&&!physical)return;
            if(exists&&old.owner!=b.owner)change_route(tx,t,old.owner,id,route_owner,false,counters.get());
            if(!exists||old.owner!=b.owner)change_route(tx,t,b.owner,id,route_owner,true,counters.get());
            if(exists&&old.consumes_layer_state&&!b.consumes_layer_state)tx.directory_erase(t.state_consumers,{0,id});
            if(b.consumes_layer_state&&(!exists||!old.consumes_layer_state))tx.directory_put(t.state_consumers,{0,id},{});
            if(exists&&!old.parent_receipts.empty()&&b.parent_receipts.empty())tx.directory_erase(t.parent_consumers,{0,id});
            if(!b.parent_receipts.empty()&&(!exists||old.parent_receipts.empty()))tx.directory_put(t.parent_consumers,{0,id},{});
            for(auto receipt:old.parent_receipts)if(!std::binary_search(b.parent_receipts.begin(),b.parent_receipts.end(),receipt))tx.directory_erase(t.parent_inputs,{id,receipt});
            for(auto receipt:b.parent_receipts)if(!std::binary_search(old.parent_receipts.begin(),old.parent_receipts.end(),receipt))tx.directory_put(t.parent_inputs,{id,receipt},{});
            for(auto input:remove)input_change(tx,t,id,input,false);
            for(auto input:add)input_change(tx,t,id,input,true);
            head.id=id;head.owner=b.owner;head.consumes_layer_state=b.consumes_layer_state;
            head.input_count+=add.size();head.input_count-=remove.size();
            if(semantic){head.generation_stamp=tx.new_change_id().value;result.generation.push_back(id);}
            if(topology){head.topology_version=tx.new_change_id().value;result.links.topology.push_back({id,old.owner,b.owner,add,remove,false});}
            replace_bundle_refs(bundle_session,old.bundles,b.bundles);b.inputs.clear();tx.directory_put(t.boundaries,{id,0},save_boundary(tx,b,head));erase_if(tx,t.generations,{id,0});
        };
        for(const auto& b:proposal.upserts)apply(b.id,b,{}, {},true);
        for(const auto& patch:proposal.patches)apply(patch.id,patch.header,patch.add_inputs,patch.remove_inputs,false);
        unique(result.generation);return result;
    }

    struct Handoff { std::vector<Id> changed_receipts,removed_receipts,changed_boundaries; LinkContext links; };
    Handoff generate(ras::Transaction& tx,const Manifest& m,const Manifest* old,const ras::ReadView& parent,std::size_t index,const LayerDelta& delta) {
        const auto& t=m.transforms[index];const auto& layer=m.layers[index];const auto& registration=pipeline.transforms[index];
        const auto selection_begin=options.diagnostics?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        BoundaryDelta proposal;BoundaryChanges changes;
        {auto a=context(tx,m,old,parent,index,Phase::Boundary,delta,delta,index);a->upper_layer=layer;Context ctx(a);ScopeEnd close{a};++counters->boundary_callbacks;
            proposal=invoke_callback(*counters,[&]{return registration.boundary(LayerView(a),ctx);});changes=admit_boundaries(tx,m,index,proposal,a);}
        std::optional<SelectionAudit> selection_audit;selection_audit.emplace(counters.get());
        std::set<Id> candidates(changes.generation.begin(),changes.generation.end()),invalidated;
        auto affected=delta.invalidated;affected.insert(affected.end(),delta.order_affected.begin(),delta.order_affected.end());unique(affected);
        for(auto input:affected){const auto visit=[&](const OutputTarget& target){if(target.input){candidates.insert(target.boundary);invalidated.insert(target.boundary);}};
            downstream(tx.read(),t,input,visit,counters.get(),false);
            if(old)downstream(parent,old->transforms[index],input,visit,counters.get(),false);}
        for(auto input:delta.removed)downstream(tx.read(),t,input,[&](const OutputTarget& target){if(target.owner)fail("retained boundary owner disappeared");},counters.get(),false);
        if(delta.layer_state_changed)for(auto id:keys(tx.read(),t.state_consumers,0)){candidates.insert(id);invalidated.insert(id);}
        // Previous-result inputs are a separate pending dependency set. Compare
        // their fixed receipt versions even when the root has no changes.
        for(auto id:keys(tx.read(),t.parent_consumers,0)){
            for(auto receipt:keys(tx.read(),t.parent_inputs,id)){
                if(!old)fail("initial Apply cannot reference a parent receipt");
                const auto version=load_receipt_version(parent,old->transforms[index],receipt,counters.get());
                if(!version)fail("parent receipt input requires complete semantic evidence");
                const auto consumed=tx.read().directory_find(t.consumed_receipts,{id,receipt});
                Id prior=0;if(consumed&&consumed->metadata){auto bytes=consumed->metadata.bytes();if(bytes.size()!=sizeof(Id))fail("invalid consumed receipt stamp");std::memcpy(&prior,bytes.span().data(),sizeof(Id));}
                if(version!=prior){candidates.insert(id);invalidated.insert(id);}
            }
        }
        std::vector<Id> selected_ids;std::map<Id,Evidence> generation;
        for(auto id:candidates){if(!tx.read().directory_find(t.boundaries,{id,0}))continue;
            auto head=load_boundary_head(tx.read(),t,id,counters.get());
            if(!tx.read().directory_find(layer.parts,{head.owner,0}))fail("retained boundary owner disappeared");
            if(invalidated.contains(id)&&!std::binary_search(changes.generation.begin(),changes.generation.end(),id)){
                head.generation_stamp=tx.new_change_id().value;
                auto writer=tx.buffer(sizeof(Id),{metadata_schema,1,1});auto bytes=writer.reserve(sizeof(Id));
                std::memcpy(bytes.bytes().data(),&head.generation_stamp,sizeof(Id));bytes.commit();
                tx.directory_put(t.generations,{id,0},{writer.finish(),{}, {}});
            }
            Evidence key;pod(key,head.generation_stamp);Evidence prior;
            const bool prior_exists=old&&parent.directory_find(old->transforms[index].receipts,{id,0});
            if(prior_exists)pod(prior,load_receipt_generation(parent,old->transforms[index],id,counters.get()));
            if(!prior_exists||key!=prior){selected_ids.push_back(id);generation.emplace(id,std::move(key));}
        }
        selection_audit.reset();
        std::vector<Boundary> selected;selected.reserve(selected_ids.size());
        for(auto id:selected_ids)selected.push_back(load_boundary(tx.read(),t,id,counters.get()));
        Handoff hand;hand.removed_receipts=proposal.removed;unique(hand.removed_receipts);hand.links=std::move(changes.links);
        for(const auto& change:hand.links.topology)hand.changed_boundaries.push_back(change.id);
        if(options.diagnostics)counters->selection_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-selection_begin).count();
        for(auto id:hand.removed_receipts){if(tx.read().directory_find(t.receipts,{id,0})){const auto prior=load_receipt(tx.read(),t,id);replace_bundle_refs(bundle_session,receipt_bundles(prior),{});replace_bundle_refs(bundle_session,prior.state.bundles,{});}erase_if(tx,t.receipts,{id,0});}
        if(!selected.empty()||!hand.removed_receipts.empty()){
            auto apply=context(tx,m,old,parent,index,Phase::Apply,delta,{},index);apply->upper_layer=layer;
            apply->selected=std::make_shared<std::vector<Boundary>>(selected);
            for(const auto& b:selected){auto isolated=*apply;isolated.grants=std::make_shared<Grants>();isolated.bundle_grants=std::make_shared<std::set<std::pair<ras::DirectoryId,Id>>>();grant(isolated,b.payload);grant_bundles(isolated,b.bundles);(*apply->boundary_grants)[b.id]=isolated.grants;(*apply->boundary_bundle_grants)[b.id]=isolated.bundle_grants;}
            apply->removed_boundaries=std::make_shared<std::vector<Id>>(hand.removed_receipts);apply->changed_boundaries=std::make_shared<std::vector<Id>>(hand.changed_boundaries);
            Context apply_context(apply);ScopeEnd apply_close{apply};++counters->apply_callbacks;counters->selected_boundaries+=selected.size();
            const auto result=invoke_callback(*counters,[&]{return registration.apply(selected,apply_context);});auto removals=result.removed;unique(removals);
            if(removals!=hand.removed_receipts||result.removed.size()!=removals.size()||result.generated.size()!=selected.size())fail("Apply returned an incomplete or duplicate receipt delta");
            AdmissionAudit receipt_admission(counters.get());
            std::set<Id> returned;
            for(const auto& result_receipt:result.generated){
                auto receipt=result_receipt;if(!generation.contains(receipt.boundary)||!returned.insert(receipt.boundary).second)fail("Apply returned an unselected or duplicate receipt");
                auto grants=apply->boundary_grants->find(receipt.boundary);
                auto checked=*apply;checked.grants=grants==apply->boundary_grants->end()?std::make_shared<Grants>():grants->second;checked.bundle_grants=apply->boundary_bundle_grants->at(receipt.boundary);require_granted(checked,receipt.payload);require_granted(checked,receipt.state.payload);require_bundles(checked,receipt_bundles(receipt));require_bundles(checked,receipt.state.bundles);require_state_evidence(receipt.state);
                const auto output=apply->chunk_outputs->find(receipt.boundary);
                if(output!=apply->chunk_outputs->end()){
                    if(receipt.chunks.data_!=output->second||!output->second->finished)fail("Apply did not return its finished chunk output");
                }else{
                    if(receipt.chunks)fail("receipt chunks are outside this Boundary output scope");
                }
                if(receipt.event_count&&!receipt.payload&&!receipt.chunks.count())fail("nonempty receipt has no payload");
                std::optional<Receipt> previous_receipt;
                if(old&&parent.directory_find(old->transforms[index].receipts,{receipt.boundary,0}))
                    previous_receipt=load_receipt(parent,old->transforms[index],receipt.boundary,nullptr,counters.get());
                receipt.semantic_version=receipt.semantic.empty()?0:tx.new_change_id().value;
                if(previous_receipt&&!receipt.semantic.empty()){
                    const auto& previous=*previous_receipt;
                    // Semantic identity is evaluator evidence. Storage packing is
                    // a separate layout property and never requires a body walk.
                    if(previous.semantic==receipt.semantic&&previous.state.semantic==receipt.state.semantic&&previous.event_count==receipt.event_count)
                        receipt.semantic_version=previous.semantic_version;
                }
                replace_bundle_refs(bundle_session,previous_receipt?receipt_bundles(*previous_receipt):std::vector<BlockBundle>{},receipt_bundles(receipt));
                replace_bundle_refs(bundle_session,previous_receipt?previous_receipt->state.bundles:std::vector<BlockBundle>{},receipt.state.bundles);
                tx.directory_put(t.receipts,{receipt.boundary,0},save_receipt(tx,receipt,generation.at(receipt.boundary)));
                const auto parent_inputs=keys(tx.read(),t.parent_inputs,receipt.boundary);
                for(auto id:keys(tx.read(),t.consumed_receipts,receipt.boundary))if(!std::binary_search(parent_inputs.begin(),parent_inputs.end(),id))tx.directory_erase(t.consumed_receipts,{receipt.boundary,id});
                for(auto id:parent_inputs){const auto version=load_receipt_version(parent,old->transforms[index],id,counters.get());
                    auto writer=tx.buffer(sizeof(Id),{metadata_schema,1,1});auto bytes=writer.reserve(sizeof(Id));std::memcpy(bytes.bytes().data(),&version,sizeof(Id));bytes.commit();tx.directory_put(t.consumed_receipts,{receipt.boundary,id},{writer.finish(),{}, {}});}
                hand.changed_receipts.push_back(receipt.boundary);++counters->generated_receipts;
            }
        }
        unique(hand.changed_receipts);return hand;
    }

    Candidate evaluate(const Anchor* anchor,const RootInput& input) {
        if(!registered)fail("Core pipeline is not registered");
        counters=std::make_shared<Counters>();counters->diagnostics_enabled=options.diagnostics;
        const auto evaluate_begin=options.diagnostics?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        std::optional<Manifest> old;if(anchor)old=checked(*anchor);auto tx=store.begin(anchor?anchor->version:ras::Version{});const auto parent=anchor?anchor->version.read():ras::ReadView{};Manifest m;
        if(old)m=*old;else{
            m.window=pipeline.identity;
            for(const auto& layer:pipeline.layers){LayerData l;l.id=layer.id;l.identity=layer.identity;l.schema=layer.payload_schema;l.layer=tx.add_layer(layer.payload_schema);
                l.parts=tx.add_directory();l.dependencies=tx.add_directory();l.producers=tx.add_directory();l.origins=tx.add_directory();l.states=tx.add_directory();l.evaluator=tx.add_directory();l.order=tx.add_order();m.layers.push_back(l);}
            for(const auto& transform:pipeline.transforms){TransformData t;t.id=transform.id;t.identity=transform.identity;
                t.boundaries=tx.add_directory();t.receipts=tx.add_directory();t.routes=tx.add_directory();t.members=tx.add_directory();
                t.consumed_receipts=tx.add_directory();t.generations=tx.add_directory();t.parent_inputs=tx.add_directory();
                t.state_consumers=tx.add_directory();t.parent_consumers=tx.add_directory();m.transforms.push_back(t);}
            m.control_layer=m.layers.front().layer;auto empty=tx.buffer(0).finish();m.state=tx.put_state(m.control_layer,maximum,empty);
        }
        operation=tx.new_change_id().value;
        m.ground_policy=ground_policy_identity(pipeline.grounds);
        if(!old){m.bundle_owners=tx.add_directory();m.ground_facts=pipeline.grounds.empty()?ras::DirectoryId{}:tx.add_directory();}
        bundle_session=make_bundle_session(tx,m.bundle_owners);
        prepared_session=std::make_shared<PreparedSession>();
        ground_session=make_ground_session(tx,m.ground_facts,parent,old?old->ground_facts:ras::DirectoryId{},pipeline.grounds,m.window);
        struct EndGround { std::shared_ptr<GroundSession>& value; std::shared_ptr<BundleSession>& bundles; std::shared_ptr<PreparedSession>& prepared; ~EndGround(){value.reset();bundles.reset();prepared.reset();} } end_ground{ground_session,bundle_session,prepared_session};
        RootCapture root;
        {auto a=context(tx,m,old?&*old:nullptr,parent,0,Phase::Capture);Context ctx(a);ScopeEnd close{a};root=invoke_callback(*counters,[&]{return pipeline.layers[0].capture(input,ctx);});require_granted(*a,root.payload);require_bundles(*a,root.bundles);require_state_evidence(root);const auto prior_root=tx.read().directory_find(m.layers[0].states,{1,0});replace_bundle_refs(bundle_session,prior_root?load_state_record(*prior_root).bundles:std::vector<BlockBundle>{},root.bundles);tx.directory_put(m.layers[0].states,{1,0},save_state(tx,root));}
        Handoff incoming;LayerDelta upper_delta;
        for(std::size_t i=0;i<m.layers.size();++i){const auto& registration=pipeline.layers[i];PartitionRequest request{!old,i==0?root:RootCapture{},incoming.changed_receipts,incoming.removed_receipts};LayerDelta delta;
            {auto a=context(tx,m,old?&*old:nullptr,parent,i,Phase::Partition,{},upper_delta,i?std::optional(i-1):std::nullopt);if(i==0){grant(*a,root.payload);grant_bundles(*a,root.bundles);}a->changed_boundaries=std::make_shared<std::vector<Id>>(incoming.changed_receipts);a->removed_boundaries=std::make_shared<std::vector<Id>>(incoming.removed_receipts);Context ctx(a);ScopeEnd close{a};++counters->partition_callbacks;const auto phase_begin=options.diagnostics?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};const auto proposed=invoke_callback(*counters,[&]{return registration.partition(request,ctx);});delta=admit_partitions(tx,m,i,proposed,a);if(options.diagnostics)counters->partition_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_begin).count();}
            {auto a=context(tx,m,old?&*old:nullptr,parent,i,Phase::Dependency,delta,upper_delta);Context ctx(a);ScopeEnd close{a};auto dependencies=invoke_callback(*counters,[&]{return registration.data_dep(LayerView(a),delta,ctx);});admit_dependencies(tx,m.layers[i],dependencies,delta);}
            closure(tx.read(),m.layers[i],parent,old?&old->layers[i]:nullptr,delta);
            if(i&&(!old||!incoming.links.topology.empty()||!delta.added.empty()||!delta.removed.empty()||!delta.topology_changed.empty())){const auto phase_begin=options.diagnostics?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};auto a=context(tx,m,old?&*old:nullptr,parent,i,Phase::Link,delta,upper_delta,i-1);a->changed_boundaries=std::make_shared<std::vector<Id>>(incoming.changed_boundaries);a->removed_boundaries=std::make_shared<std::vector<Id>>(incoming.removed_receipts);Context ctx(a);ScopeEnd close{a};
                auto upper_access=std::make_shared<Access>(*a);upper_access->layer=m.layers[i-1];const auto& callback=pipeline.transforms[i-1].link;if(callback)++counters->link_callbacks;auto links=callback?invoke_callback(*counters,[&]{return callback(LayerView(a),LayerView(upper_access),BoundaryView(a),ctx);}):LinkDelta{};admit_lineage(tx,m,i-1,links,delta,incoming.links,counters.get());if(options.diagnostics)counters->lineage_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_begin).count();}
            if(registration.build_state){const auto previous=load_state(tx.read(),m.layers[i],counters.get());auto a=context(tx,m,old?&*old:nullptr,parent,i,Phase::State,delta,upper_delta,i?std::optional(i-1):std::nullopt);a->changed_boundaries=std::make_shared<std::vector<Id>>(incoming.changed_receipts);a->removed_boundaries=std::make_shared<std::vector<Id>>(incoming.removed_receipts);if(i==0){grant(*a,root.payload);grant_bundles(*a,root.bundles);}Context ctx(a);ScopeEnd close{a};++counters->state_callbacks;auto state=invoke_callback(*counters,[&]{return registration.build_state(LayerView(a),ctx);});require_granted(*a,state.payload);require_bundles(*a,state.bundles);require_state_evidence(state);delta.layer_state_changed=state.semantic!=previous.semantic;if(delta.layer_state_changed||!same_bundles(previous.bundles,state.bundles)||bool(previous.payload)!=bool(state.payload)||(state.payload&&!previous.payload.same_storage(state.payload))){replace_bundle_refs(bundle_session,previous.bundles,state.bundles);tx.directory_put(m.layers[i].states,{0,0},save_state(tx,state));}}
            if(i<m.transforms.size())incoming=generate(tx,m,old?&*old:nullptr,parent,i,delta);
            upper_delta=std::move(delta);
        }
        for(const auto& transform:m.transforms)counters->route_entries+=tx.read().directory_size(transform.routes);
        finish_bundle_session(bundle_session);const auto manifest=save_manifest(tx,m);tx.replace_state(m.state,manifest);const auto version=tx.publish();
        if(options.diagnostics){counters->evaluate_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-evaluate_begin).count();counters->framework_seconds=counters->evaluate_seconds-counters->evaluator_callback_seconds;}
        return {version,manifest,m.window,*counters,m.state};
    }

    std::shared_ptr<Access> public_access(const Manifest& m,const ras::Version& v,std::size_t index,bool transform) const {
        if(index>=(transform?m.transforms.size():m.layers.size()))fail("Core view index outside pipeline");
        auto a=std::make_shared<Access>();a->scope=std::make_shared<Scope>();a->scope->counters=std::make_shared<Counters>();a->read=v.read();a->old=a->read;a->phase=Phase::Public;
        a->scope->counters->diagnostics_enabled=options.diagnostics;
        a->layer=m.layers[index];if(transform){a->transform=m.transforms[index];a->lower_layer=m.layers[index+1];}a->delta=std::make_shared<LayerDelta>();a->upper_delta=a->delta;a->changed_boundaries=std::make_shared<std::vector<Id>>();a->removed_boundaries=a->changed_boundaries;return a;
    }
};
} // namespace detail

Coordinator::Coordinator(ras::Store& store,CoordinatorOptions options):runtime_(std::make_unique<detail::Runtime>(store,options)){}
Coordinator::~Coordinator()=default;
void Coordinator::register_pipeline(Pipeline pipeline) {
    if(runtime_->registered)throw ras::Error("Core coordinator is already registered");
    if(!pipeline.identity||pipeline.layers.empty()||pipeline.transforms.size()+1!=pipeline.layers.size())throw ras::Error("incomplete Core pipeline registration");
    std::set<Id> layers,transforms;
    for(std::size_t i=0;i<pipeline.layers.size();++i){const auto& l=pipeline.layers[i];if(!l.id||!l.identity||!l.payload_schema||!layers.insert(l.id).second||!l.partition||!l.data_dep||(!i&&!l.capture)||(i&&l.capture))throw ras::Error("invalid Core layer registration");}
    for(std::size_t i=0;i<pipeline.transforms.size();++i){const auto& t=pipeline.transforms[i];if(!t.id||!t.identity||!transforms.insert(t.id).second||t.upper!=pipeline.layers[i].id||t.lower!=pipeline.layers[i+1].id||!t.boundary||!t.apply)throw ras::Error("invalid Core transform registration");}
    detail::validate_ground_registrations(pipeline);
    runtime_->pipeline=std::move(pipeline);runtime_->registered=true;
}
Anchor Coordinator::init(const RootInput& input){auto candidate=runtime_->evaluate(nullptr,input);return {candidate.version,candidate.manifest,candidate.window,candidate.manifest_state};}
Candidate Coordinator::refresh(const Anchor& anchor,const RootInput& input){return runtime_->evaluate(&anchor,input);}
Anchor Coordinator::commit(const Candidate& candidate) const{runtime_->checked(candidate);return {candidate.version,candidate.manifest,candidate.window,candidate.manifest_state};}
LayerView Coordinator::layer(const Anchor& anchor,std::size_t index) const{auto m=runtime_->checked(anchor);return LayerView(runtime_->public_access(m,anchor.version,index,false));}
LayerView Coordinator::layer(const Candidate& candidate,std::size_t index) const{auto m=runtime_->checked(candidate);return LayerView(runtime_->public_access(m,candidate.version,index,false));}
BoundaryView Coordinator::transform(const Anchor& anchor,std::size_t index) const{auto m=runtime_->checked(anchor);return BoundaryView(runtime_->public_access(m,anchor.version,index,true));}
BoundaryView Coordinator::transform(const Candidate& candidate,std::size_t index) const{auto m=runtime_->checked(candidate);return BoundaryView(runtime_->public_access(m,candidate.version,index,true));}
const Counters& Coordinator::last_counters() const{return *runtime_->counters;}
} // namespace flexeva::core::runtime
