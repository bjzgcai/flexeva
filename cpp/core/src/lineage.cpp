#include "runtime_internal.hpp"
#include <cstring>
#include <set>
#include <tuple>

namespace flexeva::core::runtime::detail {
namespace {
Id roles(const ras::DirectoryRecord& record) {
    if(!record.metadata||record.metadata.size()!=sizeof(Id))throw ras::Error("invalid AL route roles");
    Id result=0;const auto bytes=record.metadata.bytes();std::memcpy(&result,bytes.span().data(),sizeof(result));
    if(!result||(result&~(route_owner|route_input)))throw ras::Error("invalid AL route roles");
    return result;
}
}
void change_route(ras::Transaction& tx,const TransformData& t,Id producer,Id boundary,Id role,bool add,Counters* counters) {
    if(!producer||!boundary||(role!=route_owner&&role!=route_input))throw ras::Error("invalid AL route change");
    const auto prior=tx.read().directory_find(t.routes,{producer,boundary});
    const Id before=prior?roles(*prior):0,after=add?(before|role):(before&~role);
    if(before==after)return;
    if(!after){tx.directory_erase(t.routes,{producer,boundary});if(counters)++counters->route_removals;return;}
    auto writer=tx.buffer(sizeof(after),{metadata_schema,1,1});auto bytes=writer.reserve(sizeof(after));
    std::memcpy(bytes.bytes().data(),&after,sizeof(after));bytes.commit();
    tx.directory_put(t.routes,{producer,boundary},{writer.finish(),{}, {}});
    if(counters)++counters->route_upserts;
}
void downstream(const ras::ReadView& read,const TransformData& t,Id producer,
                const std::function<void(const OutputTarget&)>& visit,Counters* counters,bool resolve_output) {
    if(!producer)return;
    read.directory_range(t.routes,{producer,0},{producer,maximum},[&](ras::DirectoryKey key,const ras::DirectoryRecord& record){
        const auto mask=roles(record);
        if(counters){if(counters->diagnostics_enabled)++counters->metadata_reads;++counters->route_visits;}
        // Selection needs route roles and Boundary IDs only. Public queries
        // additionally resolve output presence/version from the receipt index.
        const bool has_output=resolve_output&&bool(read.directory_find(t.receipts,{key.second,0}));
        visit({key.second,has_output?load_receipt_version(read,t,key.second,counters):0,
               bool(mask&route_owner),bool(mask&route_input),has_output});
    });
}
std::optional<Lineage> lookup_lineage(const ras::ReadView& read,const TransformData& t,
                                    const LayerData& lower,const Lineage& key,Counters* counters) {
    if(counters&&counters->diagnostics_enabled)++counters->header_reads;
    if(!read.directory_find(lower.origins,{key.boundary,key.consumer})||
       !read.directory_find(t.boundaries,{key.boundary,0}))return {};
    const auto route=read.directory_find(t.routes,{key.producer,key.boundary});if(!route)return {};
    const auto mask=roles(*route);
    // Owner/input roles share one diagnostic edge as well as one stored route.
    if(key.kind==LineageKind::Boundary ? !(mask&route_owner) : !(mask&route_input)||(mask&route_owner))return {};
    auto result=key;result.id=0;return result;
}
void admit_lineage(ras::Transaction& tx,const Manifest& m,std::size_t index,const LinkDelta& delta,
                   const LayerDelta& lower_delta,const LinkContext& context,Counters* counters) {
    const auto& t=m.transforms[index];const auto& lower=m.layers[index+1];
    // Link is a protocol over the existing factorized relation. Legacy explicit
    // declarations can be checked, but never allocate edges or duplicate indexes.
    if(!delta.removed.empty())throw ras::Error("AL has no persistent edge IDs; update Boundary inputs or output origins");
    std::set<std::tuple<Id,Id,Id,LineageKind>> supplied;
    for(const auto& edge:delta.upserts){
        if(!edge.boundary||!edge.producer||!edge.consumer||
           !supplied.emplace(edge.boundary,edge.producer,edge.consumer,edge.kind).second)
            throw ras::Error("invalid or duplicate AL declaration");
        const auto b=load_boundary_head(tx.read(),t,edge.boundary,counters);
        if(edge.kind==LineageKind::Boundary){if(edge.producer!=b.owner)throw ras::Error("AL owner mismatch");}
        else if(edge.kind==LineageKind::Data){if(!tx.read().directory_find(t.members,{edge.boundary,edge.producer}))throw ras::Error("AL producer is not a declared input");}
        else throw ras::Error("unknown AL role");
        if(!tx.read().directory_find(lower.origins,{edge.boundary,edge.consumer}))throw ras::Error("AL Boundary is not an output origin");
    }
    // Admission checked every new/changed origin. Only a removed Boundary can
    // leave an untouched demanded partition pointing to a missing output.
    (void)lower_delta;
    for(const auto& change:context.topology)if(change.removed)
        tx.read().directory_range(lower.origins,{change.id,0},{change.id,maximum},
            [&](ras::DirectoryKey,const ras::DirectoryRecord&){throw ras::Error("partition retains removed Boundary origin");});
}
} // namespace flexeva::core::runtime::detail
