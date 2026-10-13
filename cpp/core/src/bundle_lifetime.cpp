#include "runtime_internal.hpp"
#include <cstring>

namespace flexeva::core::runtime::detail {
struct BundleSession {
    ras::Transaction& tx;
    ras::DirectoryId owners;
    std::set<ras::DirectoryId> possibly_dead;
    BundleSession(ras::Transaction& t,ras::DirectoryId d):tx(t),owners(d){}
};
std::shared_ptr<BundleSession> make_bundle_session(ras::Transaction& tx,ras::DirectoryId owners){
    return std::make_shared<BundleSession>(tx,owners);
}
void track_bundle_directory(Access& access,ras::DirectoryId directory){
    check(access);if(!access.bundle_session)throw ras::Error("Core bundle lacks an operation owner");
    access.bundle_session->possibly_dead.insert(directory);
}
void replace_bundle_refs(const std::shared_ptr<BundleSession>& session,const std::vector<BlockBundle>& before,const std::vector<BlockBundle>& after){
    if(before.empty()&&after.empty())return;
    if(!session)throw ras::Error("Core bundle ownership has no session");
    std::map<ras::DirectoryId,std::int64_t> changes;
    for(const auto& value:before){const auto* d=BundleAccess::get(value);if(!d)throw ras::Error("empty owned bundle");--changes[d->directory];}
    for(const auto& value:after){const auto* d=BundleAccess::get(value);if(!d)throw ras::Error("empty owned bundle");++changes[d->directory];}
    for(const auto& [directory,change]:changes){
        if(!change)continue;
        std::uint64_t count=0;
        const auto existing=session->tx.read().directory_find(session->owners,{directory.value,0});
        if(existing){if(!existing->metadata||existing->metadata.size()!=sizeof(count))throw ras::Error("invalid bundle owner count");const auto bytes=existing->metadata.bytes();std::memcpy(&count,bytes.span().data(),sizeof(count));}
        if(change<0){const auto removed=std::uint64_t(-change);if(removed>count)throw ras::Error("bundle owner count underflow");count-=removed;}
        else{if(std::uint64_t(change)>maximum-count)throw ras::Error("bundle owner count overflow");count+=std::uint64_t(change);}
        if(!count){session->tx.directory_erase(session->owners,{directory.value,0});session->possibly_dead.insert(directory);}
        else{auto writer=session->tx.buffer(sizeof(count),{metadata_schema,1,1});auto bytes=writer.reserve(sizeof(count));std::memcpy(bytes.bytes().data(),&count,sizeof(count));bytes.commit();session->tx.directory_put(session->owners,{directory.value,0},{writer.finish(),{}, {}});}
    }
}
void replace_bundle_refs(Access& access,const std::vector<BlockBundle>& before,const std::vector<BlockBundle>& after){check(access);replace_bundle_refs(access.bundle_session,before,after);}
void finish_bundle_session(const std::shared_ptr<BundleSession>& session){
    if(!session)return;
    for(const auto directory:session->possibly_dead)
        if(!session->tx.read().directory_find(session->owners,{directory.value,0}))session->tx.drop_directory(directory);
    session->possibly_dead.clear();
}
}
