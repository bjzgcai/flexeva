#include "runtime_internal.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <utility>

namespace flexeva::core::runtime {
BlockBundle::BlockBundle(std::shared_ptr<const detail::BlockBundleData> data):data_(std::move(data)){}
BlockBundle::operator bool()const noexcept{return bool(data_);}
std::size_t BlockBundle::count()const noexcept{return data_?data_->count:0;}
std::size_t BlockBundle::size()const noexcept{return data_?data_->bytes:0;}
namespace detail {
namespace {
template<class T> void put(Evidence& out,const T& value){auto bytes=std::as_bytes(std::span(&value,1));out.insert(out.end(),bytes.begin(),bytes.end());}
template<class T> T take(std::span<const std::byte>& in){if(in.size()<sizeof(T))throw ras::Error("truncated Core bundle metadata");T result;std::memcpy(&result,in.data(),sizeof(T));in=in.subspan(sizeof(T));return result;}
}
Evidence encode_bundles(const std::vector<BlockBundle>& values){
    Evidence out;put(out,std::uint64_t(values.size()));
    for(const auto& value:values){const auto* d=BundleAccess::get(value);if(!d)throw ras::Error("empty Core bundle reference");
        put(out,d->directory);put(out,d->id);put(out,std::uint64_t(d->count));put(out,std::uint64_t(d->bytes));put(out,d->layout);put(out,d->transform);put(out,d->boundary);put(out,d->operation);put(out,std::uint64_t(d->semantic.size()));out.insert(out.end(),d->semantic.begin(),d->semantic.end());}
    return out;
}
std::vector<BlockBundle> decode_bundles(std::span<const std::byte> in){
    const auto n=take<std::uint64_t>(in);if(n>in.size()/sizeof(Id))throw ras::Error("invalid Core bundle count");std::vector<BlockBundle> result;
    for(std::uint64_t i=0;i<n;++i){auto d=std::make_shared<BlockBundleData>();d->directory=take<ras::DirectoryId>(in);d->id=take<Id>(in);d->count=take<std::uint64_t>(in);d->bytes=take<std::uint64_t>(in);d->layout=take<ras::Layout>(in);d->transform=take<Id>(in);d->boundary=take<Id>(in);d->operation=take<Id>(in);auto size=take<std::uint64_t>(in);if(size>in.size()||!d->directory||!d->id)throw ras::Error("invalid Core bundle descriptor");d->semantic.assign(in.begin(),in.begin()+size);in=in.subspan(size);result.push_back(BundleAccess::make(d));}
    if(!in.empty())throw ras::Error("trailing Core bundle metadata");
    return result;
}
bool same_bundles(const std::vector<BlockBundle>& a,const std::vector<BlockBundle>& b){return encode_bundles(a)==encode_bundles(b);}
void grant_bundle(Access& access,const BlockBundle& bundle){
    check(access);const auto* d=BundleAccess::get(bundle);if(!d)throw ras::Error("empty Core bundle capability");
    const auto stored=access.read.directory_find(d->directory,{d->id,maximum});
    if(!stored||!stored->metadata)throw ras::Error("unsealed or foreign Core bundle");
    const auto expected=encode_bundles({bundle});const auto actual=stored->metadata.bytes();
    if(actual.size()!=expected.size()||std::memcmp(actual.span().data(),expected.data(),expected.size()))throw ras::Error("Core bundle descriptor mismatch");
    access.bundle_grants->insert({d->directory,d->id});
}
void require_bundle(const Access& access,const BlockBundle& bundle){check(access);const auto* d=BundleAccess::get(bundle);if(!d||!access.bundle_grants||!access.bundle_grants->contains({d->directory,d->id}))throw ras::Error("Core bundle is outside callback capabilities");}
void grant_bundles(Access& a,const std::vector<BlockBundle>& b){for(const auto& v:b)grant_bundle(a,v);}
void require_bundles(const Access& a,const std::vector<BlockBundle>& b){for(const auto& v:b)require_bundle(a,v);}
ras::Buffer read_block(Access& a,const BlockBundle& bundle,std::size_t ordinal){
    require_bundle(a,bundle);const auto& d=*BundleAccess::get(bundle);if(ordinal>=d.count)throw ras::Error("Core bundle block ordinal out of range");auto value=a.read.directory_find(d.directory,{d.id,ordinal});if(!value||!value->payload||value->payload.layout()!=d.layout)throw ras::Error("missing or invalid Core bundle block");grant(a,value->payload);return value->payload;
}
}
namespace {
void require_outgoing(const detail::Access& access,Id transform){
    detail::check(access);
    if(!access.outgoing_transform||access.outgoing_transform->id!=transform)
        throw ras::Error("output access requires this layer's registered outgoing transform");
}
void require_capture_partition(const detail::Access& access){
    detail::check(access);
    if(access.phase!=detail::Phase::Capture&&access.phase!=detail::Phase::Partition)
        throw ras::Error("output staging and parent lookup require Capture or Partition");
}
void grant_receipt(detail::Access& access,const Receipt& receipt){
    detail::grant(access,receipt.payload);detail::grant(access,receipt.state.payload);
    detail::grant_bundles(access,detail::receipt_bundles(receipt));
    detail::grant_bundles(access,receipt.state.bundles);
}
std::optional<PreparedOutput> staged(detail::Access& access,Id transform,Id boundary,const Evidence* semantic){
    if(!access.prepared_session)throw ras::Error("prepared output has no operation scope");
    const auto found=access.prepared_session->outputs.find({transform,boundary});
    if(found==access.prepared_session->outputs.end())return {};
    if(semantic&&found->second.semantic!=*semantic)
        throw ras::Error("prepared output does not match Apply Boundary semantic evidence");
    auto result=found->second.output;grant_receipt(access,result.receipt);
    if(result.body)detail::grant_bundle(access,result.body);
    return result;
}
}
struct BlockWriter::Impl {
    std::shared_ptr<detail::Access> access;std::shared_ptr<detail::BlockBundleData> output;ras::BufferWriter writer;std::size_t block_bytes;bool active=false,finished=false;
    Impl(std::shared_ptr<detail::Access> a,std::size_t bytes,ras::Layout layout,Id transform,Id boundary,Evidence semantic):access(std::move(a)),output(std::make_shared<detail::BlockBundleData>()),block_bytes(bytes){
        detail::check(*access);if(!access->scope->tx)throw ras::Error("block output requires a write callback");if(!bytes||!layout.element_size||!std::has_single_bit(layout.alignment)||layout.element_size%layout.alignment)throw ras::Error("invalid Core block layout");
        auto rem=block_bytes%layout.element_size;if(rem){auto extra=layout.element_size-rem;if(block_bytes>std::numeric_limits<std::size_t>::max()-extra)throw ras::Error("Core block size overflow");block_bytes+=extra;}
        if((transform||boundary)&&(access->phase!=detail::Phase::Capture&&access->phase!=detail::Phase::Partition))throw ras::Error("prepared outputs require Capture or Partition");
        if((transform||boundary)&&(!transform||!boundary||semantic.empty()))throw ras::Error("prepared output requires transform, Boundary and semantic evidence");
        if(transform)require_outgoing(*access,transform);
        output->directory=access->scope->tx->add_directory();detail::track_bundle_directory(*access,output->directory);output->id=access->scope->tx->new_change_id().value;output->layout=layout;output->transform=transform;output->boundary=boundary;output->semantic=std::move(semantic);output->operation=transform?access->operation:0;
    }
    void check()const{detail::check(*access);if(finished)throw ras::Error("Core block writer already finished");}
    void append(const ras::Buffer& buffer){if(!buffer||!buffer.size()||buffer.layout()!=output->layout)throw ras::Error("Core appended block has incompatible layout");if(buffer.size()>std::numeric_limits<std::size_t>::max()-output->bytes||output->count==detail::maximum)throw ras::Error("Core block size overflow");access->scope->tx->directory_put(output->directory,{output->id,output->count},{{},buffer,{}});detail::count_block_write(access->scope->counters.get());++output->count;output->bytes+=buffer.size();}
    void seal(){if(!active)return;auto bytes=writer.finish_prefix();active=false;if(bytes.size())append(bytes);}
};
BlockWriter::BlockWriter(std::shared_ptr<detail::Access> a,std::size_t bytes,ras::Layout layout,Id transform,Id boundary,Evidence semantic):impl_(std::make_unique<Impl>(std::move(a),bytes,layout,transform,boundary,std::move(semantic))){}
BlockWriter::BlockWriter(BlockWriter&&)noexcept=default;BlockWriter& BlockWriter::operator=(BlockWriter&&)noexcept=default;BlockWriter::~BlockWriter()=default;
ras::WriteSpan BlockWriter::reserve(std::size_t bytes){if(!impl_)throw ras::Error("empty Core block writer");impl_->check();if(!bytes||bytes%impl_->output->layout.element_size)throw ras::Error("Core block reservation splits an element");if(impl_->active&&bytes>impl_->writer.capacity()-impl_->writer.size())impl_->seal();if(!impl_->active){impl_->writer=impl_->access->scope->tx->buffer(std::max(bytes,impl_->block_bytes),impl_->output->layout);impl_->active=true;}return impl_->writer.reserve(bytes);}
ras::WriteSpan BlockWriter::reserve(std::size_t bytes,OutputSlice& location){auto result=reserve(bytes);location={impl_->output->count,result.offset(),bytes};return result;}
ras::WriteSpan BlockWriter::reserve_exact(std::size_t bytes){seal();if(!bytes||bytes%impl_->output->layout.element_size)throw ras::Error("invalid Core exact block reservation");impl_->writer=impl_->access->scope->tx->buffer(bytes,impl_->output->layout);impl_->active=true;return impl_->writer.reserve(bytes);}
void BlockWriter::append(const ras::Buffer& buffer){if(!impl_)throw ras::Error("empty Core block writer");impl_->check();detail::require_granted(*impl_->access,buffer);impl_->seal();impl_->append(buffer);}
void BlockWriter::seal(){if(!impl_)throw ras::Error("empty Core block writer");impl_->check();impl_->seal();}
std::size_t BlockWriter::size()const{if(!impl_)throw ras::Error("empty Core block writer");impl_->check();return impl_->output->bytes+(impl_->active?impl_->writer.size():0);}
BlockBundle BlockWriter::finish(){seal();impl_->finished=true;auto result=detail::BundleAccess::make(impl_->output);auto descriptor=detail::encode_bundles({result});auto writer=impl_->access->scope->tx->buffer(descriptor.size(),{detail::metadata_schema,1,1});auto bytes=writer.reserve(descriptor.size());std::memcpy(bytes.bytes().data(),descriptor.data(),descriptor.size());bytes.commit();impl_->access->scope->tx->directory_put(impl_->output->directory,{impl_->output->id,detail::maximum},{writer.finish(),{}, {}});detail::grant_bundle(*impl_->access,result);return result;}
BlockWriter Context::blocks(std::size_t bytes,ras::Layout layout){detail::check(*access_);return BlockWriter(access_,bytes,layout,0,0,{});}
BlockWriter Context::prepare(Id transform,Id boundary,Evidence semantic,std::size_t bytes,ras::Layout layout){detail::check(*access_);return BlockWriter(access_,bytes,layout,transform,boundary,std::move(semantic));}
std::optional<Receipt> Context::parent_output(Id transform,Id boundary)const{
    if(!access_)throw ras::Error("empty Core context");
    require_capture_partition(*access_);require_outgoing(*access_,transform);
    if(!boundary)throw ras::Error("parent output requires a Boundary identity");
    if(!access_->previous_outgoing_transform||!access_->old.directory_find(access_->previous_outgoing_transform->receipts,{boundary,0}))return {};
    auto result=detail::load_receipt(access_->old,*access_->previous_outgoing_transform,boundary,nullptr,access_->scope->counters.get());
    grant_receipt(*access_,result);return result;
}
void Context::stage_output(Id transform,Id boundary,Evidence semantic,Receipt receipt,BlockBundle body){
    if(!access_)throw ras::Error("empty Core context");
    require_capture_partition(*access_);require_outgoing(*access_,transform);
    if(!boundary||receipt.boundary!=boundary||semantic.empty()||receipt.chunks)
        throw ras::Error("staged output requires a Boundary, semantic evidence and unbound receipt");
    if(!access_->prepared_session)throw ras::Error("prepared output has no operation scope");
    detail::require_granted(*access_,receipt.payload);detail::require_granted(*access_,receipt.state.payload);
    detail::require_bundles(*access_,receipt.bundles);detail::require_bundles(*access_,receipt.state.bundles);
    if((receipt.state.payload||!receipt.state.bundles.empty())&&receipt.state.semantic.empty())
        throw ras::Error("staged receipt State payload requires semantic evidence");
    if(body)detail::require_bundle(*access_,body);
    if(!access_->prepared_session->outputs.emplace(std::pair{transform,boundary},detail::PreparedEntry{std::move(semantic),{std::move(receipt),std::move(body)}}).second)
        throw ras::Error("Boundary output was already staged in this operation");
}
std::optional<PreparedOutput> Context::prepared_output(Id transform,Id boundary)const{
    if(!access_)throw ras::Error("empty Core context");
    detail::check(*access_);
    if(access_->phase!=detail::Phase::Capture&&access_->phase!=detail::Phase::Partition&&access_->phase!=detail::Phase::Boundary)
        throw ras::Error("indexed prepared lookup requires Capture, Partition or Boundary");
    require_outgoing(*access_,transform);
    if(!boundary)throw ras::Error("prepared output requires a Boundary identity");
    return staged(*access_,transform,boundary,nullptr);
}
std::optional<PreparedOutput> Context::prepared_output()const{
    if(!access_)throw ras::Error("empty Core context");
    detail::check(*access_);
    if(access_->phase!=detail::Phase::Apply||!access_->isolated||!access_->transform)
        throw ras::Error("prepared output requires isolated Apply");
    return staged(*access_,access_->transform->id,access_->isolated->id,&access_->isolated->semantic);
}
ras::Buffer Context::block(const BlockBundle& bundle,std::size_t ordinal)const{if(!access_)throw ras::Error("empty Core context");return detail::read_block(*access_,bundle,ordinal);}
ras::Buffer Context::block(const BlockBundle& bundle,const OutputSlice& location)const{if(!location.bytes)throw ras::Error("empty Core bundle slice");return block(bundle,location.block).slice({location.offset,location.bytes});}
ChunkOutput Context::adopt_prepared(const BlockBundle& bundle){
    detail::AdmissionAudit audit(access_?access_->scope->counters.get():nullptr,true);
    if(!access_)throw ras::Error("empty Core context");
    detail::check(*access_);if(access_->phase!=detail::Phase::Apply||!access_->isolated||!access_->transform)throw ras::Error("prepared adoption requires isolated Apply");detail::require_bundle(*access_,bundle);const auto& d=*detail::BundleAccess::get(bundle);
    if(d.transform!=access_->transform->id||d.boundary!=access_->isolated->id||d.semantic!=access_->isolated->semantic||d.operation!=access_->operation)throw ras::Error("prepared output does not match Apply Boundary");
    auto output=std::make_shared<detail::ChunkOutputData>(detail::ChunkOutputData{d.boundary,d.count,d.bytes,true,d.layout,bundle});if(!access_->chunk_outputs->emplace(d.boundary,output).second)throw ras::Error("Boundary already opened output");
    ++access_->scope->counters->prepared_adoptions;++access_->scope->counters->output_root_bindings;return detail::ChunkAccess::make(output);
}
ChunkOutput Context::adopt_output(const BlockBundle& bundle){
    detail::AdmissionAudit audit(access_?access_->scope->counters.get():nullptr,true);
    if(!access_)throw ras::Error("empty Core context");
    detail::check(*access_);if(access_->phase!=detail::Phase::Apply||!access_->isolated||!access_->transform)throw ras::Error("output adoption requires isolated Apply");
    detail::require_bundle(*access_,bundle);const auto& d=*detail::BundleAccess::get(bundle);
    const auto id=access_->isolated->id;
    auto output=std::make_shared<detail::ChunkOutputData>(detail::ChunkOutputData{id,d.count,d.bytes,true,d.layout,bundle});
    if(!access_->chunk_outputs->emplace(id,output).second)throw ras::Error("Boundary already opened output");
    ++access_->scope->counters->shared_output_adoptions;++access_->scope->counters->output_root_bindings;return detail::ChunkAccess::make(output);
}
std::optional<Receipt> Context::previous_output()const{
    if(!access_)throw ras::Error("empty Core context");
    detail::check(*access_);if(access_->phase!=detail::Phase::Apply||!access_->isolated)throw ras::Error("previous output requires isolated Apply");if(!access_->previous_transform||!access_->old.directory_find(access_->previous_transform->receipts,{access_->isolated->id,0}))return {};
    auto result=detail::load_receipt(access_->old,*access_->previous_transform,access_->isolated->id,nullptr,access_->scope->counters.get());detail::grant(*access_,result.payload);detail::grant(*access_,result.state.payload);detail::grant_bundles(*access_,detail::receipt_bundles(result));detail::grant_bundles(*access_,result.state.bundles);return result;
}
ras::Buffer Context::previous_output_chunk(std::size_t ordinal)const{
    if(!access_)throw ras::Error("empty Core context");
    detail::check(*access_);if(access_->phase!=detail::Phase::Apply||!access_->isolated)throw ras::Error("previous output requires isolated Apply");
    if(!access_->previous_transform||!access_->old.directory_find(access_->previous_transform->receipts,{access_->isolated->id,0}))throw ras::Error("previous output chunk ordinal out of range");
    const auto previous=detail::load_receipt_access(access_->old,*access_->previous_transform,access_->isolated->id,access_->scope->counters.get());
    detail::grant(*access_,previous.payload);detail::grant(*access_,previous.state.payload);detail::grant_bundles(*access_,detail::receipt_bundles(previous));detail::grant_bundles(*access_,previous.state.bundles);
    if(ordinal>=previous.chunks.count())throw ras::Error("previous output chunk ordinal out of range");
    auto parent=*access_;parent.read=access_->old;return detail::read_block(parent,previous.chunks.storage(),ordinal);
}
ChunkOutput Context::adopt_previous_output(){detail::AdmissionAudit audit(access_?access_->scope->counters.get():nullptr,true);auto previous=previous_output();if(!previous)throw ras::Error("Boundary has no previous output");if(!previous->chunks)return {};
    const auto* data=detail::ChunkAccess::get(previous->chunks);auto output=std::make_shared<detail::ChunkOutputData>(*data);if(!access_->chunk_outputs->emplace(output->boundary,output).second)throw ras::Error("Boundary already opened output");
    ++access_->scope->counters->previous_output_adoptions;++access_->scope->counters->output_root_bindings;return detail::ChunkAccess::make(output);
}
}
