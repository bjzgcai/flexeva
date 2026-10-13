#include "ground_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <set>
#include <tuple>
#include <type_traits>

namespace flexeva::core::runtime {
namespace detail {
namespace {
constexpr Id ground_format = 0x47524f554e440001ULL;
constexpr Id ground_request_domain = 0x47524f554e445245ULL;
struct Encoder {
    Evidence bytes;
    template<class T> void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto data = std::as_bytes(std::span<const T>(&value, 1));
        bytes.insert(bytes.end(), data.begin(), data.end());
    }
    template<class T> void array(const std::vector<T>& values) {
        pod(static_cast<Id>(values.size()));
        const auto data = std::as_bytes(std::span<const T>(values));
        bytes.insert(bytes.end(), data.begin(), data.end());
    }
};
struct Decoder {
    ras::ByteView owner;
    std::span<const std::byte> bytes;
    explicit Decoder(const ras::Buffer& buffer) {
        if (!buffer || buffer.layout() != ras::Layout{metadata_schema, 1, 1})
            throw ras::Error("invalid Ground fact descriptor layout");
        owner = buffer.bytes();
        bytes = owner.span();
    }
    template<class T> T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        if (bytes.size() < sizeof(T)) throw ras::Error("truncated Ground fact descriptor");
        T value{};
        std::memcpy(&value, bytes.data(), sizeof(T));
        bytes = bytes.subspan(sizeof(T));
        return value;
    }
    template<class T> std::vector<T> array() {
        const auto size = pod<Id>();
        if (size > bytes.size() / sizeof(T)) throw ras::Error("invalid Ground fact descriptor count");
        std::vector<T> values(static_cast<std::size_t>(size));
        if (size) std::memcpy(values.data(), bytes.data(), size * sizeof(T));
        bytes = bytes.subspan(size * sizeof(T));
        return values;
    }
};
struct Descriptor {
    Id request=0, point=0, layer=0, registration=0, window=0, fact=0;
    ras::SchemaId schema=0;
    bool reusable=false;
    std::vector<Id> lanes;
    Evidence query, policy;
};
bool same_request(const Descriptor& a, const Descriptor& b) {
    return a.request == b.request && a.point == b.point && a.layer == b.layer &&
        a.registration == b.registration && a.window == b.window && a.schema == b.schema &&
        a.lanes == b.lanes && a.query == b.query && a.policy == b.policy;
}
ras::Buffer save_descriptor(ras::Transaction& tx, const Descriptor& value) {
    Encoder e;
    e.pod(ground_format); e.pod(value.request); e.pod(value.point); e.pod(value.layer);
    e.pod(value.registration); e.pod(value.window); e.pod(value.fact); e.pod(value.schema);
    e.pod(static_cast<Id>(value.reusable)); e.array(value.lanes); e.array(value.query); e.array(value.policy);
    auto writer = tx.buffer(e.bytes.size(), {metadata_schema, 1, 1});
    auto span = writer.reserve(e.bytes.size());
    std::memcpy(span.bytes().data(), e.bytes.data(), e.bytes.size());
    span.commit();
    return writer.finish();
}
Descriptor load_descriptor(const ras::Buffer& buffer) {
    Decoder d(buffer);
    if (d.pod<Id>() != ground_format) throw ras::Error("invalid Ground fact descriptor format");
    Descriptor result;
    result.request=d.pod<Id>(); result.point=d.pod<Id>(); result.layer=d.pod<Id>();
    result.registration=d.pod<Id>(); result.window=d.pod<Id>(); result.fact=d.pod<Id>();
    result.schema=d.pod<ras::SchemaId>();
    const auto reusable=d.pod<Id>();
    if (reusable > 1) throw ras::Error("invalid Ground fact reuse flag");
    result.reusable = reusable;
    result.lanes=d.array<Id>(); result.query=d.array<std::byte>(); result.policy=d.array<std::byte>();
    if (!d.bytes.empty() || !result.request || !result.point || !result.fact)
        throw ras::Error("invalid Ground fact descriptor");
    return result;
}
ras::DirectoryRecord required(const ras::ReadView& read, ras::DirectoryId dir, Id request, Id slot) {
    const auto record = read.directory_find(dir, {request, slot});
    if (!record) throw ras::Error("incomplete Ground fact record");
    return *record;
}
void require_state(const Access& access, const State& state) {
    if ((state.payload || !state.bundles.empty()) && state.semantic.empty())
        throw ras::Error("Ground state requires complete semantic evidence");
    require_granted(access, state.payload);
    require_bundles(access, state.bundles);
}
void grant_state(Access& access, const State& state) {
    grant(access, state.payload);
    grant_bundles(access, state.bundles);
}
using BufferKey = std::tuple<Id, Id, std::size_t, std::size_t>;
BufferKey key(const ras::Buffer& buffer) {
    const auto id=buffer.identity();
    return {id.store, id.allocation, id.bytes.offset, id.bytes.size};
}
struct CallbackEnd {
    std::shared_ptr<Access> access;
    ~CallbackEnd() { access->scope->live=false; }
};
struct Timer {
    Counters* counters;
    std::chrono::steady_clock::time_point start;
    explicit Timer(Counters* value):counters(value&&value->diagnostics_enabled?value:nullptr),
        start(counters?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{}){}
    ~Timer() {
        if(counters) counters->ground_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
};
} // namespace

struct GroundSession {
    ras::Transaction& tx;
    ras::DirectoryId current;
    ras::ReadView parent;
    ras::DirectoryId parent_facts;
    const std::vector<GroundRegistration>& registrations;
    Id window;
    // A successful equality proof is cached only for immutable storage ranges.
    std::set<std::pair<BufferKey, BufferKey>> equal_buffers;
    struct Fact { Descriptor descriptor; State inputs, value; };
    std::map<Id, Fact> consumed;

    GroundSession(ras::Transaction& transaction, ras::DirectoryId directory, ras::ReadView old,
                  ras::DirectoryId old_facts, const std::vector<GroundRegistration>& rules, Id id)
        : tx(transaction), current(directory), parent(std::move(old)), parent_facts(old_facts),
          registrations(rules), window(id) {}

    bool buffer_equal(const ras::Buffer& a, const ras::Buffer& b) {
        if (a.same_storage(b)) return true;
        if (!a || !b || a.layout()!=b.layout() || a.size()!=b.size()) return false;
        auto pair=std::make_pair(key(a),key(b));
        if(pair.second<pair.first) std::swap(pair.first,pair.second);
        if(equal_buffers.contains(pair)) return true;
        const auto av=a.bytes(), bv=b.bytes();
        if (!std::equal(av.span().begin(),av.span().end(),bv.span().begin())) return false;
        equal_buffers.insert(pair);
        return true;
    }

    ras::Buffer block(const BlockBundleData& data, std::size_t ordinal) const {
        const auto record=tx.read().directory_find(data.directory,{data.id,ordinal});
        if(!record || !record->payload) throw ras::Error("Ground input bundle is incomplete");
        return record->payload;
    }

    bool state_equal(const State& a, const State& b) {
        if(a.semantic!=b.semantic || !buffer_equal(a.payload,b.payload) || a.bundles.size()!=b.bundles.size()) return false;
        for(std::size_t n=0;n<a.bundles.size();++n) {
            const auto* ad=BundleAccess::get(a.bundles[n]);
            const auto* bd=BundleAccess::get(b.bundles[n]);
            if(!ad || !bd) throw ras::Error("Ground state contains an empty bundle");
            if(ad->count!=bd->count || ad->bytes!=bd->bytes || ad->layout!=bd->layout ||
               ad->semantic!=bd->semantic || ad->transform!=bd->transform || ad->boundary!=bd->boundary) return false;
            if(ad->directory==bd->directory && ad->id==bd->id) continue;
            for(std::size_t i=0;i<ad->count;++i) if(!buffer_equal(block(*ad,i),block(*bd,i))) return false;
        }
        return true;
    }

    Fact load(const ras::ReadView& read, ras::DirectoryId dir, Id request, const Descriptor& descriptor) {
        return {descriptor, load_state_record(required(read,dir,request,1)),
                            load_state_record(required(read,dir,request,2))};
    }

    void persist(Access& access, Id row, const Fact& fact) {
        const auto request=row;
        State previous_inputs, previous_value;
        if(tx.read().directory_find(current,{request,0})) {
            previous_inputs=load_state_record(required(tx.read(),current,request,1));
            previous_value=load_state_record(required(tx.read(),current,request,2));
        }
        replace_bundle_refs(access,previous_inputs.bundles,fact.inputs.bundles);
        replace_bundle_refs(access,previous_value.bundles,fact.value.bundles);
        tx.directory_put(current,{request,0},{save_descriptor(tx,fact.descriptor),{}, {}});
        tx.directory_put(current,{request,1},save_state(tx,fact.inputs));
        tx.directory_put(current,{request,2},save_state(tx,fact.value));
    }

    GroundResolution resolve(const std::shared_ptr<Access>& caller, const GroundRequest& request) {
        check(*caller);
        if((caller->phase!=Phase::Capture && caller->phase!=Phase::Partition) || !caller->layer)
            throw ras::Error("Ground is available only during capture or partition");
        if(caller->scope->tx!=&tx) throw ras::Error("Ground session belongs to a different transaction");
        if(!request.request_id || !request.point_id || request.query.empty() || request.requested_lanes.empty() ||
           !std::is_sorted(request.requested_lanes.begin(),request.requested_lanes.end()) ||
           std::adjacent_find(request.requested_lanes.begin(),request.requested_lanes.end())!=request.requested_lanes.end())
            throw ras::Error("Ground request requires an identity, point, query and canonical lanes");
        require_state(*caller,request.inputs);
        auto* counts=caller->scope->counters.get();
        Timer timer{counts};
        auto access=std::make_shared<Access>();
        access->scope=std::make_shared<Scope>();
        access->scope->tx=&tx; access->scope->counters=caller->scope->counters;
        access->bundle_session=caller->bundle_session;
        access->read=tx.read(); access->phase=caller->phase; access->layer=caller->layer;
        grant_state(*access,request.inputs);
        CallbackEnd end{access};
        GroundContext context(access,request.inputs);
        const GroundRegistration* selected=nullptr;
        for(const auto& registration:registrations) {
            if(registration.layer!=caller->layer->id || !registration.matches(request,context)) continue;
            if(selected) throw ras::Error("multiple Ground registrations match this request");
            selected=&registration;
        }
        if(!selected) throw ras::Error("no Ground registration matches this request");
        Descriptor descriptor{request.request_id,request.point_id,caller->layer->id,selected->id,window,0,
                              selected->result_schema,false,request.requested_lanes,request.query,selected->identity};
        const Id coordinate[]{window,caller->layer->id,request.request_id};
        const auto row=tx.intern_identity(ground_request_domain,coordinate).value;
        if(const auto found=consumed.find(row);found!=consumed.end()) {
            if(!same_request(found->second.descriptor,descriptor) || !state_equal(found->second.inputs,request.inputs))
                throw ras::Error("Ground request identity was reused with different inputs");
            if(counts) ++counts->ground_operation_hits;
            grant_state(*caller,found->second.value);
            return {request.request_id,found->second.descriptor.fact,found->second.value,found->second.descriptor.reusable};
        }
        if(parent_facts) {
            if(const auto record=parent.directory_find(parent_facts,{row,0})) {
                auto previous=load_descriptor(record->metadata);
                if(previous.reusable && same_request(previous,descriptor)) {
                    auto fact=load(parent,parent_facts,row,previous);
                    if(state_equal(fact.inputs,request.inputs)) {
                        // The inherited request cache already owns these exact
                        // records. A hit creates no directory or owner updates.
                        if(current!=parent_facts) persist(*caller,row,fact);
                        if(counts) ++counts->ground_parent_hits;
                        grant_state(*caller,fact.value);
                        auto resolution=GroundResolution{request.request_id,previous.fact,fact.value,previous.reusable};
                        consumed.emplace(row,std::move(fact));
                        return resolution;
                    }
                }
            }
        }
        if(counts) ++counts->ground_calls;
        auto result=selected->handler(request,context);
        if(result.request_id!=request.request_id) throw ras::Error("Ground result request identity mismatch");
        require_state(*access,result.value);
        if((!result.value.payload && result.value.bundles.empty()) || result.value.semantic.empty() ||
           (result.value.payload && result.value.payload.layout().schema!=selected->result_schema) ||
           std::any_of(result.value.bundles.begin(),result.value.bundles.end(),[&](const auto& bundle){
               const auto* data=BundleAccess::get(bundle);
               return !data || data->layout.schema!=selected->result_schema;
           }))
            throw ras::Error("Ground result does not match its registered payload schema");
        descriptor.fact=tx.new_change_id().value;
        descriptor.reusable=result.reusable;
        Fact fact{std::move(descriptor),request.inputs,std::move(result.value)};
        persist(*caller,row,fact);
        grant_state(*caller,fact.value);
        auto resolution=GroundResolution{request.request_id,fact.descriptor.fact,fact.value,fact.descriptor.reusable};
        consumed.emplace(row,std::move(fact));
        return resolution;
    }
};

void validate_ground_registrations(const Pipeline& pipeline) {
    std::set<Id> ids;
    for(const auto& registration:pipeline.grounds) {
        if(!registration.id || !ids.insert(registration.id).second || !registration.layer ||
           registration.identity.empty() || !registration.result_schema || !registration.matches || !registration.handler ||
           std::none_of(pipeline.layers.begin(),pipeline.layers.end(),[&](const auto& layer){return layer.id==registration.layer;}))
            throw ras::Error("invalid or duplicate Ground registration");
    }
}

Evidence ground_policy_identity(const std::vector<GroundRegistration>& registrations) {
    if(registrations.empty()) return {};
    std::vector<const GroundRegistration*> sorted;
    for(const auto& registration:registrations) sorted.push_back(&registration);
    std::sort(sorted.begin(),sorted.end(),[](const auto* a,const auto* b){return a->id<b->id;});
    Encoder e;
    e.pod(ground_format); e.pod(static_cast<Id>(sorted.size()));
    for(const auto* registration:sorted) {
        e.pod(registration->id); e.pod(registration->layer);
        e.pod(registration->result_schema); e.array(registration->identity);
    }
    return e.bytes;
}

std::shared_ptr<GroundSession> make_ground_session(ras::Transaction& tx, ras::DirectoryId current,
        ras::ReadView parent, ras::DirectoryId parent_facts, const std::vector<GroundRegistration>& registrations, Id window) {
    if(!current && !registrations.empty()) throw ras::Error("Ground session requires a fact directory");
    return std::make_shared<GroundSession>(tx,current,std::move(parent),parent_facts,registrations,window);
}
} // namespace detail

GroundContext::GroundContext(std::shared_ptr<detail::Access> access, State inputs)
    : access_(std::move(access)), inputs_(std::move(inputs)) {}
const State& GroundContext::inputs() const { detail::check(*access_); return inputs_; }
OutputWriter GroundContext::buffer(std::size_t size, ras::Layout layout) { return Context(access_).buffer(size,layout); }
ras::Buffer GroundContext::copy(std::span<const std::byte> bytes, ras::SchemaId schema) { return Context(access_).copy(bytes,schema); }
BlockWriter GroundContext::blocks(std::size_t size, ras::Layout layout) { return Context(access_).blocks(size,layout); }
ras::Buffer GroundContext::block(const BlockBundle& bundle, std::size_t ordinal) const { return Context(access_).block(bundle,ordinal); }

GroundResolution Context::ground(const GroundRequest& request) {
    if(!access_) throw ras::Error("empty Core context");
    detail::check(*access_);
    try {
        if(!access_->ground_session) throw ras::Error("Core context has no Ground session");
        return access_->ground_session->resolve(access_,request);
    } catch(...) {
        // Fail closed even when evaluator code catches the exception. Publication
        // must never make a candidate from a failed numerical fact computation.
        if(access_->scope->tx) access_->scope->tx->abort();
        throw;
    }
}
} // namespace flexeva::core::runtime
