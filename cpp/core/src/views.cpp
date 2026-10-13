#include "runtime_internal.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace flexeva::core::runtime {
namespace {
using detail::Access;
using detail::Phase;

Access& checked(const std::shared_ptr<Access>& access) {
    if (!access) throw ras::Error("empty Core context or view");
    detail::check(*access);
    return *access;
}

Counters* counters(const Access& access) {
    auto* value=access.scope->counters.get();
    return value&&value->diagnostics_enabled?value:nullptr;
}

const detail::LayerData& layer(const Access& access) {
    if (!access.layer) throw ras::Error("Core view has no layer");
    return *access.layer;
}

const detail::TransformData& transform(const Access& access) {
    if (!access.transform) throw ras::Error("Core view has no transform");
    return *access.transform;
}

bool member(const Access& access, Id id) {
    if (!access.restricted) return true;
    return access.isolated && access.transform &&
        bool(access.read.directory_find(access.transform->members, {access.isolated->id, id}));
}

void require_member(const Access& access, Id id) {
    if (!member(access, id)) throw ras::Error("partition is outside declared Boundary inputs");
}

void require_metadata(const Access& access, ras::DirectoryId directory = {}) {
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access evaluator metadata indexes");
    if (!access.scope->tx) throw ras::Error("read-only Core context cannot access evaluator metadata indexes");
    if (directory && (!access.evaluator_directories || !access.evaluator_directories->contains(directory)))
        throw ras::Error("evaluator metadata directory is outside this callback's capabilities");
}

void grant_record(Access& access, const ras::DirectoryRecord& record) {
    detail::grant(access, record.metadata);
    detail::grant(access, record.payload);
    detail::grant(access, record.auxiliary);
}

void require_unrestricted(const Access& access) {
    if (access.restricted)
        throw ras::Error("whole-layer query is outside declared Boundary inputs");
}

std::shared_ptr<Access> isolate(const std::shared_ptr<Access>& source, Id id) {
    auto& access = checked(source);
    if (access.phase != Phase::Apply)
        throw ras::Error("Boundary isolation is available only during Apply");
    if (access.isolated) {
        if (access.isolated->id != id)
            throw ras::Error("isolated Apply context cannot access another Boundary");
        return source;
    }
    if (!access.selected) throw ras::Error("Apply has no selected Boundaries");
    const auto& selected = *access.selected;
    const auto found = std::lower_bound(selected.begin(), selected.end(), id,
        [](const Boundary& value, Id key) { return value.id < key; });
    if (found == selected.end() || found->id != id)
        throw ras::Error("Boundary was not selected for Apply");
    auto result = std::make_shared<Access>(access);
    result->isolated = Boundary{found->id, found->owner, {}, found->payload, found->semantic,
                                found->state, found->consumes_layer_state, found->parent_receipts, found->bundles};
    result->restricted = true;
    result->allow_state = found->consumes_layer_state;
    auto& grants = (*access.boundary_grants)[id];
    if (!grants) grants = std::make_shared<detail::Grants>();
    result->grants = grants;
    auto& bundle_grants = (*access.boundary_bundle_grants)[id];
    if (!bundle_grants) bundle_grants = std::make_shared<std::set<std::pair<ras::DirectoryId,Id>>>();
    result->bundle_grants = bundle_grants;
    detail::grant_bundles(*result, found->bundles);
    detail::grant(*result, found->payload);
    return result;
}

std::shared_ptr<Access> effective(const std::shared_ptr<Access>& source) {
    auto& access = checked(source);
    if (access.phase != Phase::Apply || access.isolated) return source;
    if (!access.selected || access.selected->size() != 1)
        throw ras::Error("batched Apply requires for_boundary before accessing inputs or outputs");
    return isolate(source, access.selected->front().id);
}

std::vector<ras::Buffer> output_slices(Access& access, const Receipt& receipt,
                                     std::span<const OutputSlice> slices) {
    if (slices.empty()) throw ras::Error("output range must contain at least one slice");
    if (!receipt.chunks) throw ras::Error("output range requires a sealed output body");
    std::vector<ras::Buffer> result;
    result.reserve(slices.size());
    std::optional<OutputSlice> previous;
    for (const auto& slice : slices) {
        if (!slice.bytes || slice.block >= receipt.chunks.count())
            throw ras::Error("output range has an empty or absent block slice");
        if (previous && (slice.block < previous->block ||
            (slice.block == previous->block && slice.offset < previous->offset + previous->bytes)))
            throw ras::Error("output range slices must be ordered and disjoint");
        const auto block = detail::read_block(access, receipt.chunks.storage(), slice.block);
        const auto layout = block.layout();
        if (slice.offset > block.size() || slice.bytes > block.size() - slice.offset ||
            slice.offset % layout.element_size || slice.bytes % layout.element_size ||
            slice.offset % layout.alignment)
            throw ras::Error("output range slice is outside its block or element alignment");
        result.push_back(block.slice({slice.offset, slice.bytes}));
        previous = slice;
    }
    return result;
}
} // namespace

namespace detail {
void check(const Access& access) {
    if (!access.scope || !access.scope->live)
        throw ras::Error("Core context or view escaped its callback scope");
}

void grant(Access& access, const ras::Buffer& buffer) {
    check(access);
    if (!buffer) return;
    const auto identity = buffer.identity();
    if (!identity.store || !identity.allocation)
        throw ras::Error("invalid RAS buffer identity");
    if (identity.bytes.size > std::numeric_limits<std::size_t>::max() - identity.bytes.offset)
        throw ras::Error("RAS buffer capability range overflow");
    if (!access.grants) access.grants = std::make_shared<Grants>();
    auto& ranges = (*access.grants)[{identity.store, identity.allocation}];
    auto begin = identity.bytes.offset;
    auto end = begin + identity.bytes.size;
    auto first = ranges.begin();
    while (first != ranges.end() && first->offset + first->size < begin) ++first;
    auto last = first;
    while (last != ranges.end() && last->offset <= end) {
        begin = std::min(begin, last->offset);
        end = std::max(end, last->offset + last->size);
        ++last;
    }
    first = ranges.erase(first, last);
    ranges.insert(first, {begin, end - begin});
}

void grant_fresh(Access& access, const ras::Buffer& buffer) {
    grant(access, buffer);
}

bool granted(const Access& access, const ras::Buffer& buffer) {
    check(access);
    if (!buffer) return true;
    if (!access.grants) return false;
    const auto identity = buffer.identity();
    const auto found = access.grants->find({identity.store, identity.allocation});
    if (found == access.grants->end()) return false;
    for (const auto range : found->second) {
        if (identity.bytes.offset < range.offset) break;
        const auto offset = identity.bytes.offset - range.offset;
        if (offset <= range.size && identity.bytes.size <= range.size - offset) return true;
    }
    return false;
}

void require_granted(const Access& access, const ras::Buffer& buffer) {
    if (!granted(access, buffer))
        throw ras::Error("result contains a RAS buffer outside its callback capabilities");
}
} // namespace detail

LayerView::LayerView(std::shared_ptr<Access> access) : access_(std::move(access)) {}

LayerView::operator bool() const noexcept {
    return access_ && access_->scope && access_->scope->live && bool(access_->layer);
}

PartitionHead LayerView::head(Id id) const {
    auto& access = checked(access_);
    require_member(access, id);
    return detail::load_partition_head(access.read, layer(access), id, counters(access));
}

void LayerView::partitions_from(Id origin, const std::function<void(Id)>& visit) const {
    auto& access = checked(access_);
    require_unrestricted(access);
    if (!access.layer) return;
    access.read.directory_range(layer(access).origins, {origin, 0}, {origin, detail::maximum},
        [&](ras::DirectoryKey key, const ras::DirectoryRecord&) {
            if (auto* value = counters(access)) ++value->metadata_reads;
            visit(key.second);
        });
}

bool LayerView::origin_contains(Id id, Id origin) const {
    auto& access = checked(access_);
    require_member(access, id);
    if (!access.layer) return false;
    if (auto* value = counters(access)) ++value->metadata_reads;
    return bool(access.read.directory_find(layer(access).origins, {origin, id}));
}

Partition LayerView::partition(Id id) const {
    auto& access = checked(access_);
    require_member(access, id);
    auto result = detail::load_partition(access.read, layer(access), id, counters(access));
    detail::grant(access, result.payload);
    detail::grant_bundles(access, result.bundles);
    return result;
}

ras::Buffer LayerView::payload(Id id) const {
    auto& access = checked(access_);
    require_member(access, id);
    const auto record = access.read.directory_find(layer(access).parts, {id, 0});
    if (!record) throw ras::Error("missing Core metadata record");
    detail::grant(access, record->payload);
    if (auto* value = counters(access)) { ++value->payload_reads; ++value->metadata_reads; }
    return record->payload;
}

ras::Buffer LayerView::block(const BlockBundle& bundle, std::size_t ordinal) const {
    auto& access = checked(access_);
    auto result = detail::read_block(access, bundle, ordinal);
    if (auto* value = counters(access)) { ++value->payload_reads; ++value->metadata_reads; }
    return result;
}

ras::Buffer LayerView::block(const BlockBundle& bundle, const OutputSlice& location) const {
    if (!location.bytes) throw ras::Error("empty Core bundle slice");
    return block(bundle, location.block).slice({location.offset, location.bytes});
}

bool LayerView::contains(Id id) const {
    auto& access = checked(access_);
    if (!member(access, id)) return false;
    if (!access.layer) return false;
    if (auto* value = counters(access)) ++value->metadata_reads;
    return bool(access.read.directory_find(layer(access).parts, {id, 0}));
}

std::size_t LayerView::size() const {
    auto& access = checked(access_);
    require_unrestricted(access);
    if (!access.layer) return 0;
    if (auto* value = counters(access)) ++value->metadata_reads;
    return access.read.order_size(layer(access).order);
}

Id LayerView::at(std::size_t index) const {
    auto& access = checked(access_);
    require_unrestricted(access);
    if (auto* value = counters(access)) ++value->metadata_reads;
    return access.read.order_at(layer(access).order, index);
}

std::size_t LayerView::rank(Id id) const {
    auto& access = checked(access_);
    require_unrestricted(access);
    if (auto* value = counters(access)) ++value->metadata_reads;
    return access.read.order_rank(layer(access).order, id);
}

ras::OrderStats LayerView::order_stats() const {
    auto& access = checked(access_);
    require_unrestricted(access);
    if (!access.layer) return {};
    if (auto* value = counters(access)) ++value->metadata_reads;
    return access.read.order_stats(layer(access).order);
}

void LayerView::partitions(const std::function<void(const Partition&)>& visit) const {
    auto& access = checked(access_);
    require_unrestricted(access);
    if (!access.layer) return;
    const auto order = layer(access).order;
    if (auto* value = counters(access)) ++value->full_enumerations;
    access.read.order_range(order, 0, access.read.order_size(order),
        [&](Id id) { visit(partition(id)); });
}

void LayerView::consumers(Id id, const std::function<void(Id)>& visit) const {
    auto& access = checked(access_);
    require_unrestricted(access);
    if (!contains(id)) throw ras::Error("dependency producer is not in this layer");
    access.read.directory_range(layer(access).dependencies, {id, 0}, {id, detail::maximum},
        [&](ras::DirectoryKey key, const ras::DirectoryRecord&) {
            if (auto* value = counters(access)) ++value->dependency_visits;
            visit(key.second);
        });
}

void LayerView::producers(Id id, const std::function<void(Id)>& visit) const {
    auto& access = checked(access_);
    require_member(access, id);
    if (!contains(id)) throw ras::Error("dependency consumer is not in this layer");
    access.read.directory_range(layer(access).producers, {id, 0}, {id, detail::maximum},
        [&](ras::DirectoryKey key, const ras::DirectoryRecord&) {
            if (auto* value = counters(access)) ++value->dependency_visits;
            visit(key.second);
        });
}

State LayerView::state() const {
    auto& access = checked(access_);
    if (!access.allow_state)
        throw ras::Error("Boundary did not declare layer state as an input");
    if (!access.layer) return {};
    auto result = detail::load_state(access.read, layer(access), counters(access));
    detail::grant(access, result.payload);
    detail::grant_bundles(access, result.bundles);
    if (result.payload) if (auto* value = counters(access)) ++value->payload_reads;
    return result;
}

BoundaryView::BoundaryView(std::shared_ptr<Access> access) : access_(std::move(access)) {}

BoundaryHead BoundaryView::head(Id id) const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access the transform Boundary index");
    return detail::load_boundary_head(access.read, transform(access), id, counters(access));
}

void BoundaryView::inputs(Id id, const std::function<void(Id)>& visit) const {
    auto& access = checked(access_);
    if (!contains(id)) throw ras::Error("missing Core boundary");
    access.read.directory_range(transform(access).members, {id, 0}, {id, detail::maximum},
        [&](ras::DirectoryKey key, const ras::DirectoryRecord&) {
            if (auto* value = counters(access)) ++value->metadata_reads;
            visit(key.second);
        });
}

bool BoundaryView::has_input(Id id, Id input) const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access the transform Boundary index");
    if (!access.transform) return false;
    if (auto* value = counters(access)) ++value->metadata_reads;
    return bool(access.read.directory_find(transform(access).members, {id, input}));
}

Boundary BoundaryView::boundary(Id id) const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access the transform Boundary index");
    auto result = detail::load_boundary(access.read, transform(access), id, counters(access));
    detail::grant(access, result.payload);
    detail::grant_bundles(access, result.bundles);
    return result;
}

bool BoundaryView::contains(Id id) const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access the transform Boundary index");
    if (!access.transform) return false;
    if (auto* value = counters(access)) ++value->metadata_reads;
    return bool(access.read.directory_find(transform(access).boundaries, {id, 0}));
}

void BoundaryView::boundaries(const std::function<void(const Boundary&)>& visit) const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot enumerate transform Boundaries");
    if (!access.transform) return;
    if (auto* value = counters(access)) ++value->full_enumerations;
    access.read.directory_range(transform(access).boundaries, {0, 0}, {detail::maximum, detail::maximum},
        [&](ras::DirectoryKey key, const ras::DirectoryRecord&) { visit(boundary(key.first)); });
}

void BoundaryView::downstream(Id producer,const std::function<void(const OutputTarget&)>& visit) const {
    auto& access=checked(access_);
    if(access.phase==Phase::Apply)throw ras::Error("Apply cannot access downstream AL");
    if(access.transform)detail::downstream(access.read,*access.transform,producer,visit,counters(access));
}
void BoundaryView::partitions_from(Id boundary,const std::function<void(Id)>& visit) const {
    auto& access=checked(access_);
    if(access.phase==Phase::Apply)throw ras::Error("Apply cannot access downstream AL");
    if(!access.transform||!access.lower_layer)return;
    access.read.directory_range(access.lower_layer->origins,{boundary,0},{boundary,detail::maximum},
        [&](ras::DirectoryKey key,const ras::DirectoryRecord&){if(auto* c=counters(access))++c->metadata_reads;visit(key.second);});
}
Receipt BoundaryView::output(Id id) const {
    auto& access=checked(access_);
    if(access.phase==Phase::Apply)throw ras::Error("Apply cannot access unrelated outputs");
    auto result=detail::load_receipt(access.read,transform(access),id,nullptr,counters(access));
    detail::grant(access,result.payload);detail::grant(access,result.state.payload);
    detail::grant_bundles(access,detail::receipt_bundles(result));detail::grant_bundles(access,result.state.bundles);
    return result;
}
ras::Buffer BoundaryView::output_chunk(Id id,std::size_t ordinal) const {
    auto& access=checked(access_);const auto receipt=output(id);
    return detail::read_block(access,receipt.chunks.storage(),ordinal);
}
std::vector<ras::Buffer> BoundaryView::output_range(Id id,std::span<const OutputSlice> slices) const {
    auto& access=checked(access_);
    return output_slices(access,output(id),slices);
}
void BoundaryView::outputs(const std::function<void(const Receipt&)>& visit) const {
    auto& access=checked(access_);
    if(access.phase==Phase::Apply)throw ras::Error("Apply cannot enumerate unrelated outputs");
    if(!access.transform)return;
    if(auto* c=counters(access))++c->full_enumerations;
    access.read.directory_range(access.transform->receipts,{0,0},{detail::maximum,detail::maximum},
        [&](ras::DirectoryKey key,const ras::DirectoryRecord&){visit(output(key.first));});
}
void BoundaryView::lineage_to(Id consumer,const std::function<void(const Lineage&)>& visit) const {
    auto& access=checked(access_);
    if(access.phase==Phase::Apply)throw ras::Error("Apply cannot access downstream AL");
    if(!access.transform||!access.lower_layer||!access.read.directory_find(access.lower_layer->parts,{consumer,0}))return;
    const auto part=detail::load_partition(access.read,*access.lower_layer,consumer,counters(access));
    for(auto origin:part.origins){const auto b=head(origin);visit({0,origin,b.owner,consumer,LineageKind::Boundary});
        inputs(origin,[&](Id producer){if(producer!=b.owner)visit({0,origin,producer,consumer,LineageKind::Data});});}
}
std::optional<Lineage> BoundaryView::lineage(Id boundary,Id producer,Id consumer,LineageKind kind) const {
    auto& access=checked(access_);
    if(access.phase==Phase::Apply)throw ras::Error("Apply cannot access downstream AL");
    if(!access.transform||!access.lower_layer)return {};
    return detail::lookup_lineage(access.read,*access.transform,*access.lower_layer,{0,boundary,producer,consumer,kind},counters(access));
}

Evidence BoundaryView::generation_key(Id id) const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access unrelated generation evidence");
    const auto stamp = detail::load_receipt_generation(access.read, transform(access), id, counters(access));
    Evidence result(sizeof(stamp));
    std::memcpy(result.data(), &stamp, sizeof(stamp));
    return result;
}

OutputWriter::OutputWriter(ras::BufferWriter writer, std::shared_ptr<Access> access)
    : writer_(std::move(writer)), access_(std::move(access)) {}
OutputWriter::OutputWriter(OutputWriter&&) noexcept = default;
OutputWriter& OutputWriter::operator=(OutputWriter&&) noexcept = default;
OutputWriter::~OutputWriter() = default;

ras::WriteSpan OutputWriter::reserve(std::size_t bytes) {
    checked(access_);
    return writer_.reserve(bytes);
}

std::size_t OutputWriter::size() const {
    checked(access_);
    return writer_.size();
}

std::size_t OutputWriter::capacity() const {
    checked(access_);
    return writer_.capacity();
}

ras::Buffer OutputWriter::finish() {
    auto& access = checked(access_);
    auto result = writer_.finish();
    detail::grant_fresh(access, result);
    return result;
}

Context::Context(std::shared_ptr<Access> access) : access_(std::move(access)) {}

OutputWriter Context::buffer(std::size_t bytes, ras::Layout layout) {
    auto access = effective(access_);
    if (!access->scope->tx) throw ras::Error("read-only Core context cannot allocate output");
    auto writer = access->scope->tx->buffer(bytes, layout);
    return OutputWriter(std::move(writer), std::move(access));
}

ChunkWriter Context::chunks(std::size_t bytes, ras::Layout layout, bool intern) {
    return ChunkWriter(effective(access_), bytes, layout, intern);
}

ras::Buffer Context::copy(std::span<const std::byte> bytes, ras::SchemaId schema) {
    auto writer = buffer(bytes.size(), {schema, 1, 1});
    if (!bytes.empty()) {
        auto span = writer.reserve(bytes.size());
        std::memcpy(span.bytes().data(), bytes.data(), bytes.size());
        span.commit();
    }
    return writer.finish();
}

Partition Context::materialize_range(Partition header, Id boundary,
                                     std::span<const OutputSlice> slices,
                                     const OutputRangeValidator& validate) {
    auto& access = checked(access_);
    if (access.phase != Phase::Partition || !access.transform || !access.layer)
        throw ras::Error("output ranges can be materialized only in an incoming Partition callback");
    if (!validate) throw ras::Error("output range requires an evaluator membership validator");
    if (!header.id || header.payload || !header.bundles.empty() || !header.origins.empty() ||
        !header.event_count || header.semantic.empty())
        throw ras::Error("output range requires an unbound complete partition header");
    const auto canonical = [](const std::vector<Id>& values) {
        return !values.empty() && std::is_sorted(values.begin(),values.end()) &&
            std::adjacent_find(values.begin(),values.end()) == values.end();
    };
    if (!canonical(header.lanes) || !canonical(header.event_kinds))
        throw ras::Error("output range needs canonical common active lanes and event kinds");
    const auto input = receipt(boundary);
    (void)output_slices(access,input,slices);
    if (header.event_count > input.event_count || !validate(input,slices,header))
        throw ras::Error("output range evaluator membership or lane proof failed");
    if (slices.size() > (std::numeric_limits<std::size_t>::max() - sizeof(OutputRangeHeader)) / sizeof(OutputSlice))
        throw ras::Error("output range descriptor size overflow");
    const auto bytes = sizeof(OutputRangeHeader) + slices.size_bytes();
    Evidence descriptor(bytes);
    const OutputRangeHeader range{boundary,input.semantic_version,slices.size()};
    std::memcpy(descriptor.data(),&range,sizeof(range));
    std::memcpy(descriptor.data()+sizeof(range),slices.data(),slices.size_bytes());
    header.payload = copy(descriptor,access.layer->schema);
    // Address changes must survive semantic-body reuse during admission.
    header.layout = std::move(descriptor);
    header.origins = {boundary};
    header.bundles = {input.chunks.storage()};
    return header;
}

std::vector<Id> Context::input_order(Id lane_id) const {
    auto access = effective(access_);
    if (access->phase != Phase::Apply || !access->isolated || !access->upper_layer)
        throw ras::Error("input_order requires an isolated Apply Boundary");
    if (auto* value = counters(*access)) ++value->metadata_reads;
    auto boundary = *access->isolated;
    access->read.directory_range(transform(*access).members, {boundary.id, 0}, {boundary.id, detail::maximum},
        [&](ras::DirectoryKey key, const ras::DirectoryRecord&) { boundary.inputs.push_back(key.second); });
    return detail::projected_inputs(access->read, *access->upper_layer, boundary, lane_id);
}

Context Context::for_boundary(Id id) const { return Context(isolate(access_, id)); }

LayerView Context::previous() const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access the previous lower layer");
    auto view = std::make_shared<Access>(access);
    view->read = access.old;
    view->layer = access.previous_layer;
    return LayerView(std::move(view));
}

LayerView Context::upper() const {
    auto access = effective(access_);
    if (!access->upper_layer) return {};
    auto view = std::make_shared<Access>(*access);
    view->layer = access->upper_layer;
    return LayerView(std::move(view));
}

BoundaryView Context::boundaries() const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access the transform Boundary index");
    if (access.phase == Phase::Boundary) {
        auto view = std::make_shared<Access>(access);
        view->read = access.old;
        view->transform = access.previous_transform;
        return BoundaryView(std::move(view));
    }
    if (!access.transform) return {};
    return BoundaryView(access_);
}

Receipt Context::receipt(Id id) const {
    auto access = effective(access_);
    Receipt result;
    if (access->phase == Phase::Apply) {
        if (!access->isolated ||
            std::find(access->isolated->parent_receipts.begin(), access->isolated->parent_receipts.end(), id)
                == access->isolated->parent_receipts.end())
            throw ras::Error("parent receipt was not declared by this Boundary");
        if (!access->previous_transform)
            throw ras::Error("Apply has no parent receipt snapshot");
        result = detail::load_receipt(access->old, *access->previous_transform, id, nullptr, counters(*access));
    } else if (access->phase == Phase::Partition || access->phase == Phase::State) {
        result = detail::load_receipt(access->read, transform(*access), id, nullptr, counters(*access));
    } else {
        throw ras::Error("receipt access requires Partition, State or declared Apply parent");
    }
    detail::grant(*access, result.payload);
    detail::grant(*access, result.state.payload);
    detail::grant_bundles(*access, detail::receipt_bundles(result));
    detail::grant_bundles(*access, result.state.bundles);
    if (auto* value = counters(*access)) {
        value->payload_reads += bool(result.payload) + bool(result.state.payload);
    }
    return result;
}

ras::Buffer Context::receipt_chunk(Id id, std::size_t ordinal) const {
    auto access = effective(access_);
    Receipt receipt_value;
    if (access->phase == Phase::Apply) {
        if (!access->isolated ||
            std::find(access->isolated->parent_receipts.begin(), access->isolated->parent_receipts.end(), id)
                == access->isolated->parent_receipts.end())
            throw ras::Error("parent receipt was not declared by this Boundary");
        if (!access->previous_transform)
            throw ras::Error("Apply has no parent receipt snapshot");
        receipt_value = detail::load_receipt_access(access->old, *access->previous_transform, id, counters(*access));
    } else if (access->phase == Phase::Partition || access->phase == Phase::State) {
        receipt_value = detail::load_receipt_access(access->read, transform(*access), id, counters(*access));
    } else {
        throw ras::Error("receipt access requires Partition, State or declared Apply parent");
    }
    // Preserve receipt()'s payload/bundle grants without copying its potentially
    // large evidence vectors for every physical chunk.
    detail::grant(*access, receipt_value.payload);
    detail::grant(*access, receipt_value.state.payload);
    detail::grant_bundles(*access, detail::receipt_bundles(receipt_value));
    detail::grant_bundles(*access, receipt_value.state.bundles);
    if (auto* value = counters(*access))
        value->payload_reads += bool(receipt_value.payload) + bool(receipt_value.state.payload);
    const auto* chunks = detail::ChunkAccess::get(receipt_value.chunks);
    if (!chunks || ordinal >= chunks->count) throw ras::Error("Core receipt chunk ordinal is out of range");
    auto view=*access;
    if(access->phase==Phase::Apply)view.read=access->old;
    if(auto* value=counters(*access))++value->payload_reads;
    return detail::read_block(view,receipt_value.chunks.storage(),ordinal);
}
void Context::receipts(const std::function<void(const Receipt&)>& visit) const {
    auto& access=checked(access_);
    if(access.phase!=Phase::Partition&&access.phase!=Phase::State)throw ras::Error("output enumeration requires Partition or State");
    if(!access.transform)return;
    if(auto* value=counters(access))++value->full_enumerations;
    access.read.directory_range(access.transform->receipts,{0,0},{detail::maximum,detail::maximum},
        [&](ras::DirectoryKey key,const ras::DirectoryRecord&){visit(receipt(key.first));});
}
void Context::receipt_chunks(Id id,const std::function<void(std::size_t,const ras::Buffer&)>& visit) const {
    auto access=effective(access_);const auto value=receipt(id);if(!value.chunks)return;
    auto view=*access;if(access->phase==Phase::Apply)view.read=access->old;
    for(std::size_t i=0;i<value.chunks.count();++i){if(auto* c=counters(*access))++c->payload_reads;visit(i,detail::read_block(view,value.chunks.storage(),i));}
}
void Context::retain_receipt_chunk(Id id,std::size_t ordinal,const ras::Buffer& equal_backing) {
    auto& access=checked(access_);
    if(access.phase!=Phase::Partition)throw ras::Error("receipt chunk check requires Partition");
    detail::require_granted(access,equal_backing);
    const auto value=receipt_chunk(id,ordinal);
    if(!value.same_storage(equal_backing))throw ras::Error("sealed output is immutable; adopt the complete output before admission");
}

const LayerDelta& Context::delta() const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access whole-layer changes");
    static const LayerDelta empty;
    return access.delta ? *access.delta : empty;
}

const LayerDelta& Context::upper_delta() const {
    auto& access = checked(access_);
    if (access.phase == Phase::Apply)
        throw ras::Error("Apply cannot access whole-layer changes");
    static const LayerDelta empty;
    return access.upper_delta ? *access.upper_delta : empty;
}

std::span<const Id> Context::changed_boundaries() const {
    auto& access = checked(access_);
    return access.changed_boundaries ? std::span<const Id>(*access.changed_boundaries) : std::span<const Id>{};
}

std::span<const Id> Context::removed_boundaries() const {
    auto& access = checked(access_);
    return access.removed_boundaries ? std::span<const Id>(*access.removed_boundaries) : std::span<const Id>{};
}

ras::DirectoryId Context::evaluator_directory(Id slot) {
    auto& access = checked(access_);
    require_metadata(access);
    const auto& current = layer(access);
    if (auto* value = counters(access)) ++value->metadata_reads;
    const auto entry = access.read.directory_find(current.evaluator, {slot, 0});
    ras::DirectoryId directory;
    if (entry) {
        if (!entry->metadata || entry->metadata.size() != sizeof(directory))
            throw ras::Error("invalid evaluator directory identity");
        const auto bytes = entry->metadata.bytes();
        std::memcpy(&directory, bytes.span().data(), sizeof(directory));
        if (!directory) throw ras::Error("invalid evaluator directory identity");
    } else {
        directory = access.scope->tx->add_directory();
        const auto data = copy(std::as_bytes(std::span<const ras::DirectoryId>(&directory, 1)), detail::metadata_schema);
        access.scope->tx->directory_put(current.evaluator, {slot, 0}, {data, {}, {}});
    }
    access.evaluator_directories->insert(directory);
    return directory;
}

std::optional<ras::DirectoryRecord> Context::metadata_find(ras::DirectoryId directory, ras::DirectoryKey key) const {
    auto& access = checked(access_);
    require_metadata(access, directory);
    if (!directory) throw ras::Error("empty evaluator metadata directory");
    if (auto* value = counters(access)) ++value->metadata_reads;
    auto result = access.read.directory_find(directory, key);
    if (result) grant_record(access, *result);
    return result;
}

void Context::metadata_put(ras::DirectoryId directory, ras::DirectoryKey key, const ras::DirectoryRecord& record) {
    auto& access = checked(access_);
    require_metadata(access, directory);
    if (!directory) throw ras::Error("empty evaluator metadata directory");
    detail::require_granted(access, record.metadata);
    detail::require_granted(access, record.payload);
    detail::require_granted(access, record.auxiliary);
    access.scope->tx->directory_put(directory, key, record);
}

void Context::metadata_erase(ras::DirectoryId directory, ras::DirectoryKey key) {
    auto& access = checked(access_);
    require_metadata(access, directory);
    if (!directory) throw ras::Error("empty evaluator metadata directory");
    access.scope->tx->directory_erase(directory, key);
}

void Context::metadata_range(ras::DirectoryId directory, ras::DirectoryKey first, ras::DirectoryKey last,
                             const std::function<void(ras::DirectoryKey, const ras::DirectoryRecord&)>& visit) const {
    auto& access = checked(access_);
    require_metadata(access, directory);
    if (!directory) throw ras::Error("empty evaluator metadata directory");
    access.read.directory_range(directory, first, last,
        [&](ras::DirectoryKey key, const ras::DirectoryRecord& record) {
            detail::check(access);
            grant_record(access, record);
            if (auto* value = counters(access)) ++value->metadata_reads;
            visit(key, record);
        });
}

std::optional<std::pair<ras::DirectoryKey, ras::DirectoryRecord>> Context::metadata_next(
    ras::DirectoryId directory, ras::DirectoryKey key) const {
    auto& access = checked(access_);
    require_metadata(access, directory);
    if (!directory) throw ras::Error("empty evaluator metadata directory");
    if (auto* value = counters(access)) ++value->metadata_reads;
    auto result = access.read.directory_lower_bound(directory, key);
    if (result) grant_record(access, result->second);
    return result;
}

std::optional<std::pair<ras::DirectoryKey, ras::DirectoryRecord>> Context::metadata_prev(
    ras::DirectoryId directory, ras::DirectoryKey key) const {
    auto& access = checked(access_);
    require_metadata(access, directory);
    if (!directory) throw ras::Error("empty evaluator metadata directory");
    if (auto* value = counters(access)) ++value->metadata_reads;
    auto result = access.read.directory_predecessor(directory, key);
    if (result) grant_record(access, result->second);
    return result;
}

Id Context::new_identity() {
    auto& access = checked(access_);
    if (!access.scope->tx) throw ras::Error("read-only Core context cannot allocate an identity");
    return access.scope->tx->new_change_id().value;
}

Id Context::identity(Id domain, std::span<const Id> coordinate) {
    auto& access = checked(access_);
    if (!access.scope->tx) throw ras::Error("read-only Core context cannot allocate an identity");
    // The stable RAS layer identity namespaces coordinates across pipelines;
    // Store-owned interning also covers first introduction in sibling branches.
    std::vector<Id> scoped{layer(access).layer.value, domain};
    scoped.insert(scoped.end(),coordinate.begin(),coordinate.end());
    return access.scope->tx->intern_identity(0x434f52454944454eULL,scoped).value;
}

void Context::ground() const {
    checked(access_);
    throw ras::Error("Ground requires a typed request");
}

} // namespace flexeva::core::runtime
