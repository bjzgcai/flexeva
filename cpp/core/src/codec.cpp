#include "runtime_internal.hpp"
#include <cstring>
#include <type_traits>

namespace flexeva::core::runtime::detail {
thread_local Counters* current_selection_counter = nullptr;
SelectionAudit::SelectionAudit(Counters* counters) noexcept
    : previous_(current_selection_counter) { current_selection_counter = counters&&counters->diagnostics_enabled?counters:nullptr; }
SelectionAudit::~SelectionAudit() { current_selection_counter = previous_; }
namespace { thread_local Counters* current_adoption_counter=nullptr; }
AdmissionAudit::AdmissionAudit(Counters* value,bool adopt) noexcept
    :counters(value&&value->diagnostics_enabled?value:nullptr),adoption(adopt),previous_adoption(current_adoption_counter){
    if(counters)read.emplace(observe,this);
    if(adopt)current_adoption_counter=counters;
}
AdmissionAudit::~AdmissionAudit(){current_adoption_counter=previous_adoption;}
void AdmissionAudit::observe(void* opaque,const ras::BufferIdentity&,const ras::Layout& layout,std::size_t bytes){
    auto& audit=*static_cast<AdmissionAudit*>(opaque);if(!audit.counters)return;
    if(layout.schema==metadata_schema)audit.counters->admission_metadata_bytes_read+=bytes;
    else if(audit.adoption)audit.counters->adoption_body_bytes_read+=bytes;
    else audit.counters->admission_body_bytes_read+=bytes;
}
void count_block_write(Counters* counters){
    if(counters)++counters->output_block_writes;
    if(current_adoption_counter)++current_adoption_counter->adopted_block_index_writes;
}
namespace {
constexpr std::uint64_t magic = 0x464558434f444543ULL;
constexpr std::uint64_t format = 5;
enum class Kind : std::uint64_t { Partition = 1, Boundary, Receipt, Lineage, State, Manifest, NativePartition, Relation };
struct Header { std::uint64_t magic, format, kind, body_size; };
struct StoredBoundaryHead {
    Id id, owner, generation_stamp, topology_version;
    std::uint64_t input_count, consumes_layer_state;
};

struct Encoder {
    std::vector<std::byte> bytes;
    explicit Encoder(Kind kind) { put(Header{magic, format, static_cast<std::uint64_t>(kind), 0}); }
    template<class T> void put(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        append(std::as_bytes(std::span<const T>(&value, 1)));
    }
    void append(std::span<const std::byte> data) {
        if (data.size() > bytes.max_size() - bytes.size()) throw ras::Error("Core metadata size overflow");
        bytes.insert(bytes.end(), data.begin(), data.end());
    }
    template<class T> void array(std::span<const T> values) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (values.size() > maximum || values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw ras::Error("Core metadata count overflow");
        put(static_cast<std::uint64_t>(values.size()));
        append(std::as_bytes(values));
    }
    template<class T> void array(const std::vector<T>& values) { array(std::span<const T>(values)); }
    ras::Buffer finish(ras::Transaction& tx) {
        Header header{};
        std::memcpy(&header, bytes.data(), sizeof(header));
        header.body_size = bytes.size() - sizeof(header);
        std::memcpy(bytes.data(), &header, sizeof(header));
        auto writer = tx.buffer(bytes.size(), {metadata_schema, 1, 1});
        auto span = writer.reserve(bytes.size());
        std::memcpy(span.bytes().data(), bytes.data(), bytes.size());
        span.commit();
        return writer.finish();
    }
};

struct Decoder {
    ras::ByteView owner;
    std::span<const std::byte> bytes;
    explicit Decoder(const ras::Buffer& buffer, Kind kind) {
        if (!buffer || buffer.layout().schema != metadata_schema || buffer.layout().element_size != 1)
            throw ras::Error("invalid Core metadata layout");
        owner = buffer.bytes();
        bytes = owner.span();
        const auto header = get<Header>();
        if (header.magic != magic || header.format != format || header.kind != static_cast<std::uint64_t>(kind) ||
            header.body_size != bytes.size()) throw ras::Error("invalid Core metadata header");
    }
    template<class T> T get() {
        static_assert(std::is_trivially_copyable_v<T>);
        if (bytes.size() < sizeof(T)) throw ras::Error("truncated Core metadata");
        T value{};
        std::memcpy(&value, bytes.data(), sizeof(T));
        bytes = bytes.subspan(sizeof(T));
        return value;
    }
    template<class T> std::vector<T> array() {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto count = get<std::uint64_t>();
        if (count > bytes.size() / sizeof(T)) throw ras::Error("invalid Core metadata count");
        std::vector<T> result(static_cast<std::size_t>(count));
        const auto size = result.size() * sizeof(T);
        if (size) std::memcpy(result.data(), bytes.data(), size);
        bytes = bytes.subspan(size);
        return result;
    }
    template<class T> void skip_array() {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto count = get<std::uint64_t>();
        if (count > bytes.size() / sizeof(T)) throw ras::Error("invalid Core metadata count");
        bytes = bytes.subspan(static_cast<std::size_t>(count) * sizeof(T));
    }
    void end() const { if (!bytes.empty()) throw ras::Error("trailing Core metadata bytes"); }
};

ras::DirectoryRecord required(const ras::ReadView& read, ras::DirectoryId directory, Id id) {
    auto record = read.directory_find(directory, {id, 0});
    if (!record) throw ras::Error("missing Core metadata record");
    return *record;
}
void count(Counters* counters) { if (counters&&counters->diagnostics_enabled) ++counters->metadata_reads; }
void count_head(Counters* counters) { if (counters&&counters->diagnostics_enabled) ++counters->header_reads; }
void count_evidence(Counters* counters, std::size_t size) {
    if (counters && current_selection_counter == counters) counters->selection_evidence_bytes += size;
}
void generation_override(const ras::ReadView& read, const TransformData& transform, BoundaryHead& head) {
    if (!transform.generations) return;
    const auto record = read.directory_find(transform.generations, {head.id, 0});
    if (!record) return;
    if (!record->metadata || record->metadata.size() != sizeof(Id))
        throw ras::Error("invalid Core boundary generation stamp");
    const auto bytes = record->metadata.bytes();
    std::memcpy(&head.generation_stamp, bytes.span().data(), sizeof(Id));
}
BoundaryHead boundary_head(Decoder& decoder) {
    const auto stored = decoder.get<StoredBoundaryHead>();
    if (stored.consumes_layer_state > 1 || stored.input_count > std::numeric_limits<std::size_t>::max())
        throw ras::Error("invalid Core boundary header");
    return {stored.id, stored.owner, stored.generation_stamp, stored.topology_version,
            static_cast<std::size_t>(stored.input_count), stored.consumes_layer_state != 0};
}


} // namespace

ras::DirectoryRecord save_partition(ras::Transaction& tx, const Partition& value, ras::PartitionId native) {
    Encoder encoder(Kind::Partition);
    auto head = value.versions;
    head.id = value.id;
    head.event_count = value.event_count;
    encoder.put(head);
    encoder.array(value.lanes);
    encoder.array(value.semantic);
    encoder.array(value.state);
    encoder.array(value.origins);
    encoder.array(value.event_kinds);
    encoder.array(value.layout);
    encoder.array(encode_bundles(value.bundles));
    ras::DirectoryRecord result{encoder.finish(tx), value.payload, {}};
    if (native) bind_partition(tx, result, native);
    return result;
}

void bind_partition(ras::Transaction& tx, ras::DirectoryRecord& record, ras::PartitionId native) {
    if (!native) throw ras::Error("missing Core native partition identity");
    Encoder encoder(Kind::NativePartition);
    encoder.put(native);
    record.auxiliary = encoder.finish(tx);
}

Partition load_partition(const ras::ReadView& read, const LayerData& layer, Id id, Counters* counters) {
    count(counters);
    const auto record = required(read, layer.parts, id);
    Decoder decoder(record.metadata, Kind::Partition);
    Partition value;
    value.versions = decoder.get<PartitionHead>();
    value.id = value.versions.id;
    value.event_count = value.versions.event_count;
    value.lanes = decoder.array<Id>();
    value.semantic = decoder.array<std::byte>();
    count_evidence(counters, value.semantic.size());
    value.state = decoder.array<std::byte>();
    count_evidence(counters, value.state.size());
    value.origins = decoder.array<Id>();
    value.event_kinds = decoder.array<Id>();
    value.layout = decoder.array<std::byte>();
    count_evidence(counters, value.layout.size());
    value.bundles = decode_bundles(decoder.array<std::byte>());
    decoder.end();
    if (value.id != id) throw ras::Error("Core partition identity mismatch");
    value.payload = record.payload;
    return value;
}

ras::PartitionId native_partition(const ras::ReadView& read, const LayerData& layer, Id id) {
    const auto record = required(read, layer.parts, id);
    Decoder decoder(record.auxiliary, Kind::NativePartition);
    const auto result = decoder.get<ras::PartitionId>();
    decoder.end();
    if (!result) throw ras::Error("missing Core native partition identity");
    return result;
}

PartitionHead load_partition_head(const ras::ReadView& read, const LayerData& layer, Id id, Counters* counters) {
    count_head(counters);
    Decoder decoder(required(read, layer.parts, id).metadata, Kind::Partition);
    const auto head = decoder.get<PartitionHead>();
    if (head.id != id) throw ras::Error("Core partition identity mismatch");
    return head;
}

ras::DirectoryRecord save_boundary(ras::Transaction& tx, const Boundary& value, const BoundaryHead& head) {
    if (head.id != value.id || head.owner != value.owner || head.consumes_layer_state != value.consumes_layer_state)
        throw ras::Error("Core boundary header mismatch");
    Encoder encoder(Kind::Boundary);
    encoder.put(StoredBoundaryHead{head.id, head.owner, head.generation_stamp, head.topology_version,
                                  head.input_count, static_cast<std::uint64_t>(head.consumes_layer_state)});
    encoder.array(value.semantic);
    encoder.array(value.state);
    encoder.array(value.parent_receipts);
    encoder.array(encode_bundles(value.bundles));
    return {encoder.finish(tx), value.payload, {}};
}

BoundaryHead load_boundary_head(const ras::ReadView& read, const TransformData& transform, Id id, Counters* counters) {
    count_head(counters);
    Decoder decoder(required(read, transform.boundaries, id).metadata, Kind::Boundary);
    auto head = boundary_head(decoder);
    if (head.id != id) throw ras::Error("Core boundary identity mismatch");
    generation_override(read, transform, head);
    return head;
}

Boundary load_boundary(const ras::ReadView& read, const TransformData& transform, Id id, Counters* counters, bool with_inputs) {
    count(counters);
    const auto record = required(read, transform.boundaries, id);
    Decoder decoder(record.metadata, Kind::Boundary);
    auto head = boundary_head(decoder);
    generation_override(read, transform, head);
    Boundary value;
    value.id = head.id;
    value.owner = head.owner;
    value.consumes_layer_state = head.consumes_layer_state;
    value.semantic = decoder.array<std::byte>();
    count_evidence(counters, value.semantic.size());
    value.state = decoder.array<std::byte>();
    count_evidence(counters, value.state.size());
    value.parent_receipts = decoder.array<Id>();
    value.bundles = decode_bundles(decoder.array<std::byte>());
    decoder.end();
    if (value.id != id) throw ras::Error("Core boundary identity mismatch");
    if (with_inputs) {
        value.inputs.reserve(head.input_count);
        read.directory_range(transform.members, {id, 0}, {id, maximum},
            [&](ras::DirectoryKey key, const ras::DirectoryRecord&) { value.inputs.push_back(key.second); });
        if (value.inputs.size() != head.input_count) throw ras::Error("Core boundary input index mismatch");
    }
    value.payload = record.payload;
    return value;
}

ras::DirectoryRecord save_receipt(ras::Transaction& tx, const Receipt& value, const Evidence& generation_key) {
    Encoder encoder(Kind::Receipt);
    if (generation_key.size() != sizeof(Id)) throw ras::Error("invalid Core generation stamp");
    Id generation_stamp = 0;
    std::memcpy(&generation_stamp, generation_key.data(), sizeof(generation_stamp));
    encoder.put(value.boundary);
    encoder.put(value.semantic_version);
    encoder.put(generation_stamp);
    encoder.put(value.event_count);
    encoder.put(static_cast<std::uint64_t>(bool(value.chunks)));
    encoder.put(static_cast<std::uint64_t>(value.chunks.count()));
    encoder.put(static_cast<std::uint64_t>(value.chunks.size()));
    encoder.put(value.chunks ? ChunkAccess::get(value.chunks)->layout : ras::Layout{});
    encoder.array(encode_bundles(value.chunks ? std::vector<BlockBundle>{value.chunks.storage()} : std::vector<BlockBundle>{}));
    encoder.array(value.semantic);
    encoder.array(value.state.semantic);
    encoder.array(encode_bundles(value.bundles));
    encoder.array(encode_bundles(value.state.bundles));
    return {encoder.finish(tx), value.payload, value.state.payload};
}

static Receipt load_receipt_impl(const ras::ReadView& read, const TransformData& transform, Id id,
                                 Evidence* key, Counters* counters, bool evidence) {
    count(counters);
    const auto record = required(read, transform.receipts, id);
    Decoder decoder(record.metadata, Kind::Receipt);
    Receipt value;
    value.boundary = decoder.get<Id>();
    value.semantic_version = decoder.get<Id>();
    const auto generation_stamp = decoder.get<Id>();
    value.event_count = decoder.get<std::uint64_t>();
    const auto has_chunks = decoder.get<std::uint64_t>();
    const auto chunk_count = decoder.get<std::uint64_t>();
    const auto chunk_bytes = decoder.get<std::uint64_t>();
    const auto chunk_layout = decoder.get<ras::Layout>();
    const auto chunk_storage=decode_bundles(decoder.array<std::byte>());
    if (has_chunks > 1 || (!has_chunks && (chunk_count || chunk_bytes)) ||
        (chunk_count == 0) != (chunk_bytes == 0) ||
        chunk_count > std::numeric_limits<std::size_t>::max() || chunk_bytes > std::numeric_limits<std::size_t>::max())
        throw ras::Error("invalid Core receipt chunk metadata");
    if(chunk_storage.size()!=has_chunks)throw ras::Error("invalid Core receipt storage root");
    if (has_chunks) {
        const auto& stored=*BundleAccess::get(chunk_storage.front());
        if(stored.count!=chunk_count||stored.bytes!=chunk_bytes||stored.layout!=chunk_layout)throw ras::Error("receipt disagrees with immutable storage root");
        value.chunks = ChunkAccess::make(std::make_shared<ChunkOutputData>(ChunkOutputData{
            id, static_cast<std::size_t>(chunk_count), static_cast<std::size_t>(chunk_bytes), true, chunk_layout,chunk_storage.front()}));
    }
    if (evidence) {
        value.semantic = decoder.array<std::byte>();
        count_evidence(counters, value.semantic.size());
        value.state.semantic = decoder.array<std::byte>();
        count_evidence(counters, value.state.semantic.size());
    } else {
        decoder.skip_array<std::byte>();
        decoder.skip_array<std::byte>();
    }
    value.bundles = decode_bundles(decoder.array<std::byte>());
    value.state.bundles = decode_bundles(decoder.array<std::byte>());
    decoder.end();
    if (value.boundary != id) throw ras::Error("Core receipt identity mismatch");
    value.payload = record.payload;
    value.state.payload = record.auxiliary;
    if (key) {
        key->resize(sizeof(generation_stamp));
        std::memcpy(key->data(), &generation_stamp, sizeof(generation_stamp));
    }
    return value;
}

Receipt load_receipt(const ras::ReadView& read, const TransformData& transform, Id id,
                     Evidence* key, Counters* counters) {
    return load_receipt_impl(read, transform, id, key, counters, true);
}

Receipt load_receipt_access(const ras::ReadView& read, const TransformData& transform, Id id,
                            Counters* counters) {
    return load_receipt_impl(read, transform, id, nullptr, counters, false);
}

Id load_receipt_version(const ras::ReadView& read, const TransformData& transform, Id id, Counters* counters) {
    count_head(counters);
    Decoder decoder(required(read, transform.receipts, id).metadata, Kind::Receipt);
    if (decoder.get<Id>() != id) throw ras::Error("Core receipt identity mismatch");
    return decoder.get<Id>();
}

Id load_receipt_generation(const ras::ReadView& read, const TransformData& transform, Id id, Counters* counters) {
    count_head(counters);
    Decoder decoder(required(read, transform.receipts, id).metadata, Kind::Receipt);
    if (decoder.get<Id>() != id) throw ras::Error("Core receipt identity mismatch");
    (void)decoder.get<Id>();
    return decoder.get<Id>();
}

ras::DirectoryRecord save_state(ras::Transaction& tx, const State& value) {
    Encoder encoder(Kind::State);
    encoder.array(value.semantic);
    encoder.array(encode_bundles(value.bundles));
    return {encoder.finish(tx), value.payload, {}};
}

State load_state(const ras::ReadView& read, const LayerData& layer, Counters* counters) {
    count(counters);
    const auto record = read.directory_find(layer.states, {0, 0});
    if (!record) return {};
    return load_state_record(*record, counters);
}

State load_state_record(const ras::DirectoryRecord& record, Counters* counters) {
    Decoder decoder(record.metadata, Kind::State);
    State result;
    result.semantic = decoder.array<std::byte>();
    count_evidence(counters, result.semantic.size());
    result.bundles = decode_bundles(decoder.array<std::byte>());
    decoder.end();
    result.payload = record.payload;
    return result;
}


ras::DirectoryRecord save_relation(ras::Transaction& tx, ras::RelationId relation) {
    Encoder encoder(Kind::Relation);
    encoder.put(relation);
    return {encoder.finish(tx), {}, {}};
}

ras::RelationId load_relation(const ras::DirectoryRecord& record) {
    Decoder decoder(record.metadata, Kind::Relation);
    const auto relation = decoder.get<ras::RelationId>();
    decoder.end();
    return relation;
}

ras::Buffer save_manifest(ras::Transaction& tx, const Manifest& value) {
    Encoder encoder(Kind::Manifest);
    encoder.put(value.window);
    encoder.put(value.control_layer);
    encoder.put(value.state);
    encoder.array(value.layers);
    encoder.array(value.transforms);
    encoder.put(value.ground_facts);
    encoder.put(value.bundle_owners);
    encoder.array(value.ground_policy);
    return encoder.finish(tx);
}

Manifest load_manifest(const ras::Buffer& buffer) {
    Decoder decoder(buffer, Kind::Manifest);
    Manifest value;
    value.window = decoder.get<Id>();
    value.control_layer = decoder.get<ras::LayerId>();
    value.state = decoder.get<ras::StateId>();
    value.layers = decoder.array<LayerData>();
    value.transforms = decoder.array<TransformData>();
    value.ground_facts = decoder.get<ras::DirectoryId>();
    value.bundle_owners = decoder.get<ras::DirectoryId>();
    value.ground_policy = decoder.array<std::byte>();
    decoder.end();
    return value;
}
} // namespace flexeva::core::runtime::detail
