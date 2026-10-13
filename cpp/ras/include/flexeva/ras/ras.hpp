#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace flexeva::ras {

class Error : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// IDs are process-local, typed and never recycled. They are not content hashes.
template<class Tag> struct Id {
    std::uint64_t store = 0;
    std::uint64_t value = 0;
    explicit operator bool() const noexcept { return store != 0 && value != 0; }
    auto operator<=>(const Id&) const = default;
};
using LayerId = Id<struct LayerTag>;
using LaneId = Id<struct LaneTag>;
using PartitionId = Id<struct PartitionTag>;
using EventId = Id<struct EventTag>;
using RelationId = Id<struct RelationTag>;
using StateId = Id<struct StateTag>;
using DirectoryId = Id<struct DirectoryTag>;
using OrderId = Id<struct OrderTag>;
using ChangeId = Id<struct ChangeTag>;
using SchemaId = std::uint64_t;

struct Layout {
    SchemaId schema = 0;
    std::size_t element_size = 1;
    std::size_t alignment = 1;
    auto operator<=>(const Layout&) const = default;
};

struct Range {
    std::size_t offset = 0;
    std::size_t size = 0;
};

// Stable allocation identity plus this handle's allocation-relative byte range.
// Slices share the allocation token; released allocation tokens are not reused.
struct BufferIdentity {
    std::uint64_t store = 0;
    std::uint64_t allocation = 0;
    Range bytes{};
};

// Tracked RAS storage, not RSS. capacity_bytes covers payload AND metadata-array
// slabs/extents; metadata_bytes covers management objects, index_bytes covers
// index pages. Their sum has no overlap. payload_bytes and metadata_array_bytes
// describe logical allocations within capacity_bytes, not additional memory.
// Allocator headers, shared_ptr control blocks and the fixed Store are excluded.
struct MemoryStats {
    std::size_t payload_bytes = 0;
    std::size_t capacity_bytes = 0;
    std::size_t index_bytes = 0;
    std::size_t metadata_bytes = 0;
    std::size_t blocks = 0;
    std::size_t index_pages = 0;
    std::size_t versions = 0;
    std::size_t pending_writers = 0;
    std::size_t pending_reservations = 0;
    std::size_t metadata_array_bytes = 0;
    std::size_t live_payload_allocations = 0;
    std::size_t peak_capacity_bytes = 0;
    std::uint64_t allocated_payload_bytes = 0;
    std::uint64_t copied_payload_bytes = 0;
    std::uint64_t created_index_pages = 0;
    std::uint64_t view_requests = 0;
    std::uint64_t allocation_requests = 0;
    std::uint64_t reused_payload_slots = 0;
    std::uint64_t copied_metadata_bytes = 0;
    std::uint64_t index_page_visits = 0;
    std::uint64_t publish_count = 0;
};

namespace detail {
struct Store;
struct Buffer;
struct Writer;
struct Reservation;
struct Snapshot;
struct Transaction;
struct PartitionWriter;
struct PartitionMetadata;
struct Catalog;
}

class ByteView {
public:
    ByteView() = default;
    ByteView(const ByteView&) = default;
    ByteView& operator=(const ByteView&) = default;
    ByteView(ByteView&& other) noexcept
        : owner_(std::move(other.owner_)), data_(std::exchange(other.data_, {})) {}
    ByteView& operator=(ByteView&& other) noexcept {
        if (this != &other) { owner_ = std::move(other.owner_); data_ = std::exchange(other.data_, {}); }
        return *this;
    }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return size() == 0; }
    // The returned non-owning span is valid while this view or another owner
    // of the same storage remains alive. A moved-from view is empty.
    std::span<const std::byte> span() const noexcept { return data_; }
    // Every slice pins its entire allocation, but no version or catalog.
    ByteView slice(Range range) const;
private:
    friend class Buffer;
    friend class PartitionView;
    friend class ReadView;
    ByteView(std::shared_ptr<const void>, std::span<const std::byte>);
    std::shared_ptr<const void> owner_;
    std::span<const std::byte> data_;
};

template<class T> class ArrayView {
    static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
public:
    std::size_t size() const noexcept { return bytes_.size() / sizeof(T); }
    bool empty() const noexcept { return size() == 0; }
    const T& operator[](std::size_t index) const {
        if (index >= size()) throw Error("array index outside view");
        return span()[index];
    }
    std::span<const T> span() const noexcept {
        auto bytes = bytes_.span();
        return {reinterpret_cast<const T*>(bytes.data()), bytes.size() / sizeof(T)};
    }
    ArrayView slice(Range elements) const {
        if (elements.offset > size() || elements.size > size() - elements.offset)
            throw Error("array slice outside view");
        return ArrayView(bytes_.slice({elements.offset * sizeof(T), elements.size * sizeof(T)}));
    }
private:
    friend class Buffer;
    friend class PartitionView;
    explicit ArrayView(ByteView bytes) : bytes_(std::move(bytes)) {}
    ByteView bytes_;
};

// Scoped byte-view acquisition audit, not a count of CPU memory loads. Existing
// borrowed views do not acquire bytes again. Thread-local and nestable; disabled
// reads perform one pointer check. The callback must outlive this scope.
class BufferReadAudit {
public:
    using Callback = void (*)(void*, const BufferIdentity&, const Layout&, std::size_t);
    BufferReadAudit(Callback, void*) noexcept;
    ~BufferReadAudit();
    BufferReadAudit(const BufferReadAudit&) = delete;
    BufferReadAudit& operator=(const BufferReadAudit&) = delete;
private:
    friend class Buffer;
    Callback callback_;
    void* context_;
    BufferReadAudit* previous_;
};

// An immutable, owning capability for RAS storage. Keeping this handle or a view
// intentionally pins these blocks, but does not pin an entire version/catalog.
class Buffer {
public:
    Buffer() = default;
    Buffer(const Buffer&) = default;
    Buffer& operator=(const Buffer&) = default;
    Buffer(Buffer&& other) noexcept
        : buffer_(std::move(other.buffer_)), range_(std::exchange(other.range_, {})) {}
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) { buffer_ = std::move(other.buffer_); range_ = std::exchange(other.range_, {}); }
        return *this;
    }
    explicit operator bool() const noexcept { return bool(buffer_); }
    std::size_t size() const noexcept;
    BufferIdentity identity() const noexcept;
    Layout layout() const;
    ByteView bytes() const;
    // An aligned subrange shares and pins the complete underlying allocation.
    // This creates no allocation and reads/copies no payload.
    Buffer slice(Range bytes) const;
    template<class T> ArrayView<T> array(SchemaId schema) const {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        const auto actual = layout();
        if (actual.schema != schema || actual.element_size != sizeof(T) || actual.alignment < alignof(T))
            throw Error("buffer layout does not match requested array");
        return ArrayView<T>(bytes());
    }
    bool same_storage(const Buffer& other) const noexcept {
        return buffer_ == other.buffer_ && range_.offset == other.range_.offset && range_.size == other.range_.size;
    }
private:
    friend class Transaction;
    friend class BufferWriter;
    friend class PartitionWriter;
    friend struct detail::Transaction;
    friend struct detail::PartitionWriter;
    Buffer(std::shared_ptr<const detail::Buffer> p, Range range) : buffer_(std::move(p)), range_(range) {}
    std::shared_ptr<const detail::Buffer> buffer_;
    Range range_;
};

// A single writable borrow. Initialize every committed byte. Pointers/spans
// must not be used after commit or destruction. Abandoning it cancels its writer.
class WriteSpan {
public:
    WriteSpan();
    WriteSpan(WriteSpan&&) noexcept;
    WriteSpan& operator=(WriteSpan&&) noexcept;
    WriteSpan(const WriteSpan&) = delete;
    WriteSpan& operator=(const WriteSpan&) = delete;
    ~WriteSpan();
    std::span<std::byte> bytes();
    std::size_t offset() const;
    template<class T> std::span<T> array() {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        check_type(sizeof(T), alignof(T));
        auto span = bytes();
        return {reinterpret_cast<T*>(span.data()), span.size() / sizeof(T)};
    }
    void commit();
    void commit(std::size_t bytes);
private:
    friend class BufferWriter;
    explicit WriteSpan(std::unique_ptr<detail::Reservation>);
    void check_type(std::size_t, std::size_t) const;
    std::unique_ptr<detail::Reservation> reservation_;
};

// A fixed-size allocation. reserve(n) borrows exactly n consecutive bytes at
// the committed end or throws; it never grows or relocates storage. finish()
// requires the entire declared capacity to be committed and no active borrow.
class BufferWriter {
public:
    BufferWriter() = default;
    BufferWriter(BufferWriter&&) noexcept;
    BufferWriter& operator=(BufferWriter&&) noexcept;
    BufferWriter(const BufferWriter&) = delete;
    BufferWriter& operator=(const BufferWriter&) = delete;
    ~BufferWriter();
    WriteSpan reserve(std::size_t bytes);
    std::size_t size() const;
    std::size_t capacity() const;
    Buffer finish();
    // Seal only the committed prefix; retain the fixed allocation without
    // copying, resizing, or exposing its unwritten tail.
    Buffer finish_prefix();
private:
    friend class Transaction;
    friend class PartitionWriter;
    friend struct detail::PartitionWriter;
    explicit BufferWriter(std::shared_ptr<detail::Writer>);
    std::shared_ptr<detail::Writer> writer_;
};

struct Layer {
    SchemaId representation;
    std::uint64_t ordinal;
};
struct Lane {
    LayerId layer;
    std::uint64_t label;
};
struct LanePosition {
    LaneId lane;
    std::uint64_t order;
};
// Framework metadata is separate from evaluator-defined payload bytes.
struct Event {
    std::uint64_t kind;
    Range payload;
    Range positions;
    EventId id{};
};
inline constexpr SchemaId event_schema = 0xfffffffffffffff1ULL;
inline constexpr SchemaId position_schema = 0xfffffffffffffff2ULL;

class PartitionView {
public:
    PartitionId id() const noexcept { return id_; }
    LayerId layer() const noexcept { return layer_; }
    std::size_t event_count() const noexcept;
    ArrayView<Event> events() const;
    ArrayView<LanePosition> positions() const;
    const Buffer& payload() const noexcept { return payload_; }
    const Buffer& descriptor() const noexcept { return descriptor_; }
    bool opaque() const noexcept { return bool(descriptor_); }
    ByteView event_payload(std::size_t index) const;
    ArrayView<LanePosition> event_positions(std::size_t index) const;
private:
    friend class Version;
    friend class ReadView;
    PartitionId id_;
    LayerId layer_;
    std::shared_ptr<const detail::PartitionMetadata> metadata_;
    Buffer payload_;
    Buffer descriptor_;
};

inline constexpr std::uint64_t whole_partition = std::numeric_limits<std::uint64_t>::max();
struct Endpoint {
    PartitionId partition;
    std::uint64_t event = whole_partition;
};
// Lineage points from an upper producer to a lower consumer; incoming queries
// provide the reverse AL lookup. Data connects two same-layer event endpoints
// or two whole partitions; Order connects same-layer event endpoints only.
enum class RelationKind { Data, Order, Lineage };
struct Relation {
    Endpoint source;
    Endpoint target;
    RelationKind kind;
    std::uint64_t tag = 0;
};
struct StoredState {
    LayerId layer;
    Buffer data;
};

struct DirectoryKey {
    std::uint64_t first = 0, second = 0;
    auto operator<=>(const DirectoryKey&) const = default;
};
struct DirectoryRecord {
    Buffer metadata, payload, auxiliary;
};
struct OrderStats {
    std::size_t max_label_words = 0;
    std::size_t label_bytes = 0;
};

// A version view is immutable. A transaction view sees its current sealed
// records and indexes, and rejects queries after publication, abort or failure.
// Extracted buffers/views remain owning capabilities with their usual lifetime.
class ReadView {
public:
    ReadView() = default;
    Layer layer(LayerId) const;
    Lane lane(LaneId) const;
    PartitionView partition(PartitionId) const;
    Endpoint locate(EventId) const;
    Relation relation(RelationId) const;
    StoredState state(StateId) const;
    StateId find_state(LayerId, std::uint64_t key) const;
    std::size_t partition_count() const;
    std::size_t relation_count() const;
    void partitions(LayerId, const std::function<void(PartitionId)>&) const;
    void incoming(PartitionId, const std::function<void(RelationId, const Relation&)>&) const;
    void outgoing(PartitionId, const std::function<void(RelationId, const Relation&)>&) const;
    std::optional<DirectoryRecord> directory_find(DirectoryId, DirectoryKey) const;
    // Indexed neighbors, without opening any stored buffer. lower_bound returns
    // the first key >= key; predecessor returns the last key strictly < key.
    // Both perform O(log N) index work and return owning buffer handles.
    std::optional<std::pair<DirectoryKey, DirectoryRecord>> directory_lower_bound(DirectoryId, DirectoryKey) const;
    std::optional<std::pair<DirectoryKey, DirectoryRecord>> directory_predecessor(DirectoryId, DirectoryKey) const;
    // Both key bounds are inclusive; reversed bounds produce no entries.
    void directory_range(DirectoryId, DirectoryKey low, DirectoryKey high,
                         const std::function<void(DirectoryKey, const DirectoryRecord&)>&) const;
    std::size_t directory_count() const;
    std::size_t directory_size(DirectoryId) const;
    std::size_t order_size(OrderId) const;
    std::uint64_t order_at(OrderId, std::size_t position) const;
    std::size_t order_rank(OrderId, std::uint64_t value) const;
    void order_range(OrderId, std::size_t start, std::size_t count,
                     const std::function<void(std::uint64_t)>&) const;
    OrderStats order_stats(OrderId) const;
private:
    friend class Version;
    friend class Transaction;
    explicit ReadView(std::shared_ptr<const detail::Snapshot>);
    explicit ReadView(std::shared_ptr<detail::Transaction>);
    const detail::Catalog& catalog() const;
    const detail::Store& store() const;
    std::shared_ptr<const detail::Snapshot> snapshot_;
    std::shared_ptr<detail::Transaction> transaction_;
};

class Version {
public:
    Version() = default;
    explicit operator bool() const noexcept { return bool(snapshot_); }
    ReadView read() const;
    Layer layer(LayerId) const;
    Lane lane(LaneId) const;
    PartitionView partition(PartitionId) const;
    // Stable event identity resolves independently in every immutable version.
    // Throws if this event is absent from the version or belongs to another Store.
    Endpoint locate(EventId) const;
    Relation relation(RelationId) const;
    StoredState state(StateId) const;
    // Names are layer-local. A missing key returns an empty handle; an invalid
    // layer still throws. Anonymous add_state() entries have no name.
    StateId find_state(LayerId, std::uint64_t key) const;
    std::size_t partition_count() const;
    std::size_t relation_count() const;
    void partitions(LayerId, const std::function<void(PartitionId)>&) const;
    void incoming(PartitionId, const std::function<void(RelationId, const Relation&)>&) const;
    void outgoing(PartitionId, const std::function<void(RelationId, const Relation&)>&) const;
private:
    friend class Store;
    friend class Transaction;
    explicit Version(std::shared_ptr<const detail::Snapshot> p) : snapshot_(std::move(p)) {}
    std::shared_ptr<const detail::Snapshot> snapshot_;
};

class PartitionWriter {
public:
    PartitionWriter(PartitionWriter&&) noexcept;
    PartitionWriter& operator=(PartitionWriter&&) noexcept;
    PartitionWriter(const PartitionWriter&) = delete;
    PartitionWriter& operator=(const PartitionWriter&) = delete;
    ~PartitionWriter();
    // The enclosing finish() seals payload; do not finish/move this separately.
    // A partition_from() writer reuses immutable payload and has no writer.
    BufferWriter& payload();
    // Positions are validated and metadata constructed during production.
    // Payload can be any layer-defined layout; no payload parser is invoked.
    // Omit preserve to assign a new identity. Preserved IDs must be unique
    // among live events, including events in other layers. Repartition by
    // erasing old partitions before sealing their replacement partitions.
    EventId event(std::uint64_t kind, Range payload, std::span<const LanePosition>, EventId preserve = {});
    PartitionId finish();
private:
    friend class Transaction;
    explicit PartitionWriter(std::unique_ptr<detail::PartitionWriter>);
    std::unique_ptr<detail::PartitionWriter> writer_;
};

class Transaction {
public:
    Transaction(Transaction&&) noexcept;
    Transaction& operator=(Transaction&&) noexcept;
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    ~Transaction();
    ReadView read() const;
    LayerId add_layer(SchemaId representation);
    LaneId add_lane(LayerId, std::uint64_t label);
    // Reserve a store-local identity before constructing its event metadata.
    // The identity becomes locatable only after its partition is sealed.
    EventId new_event_id();
    // Store-wide change identity; unique across branches and never reused after
    // abort. Allocating a stamp creates no catalog entry or payload allocation.
    ChangeId new_change_id();
    // Stable logical identity, interned by complete coordinate in this Store.
    // Registry entries survive abort and branches, but hold no payload/version.
    ChangeId intern_identity(std::uint64_t domain, std::span<const std::uint64_t> coordinate);
    BufferWriter buffer(std::size_t bytes, Layout layout = {});
    template<class T> BufferWriter array(SchemaId schema, std::size_t count) {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw Error("array size overflow");
        return buffer(count * sizeof(T), {schema, sizeof(T), alignof(T)});
    }
    PartitionWriter partition(LayerId, std::size_t payload_bytes, Layout payload = {}, PartitionId replace = {});
    PartitionWriter partition_from(LayerId, const Buffer& payload, PartitionId replace = {});
    // A complete evaluator-owned body plus its separately stored descriptor.
    // This path admits no event membership and creates no event index entries.
    PartitionId opaque_partition(LayerId, const Buffer& payload, const Buffer& descriptor,
                                 PartitionId replace = {});
    void replace_payload(PartitionId, const Buffer&);
    // Erasing a partition with incident relations is rejected. Remove those
    // relations explicitly, so storage never silently changes dependencies.
    void erase_partition(PartitionId);
    RelationId add_relation(Relation);
    void erase_relation(RelationId);
    StateId add_state(LayerId, const Buffer&);
    // Insert or replace a layer-local named state, preserving its StateId.
    StateId put_state(LayerId, std::uint64_t key, const Buffer&);
    void replace_state(StateId, const Buffer&);
    void erase_state(StateId);
    DirectoryId add_directory();
    // Drop only this transaction catalog's root; parent snapshots remain valid.
    void drop_directory(DirectoryId);
    void directory_put(DirectoryId, DirectoryKey, const DirectoryRecord&);
    void directory_erase(DirectoryId, DirectoryKey);
    OrderId add_order();
    // Values are unique within one order. A splice can reinsert deleted values.
    // Positions refer to the current transaction view, not its parent version.
    void order_splice(OrderId, std::size_t start, std::size_t delete_count,
                      std::span<const std::uint64_t> values);
    // Catalog mutation errors poison the transaction; abort it. An attempted
    // publish with outstanding writers/borrows may be retried after completion.
    Version publish();
    void abort() noexcept;
private:
    friend class Store;
    explicit Transaction(std::shared_ptr<detail::Transaction>);
    std::shared_ptr<detail::Transaction> transaction_;
};

// Single-process library; calls sharing a Store must be externally serialized.
// Views survive Store destruction. No disk, IPC, parser or evaluator runtime is
// involved. The Store retains a compact logical identity registry, but no catalog
// of old versions or detached buffers.
class Store {
public:
    Store();
    Transaction begin(const Version& base = {});
    MemoryStats stats() const noexcept;
private:
    std::shared_ptr<detail::Store> store_;
};

} // namespace flexeva::ras
