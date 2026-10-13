#include <flexeva/ras/ras.hpp>

#include <utility>

#include "index.hpp"
#include "storage.hpp"
#include "order.hpp"

namespace flexeva::ras::detail {

struct Partition {
    LayerId layer;
    std::shared_ptr<const PartitionMetadata> metadata;
    flexeva::ras::Buffer payload;
    flexeva::ras::Buffer descriptor{};
};

struct State {
    LayerId layer;
    flexeva::ras::Buffer data;
    std::uint64_t key = 0;
    bool named = false;
};

struct Catalog {
    Index<Layer> layers;
    Index<Lane> lanes;
    Index<LaneId> lane_labels;
    Index<Partition> partitions;
    Index<PartitionId> layer_partitions;
    Index<Endpoint> event_locations;
    Index<Relation> relations;
    Index<RelationId> incoming, outgoing;
    Index<State> states;
    Index<StateId> state_labels;
    Index<Index<DirectoryRecord>> directories;
    Index<Order> orders;
    explicit Catalog(std::shared_ptr<MemoryStats> stats)
        : layers(stats), lanes(stats), lane_labels(stats), partitions(stats), layer_partitions(stats), event_locations(stats),
          relations(stats), incoming(stats), outgoing(stats), states(stats), state_labels(stats),
          directories(stats), orders(stats) {}
};

struct Snapshot {
    Charge charge;
    std::shared_ptr<Store> store;
    Catalog catalog;
    Snapshot(std::shared_ptr<Store> s, Catalog c)
        : charge(s->stats, sizeof(Snapshot)), store(std::move(s)), catalog(std::move(c)) { ++store->stats->versions; }
    ~Snapshot() { --store->stats->versions; }
};

template<class T, class I> const T& lookup(const Store& store, const Index<T>& index, I id) {
    store.check(id);
    const auto* value = index.find({id.value, 0});
    require(value != nullptr, "RAS handle absent from this version");
    return *value;
}

struct Transaction {
    Charge charge;
    std::shared_ptr<Store> store;
    Catalog catalog;
    bool open = true, failed = false;
    std::size_t writers = 0, reservations = 0;
    Transaction(std::shared_ptr<Store> s, const Catalog* base)
        : charge(s->stats, sizeof(Transaction)), store(std::move(s)), catalog(base ? *base : Catalog(store->stats)) {}
    void check() const { require(open && !failed, "transaction is closed or failed"); }
    void buffer_check(const flexeva::ras::Buffer& buffer) const {
        check();
        require(buffer.buffer_ && buffer.buffer_->store == store, "foreign or null RAS buffer");
    }
    const Partition& partition(PartitionId id) const { return lookup(*store, catalog.partitions, id); }
    void endpoint(Endpoint endpoint) const {
        const auto& part = partition(endpoint.partition);
        require(endpoint.event == whole_partition || (part.metadata && endpoint.event < part.metadata->events.size()),
                "relation event outside partition");
    }
    void relation_check(const Relation& relation) const {
        endpoint(relation.source); endpoint(relation.target);
        const auto source = partition(relation.source.partition).layer;
        const auto target = partition(relation.target.partition).layer;
        switch (relation.kind) {
        case RelationKind::Data:
            require(source == target &&
                    ((relation.source.event == whole_partition) == (relation.target.event == whole_partition)),
                    "data relation requires same-layer endpoints of the same granularity");
            break;
        case RelationKind::Order:
            require(source == target && relation.source.event != whole_partition && relation.target.event != whole_partition,
                    "same-layer relation requires two event endpoints");
            break;
        case RelationKind::Lineage:
            require(relation.source.event == whole_partition && relation.target.event == whole_partition &&
                    lookup(*store, catalog.layers, source).ordinal + 1 == lookup(*store, catalog.layers, target).ordinal,
                    "lineage requires adjacent upper/lower partitions");
            break;
        default: throw Error("unknown relation kind");
        }
    }
    void put_partition(PartitionId id, Partition part) {
        check();
        const auto* old = catalog.partitions.find({id.value, 0});
        const bool existing = old != nullptr;
        if (old) {
            require(old->layer == part.layer, "partition replacement cannot change layer");
            // Opaque partitions only admit whole-partition endpoints. Replacing
            // their body/descriptor cannot invalidate any existing relation.
            if (old->metadata || part.metadata) {
                auto validate = [&](Key, RelationId edge) {
                    const auto& relation = lookup(*store, catalog.relations, edge);
                    for (const auto& endpoint : {relation.source, relation.target})
                        if (endpoint.partition == id)
                            require(endpoint.event == whole_partition || (part.metadata && endpoint.event < part.metadata->events.size()),
                                    "partition replacement would invalidate relation endpoints");
                };
                catalog.incoming.range({id.value, 0}, {id.value, maximum}, validate);
                catalog.outgoing.range({id.value, 0}, {id.value, maximum}, validate);
            }
        }
        // Touch only replaced metadata. All mutations are guarded by mutate(),
        // so a duplicate or allocation failure cannot publish a partial index.
        if (old && old->metadata)
            for (const auto& event : old->metadata->events.span())
                catalog.event_locations.erase({event.id.value, 0});
        std::uint64_t ordinal = 0;
        if (part.metadata)
            for (const auto& event : part.metadata->events.span()) {
                require(!catalog.event_locations.find({event.id.value, 0}), "duplicate live event identity");
                catalog.event_locations.put({event.id.value, 0}, {id, ordinal++});
            }
        catalog.partitions.put({id.value, 0}, part);
        if (!existing) catalog.layer_partitions.put({part.layer.value, id.value}, id);
    }
};

// If a multi-index update throws, do not publish a partly changed catalog.
template<class Fn> decltype(auto) mutate(const std::shared_ptr<Transaction>& tx, Fn&& fn) {
    require(bool(tx), "empty transaction");
    tx->check();
    try { return fn(*tx); }
    catch (...) { tx->failed = true; throw; }
}

struct Writer {
    Charge charge;
    std::shared_ptr<Transaction> tx;
    std::shared_ptr<Buffer> buffer;
    std::size_t committed = 0;
    bool open = true, borrowed = false, failed = false;
    Writer(std::shared_ptr<Transaction> t, Layout layout, std::size_t bytes)
        : charge(t->store->stats, sizeof(Writer)), tx(std::move(t)),
          buffer(std::make_shared<Buffer>(tx->store, layout, bytes)) {
        ++tx->writers; ++charge.stats->pending_writers;
    }
    ~Writer() { cancel(); }
    void check() const { tx->check(); require(open && !failed, "writer is closed or failed"); }
    void cancel() noexcept {
        if (!open) return;
        open = false;
        --tx->writers; --charge.stats->pending_writers;
        buffer.reset();
    }
};

struct Reservation {
    Charge charge;
    std::shared_ptr<Writer> writer;
    std::size_t logical, length;
    bool done = false;
    Reservation(std::shared_ptr<Writer> w, std::size_t n)
        : charge(w->tx->store->stats, sizeof(Reservation)), writer(std::move(w)),
          logical(writer->committed), length(n) {
        writer->borrowed = true;
        ++writer->tx->reservations; ++charge.stats->pending_reservations;
    }
    void release() noexcept {
        done = true; writer->borrowed = false;
        --writer->tx->reservations; --charge.stats->pending_reservations;
    }
    ~Reservation() { if (!done) { writer->failed = true; release(); } }
    void check() const { require(!done, "write borrow is closed"); writer->check(); }
    void commit(std::size_t n) {
        check();
        require(n <= length && n % writer->buffer->layout.element_size == 0, "invalid write commit length");
        writer->committed += n;
        release();
    }
};

std::unique_ptr<Reservation> reserve(const std::shared_ptr<Writer>& writer, std::size_t requested) {
    require(bool(writer), "empty writer"); writer->check();
    require(!writer->borrowed, "writer already has a write borrow");
    require(requested > 0 && requested % writer->buffer->layout.element_size == 0 &&
            requested <= writer->buffer->size - writer->committed, "write exceeds fixed buffer size or element boundary");
    return std::make_unique<Reservation>(writer, requested);
}

std::shared_ptr<const Buffer> finish(const std::shared_ptr<Writer>& writer) {
    require(bool(writer), "empty writer"); writer->check();
    require(!writer->borrowed, "writer has an outstanding write borrow");
    require(writer->committed == writer->buffer->size, "fixed buffer is not completely committed");
    std::shared_ptr<const Buffer> result = writer->buffer;
    writer->cancel();
    return result;
}

struct PartitionWriter {
    Charge charge;
    std::shared_ptr<Transaction> tx;
    PartitionId id;
    LayerId layer;
    flexeva::ras::BufferWriter payload;
    flexeva::ras::Buffer reused_payload;
    std::shared_ptr<PartitionMetadata> metadata;
    bool open = true;
    PartitionWriter(std::shared_ptr<Transaction> t, PartitionId p, LayerId l, Layout layout,
                    std::size_t bytes, flexeva::ras::Buffer reused = {})
        : charge(t->store->stats, sizeof(PartitionWriter)), tx(std::move(t)), id(p), layer(l),
          payload(reused ? flexeva::ras::BufferWriter{} :
                  flexeva::ras::BufferWriter(std::make_shared<Writer>(tx, layout, bytes))),
          reused_payload(std::move(reused)), metadata(std::make_shared<PartitionMetadata>(tx->store)) {
        ++tx->writers; ++charge.stats->pending_writers;
    }
    ~PartitionWriter() { cancel(); }
    void cancel() noexcept {
        if (!open) return;
        open = false;
        --tx->writers; --charge.stats->pending_writers;
    }
    void check() const { tx->check(); require(open, "partition writer is closed"); }
    EventId event(std::uint64_t kind, Range data, std::span<const LanePosition> lanes, EventId preserve) {
        check(); bounds(data, reused_payload ? reused_payload.size() : payload.size());
        require(!lanes.empty(), "event needs at least one lane position");
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            require(lookup(*tx->store, tx->catalog.lanes, lanes[i].lane).layer == layer, "event lane belongs to another layer");
            for (std::size_t j = 0; j < i; ++j) require(lanes[i].lane != lanes[j].lane, "duplicate lane in event context");
        }
        if (preserve) {
            tx->store->check(preserve);
            require(preserve.value < tx->store->next_id, "unreserved event identity");
        } else require(preserve.store == 0 && preserve.value == 0, "malformed event identity");
        const auto event_id = preserve ? preserve : tx->store->fresh<EventId>();
        Range positions{metadata->positions.size(), lanes.size()};
        for (const auto& lane : lanes) metadata->positions.push(tx->store->pool, lane);
        metadata->events.push(tx->store->pool, Event{kind, data, positions, event_id});
        metadata->payload_extent = std::max(metadata->payload_extent, data.offset + data.size);
        return event_id;
    }
    PartitionId finish() {
        check();
        auto data = reused_payload ? reused_payload : payload.finish();
        tx->put_partition(id, {layer, metadata, std::move(data)});
        reused_payload = {}; metadata.reset();
        cancel(); return id;
    }
};

} // namespace flexeva::ras::detail

namespace flexeva::ras {

namespace { thread_local BufferReadAudit* current_read_audit=nullptr; }
BufferReadAudit::BufferReadAudit(Callback callback,void* context) noexcept
    :callback_(callback),context_(context),previous_(current_read_audit){current_read_audit=this;}
BufferReadAudit::~BufferReadAudit(){current_read_audit=previous_;}

ByteView::ByteView(std::shared_ptr<const void> owner, std::span<const std::byte> data)
    : owner_(std::move(owner)), data_(data) {}
ByteView ByteView::slice(Range range) const {
    detail::bounds(range, size());
    return ByteView(owner_, data_.subspan(range.offset, range.size));
}
std::size_t Buffer::size() const noexcept { return range_.size; }
BufferIdentity Buffer::identity() const noexcept {
    return buffer_ ? BufferIdentity{buffer_->store->id, buffer_->allocation, range_} : BufferIdentity{};
}
Layout Buffer::layout() const { detail::require(bool(buffer_), "empty buffer"); return buffer_->layout; }
ByteView Buffer::bytes() const {
    detail::require(bool(buffer_), "empty buffer");
    if(current_read_audit&&current_read_audit->callback_)
        current_read_audit->callback_(current_read_audit->context_,identity(),buffer_->layout,range_.size);
    ++buffer_->store->stats->view_requests;
    auto* start = buffer_->region.data;
    if (range_.offset) start += range_.offset;
    return ByteView(buffer_, {start, range_.size});
}
Buffer Buffer::slice(Range range) const {
    detail::require(bool(buffer_), "empty buffer"); detail::bounds(range, size());
    const auto stride = buffer_->layout.element_size;
    detail::require(range.offset % stride == 0 && range.size % stride == 0, "slice splits a stored element");
    return Buffer(buffer_, {range_.offset + range.offset, range.size});
}

WriteSpan::WriteSpan() = default;
WriteSpan::WriteSpan(std::unique_ptr<detail::Reservation> r) : reservation_(std::move(r)) {}
WriteSpan::WriteSpan(WriteSpan&&) noexcept = default;
WriteSpan& WriteSpan::operator=(WriteSpan&&) noexcept = default;
WriteSpan::~WriteSpan() = default;
std::span<std::byte> WriteSpan::bytes() {
    detail::require(bool(reservation_), "empty write borrow");
    reservation_->check();
    return {reservation_->writer->buffer->region.data + reservation_->logical, reservation_->length};
}
std::size_t WriteSpan::offset() const {
    detail::require(bool(reservation_), "empty write borrow");
    reservation_->check(); return reservation_->logical;
}
void WriteSpan::check_type(std::size_t size, std::size_t alignment) const {
    detail::require(bool(reservation_), "empty write borrow");
    reservation_->check();
    auto layout = reservation_->writer->buffer->layout;
    detail::require(layout.element_size == size && layout.alignment >= alignment, "write borrow has another element layout");
}
void WriteSpan::commit() { detail::require(bool(reservation_), "empty write borrow"); commit(reservation_->length); }
void WriteSpan::commit(std::size_t n) {
    detail::require(bool(reservation_), "empty write borrow");
    try { reservation_->commit(n); }
    catch (const std::bad_alloc&) { reservation_->writer->tx->failed = true; throw; }
    reservation_.reset();
}

BufferWriter::BufferWriter(std::shared_ptr<detail::Writer> writer) : writer_(std::move(writer)) {}
BufferWriter::BufferWriter(BufferWriter&&) noexcept = default;
BufferWriter& BufferWriter::operator=(BufferWriter&& other) noexcept {
    if (this != &other) { if (writer_) writer_->cancel(); writer_ = std::move(other.writer_); } return *this;
}
BufferWriter::~BufferWriter() { if (writer_) writer_->cancel(); }
WriteSpan BufferWriter::reserve(std::size_t n) {
    try { return WriteSpan(detail::reserve(writer_, n)); }
    catch (const std::bad_alloc&) { if (writer_) writer_->tx->failed = true; throw; }
}
std::size_t BufferWriter::size() const {
    detail::require(bool(writer_), "empty writer"); writer_->check(); return writer_->committed;
}
std::size_t BufferWriter::capacity() const {
    detail::require(bool(writer_), "empty writer"); writer_->check(); return writer_->buffer->size;
}
Buffer BufferWriter::finish() {
    auto data = detail::finish(writer_);
    auto result = Buffer(data, {0, data->size});
    writer_.reset();
    return result;
}
Buffer BufferWriter::finish_prefix() {
    detail::require(bool(writer_), "empty writer");
    writer_->check();
    detail::require(!writer_->borrowed, "writer has an outstanding write borrow");
    const auto committed = writer_->committed;
    std::shared_ptr<const detail::Buffer> data = writer_->buffer;
    auto result = Buffer(std::move(data), {0, committed});
    writer_->cancel();
    writer_.reset();
    return result;
}

std::size_t PartitionView::event_count() const noexcept { return metadata_ ? metadata_->events.size() : 0; }
ArrayView<Event> PartitionView::events() const {
    detail::require(bool(metadata_), "empty partition view");
    return ArrayView<Event>(ByteView(metadata_, std::as_bytes(metadata_->events.span())));
}
ArrayView<LanePosition> PartitionView::positions() const {
    detail::require(bool(metadata_), "empty partition view");
    return ArrayView<LanePosition>(ByteView(metadata_, std::as_bytes(metadata_->positions.span())));
}
ByteView PartitionView::event_payload(std::size_t index) const { return payload_.bytes().slice(events()[index].payload); }
ArrayView<LanePosition> PartitionView::event_positions(std::size_t index) const { return positions().slice(events()[index].positions); }

static const detail::Snapshot& snapshot(const std::shared_ptr<const detail::Snapshot>& ptr) {
    detail::require(bool(ptr), "empty version"); return *ptr;
}
ReadView::ReadView(std::shared_ptr<const detail::Snapshot> ptr) : snapshot_(std::move(ptr)) {}
ReadView::ReadView(std::shared_ptr<detail::Transaction> ptr) : transaction_(std::move(ptr)) {}
const detail::Catalog& ReadView::catalog() const {
    if (transaction_) { transaction_->check(); return transaction_->catalog; }
    return snapshot(snapshot_).catalog;
}
const detail::Store& ReadView::store() const {
    if (transaction_) { transaction_->check(); return *transaction_->store; }
    return *snapshot(snapshot_).store;
}
Layer ReadView::layer(LayerId id) const { return detail::lookup(store(), catalog().layers, id); }
Lane ReadView::lane(LaneId id) const { return detail::lookup(store(), catalog().lanes, id); }
PartitionView ReadView::partition(PartitionId id) const {
    const auto& part = detail::lookup(store(), catalog().partitions, id);
    PartitionView view;
    view.id_ = id; view.layer_ = part.layer;
    view.metadata_ = part.metadata; view.payload_ = part.payload; view.descriptor_ = part.descriptor;
    return view;
}
Endpoint ReadView::locate(EventId id) const { return detail::lookup(store(), catalog().event_locations, id); }
Relation ReadView::relation(RelationId id) const { return detail::lookup(store(), catalog().relations, id); }
StoredState ReadView::state(StateId id) const {
    const auto& state = detail::lookup(store(), catalog().states, id);
    return {state.layer, state.data};
}
StateId ReadView::find_state(LayerId layer, std::uint64_t key) const {
    const auto& c = catalog();
    detail::lookup(store(), c.layers, layer);
    const auto* id = c.state_labels.find({layer.value, key});
    return id ? *id : StateId{};
}
std::size_t ReadView::partition_count() const { return catalog().partitions.size(); }
std::size_t ReadView::relation_count() const { return catalog().relations.size(); }
void ReadView::partitions(LayerId layer_id, const std::function<void(PartitionId)>& fn) const {
    detail::lookup(store(), catalog().layers, layer_id);
    const auto entries = catalog().layer_partitions;
    entries.range({layer_id.value, 0}, {layer_id.value, detail::maximum}, [&](detail::Key, PartitionId id) { fn(id); });
}
void ReadView::incoming(PartitionId id, const std::function<void(RelationId, const Relation&)>& fn) const {
    const auto c = catalog();
    const auto& s = store();
    detail::lookup(s, c.partitions, id);
    c.incoming.range({id.value, 0}, {id.value, detail::maximum}, [&](detail::Key, RelationId edge) {
        fn(edge, detail::lookup(s, c.relations, edge));
    });
}
void ReadView::outgoing(PartitionId id, const std::function<void(RelationId, const Relation&)>& fn) const {
    const auto c = catalog();
    const auto& s = store();
    detail::lookup(s, c.partitions, id);
    c.outgoing.range({id.value, 0}, {id.value, detail::maximum}, [&](detail::Key, RelationId edge) {
        fn(edge, detail::lookup(s, c.relations, edge));
    });
}
std::optional<DirectoryRecord> ReadView::directory_find(DirectoryId id, DirectoryKey key) const {
    const auto& entries = detail::lookup(store(), catalog().directories, id);
    const auto* found = entries.find({key.first, key.second});
    return found ? std::optional(*found) : std::nullopt;
}
std::optional<std::pair<DirectoryKey, DirectoryRecord>> ReadView::directory_lower_bound(DirectoryId id, DirectoryKey key) const {
    const auto& entries = detail::lookup(store(), catalog().directories, id);
    const auto rank = entries.rank({key.first, key.second});
    if (rank == entries.size()) return std::nullopt;
    const auto [found, record] = entries.at(rank);
    return std::pair{DirectoryKey{found.first, found.second}, record};
}
std::optional<std::pair<DirectoryKey, DirectoryRecord>> ReadView::directory_predecessor(DirectoryId id, DirectoryKey key) const {
    const auto& entries = detail::lookup(store(), catalog().directories, id);
    const auto rank = entries.rank({key.first, key.second});
    if (rank == 0) return std::nullopt;
    const auto [found, record] = entries.at(rank - 1);
    return std::pair{DirectoryKey{found.first, found.second}, record};
}
void ReadView::directory_range(DirectoryId id, DirectoryKey low, DirectoryKey high,
                               const std::function<void(DirectoryKey, const DirectoryRecord&)>& fn) const {
    const auto entries = detail::lookup(store(), catalog().directories, id);
    if (high < low) return;
    entries.range({low.first, low.second}, {high.first, high.second}, [&](detail::Key key, const DirectoryRecord& record) {
        fn({key.first, key.second}, record);
    });
}
std::size_t ReadView::directory_count() const { return catalog().directories.size(); }
std::size_t ReadView::directory_size(DirectoryId id) const { return detail::lookup(store(), catalog().directories, id).size(); }
std::size_t ReadView::order_size(OrderId id) const { return detail::lookup(store(), catalog().orders, id).size(); }
std::uint64_t ReadView::order_at(OrderId id, std::size_t position) const { return detail::lookup(store(), catalog().orders, id).at(position); }
std::size_t ReadView::order_rank(OrderId id, std::uint64_t value) const { return detail::lookup(store(), catalog().orders, id).rank(value); }
void ReadView::order_range(OrderId id, std::size_t start, std::size_t count,
                           const std::function<void(std::uint64_t)>& fn) const {
    const auto order = detail::lookup(store(), catalog().orders, id);
    order.range(start, count, fn);
}
OrderStats ReadView::order_stats(OrderId id) const { return detail::lookup(store(), catalog().orders, id).stats(); }

ReadView Version::read() const { snapshot(snapshot_); return ReadView(snapshot_); }
Layer Version::layer(LayerId id) const { return read().layer(id); }
Lane Version::lane(LaneId id) const { return read().lane(id); }
PartitionView Version::partition(PartitionId id) const { return read().partition(id); }
Endpoint Version::locate(EventId id) const { return read().locate(id); }
Relation Version::relation(RelationId id) const { return read().relation(id); }
StoredState Version::state(StateId id) const { return read().state(id); }
StateId Version::find_state(LayerId layer, std::uint64_t key) const { return read().find_state(layer, key); }
std::size_t Version::partition_count() const { return read().partition_count(); }
std::size_t Version::relation_count() const { return read().relation_count(); }
void Version::partitions(LayerId layer, const std::function<void(PartitionId)>& fn) const { read().partitions(layer, fn); }
void Version::incoming(PartitionId id, const std::function<void(RelationId, const Relation&)>& fn) const { read().incoming(id, fn); }
void Version::outgoing(PartitionId id, const std::function<void(RelationId, const Relation&)>& fn) const { read().outgoing(id, fn); }

PartitionWriter::PartitionWriter(std::unique_ptr<detail::PartitionWriter> p) : writer_(std::move(p)) {}
PartitionWriter::PartitionWriter(PartitionWriter&&) noexcept = default;
PartitionWriter& PartitionWriter::operator=(PartitionWriter&&) noexcept = default;
PartitionWriter::~PartitionWriter() = default;
BufferWriter& PartitionWriter::payload() {
    detail::require(bool(writer_), "empty partition writer"); writer_->check();
    detail::require(!writer_->reused_payload, "partition reuses immutable payload");
    return writer_->payload;
}
EventId PartitionWriter::event(std::uint64_t kind, Range data, std::span<const LanePosition> lanes, EventId preserve) {
    detail::require(bool(writer_), "empty partition writer");
    return detail::mutate(writer_->tx, [&](detail::Transaction&) { return writer_->event(kind, data, lanes, preserve); });
}
PartitionId PartitionWriter::finish() {
    detail::require(bool(writer_), "empty partition writer");
    auto result = detail::mutate(writer_->tx, [&](detail::Transaction&) { return writer_->finish(); });
    writer_.reset();
    return result;
}

Transaction::Transaction(std::shared_ptr<detail::Transaction> tx) : transaction_(std::move(tx)) {}
Transaction::Transaction(Transaction&&) noexcept = default;
Transaction& Transaction::operator=(Transaction&& other) noexcept {
    if (this != &other) { abort(); transaction_ = std::move(other.transaction_); } return *this;
}
Transaction::~Transaction() { abort(); }
ReadView Transaction::read() const {
    detail::require(bool(transaction_), "empty transaction");
    transaction_->check();
    return ReadView(transaction_);
}
LayerId Transaction::add_layer(SchemaId representation) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        auto id = tx.store->fresh<LayerId>();
        tx.catalog.layers.put({id.value, 0}, {representation, tx.catalog.layers.size()}); return id;
    });
}
LaneId Transaction::add_lane(LayerId layer, std::uint64_t label) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        detail::lookup(*tx.store, tx.catalog.layers, layer);
        if (auto old = tx.catalog.lane_labels.find({layer.value, label})) return *old;
        auto id = tx.store->fresh<LaneId>();
        tx.catalog.lanes.put({id.value, 0}, {layer, label});
        tx.catalog.lane_labels.put({layer.value, label}, id); return id;
    });
}
EventId Transaction::new_event_id() {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) { return tx.store->fresh<EventId>(); });
}
ChangeId Transaction::new_change_id() {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) { return tx.store->fresh<ChangeId>(); });
}
ChangeId Transaction::intern_identity(std::uint64_t domain, std::span<const std::uint64_t> coordinate) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) { return tx.store->intern(domain, coordinate); });
}
BufferWriter Transaction::buffer(std::size_t bytes, Layout layout) {
    return detail::mutate(transaction_, [&](detail::Transaction&) {
        detail::layout_valid(layout);
        detail::require(bytes % layout.element_size == 0, "buffer size splits a stored element");
        return BufferWriter(std::make_shared<detail::Writer>(transaction_, layout, bytes));
    });
}
PartitionWriter Transaction::partition(LayerId layer, std::size_t bytes, Layout layout, PartitionId replace) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        detail::lookup(*tx.store, tx.catalog.layers, layer); detail::layout_valid(layout);
        detail::require(bytes % layout.element_size == 0, "buffer size splits a stored element");
        if (replace) detail::require(tx.partition(replace).layer == layer, "partition replacement cannot change layer");
        else detail::require(replace.store == 0 && replace.value == 0, "malformed partition handle");
        auto id = replace ? replace : tx.store->fresh<PartitionId>();
        return PartitionWriter(std::make_unique<detail::PartitionWriter>(transaction_, id, layer, layout, bytes));
    });
}
PartitionWriter Transaction::partition_from(LayerId layer, const Buffer& payload, PartitionId replace) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        detail::lookup(*tx.store, tx.catalog.layers, layer); tx.buffer_check(payload);
        if (replace) detail::require(tx.partition(replace).layer == layer, "partition replacement cannot change layer");
        else detail::require(replace.store == 0 && replace.value == 0, "malformed partition handle");
        auto id = replace ? replace : tx.store->fresh<PartitionId>();
        return PartitionWriter(std::make_unique<detail::PartitionWriter>(transaction_, id, layer, payload.layout(), payload.size(), payload));
    });
}
PartitionId Transaction::opaque_partition(LayerId layer, const Buffer& payload,
                                           const Buffer& descriptor, PartitionId replace) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        detail::lookup(*tx.store, tx.catalog.layers, layer);
        tx.buffer_check(payload); tx.buffer_check(descriptor);
        if (replace) detail::require(tx.partition(replace).layer == layer, "partition replacement cannot change layer");
        else detail::require(replace.store == 0 && replace.value == 0, "malformed partition handle");
        const auto id = replace ? replace : tx.store->fresh<PartitionId>();
        tx.put_partition(id, {layer, {}, payload, descriptor});
        return id;
    });
}
void Transaction::replace_payload(PartitionId id, const Buffer& buffer) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        tx.buffer_check(buffer);
        auto part = tx.partition(id);
        detail::require(part.payload.layout() == buffer.layout() && (!part.metadata || part.metadata->payload_extent <= buffer.size()),
                        "replacement payload layout or extent mismatch");
        part.payload = buffer;
        tx.catalog.partitions.put({id.value, 0}, part);
    });
}
void Transaction::erase_partition(PartitionId id) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        auto part = tx.partition(id);
        bool incident = false;
        auto exists = [&](detail::Key, RelationId) { incident = true; };
        tx.catalog.incoming.range({id.value, 0}, {id.value, detail::maximum}, exists);
        tx.catalog.outgoing.range({id.value, 0}, {id.value, detail::maximum}, exists);
        detail::require(!incident, "remove incident relations before erasing partition");
        if (part.metadata)
            for (const auto& event : part.metadata->events.span())
                tx.catalog.event_locations.erase({event.id.value, 0});
        tx.catalog.partitions.erase({id.value, 0});
        tx.catalog.layer_partitions.erase({part.layer.value, id.value});
    });
}
RelationId Transaction::add_relation(Relation relation) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        tx.relation_check(relation);
        auto id = tx.store->fresh<RelationId>();
        tx.catalog.relations.put({id.value, 0}, relation);
        tx.catalog.outgoing.put({relation.source.partition.value, id.value}, id);
        tx.catalog.incoming.put({relation.target.partition.value, id.value}, id); return id;
    });
}
void Transaction::erase_relation(RelationId id) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        auto relation = detail::lookup(*tx.store, tx.catalog.relations, id);
        tx.catalog.relations.erase({id.value, 0});
        tx.catalog.outgoing.erase({relation.source.partition.value, id.value});
        tx.catalog.incoming.erase({relation.target.partition.value, id.value});
    });
}
StateId Transaction::add_state(LayerId layer, const Buffer& buffer) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        detail::lookup(*tx.store, tx.catalog.layers, layer); tx.buffer_check(buffer);
        auto id = tx.store->fresh<StateId>(); tx.catalog.states.put({id.value, 0}, {layer, buffer}); return id;
    });
}
StateId Transaction::put_state(LayerId layer, std::uint64_t key, const Buffer& buffer) {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        detail::lookup(*tx.store, tx.catalog.layers, layer); tx.buffer_check(buffer);
        if (const auto* existing = tx.catalog.state_labels.find({layer.value, key})) {
            const auto id = *existing;
            auto state = detail::lookup(*tx.store, tx.catalog.states, id);
            detail::require(state.layer == layer && state.named && state.key == key, "named state index mismatch");
            state.data = buffer;
            tx.catalog.states.put({id.value, 0}, state);
            return id;
        }
        const auto id = tx.store->fresh<StateId>();
        tx.catalog.states.put({id.value, 0}, {layer, buffer, key, true});
        tx.catalog.state_labels.put({layer.value, key}, id);
        return id;
    });
}
void Transaction::replace_state(StateId id, const Buffer& buffer) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        tx.buffer_check(buffer);
        auto state = detail::lookup(*tx.store, tx.catalog.states, id);
        state.data = buffer; tx.catalog.states.put({id.value, 0}, state);
    });
}
void Transaction::erase_state(StateId id) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        const auto state = detail::lookup(*tx.store, tx.catalog.states, id);
        if (state.named) tx.catalog.state_labels.erase({state.layer.value, state.key});
        tx.catalog.states.erase({id.value, 0});
    });
}
DirectoryId Transaction::add_directory() {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        const auto id = tx.store->fresh<DirectoryId>();
        tx.catalog.directories.put({id.value, 0}, detail::Index<DirectoryRecord>(tx.store->stats));
        return id;
    });
}
void Transaction::drop_directory(DirectoryId id) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        (void)detail::lookup(*tx.store, tx.catalog.directories, id);
        tx.catalog.directories.erase({id.value, 0});
    });
}
void Transaction::directory_put(DirectoryId id, DirectoryKey key, const DirectoryRecord& record) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        for (const auto* buffer : {&record.metadata, &record.payload, &record.auxiliary})
            if (*buffer) tx.buffer_check(*buffer);
        tx.store->check(id);
        detail::require(tx.catalog.directories.mutate_existing({id.value, 0}, [&](auto& entries) {
            entries.put({key.first, key.second}, record);
        }), "RAS handle absent from this version");
    });
}
void Transaction::directory_erase(DirectoryId id, DirectoryKey key) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        const auto& entries = detail::lookup(*tx.store, tx.catalog.directories, id);
        if (!entries.find({key.first, key.second})) return;
        tx.catalog.directories.mutate_existing({id.value, 0}, [&](auto& writable) {
            writable.erase({key.first, key.second});
        });
    });
}
OrderId Transaction::add_order() {
    return detail::mutate(transaction_, [&](detail::Transaction& tx) {
        const auto id = tx.store->fresh<OrderId>();
        tx.catalog.orders.put({id.value, 0}, detail::Order(tx.store));
        return id;
    });
}
void Transaction::order_splice(OrderId id, std::size_t start, std::size_t delete_count,
                                std::span<const std::uint64_t> values) {
    detail::mutate(transaction_, [&](detail::Transaction& tx) {
        tx.store->check(id);
        detail::require(tx.catalog.orders.mutate_existing({id.value, 0}, [&](auto& order) {
            order.splice_in_place(start, delete_count, values);
        }), "RAS handle absent from this version");
    });
}
Version Transaction::publish() {
    detail::require(bool(transaction_), "empty transaction");
    auto& tx = *transaction_; tx.check();
    detail::require(tx.writers == 0 && tx.reservations == 0, "cannot publish with unfinished writers or borrows");
    auto result = std::make_shared<detail::Snapshot>(tx.store, std::move(tx.catalog));
    ++tx.store->stats->publish_count;
    tx.open = false;
    transaction_.reset();
    return Version(std::move(result));
}
void Transaction::abort() noexcept {
    if (!transaction_) return;
    transaction_->open = false;
    transaction_->catalog = detail::Catalog(transaction_->store->stats);
    transaction_.reset();
}

Store::Store() : store_(std::make_shared<detail::Store>()) {}
Transaction Store::begin(const Version& base) {
    detail::require(!base.snapshot_ || base.snapshot_->store == store_, "base version belongs to another Store");
    return Transaction(std::make_shared<detail::Transaction>(store_, base.snapshot_ ? &base.snapshot_->catalog : nullptr));
}
MemoryStats Store::stats() const noexcept { return *store_->stats; }

} // namespace flexeva::ras
