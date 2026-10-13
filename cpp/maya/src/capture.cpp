#include <flexeva/maya/maya.hpp>
#include <flexeva/core/layer.hpp>

#include <flexeva/maya/engine.hpp>
// The legacy ABI header also declares a TU-local default-stream object.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <fake_types.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace flexeva::maya {
namespace {

struct SourceValue {
    std::uint64_t key = 0, revision = 0, order = 0;
    std::uint32_t rank = 0, reserved = 0;
};
struct Receipt {
    ras::EventId id;
    std::uint64_t sequence = 0, opcode = 0;
};
struct ReceiptHeader {
    SourceValue source;
    ras::PartitionId partition;
    ras::EventId event;
    std::uint64_t count = 0;
};
struct CaptureSummary { std::uint64_t boundaries = 0; };

ReceiptHeader read_header(const ras::Version& version, ras::LayerId layer, std::uint64_t key) {
    const auto state = version.find_state(layer, key);
    if (!state) throw ras::Error("unknown Maya boundary");
    const auto bytes = version.state(state).data.bytes();
    if (bytes.size() < sizeof(ReceiptHeader)) throw ras::Error("invalid Maya boundary receipt");
    ReceiptHeader header;
    std::memcpy(&header, bytes.span().data(), sizeof(header));
    if (header.source.key != key || header.count > (bytes.size() - sizeof(header)) / sizeof(Receipt) ||
        bytes.size() != sizeof(header) + header.count * sizeof(Receipt))
        throw ras::Error("invalid Maya boundary receipt extent");
    return header;
}

// Only compact receipts are staged. Their backing is RAS-owned, including while
// a boundary is running; no duplicate native trace container is constructed.
class Receipts {
public:
    explicit Receipts(ras::Transaction& tx) : tx_(tx) {}
    void append(Receipt value) {
        if (!writer_) writer_.emplace(tx_.array<Receipt>(state_schema, chunk_count));
        auto borrow = writer_->reserve(sizeof(value));
        std::memcpy(borrow.bytes().data(), &value, sizeof(value));
        borrow.commit();
        ++count_;
        if (writer_->size() == writer_->capacity()) seal();
    }
    std::size_t size() const noexcept { return count_; }
    void store(ras::LayerId source, ras::LayerId trace, std::uint64_t key, ReceiptHeader header) {
        if (writer_) seal();
        header.count = count_;
        if (count_ > (std::numeric_limits<std::size_t>::max() - sizeof(header)) / sizeof(Receipt))
            throw ras::Error("Maya receipt extent overflow");
        auto source_writer = tx_.buffer(sizeof(header) + count_ * sizeof(Receipt),
                                         {state_schema, alignof(ReceiptHeader), alignof(ReceiptHeader)});
        auto h = source_writer.reserve(sizeof(header));
        std::memcpy(h.bytes().data(), &header, sizeof(header)); h.commit();
        auto trace_writer = tx_.array<ras::EventId>(state_schema, count_);
        std::size_t remaining = count_;
        for (const auto& chunk : chunks_) {
            const auto values = chunk.array<Receipt>(state_schema);
            const auto count = std::min(remaining, chunk_count);
            auto body = source_writer.reserve(count * sizeof(Receipt));
            std::memcpy(body.bytes().data(), values.span().data(), body.bytes().size()); body.commit();
            auto ids = trace_writer.reserve(count * sizeof(ras::EventId));
            auto output = ids.array<ras::EventId>();
            for (std::size_t i = 0; i != count; ++i) output[i] = values[i].id;
            ids.commit();
            remaining -= count;
        }
        tx_.put_state(source, key, source_writer.finish());
        tx_.put_state(trace, key, trace_writer.finish());
    }
private:
    void seal() {
        if (const auto remaining = writer_->capacity() - writer_->size()) {
            auto tail = writer_->reserve(remaining);
            std::memset(tail.bytes().data(), 0, remaining); tail.commit();
        }
        chunks_.push_back(writer_->finish()); writer_.reset();
    }
    static constexpr std::size_t chunk_count = 128;
    ras::Transaction& tx_;
    std::optional<ras::BufferWriter> writer_;
    std::vector<ras::Buffer> chunks_;
    std::size_t count_ = 0;
};

void erase_partitions(ras::Transaction& tx, const ras::Version& version,
                      const std::set<ras::PartitionId>& partitions) {
    std::set<ras::RelationId> relations;
    for (auto partition : partitions) {
        auto collect = [&](ras::RelationId id, const ras::Relation&) { relations.insert(id); };
        version.incoming(partition, collect); version.outgoing(partition, collect);
    }
    for (auto relation : relations) tx.erase_relation(relation);
    for (auto partition : partitions) tx.erase_partition(partition);
}

void link_sources(ras::Store& store, ras::Version& version, const Layers& layers,
                  std::span<const ras::PartitionId> partitions) {
    if (partitions.empty()) return;
    auto tx = store.begin(version);
    for (auto id : partitions) {
        const auto part = version.partition(id);
        std::set<std::uint64_t> producers;
        for (const auto& event : part.events().span()) {
            auto value = core::event_view<Operation>(version, event.id, trace_schema);
            const auto& op = value.get();
            if (!op.producer_count || op.producer_count > std::size(op.producers))
                throw ras::Error("invalid Maya producer set");
            producers.insert(op.producers, op.producers + op.producer_count);
        }
        for (auto key : producers) {
            const auto source = read_header(version, layers.source, key).partition;
            tx.add_relation({{source}, {id}, ras::RelationKind::Lineage});
        }
    }
    version = tx.publish();
}

// The partial worklist contains handles and keys only. Owner closure includes
// every record sharing a replaced physical partition, without walking unrelated
// Trace directories. Inputs referenced by these records may remain retained.
void select_capture(const Candidate& parent, std::span<const std::uint64_t> dirty,
                    std::set<std::uint64_t>& boundaries, std::set<ras::PartitionId>& partitions,
                    Statistics& stats) {
    std::vector<ras::PartitionId> queue;
    auto add_partition = [&](ras::PartitionId id) {
        if (partitions.insert(id).second) queue.push_back(id);
    };
    auto add_owner = [&](std::uint64_t key) {
        if (!boundaries.insert(key).second) return;
        const auto ids = core::state_view<ras::EventId>(parent.version, parent.layers.trace, key, state_schema);
        for (auto id : ids.span()) add_partition(parent.version.locate(id).partition);
        const auto source = read_header(parent.version, parent.layers.source, key).partition;
        parent.version.outgoing(source, [&](ras::RelationId, const ras::Relation& edge) {
            if (edge.kind == ras::RelationKind::Lineage &&
                parent.version.partition(edge.target.partition).layer() == parent.layers.trace)
                add_partition(edge.target.partition);
        });
    };
    for (auto key : dirty) {
        if (!key) throw ras::Error("Maya boundary keys must be nonzero");
        add_owner(key);
    }
    for (std::size_t i = 0; i != queue.size(); ++i) {
        ++stats.trace_partitions_read;
        const auto part = parent.version.partition(queue[i]);
        for (const auto& event : part.events().span()) {
            const auto value = core::event_view<Operation>(parent.version, event.id, trace_schema);
            add_owner(value.get().boundary);
        }
    }
}

bool same_topology(const Operation& a, const Operation& b) {
    const auto& x = a.call; const auto& y = b.call;
    return a.rank == b.rank && a.sequence == b.sequence && x.op == y.op &&
        x.stream == y.stream && x.event == y.event && x.handle == y.handle &&
        x.comm == y.comm && x.src == y.src && x.dst == y.dst &&
        x.a == y.a && x.b == y.b && x.c == y.c && x.flags == y.flags &&
        x.copy_kind == y.copy_kind && x.rank == y.rank && x.nranks == y.nranks &&
        x.peer == y.peer && x.datatype == y.datatype && x.reduction == y.reduction &&
        a.src_offset == b.src_offset && a.dst_offset == b.dst_offset &&
        a.a_offset == b.a_offset && a.b_offset == b.b_offset && a.c_offset == b.c_offset &&
        std::memcmp(x.comm_uid, y.comm_uid, sizeof(x.comm_uid)) == 0;
}

void validate_options(const RunOptions& options) {
    if (!options.context || !options.policies.trace.max_payload_bytes ||
        !options.policies.execution.max_payload_bytes || !options.policies.timing.max_payload_bytes)
        throw ras::Error("Maya context and partition capacities must be nonzero");
}

std::uint64_t scalar_width(std::int32_t type) {
    switch (type) {
    case CUDA_R_8I: case CUDA_R_8U: return 1;
    case CUDA_R_16F: case CUDA_R_16BF: case CUDA_R_16I: case CUDA_R_16U: return 2;
    case CUDA_R_32F: case CUDA_R_32I: case CUDA_R_32U: return 4;
    case CUDA_R_64F: case CUDA_R_64I: case CUDA_R_64U: return 8;
    default: throw ras::Error("unsupported Maya GEMM scalar dtype");
    }
}

std::uint64_t matrix_extent(std::int32_t rows, std::int32_t columns, std::int32_t leading,
                            std::uint64_t width, bool empty_output) {
    if (rows < 0 || columns < 0 || leading < std::max(1, rows))
        throw ras::Error("invalid Maya GEMM matrix dimensions or leading dimension");
    if (!rows || !columns || empty_output) return 0;
    const auto r = static_cast<std::uint64_t>(rows);
    const auto c = static_cast<std::uint64_t>(columns - 1);
    const auto ld = static_cast<std::uint64_t>(leading);
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (c > (maximum - r) / ld) throw ras::Error("Maya GEMM matrix extent overflow");
    const auto elements = c * ld + r;
    if (elements > maximum / width) throw ras::Error("Maya GEMM matrix byte extent overflow");
    return elements * width;
}

std::array<std::uint64_t, 3> gemm_extents(const FakeCudaRasRecord& op) {
    if (op.m < 0 || op.n < 0 || op.k < 0 ||
        op.trans_a < CUBLAS_OP_N || op.trans_a > CUBLAS_OP_C ||
        op.trans_b < CUBLAS_OP_N || op.trans_b > CUBLAS_OP_C)
        throw ras::Error("unsupported Maya GEMM dimensions or transpose mode");
    const auto aw = scalar_width(op.a_type), bw = scalar_width(op.b_type), cw = scalar_width(op.c_type);
    const bool empty_output = op.m == 0 || op.n == 0;
    return {
        matrix_extent(op.trans_a == CUBLAS_OP_N ? op.m : op.k,
                      op.trans_a == CUBLAS_OP_N ? op.k : op.m, op.lda, aw, empty_output),
        matrix_extent(op.trans_b == CUBLAS_OP_N ? op.k : op.n,
                      op.trans_b == CUBLAS_OP_N ? op.n : op.k, op.ldb, bw, empty_output),
        matrix_extent(op.m, op.n, op.ldc, cw, empty_output)};
}

struct Resource {
    ras::EventId id;
    std::uint64_t boundary = 0, last_writer = 0, extent = 0;
    std::uint32_t rank = 0;
};
using Resources = std::map<std::uint64_t, Resource>;
using UniqueId = std::array<std::uint8_t, 128>;
struct UidResource { Resource resource; int nranks = 0, remaining = 0; std::set<int> initialized; };

} // namespace

struct Capture::Impl {
    ras::Store& store;
    const Candidate* parent;
    Layers layers;
    const RunOptions& options;
    const Changes& changes;
    Statistics& stats;
    TraceDelta& delta;
    bool full;
    ras::Transaction tx;
    core::PartitionBuilder<Operation> builder;
    std::set<std::uint64_t> selected, explicitly_dirty, seen;
    std::set<ras::PartitionId> removed;
    std::map<std::uint32_t, std::uint64_t> rank_sequence;
    Resources allocations, streams, events, handles, comms;
    std::map<UniqueId, UidResource> uids;
    std::map<std::uint32_t, unsigned> group_depth;
    std::exception_ptr failure;
    std::uint64_t boundary = 0, order = 0, cursor = 0;
    std::uint32_t rank = 0;
    bool recording = false, active = false;
    ReceiptHeader old{};
    ras::ByteView old_receipts;
    std::unique_ptr<Receipts> receipts;

    Impl(ras::Store& s, const Candidate* p, Layers l, const RunOptions& o, const Changes& c,
         Statistics& st, TraceDelta& d, bool all)
        : store(s), parent(p), layers(l), options(o), changes(c), stats(st), delta(d), full(all),
          tx(s.begin(p ? p->version : ras::Version{})),
          builder(tx, l.trace, trace_schema, o.policies.trace.max_payload_bytes),
          explicitly_dirty(c.boundaries.begin(), c.boundaries.end()) {
        if (parent) {
            if (full) parent->version.partitions(layers.trace, [&](ras::PartitionId id) { removed.insert(id); });
            else select_capture(*parent, changes.boundaries, selected, removed, stats);
            erase_partitions(tx, parent->version, removed);
            delta.removed.assign(removed.begin(), removed.end());
        }
    }

    static int sink(void* context, const FakeCudaRasRecord* record) noexcept {
        auto& self = *static_cast<Impl*>(context);
        try { self.emit(*record); return 0; }
        catch (...) { self.failure = std::current_exception(); return -1; }
    }

    void depend(Operation* output, std::uint64_t key) {
        if (!output || !key) return;
        for (std::size_t i = 0; i != output->producer_count; ++i)
            if (output->producers[i] == key) return;
        if (output->producer_count == std::size(output->producers))
            throw ras::Error("Maya operation exceeds producer bound");
        output->producers[output->producer_count++] = key;
    }
    void depend(Operation* output, const Resource& resource) {
        depend(output, resource.boundary); depend(output, resource.last_writer);
    }
    Resource& lookup(Resources& resources, std::uint64_t address, Operation* output) {
        const auto found = resources.find(address);
        if (found == resources.end() || found->second.rank != rank)
            throw ras::Error("Maya resource is absent or belongs to another rank");
        depend(output, found->second); return found->second;
    }
    void create(Resources& resources, std::uint64_t address, ras::EventId id, std::uint64_t extent = 0) {
        if (!address || !resources.emplace(address, Resource{id, boundary, boundary, extent, rank}).second)
            throw ras::Error("Maya resource creation reuses a live address");
    }
    std::pair<Resource*, std::uint64_t> memory(std::uint64_t address, std::uint64_t bytes, Operation* output) {
        if (!address && bytes == 0) return {nullptr, 0};
        auto found = allocations.upper_bound(address);
        if (found == allocations.begin()) throw ras::Error("Maya buffer has no live allocation");
        --found;
        const auto offset = address - found->first;
        if (found->second.rank != rank || offset > found->second.extent || bytes > found->second.extent - offset)
            throw ras::Error("Maya buffer range exceeds its live allocation");
        depend(output, found->second); return {&found->second, offset};
    }
    void buffer(std::uint64_t address, std::uint64_t bytes, bool write, Operation* output,
                std::uint64_t* normalized, std::uint64_t* offset) {
        const auto [resource, displacement] = memory(address, bytes, output);
        if (normalized) *normalized = resource ? resource->id.value : 0;
        if (offset) *offset = displacement;
        if (write && resource) resource->last_writer = boundary;
    }
    void stream(std::uint64_t address, Operation* output) {
        if (!address) { if (output) output->call.stream = 0; return; }
        auto& value = lookup(streams, address, output);
        if (output) output->call.stream = value.id.value;
    }

    void normalize(const FakeCudaRasRecord& raw, ras::EventId id, Operation* out) {
        if (out) out->call = raw;
        depend(out, boundary);
        auto change = [&](Resources& resources, std::uint64_t address, std::uint64_t* field, bool destroy) {
            auto& resource = lookup(resources, address, out);
            if (field) *field = resource.id.value;
            if (destroy) resources.erase(address);
        };
        auto op = raw.op;
        switch (op) {
        case FAKECUDA_RAS_ALLOC:
            create(allocations, raw.dst, id, raw.bytes);
            if (out) out->call.dst = id.value;
            break;
        case FAKECUDA_RAS_FREE:
            change(allocations, raw.src, out ? &out->call.src : nullptr, true); break;
        case FAKECUDA_RAS_STREAM_CREATE:
            create(streams, raw.stream, id);
            if (out) out->call.stream = id.value;
            break;
        case FAKECUDA_RAS_STREAM_DESTROY:
            change(streams, raw.stream, out ? &out->call.stream : nullptr, true); break;
        case FAKECUDA_RAS_EVENT_CREATE:
            create(events, raw.event, id);
            if (out) out->call.event = id.value;
            break;
        case FAKECUDA_RAS_EVENT_DESTROY:
            change(events, raw.event, out ? &out->call.event : nullptr, true); break;
        case FAKECUDA_RAS_EVENT_RECORD:
        case FAKECUDA_RAS_EVENT_SYNC:
        case FAKECUDA_RAS_STREAM_WAIT_EVENT: {
            auto& event = lookup(events, raw.event, out);
            if (out) out->call.event = event.id.value;
            stream(raw.stream, out);
            if (op == FAKECUDA_RAS_EVENT_RECORD) event.last_writer = boundary;
            break;
        }
        case FAKECUDA_RAS_STREAM_SYNC: stream(raw.stream, out); break;
        case FAKECUDA_RAS_DEVICE_SYNC: break;
        case FAKECUDA_RAS_MEMCPY_ASYNC:
            stream(raw.stream, out);
            buffer(raw.src, raw.bytes, false, out, out ? &out->call.src : nullptr, out ? &out->src_offset : nullptr);
            buffer(raw.dst, raw.bytes, true, out, out ? &out->call.dst : nullptr, out ? &out->dst_offset : nullptr);
            break;
        case FAKECUDA_RAS_MEMSET_ASYNC:
            stream(raw.stream, out);
            buffer(raw.dst, raw.bytes, true, out, out ? &out->call.dst : nullptr, out ? &out->dst_offset : nullptr);
            break;
        case FAKECUDA_RAS_BLAS_CREATE:
            create(handles, raw.handle, id); if (out) out->call.handle = id.value; break;
        case FAKECUDA_RAS_BLAS_DESTROY:
            change(handles, raw.handle, out ? &out->call.handle : nullptr, true); break;
        case FAKECUDA_RAS_BLAS_SET_STREAM: {
            auto& handle = lookup(handles, raw.handle, out);
            if (out) out->call.handle = handle.id.value;
            stream(raw.stream, out); handle.last_writer = boundary; break;
        }
        case FAKECUDA_RAS_GEMM: {
            change(handles, raw.handle, out ? &out->call.handle : nullptr, false);
            stream(raw.stream, out);
            const auto extents = gemm_extents(raw);
            buffer(raw.a, extents[0], false, out, out ? &out->call.a : nullptr, out ? &out->a_offset : nullptr);
            buffer(raw.b, extents[1], false, out, out ? &out->call.b : nullptr, out ? &out->b_offset : nullptr);
            buffer(raw.c, extents[2], extents[2] != 0, out, out ? &out->call.c : nullptr, out ? &out->c_offset : nullptr);
            break;
        }
        case FAKECUDA_RAS_NCCL_UID: {
            UniqueId key; std::copy(std::begin(raw.comm_uid), std::end(raw.comm_uid), key.begin());
            if (!uids.emplace(key, UidResource{Resource{id, boundary, boundary, 0, rank}, 0, 0, {}}).second)
                throw ras::Error("duplicate Maya communication UID");
            if (out) { std::memset(out->call.comm_uid, 0, sizeof(out->call.comm_uid));
                std::memcpy(out->call.comm_uid, &id.value, sizeof(id.value)); }
            break;
        }
        case FAKECUDA_RAS_COMM_INIT: {
            if (raw.rank < 0 || std::uint32_t(raw.rank) != rank || raw.nranks <= 0 || raw.rank >= raw.nranks)
                throw ras::Error("Maya communicator rank disagrees with boundary rank");
            UniqueId key; std::copy(std::begin(raw.comm_uid), std::end(raw.comm_uid), key.begin());
            auto found = uids.find(key);
            if (found == uids.end()) throw ras::Error("Maya communicator UID was not generated in this capture");
            auto& uid = found->second;
            if (!uid.nranks) { uid.nranks = raw.nranks; uid.remaining = raw.nranks; }
            if (uid.nranks != raw.nranks || !uid.initialized.insert(raw.rank).second)
                throw ras::Error("Maya communicator UID has inconsistent participants");
            depend(out, uid.resource);
            create(comms, raw.comm, id);
            if (out) { out->call.comm = id.value; std::memset(out->call.comm_uid, 0, sizeof(out->call.comm_uid));
                std::memcpy(out->call.comm_uid, &uid.resource.id.value, sizeof(uid.resource.id.value)); }
            if (!--uid.remaining) uids.erase(found);
            break;
        }
        case FAKECUDA_RAS_COMM_DESTROY:
            change(comms, raw.comm, out ? &out->call.comm : nullptr, true); break;
        case FAKECUDA_RAS_GROUP_START:
            if (group_depth[rank] == std::numeric_limits<unsigned>::max()) throw ras::Error("Maya group nesting overflow");
            ++group_depth[rank]; break;
        case FAKECUDA_RAS_GROUP_END:
            if (!group_depth[rank]) throw ras::Error("Maya NCCL group end has no start");
            --group_depth[rank]; break;
        case FAKECUDA_RAS_SEND:
        case FAKECUDA_RAS_RECV:
        case FAKECUDA_RAS_ALLREDUCE:
            if (raw.rank < 0 || std::uint32_t(raw.rank) != rank) throw ras::Error("Maya transfer rank disagrees with boundary rank");
            change(comms, raw.comm, out ? &out->call.comm : nullptr, false);
            stream(raw.stream, out);
            if (op != FAKECUDA_RAS_RECV)
                buffer(raw.src, raw.bytes, false, out, out ? &out->call.src : nullptr, out ? &out->src_offset : nullptr);
            if (op != FAKECUDA_RAS_SEND)
                buffer(raw.dst, raw.bytes, true, out, out ? &out->call.dst : nullptr, out ? &out->dst_offset : nullptr);
            break;
        default: throw ras::Error("unsupported Maya operation");
        }
    }

    Receipt receipt(std::size_t index) const {
        if (index >= old.count) throw ras::Error("Maya retained boundary emitted additional operations; Changes.structure_changed required");
        Receipt value;
        std::memcpy(&value, old_receipts.span().data() + sizeof(ReceiptHeader) + index * sizeof(Receipt), sizeof(value));
        return value;
    }

    void emit(const FakeCudaRasRecord& raw) {
        if (!active) throw ras::Error("Maya native operation occurred outside a Boundary");
        ++stats.hooks;
        auto& sequence = rank_sequence[rank];
        if (sequence == std::numeric_limits<std::uint64_t>::max()) throw ras::Error("Maya operation sequence overflow");
        const auto current_sequence = sequence++;
        if (!recording) {
            const auto previous = receipt(cursor++);
            if (previous.opcode != raw.op || previous.sequence != current_sequence)
                throw ras::Error("retained Maya boundary changed operation order; Changes.structure_changed required");
            normalize(raw, previous.id, nullptr);
            ++stats.retained_hooks;
            return;
        }
        ras::EventId id{};
        if (cursor < old.count) {
            const auto previous = receipt(cursor);
            if (previous.opcode == raw.op) id = previous.id;
            else {
                if (!full) throw ras::Error("Maya API opcode/order changed; Changes.structure_changed required");
                delta.structural = true;
            }
        } else if (parent) {
            if (!full) throw ras::Error("Maya API count changed; Changes.structure_changed required");
            delta.structural = true;
        }
        if (!id) id = tx.new_event_id();
        Operation value{};
        value.boundary = boundary; value.sequence = current_sequence; value.rank = rank;
        normalize(raw, id, &value);
        if (parent && cursor < old.count && id == receipt(cursor).id) {
            const auto previous = core::event_view<Operation>(parent->version, id, trace_schema);
            if (!same_topology(previous.get(), value)) delta.structural = true;
        }
        const auto lane = tx.add_lane(layers.trace, rank);
        const ras::LanePosition position{lane, current_sequence};
        builder.append(value, raw.op, {&position, 1}, id);
        receipts->append({id, current_sequence, raw.op});
        delta.changed_events.push_back(id);
        ++stats.raw_records_generated; stats.raw_payload_bytes_written += sizeof(Operation);
        ++cursor;
    }

    void enter(std::uint64_t key, std::uint32_t logical_rank, std::uint64_t revision,
               const std::function<void()>& callback) {
        if (active || !key || !callback || !seen.insert(key).second)
            throw ras::Error("Maya boundaries must be nonnested, nonzero, unique, and callable");
        boundary = key; rank = logical_rank; cursor = 0; old = {}; old_receipts = {};
        const auto old_state = parent ? parent->version.find_state(layers.source, key) : ras::StateId{};
        if (old_state) {
            old = read_header(parent->version, layers.source, key);
            old_receipts = parent->version.state(old_state).data.bytes();
            if (!full && (old.source.rank != rank || old.source.order != order))
                throw ras::Error("Maya boundary rank or order changed without Changes.structure_changed");
            if (!full && old.source.revision != revision && !explicitly_dirty.contains(key))
                throw ras::Error("Maya boundary revision changed without an input delta");
        } else if (parent && !full) throw ras::Error("new Maya boundary requires Changes.structure_changed");
        recording = full || selected.contains(key);
        if (recording) receipts = std::make_unique<Receipts>(tx);
        active = true;
        try {
            callback();
            active = false;
            if (failure) std::rethrow_exception(failure);
            if (fakecuda_ras_status()) throw ras::Error(fakecuda_ras_last_error());
            if (!recording && cursor != old.count) throw ras::Error("retained Maya boundary omitted operations; Changes.structure_changed required");
            if (recording && parent && cursor != old.count) {
                if (!full) throw ras::Error("Maya API count changed; Changes.structure_changed required");
                delta.structural = true;
            }
            if (recording) {
                const SourceValue value{key, revision, order, rank, 0};
                ReceiptHeader header = old;
                header.source = value;
                if (!old_state || std::memcmp(&old.source, &value, sizeof(value)) != 0) {
                    auto writer = tx.partition(layers.source, sizeof(value),
                        {source_schema, sizeof(value), alignof(SourceValue)}, old.partition);
                    auto borrow = writer.payload().reserve(sizeof(value));
                    std::memcpy(borrow.bytes().data(), &value, sizeof(value)); borrow.commit();
                    const ras::LanePosition position{tx.add_lane(layers.source, rank), order};
                    header.event = writer.event(1, {0, sizeof(value)}, {&position, 1}, old.event);
                    header.partition = writer.finish();
                }
                receipts->store(layers.source, layers.trace, key, header);
                receipts.reset();
            }
            ++order;
        } catch (...) {
            active = false; tx.abort();
            if (failure) std::rethrow_exception(failure);
            throw;
        }
    }

    ras::Version finish() {
        if (failure) std::rethrow_exception(failure);
        if (fakecuda_ras_status()) throw ras::Error(fakecuda_ras_last_error());
        for (auto [logical_rank, depth] : group_depth) {
            (void)logical_rank;
            if (depth) throw ras::Error("Maya capture ended with an open NCCL group");
        }
        if (!allocations.empty() || !streams.empty() || !events.empty() || !handles.empty() || !comms.empty())
            throw ras::Error("Maya workload must close native resources within its capture session");
        for (const auto& [token, uid] : uids) {
            (void)token;
            if (uid.nranks) throw ras::Error("Maya communicator is missing participants");
        }
        if (parent && !full) {
            const auto previous = core::state_view<CaptureSummary>(parent->version, layers.source, 0, state_schema);
            if (previous.size() != 1 || order != previous[0].boundaries)
                throw ras::Error("Maya workload omitted a retained boundary; Changes.structure_changed required");
        }
        if (parent && full) {
            std::set<ras::PartitionId> absent;
            parent->version.partitions(layers.source, [&](ras::PartitionId id) {
                const auto part = parent->version.partition(id);
                const auto data = part.payload().array<SourceValue>(source_schema);
                const auto key = data[0].key;
                if (!seen.contains(key)) {
                    absent.insert(id);
                    tx.erase_state(parent->version.find_state(layers.source, key));
                    tx.erase_state(parent->version.find_state(layers.trace, key));
                }
            });
            // All old Source outgoing edges were incident on removed Trace.
            for (auto id : absent) tx.erase_partition(id);
        }
        const CaptureSummary summary{order};
        core::put_state(tx, layers.source, 0, std::span(&summary, 1), state_schema);
        builder.finish();
        delta.changed = builder.partitions();
        stats.trace_partitions_written += delta.changed.size();
        auto version = tx.publish();
        link_sources(store, version, layers, delta.changed);
        return version;
    }
};

void Capture::boundary(std::uint64_t key, std::uint32_t rank, std::uint64_t revision,
                       const std::function<void()>& callback) {
    try { impl_->enter(key, rank, revision, callback); }
    catch (...) {
        impl_->tx.abort();
        if (!impl_->failure) impl_->failure = std::current_exception();
        throw;
    }
}

namespace {
struct SinkGuard {
    bool installed = false;
    ~SinkGuard() { if (installed) fakecuda_ras_reset(); }
};

ras::Version repack_trace(ras::Store& store, const Candidate& parent, const Policies& policies,
                          TraceDelta& delta, Statistics& stats) {
    std::set<ras::PartitionId> old;
    parent.version.partitions(parent.layers.trace, [&](ras::PartitionId id) { old.insert(id); });
    auto tx = store.begin(parent.version);
    erase_partitions(tx, parent.version, old);
    core::PartitionBuilder<Operation> builder(tx, parent.layers.trace, trace_schema, policies.trace.max_payload_bytes);
    for (auto id : old) {
        ++stats.trace_partitions_read;
        const auto partition = parent.version.partition(id);
        const auto events = partition.events();
        for (std::size_t i = 0; i != events.size(); ++i) {
            const auto& event = events[i];
            const auto value = core::event_view<Operation>(parent.version, event.id, trace_schema);
            const auto positions = partition.event_positions(i);
            builder.append(value.get(), event.kind, positions.span(), event.id);
            stats.repacked_bytes += sizeof(Operation);
        }
    }
    builder.finish();
    delta.changed = builder.partitions(); delta.removed.assign(old.begin(), old.end()); delta.layout_only = true;
    stats.trace_partitions_written += delta.changed.size();
    auto version = tx.publish();
    link_sources(store, version, parent.layers, delta.changed);
    return version;
}
} // namespace

Evaluation Evaluator::run(const Workload& workload, const CostProvider& costs, RunOptions options) {
    validate_options(options);
    if (!workload || !costs) throw ras::Error("Maya workload and cost provider must be callable");
    const auto initial_visits = store_.stats().index_page_visits;
    Evaluation result;
    auto tx = store_.begin();
    Layers layers{tx.add_layer(source_schema), tx.add_layer(trace_schema),
                  tx.add_layer(execution_schema), tx.add_layer(timing_schema)};
    Candidate seed{tx.publish(), layers, options.policies, options.context, {}};
    Changes changes;
    TraceDelta delta; delta.structural = true;
    Capture::Impl impl(store_, &seed, layers, options, changes, result.statistics, delta, true);
    // An empty seed supplies registered layers but no previous workload state.
    SinkGuard guard;
    if (fakecuda_ras_install(&impl, &Capture::Impl::sink)) throw ras::Error("another Maya capture session is active");
    guard.installed = true;
    Capture capture(impl);
    workload(capture);
    auto trace = impl.finish();
    fakecuda_ras_reset(); guard.installed = false;
    auto engine = evaluate_engine(store_, nullptr, trace, layers, options.policies, delta, {}, costs, result.statistics);
    result.candidate = {std::move(engine.version), layers, options.policies, options.context, std::move(engine.feedback)};
    result.statistics.index_visits = store_.stats().index_page_visits - initial_visits;
    return result;
}

Evaluation Evaluator::refresh(const Candidate& parent, const Workload& workload, const CostProvider& costs,
                              const Changes& changes, std::optional<RunOptions> requested) {
    const RunOptions options = requested.value_or(RunOptions{parent.policies, parent.context});
    validate_options(options);
    if (!parent.version || !workload || !costs) throw ras::Error("invalid Maya refresh input");
    // Store ownership is checked even for a no-op result.
    auto check = store_.begin(parent.version); check.abort();
    if (options.context != parent.context) return run(workload, costs, options);
    const auto initial_visits = store_.stats().index_page_visits;
    Evaluation result;
    if (changes.boundaries.empty() && changes.cost_boundaries.empty() && !changes.structure_changed && options.policies == parent.policies) {
        result.candidate = parent;
        return result;
    }
    TraceDelta delta;
    auto trace = parent.version;
    if (changes.structure_changed || !changes.boundaries.empty()) {
        delta.structural = changes.structure_changed;
        Capture::Impl impl(store_, &parent, parent.layers, options, changes, result.statistics, delta, changes.structure_changed);
        SinkGuard guard;
        if (fakecuda_ras_install(&impl, &Capture::Impl::sink)) throw ras::Error("another Maya capture session is active");
        guard.installed = true;
        Capture capture(impl);
        workload(capture);
        trace = impl.finish();
        fakecuda_ras_reset(); guard.installed = false;
        if (!changes.structure_changed && options.policies.trace != parent.policies.trace) {
            Candidate intermediate{trace, parent.layers, parent.policies, parent.context, {}};
            TraceDelta repacked;
            trace = repack_trace(store_, intermediate, options.policies, repacked, result.statistics);
            delta.removed.clear();
            parent.version.partitions(parent.layers.trace, [&](ras::PartitionId id) { delta.removed.push_back(id); });
            delta.changed = std::move(repacked.changed);
        }
    } else if (options.policies.trace != parent.policies.trace) {
        trace = repack_trace(store_, parent, options.policies, delta, result.statistics);
    }
    auto engine = evaluate_engine(store_, &parent, trace, parent.layers, options.policies,
                                   delta, changes.cost_boundaries, costs, result.statistics);
    result.candidate = {std::move(engine.version), parent.layers, options.policies, parent.context, std::move(engine.feedback)};
    result.statistics.index_visits = store_.stats().index_page_visits - initial_visits;
    return result;
}

} // namespace flexeva::maya
