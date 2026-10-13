#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <new>

#include "index.hpp"

namespace flexeva::ras::detail {

constexpr std::size_t block_size = 64 * 1024;
constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
inline void require(bool condition, const char* message) { if (!condition) throw Error(message); }
inline void bounds(Range range, std::size_t size) {
    require(range.offset <= size && range.size <= size - range.offset, "range outside stored data");
}
inline void layout_valid(Layout layout) {
    require(layout.element_size > 0 && std::has_single_bit(layout.alignment) &&
            layout.element_size % layout.alignment == 0, "invalid buffer layout");
}

struct Charge {
    std::shared_ptr<MemoryStats> stats;
    std::size_t bytes;
    Charge(std::shared_ptr<MemoryStats> s, std::size_t n) : stats(std::move(s)), bytes(n) { stats->metadata_bytes += bytes; }
    ~Charge() { stats->metadata_bytes -= bytes; }
    Charge(const Charge&) = delete;
    Charge& operator=(const Charge&) = delete;
};

enum class Use : std::size_t { Payload, Metadata };

// Only non-full slabs appear in a size-class list. Full slabs are owned by
// their live regions; the last release deletes the slab, with no retained cache.
struct Slab {
    Charge charge;
    std::byte* data;
    std::size_t stride, bucket, next_unused = 0, live = 0;
    Use use;
    std::byte* free = nullptr;
    Slab *previous = nullptr, *next = nullptr;
    Slab(std::shared_ptr<MemoryStats> stats, std::size_t size, std::size_t index, Use kind)
        : charge(std::move(stats), sizeof(Slab)),
          data(static_cast<std::byte*>(::operator new(block_size, std::align_val_t(size)))),
          stride(size), bucket(index), use(kind) {
        charge.stats->capacity_bytes += block_size;
        ++charge.stats->blocks;
    }
    ~Slab() {
        ::operator delete(data, std::align_val_t(stride));
        charge.stats->capacity_bytes -= block_size;
        --charge.stats->blocks;
    }
    std::size_t slots() const { return block_size / stride; }
};

// The owning Buffer/metadata object keeps Store alive until it returns this
// region. No region points back to an owning handle, avoiding ownership cycles.
struct Region { std::byte* data = nullptr; Slab* slab = nullptr; };

class Pool {
    std::array<std::array<Slab*, 12>, 2> partial_{};
    Slab*& head(const Slab* slab) { return partial_[std::size_t(slab->use)][slab->bucket]; }
    void link(Slab* slab) noexcept {
        auto& first = head(slab);
        slab->previous = nullptr; slab->next = first;
        if (first) first->previous = slab;
        first = slab;
    }
    void unlink(Slab* slab) noexcept {
        if (slab->previous) slab->previous->next = slab->next;
        else head(slab) = slab->next;
        if (slab->next) slab->next->previous = slab->previous;
        slab->previous = slab->next = nullptr;
    }
    static std::size_t alignment(std::size_t requested) { return std::max(requested, alignof(std::max_align_t)); }
public:
    std::shared_ptr<MemoryStats> stats;
    explicit Pool(std::shared_ptr<MemoryStats> s) : stats(std::move(s)) {}
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Region allocate(std::size_t bytes, std::size_t align, Use use) {
        if (!bytes) return {};
        ++stats->allocation_requests;
        Region result;
        if (bytes <= block_size / 2 && align <= block_size / 2) {
            const auto stride = std::bit_ceil(std::max({bytes, align, std::size_t(16)}));
            const auto bucket = std::countr_zero(stride) - 4;
            auto* slab = partial_[std::size_t(use)][bucket];
            if (!slab) { slab = new Slab(stats, stride, bucket, use); link(slab); }
            if (slab->free) {
                result.data = slab->free;
                std::memcpy(&slab->free, result.data, sizeof(slab->free));
                if (use == Use::Payload) ++stats->reused_payload_slots;
            } else result.data = slab->data + slab->next_unused++ * stride;
            result.slab = slab;
            if (++slab->live == slab->slots()) unlink(slab);
        } else {
            result.data = static_cast<std::byte*>(::operator new(bytes, std::align_val_t(alignment(align))));
            stats->capacity_bytes += bytes;
            ++stats->blocks;
        }
        stats->peak_capacity_bytes = std::max(stats->peak_capacity_bytes, stats->capacity_bytes);
        return result;
    }
    void release(Region region, std::size_t bytes, std::size_t align) noexcept {
        if (!region.data) return;
        if (auto* slab = region.slab) {
            if (slab->live == slab->slots()) link(slab);
            std::memcpy(region.data, &slab->free, sizeof(slab->free));
            slab->free = region.data;
            if (--slab->live == 0) { unlink(slab); delete slab; }
        } else {
            ::operator delete(region.data, std::align_val_t(alignment(align)));
            stats->capacity_bytes -= bytes;
            --stats->blocks;
        }
    }
};

struct IdentityWords {
    std::vector<std::uint64_t> values;
    Charge charge;
    IdentityWords(std::shared_ptr<MemoryStats> stats, std::span<const std::uint64_t> coordinate)
        : values(coordinate.begin(), coordinate.end()),
          charge(std::move(stats), sizeof(IdentityWords) + values.capacity() * sizeof(std::uint64_t)) {}
};
struct IdentityCoordinate {
    std::uint64_t domain = 0;
    std::span<const std::uint64_t> words;
    std::shared_ptr<const IdentityWords> owner;
    friend bool operator<(const IdentityCoordinate& a, const IdentityCoordinate& b) {
        if (a.domain != b.domain) return a.domain < b.domain;
        return std::lexicographical_compare(a.words.begin(), a.words.end(), b.words.begin(), b.words.end());
    }
};

struct Store {
    const std::uint64_t id;
    std::shared_ptr<MemoryStats> stats = std::make_shared<MemoryStats>();
    Pool pool{stats};
    Index<ChangeId, IdentityCoordinate> identities{stats};
    std::uint64_t next_id = 1;
    static std::uint64_t identity() {
        static std::atomic<std::uint64_t> next{1};
        const auto value = next.fetch_add(1, std::memory_order_relaxed);
        if (value == 0 || value == maximum) std::terminate();
        return value;
    }
    Store() : id(identity()) {}
    template<class I> I fresh() {
        require(next_id != maximum, "RAS identifier space exhausted");
        return {id, next_id++};
    }
    ChangeId intern(std::uint64_t domain, std::span<const std::uint64_t> coordinate) {
        if (const auto* found = identities.find({domain, coordinate, {}})) return *found;
        auto owner = std::make_shared<IdentityWords>(stats, coordinate);
        const auto id = fresh<ChangeId>();
        auto updated = identities;
        updated.put({domain, owner->values, owner}, id);
        identities = std::move(updated);
        return id;
    }
    template<class I> void check(I key) const { require(key && key.store == id, "foreign or null RAS handle"); }
};

// One owner object per immutable payload, with no segment/Allocation wrappers.
struct Buffer {
    Charge charge;
    std::shared_ptr<Store> store;
    const std::uint64_t allocation;
    Layout layout;
    std::size_t size;
    Region region;
    Buffer(std::shared_ptr<Store> s, Layout l, std::size_t bytes)
        : charge(s->stats, sizeof(Buffer)), store(std::move(s)),
          allocation(store->fresh<Id<struct BufferAllocationTag>>().value), layout(l), size(bytes),
          region(store->pool.allocate(bytes, l.alignment, Use::Payload)) {
        charge.stats->payload_bytes += size;
        charge.stats->allocated_payload_bytes += size;
        charge.stats->live_payload_allocations += size != 0;
    }
    ~Buffer() {
        store->pool.release(region, size, layout.alignment);
        charge.stats->payload_bytes -= size;
        charge.stats->live_payload_allocations -= size != 0;
    }
};

// Only framework metadata grows. The one-record case is inline; larger arrays
// use the same RAS pool, doubling during production without touching payload.
template<class T> class MetaArray {
    union Storage {
        T one;
        Region region;
        Storage() : one{} {}
    } storage_;
    std::size_t size_ = 0, capacity_ = 1;
    T* data() { return capacity_ == 1 ? &storage_.one : reinterpret_cast<T*>(storage_.region.data); }
public:
    MetaArray() = default;
    MetaArray(const MetaArray&) = delete;
    MetaArray& operator=(const MetaArray&) = delete;
    std::size_t size() const { return size_; }
    std::span<const T> span() const {
        return {capacity_ == 1 ? &storage_.one : reinterpret_cast<const T*>(storage_.region.data), size_};
    }
    void push(Pool& pool, const T& value) {
        if (size_ == capacity_) {
            require(capacity_ <= std::numeric_limits<std::size_t>::max() / sizeof(T) / 2, "metadata size overflow");
            const auto next = capacity_ * 2;
            auto region = pool.allocate(next * sizeof(T), alignof(T), Use::Metadata);
            std::memcpy(region.data, data(), size_ * sizeof(T));
            pool.stats->copied_metadata_bytes += size_ * sizeof(T);
            if (capacity_ > 1) {
                pool.release(storage_.region, capacity_ * sizeof(T), alignof(T));
                pool.stats->metadata_array_bytes -= capacity_ * sizeof(T);
            }
            storage_.region = region;
            capacity_ = next;
            pool.stats->metadata_array_bytes += capacity_ * sizeof(T);
        }
        std::memcpy(data() + size_++, &value, sizeof(T));
    }
    void release(Pool& pool) noexcept {
        if (capacity_ > 1) {
            pool.release(storage_.region, capacity_ * sizeof(T), alignof(T));
            pool.stats->metadata_array_bytes -= capacity_ * sizeof(T);
        }
    }
};

struct PartitionMetadata {
    Charge charge;
    std::shared_ptr<Store> store;
    MetaArray<Event> events;
    MetaArray<LanePosition> positions;
    std::size_t payload_extent = 0;
    explicit PartitionMetadata(std::shared_ptr<Store> s)
        : charge(s->stats, sizeof(PartitionMetadata)), store(std::move(s)) {}
    ~PartitionMetadata() { events.release(store->pool); positions.release(store->pool); }
};

} // namespace flexeva::ras::detail
