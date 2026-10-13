#pragma once

#include <flexeva/ras/ras.hpp>

#include <algorithm>
#include <cstring>
#include <optional>
#include <vector>

namespace flexeva::core {

// Batches fixed-size evaluator records directly into one active RAS partition.
// Capacity is a physical choice; owner changes do not alter these boundaries.
// Finish explicitly before publishing. A failed or abandoned nonempty builder
// aborts its transaction, so partially sealed output cannot become a candidate.
template<class T> class PartitionBuilder {
    static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
public:
    PartitionBuilder(ras::Transaction& tx, ras::LayerId layer, ras::SchemaId schema,
                     std::size_t max_payload_bytes)
        : tx_(tx), layer_(layer), schema_(schema),
          capacity_(std::max(std::size_t(1), max_payload_bytes / sizeof(T)) * sizeof(T)) {}
    PartitionBuilder(const PartitionBuilder&) = delete;
    PartitionBuilder& operator=(const PartitionBuilder&) = delete;
    PartitionBuilder(PartitionBuilder&&) = delete;
    PartitionBuilder& operator=(PartitionBuilder&&) = delete;
    ~PartitionBuilder() {
        if (!finished_ && (active_ || !partitions_.empty())) tx_.abort();
    }

    ras::EventId append(const T& value, std::uint64_t kind,
                        std::span<const ras::LanePosition> positions, ras::EventId preserve = {}) {
        try {
            if (finished_) throw ras::Error("partition builder is finished");
            if (!active_) active_.emplace(tx_.partition(layer_, capacity_, {schema_, sizeof(T), alignof(T)}));
            const auto id = preserve.store == 0 && preserve.value == 0 ? tx_.new_event_id() : preserve;
            auto borrow = active_->payload().reserve(sizeof(T));
            const auto offset = borrow.offset();
            std::memcpy(borrow.bytes().data(), &value, sizeof(T));
            borrow.commit();
            active_->event(kind, {offset, sizeof(T)}, positions, id);
            if (active_->payload().size() == capacity_) seal();
            return id;
        } catch (...) { tx_.abort(); throw; }
    }

    void finish() {
        if (finished_) return;
        try {
            if (active_) seal();
            finished_ = true;
        } catch (...) { tx_.abort(); throw; }
    }
    const std::vector<ras::PartitionId>& partitions() const noexcept { return partitions_; }

private:
    void seal() {
        const auto remaining = capacity_ - active_->payload().size();
        if (remaining) {
            auto tail = active_->payload().reserve(remaining);
            std::memset(tail.bytes().data(), 0, remaining);
            tail.commit();
        }
        const auto id = active_->finish();
        active_.reset();
        partitions_.push_back(id);
    }
    ras::Transaction& tx_;
    ras::LayerId layer_;
    ras::SchemaId schema_;
    std::size_t capacity_;
    std::optional<ras::PartitionWriter> active_;
    std::vector<ras::PartitionId> partitions_;
    bool finished_ = false;
};

template<class T> class EventView {
    static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
public:
    explicit EventView(ras::ByteView bytes) : bytes_(std::move(bytes)) {
        if (bytes_.size() != sizeof(T) ||
            reinterpret_cast<std::uintptr_t>(bytes_.span().data()) % alignof(T) != 0)
            throw ras::Error("event payload size or alignment does not match requested type");
    }
    // The returned reference remains valid while this view or another owner of
    // its allocation remains alive. Moving the view transfers that ownership.
    const T& get() const {
        if (bytes_.size() != sizeof(T)) throw ras::Error("empty event view");
        return *reinterpret_cast<const T*>(bytes_.span().data());
    }
private:
    ras::ByteView bytes_;
};

template<class T> EventView<T> event_view(const ras::Version& version, ras::EventId id, ras::SchemaId schema) {
    const auto endpoint = version.locate(id);
    const auto partition = version.partition(endpoint.partition);
    const auto layout = partition.payload().layout();
    if (layout.schema != schema || layout.element_size != sizeof(T) || layout.alignment < alignof(T))
        throw ras::Error("event layout does not match requested type");
    return EventView<T>(partition.event_payload(endpoint.event));
}

template<class T> ras::StateId put_state(ras::Transaction& tx, ras::LayerId layer, std::uint64_t key,
                                        std::span<const T> values, ras::SchemaId schema) {
    try {
        auto writer = tx.array<T>(schema, values.size());
        if (!values.empty()) {
            auto borrow = writer.reserve(values.size_bytes());
            std::memcpy(borrow.bytes().data(), values.data(), values.size_bytes());
            borrow.commit();
        }
        return tx.put_state(layer, key, writer.finish());
    } catch (...) { tx.abort(); throw; }
}

template<class T> ras::ArrayView<T> state_view(const ras::Version& version, ras::LayerId layer,
                                             std::uint64_t key, ras::SchemaId schema) {
    const auto id = version.find_state(layer, key);
    if (!id) throw ras::Error("named RAS state is absent");
    return version.state(id).data.template array<T>(schema);
}

} // namespace flexeva::core
