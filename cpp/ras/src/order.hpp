#pragma once

#include <flexeva/ras/ras.hpp>

#include <optional>
#include <unordered_set>

#include "index.hpp"
#include "storage.hpp"

namespace flexeva::ras::detail {

// The leaf key, parent separators, reverse directory, and historical roots all
// share this immutable RAS allocation. Copying an index page copies references.
class OrderLabel {
    std::shared_ptr<const Buffer> data_;
    explicit OrderLabel(std::shared_ptr<const Buffer> data) : data_(std::move(data)) {}
public:
    OrderLabel() = default;
    std::span<const std::uint32_t> words() const noexcept {
        if (!data_) return {};
        return {reinterpret_cast<const std::uint32_t*>(data_->region.data), data_->size / sizeof(std::uint32_t)};
    }
    std::size_t size() const noexcept { return data_ ? data_->size / sizeof(std::uint32_t) : 0; }
    bool operator<(const OrderLabel& other) const noexcept {
        if (data_ == other.data_) return false;
        const auto a = words(), b = other.words();
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    }
    // Same dense-label rule as Python _between: missing left digits are zero,
    // a missing right bound is 2^32, and generated labels never end in zero.
    static OrderLabel between(const std::shared_ptr<Store>& store,
                              const OrderLabel* left, const OrderLabel* right) {
        const auto a = left ? left->words() : std::span<const std::uint32_t>{};
        const auto b = right ? right->words() : std::span<const std::uint32_t>{};
        std::size_t offset = 0;
        std::uint32_t last = 0;
        bool bounded = right != nullptr;
        for (;;) {
            const std::uint64_t low = offset < a.size() ? a[offset] : 0;
            const std::uint64_t high = bounded && offset < b.size() ? b[offset] : (std::uint64_t{1} << 32);
            if (high - low > 1) { last = std::uint32_t((low + high) / 2); break; }
            if (low != high) bounded = false;
            require(offset < std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) - 1,
                    "order label size overflow");
            ++offset;
        }
        const auto bytes = (offset + 1) * sizeof(std::uint32_t);
        auto data = std::make_shared<Buffer>(store, Layout{0, sizeof(std::uint32_t), alignof(std::uint32_t)}, bytes);
        auto* output = reinterpret_cast<std::uint32_t*>(data->region.data);
        for (std::size_t i = 0; i < offset; ++i) output[i] = i < a.size() ? a[i] : 0;
        output[offset] = last;
        return OrderLabel(std::move(data));
    }
};

// Values are unique within an order. Labels stay stable outside the splice;
// arbitrary insertion only grows labels in that gap, never renumbers others.
class Order {
    std::shared_ptr<Store> store_;
    Index<std::uint64_t, OrderLabel> forward_;
    Index<OrderLabel> reverse_;
    Index<std::size_t> lengths_;
    std::size_t label_bytes_ = 0;

    void remove_length(const OrderLabel& label) {
        const Key key{label.size(), 0};
        const auto count = *lengths_.find(key);
        if (count == 1) lengths_.erase(key); else lengths_.put(key, count - 1);
        label_bytes_ -= label.size() * sizeof(std::uint32_t);
    }
    void add_length(const OrderLabel& label) {
        const Key key{label.size(), 0};
        const auto* old = lengths_.find(key);
        lengths_.put(key, old ? *old + 1 : 1);
        const auto bytes = label.size() * sizeof(std::uint32_t);
        require(bytes <= std::numeric_limits<std::size_t>::max() - label_bytes_, "order label byte count overflow");
        label_bytes_ += bytes;
    }
    void insert(std::span<const std::uint64_t> values, const OrderLabel* left, const OrderLabel* right) {
        if (values.empty()) return;
        const auto middle = values.size() / 2;
        const auto label = OrderLabel::between(store_, left, right);
        insert(values.first(middle), left, &label);
        forward_.put(label, values[middle]);
        reverse_.put({values[middle], 0}, label);
        add_length(label);
        insert(values.subspan(middle + 1), &label, right);
    }
public:
    explicit Order(std::shared_ptr<Store> store)
        : store_(std::move(store)), forward_(store_->stats), reverse_(store_->stats), lengths_(store_->stats) {}
    std::size_t size() const noexcept { return forward_.size(); }
    std::uint64_t at(std::size_t position) const { return forward_.at(position).second; }
    std::size_t rank(std::uint64_t value) const {
        const auto* label = reverse_.find({value, 0});
        require(label != nullptr, "value absent from order");
        return forward_.rank(*label);
    }
    template<class Fn> void range(std::size_t start, std::size_t count, Fn&& fn) const {
        forward_.slice(start, count, [&](const OrderLabel&, std::uint64_t value) { fn(value); });
    }
    void splice(std::size_t start, std::size_t delete_count, std::span<const std::uint64_t> values) {
        // Standalone use retains the strong exception guarantee. Transactions
        // use splice_in_place and poison themselves if any mutation fails.
        auto next = *this;
        next.splice_in_place(start, delete_count, values);
        *this = std::move(next);
    }
    void splice_in_place(std::size_t start, std::size_t delete_count, std::span<const std::uint64_t> values) {
        require(start <= size() && delete_count <= size() - start, "order splice outside sequence");
        require(values.size() <= std::numeric_limits<std::size_t>::max() - (size() - delete_count),
                "order size overflow");
        if (delete_count == 0 && values.empty()) return;
        const auto left = start ? std::optional(forward_.at(start - 1).first) : std::nullopt;
        const auto right = start + delete_count < size()
                         ? std::optional(forward_.at(start + delete_count).first) : std::nullopt;
        std::unordered_set<std::uint64_t> unique;
        unique.reserve(values.size());
        for (auto value : values) {
            require(unique.insert(value).second, "duplicate inserted order value");
            if (const auto* label = reverse_.find({value, 0}))
                require(delete_count != 0 && (!left || *left < *label) && (!right || *label < *right),
                        "inserted value already exists outside order splice");
        }
        std::vector<std::pair<OrderLabel, std::uint64_t>> deleted;
        deleted.reserve(delete_count);
        forward_.slice(start, delete_count, [&](const OrderLabel& label, std::uint64_t value) {
            deleted.emplace_back(label, value);
        });
        // The caller must discard/poison this order after a mutation failure.
        // Historical roots remain protected by each index's path-level COW.
        for (const auto& [label, value] : deleted) {
            forward_.erase(label);
            reverse_.erase({value, 0});
            remove_length(label);
        }
        insert(values, left ? &*left : nullptr, right ? &*right : nullptr);
    }
    // Logical label bytes for this order only. Retained historical allocations
    // and sharing across index pages are accounted separately by MemoryStats.
    OrderStats stats() const {
        const auto maximum_words = lengths_.size() ? lengths_.at(lengths_.size() - 1).first.first : 0;
        return {std::size_t(maximum_words), label_bytes_};
    }
};

} // namespace flexeva::ras::detail
