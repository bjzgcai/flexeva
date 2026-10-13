#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace flexeva::ras::detail {

struct Key {
    std::uint64_t first = 0, second = 0;
    auto operator<=>(const Key&) const = default;
};

// Mutable only when uniquely owned by a transaction. Published roots share
// immutable pages; subsequent writes copy their search path, never a catalog.
// Keys need only provide a strict ordering; separators share key ownership.
template<class T, class K = Key> class Index {
    static constexpr std::size_t fanout = 32;
    struct Page;
    using Ptr = std::shared_ptr<Page>;
    using Entry = std::pair<K, T>;
    struct Child { K maximum; Ptr page; };
    struct Page {
        std::shared_ptr<MemoryStats> stats;
        bool leaf;
        std::vector<Entry> entries;
        std::vector<Child> children;
        std::size_t charge, count = 0;
        Page(std::shared_ptr<MemoryStats> s, bool l) : stats(std::move(s)), leaf(l) {
            if (leaf) entries.reserve(fanout + 1); else children.reserve(fanout + 1);
            charge = sizeof(Page) + entries.capacity() * sizeof(Entry) + children.capacity() * sizeof(Child);
            stats->index_bytes += charge;
            ++stats->index_pages;
            ++stats->created_index_pages;
        }
        ~Page() { stats->index_bytes -= charge; --stats->index_pages; }
        K maximum() const { return leaf ? entries.back().first : children.back().maximum; }
        std::size_t size() const { return leaf ? entries.size() : children.size(); }
        void recount() {
            count = entries.size();
            for (const auto& child : children) count += child.page->count;
        }
    };
public:
    explicit Index(std::shared_ptr<MemoryStats> s = {}) : stats_(std::move(s)) {}
    std::size_t size() const noexcept { return size_; }
    const T* find(const K& key) const {
        auto* page = root_.get();
        while (page) {
            ++stats_->index_page_visits;
            if (page->leaf) {
                auto it = lower(page->entries, key);
                return it != page->entries.end() && equal(it->first, key) ? &it->second : nullptr;
            }
            auto it = child_lower(page->children, key);
            if (it == page->children.end()) return nullptr;
            page = it->page.get();
        }
        return nullptr;
    }
    // Copying a key/value keeps any owning references valid across mutations.
    Entry at(std::size_t position) const {
        if (position >= size_) throw Error("index position outside sequence");
        const auto* page = root_.get();
        for (;;) {
            ++stats_->index_page_visits;
            if (page->leaf) return page->entries[position];
            for (const auto& child : page->children) {
                if (position < child.page->count) { page = child.page.get(); break; }
                position -= child.page->count;
            }
        }
    }
    // Number of keys strictly less than key (also its insertion rank).
    std::size_t rank(const K& key) const {
        std::size_t result = 0;
        const auto* page = root_.get();
        while (page) {
            ++stats_->index_page_visits;
            if (page->leaf) return result + std::size_t(lower(page->entries, key) - page->entries.begin());
            const Page* next = nullptr;
            for (const auto& child : page->children) {
                if (!(child.maximum < key)) { next = child.page.get(); break; }
                result += child.page->count;
            }
            page = next;
        }
        return result;
    }
    void put(const K& key, const T& value) {
        bool added = false;
        if (!root_) root_ = make(true);
        auto right = insert(root_, key, value, added);
        if (right) {
            auto next = make(false);
            next->children.push_back({root_->maximum(), root_});
            next->children.push_back({right->maximum(), std::move(right)});
            next->recount();
            root_ = std::move(next);
        }
        size_ += added;
    }
    bool erase(const K& key) {
        if (!find(key)) return false;
        remove(root_, key);
        --size_;
        if (root_ && root_->size() == 0) root_.reset();
        while (root_ && !root_->leaf && root_->children.size() == 1) {
            auto next = root_->children.front().page;
            root_ = std::move(next);
        }
        return true;
    }
    // Own only the existing key's path before editing its value. The callback
    // must not structurally mutate this index or retain the value reference.
    // A throwing callback requires its enclosing transaction to be abandoned.
    template<class Fn> bool mutate_existing(const K& key, Fn&& fn) {
        if (!find(key)) return false;
        mutate_value(root_, key, fn);
        return true;
    }
    template<class Fn> void range(const K& low, const K& high, Fn&& fn) const {
        // Retaining the root also protects against reentrant mutation.
        auto pin = root_;
        if (!(high < low)) walk(pin.get(), low, high, fn);
    }
    template<class Fn> void slice(std::size_t start, std::size_t count, Fn&& fn) const {
        if (start > size_ || count > size_ - start) throw Error("index slice outside sequence");
        auto pin = root_;
        if (count) walk_positions(pin.get(), start, count, fn);
    }
private:
    std::shared_ptr<MemoryStats> stats_;
    Ptr root_;
    std::size_t size_ = 0;
    static bool equal(const K& a, const K& b) { return !(a < b) && !(b < a); }
    static auto lower(auto& entries, const K& key) {
        return std::lower_bound(entries.begin(), entries.end(), key,
                                [](const auto& entry, const K& k) { return entry.first < k; });
    }
    static auto child_lower(auto& children, const K& key) {
        return std::lower_bound(children.begin(), children.end(), key,
                                [](const auto& child, const K& k) { return child.maximum < k; });
    }
    Ptr make(bool leaf) { return std::make_shared<Page>(stats_, leaf); }
    void own(Ptr& page) {
        if (page.unique()) return;
        auto copy = make(page->leaf);
        copy->entries = page->entries;
        copy->children = page->children;
        copy->count = page->count;
        page = std::move(copy);
    }
    template<class Fn> void mutate_value(Ptr& page, const K& key, Fn& fn) {
        ++stats_->index_page_visits;
        own(page);
        if (page->leaf) fn(lower(page->entries, key)->second);
        else mutate_value(child_lower(page->children, key)->page, key, fn);
    }
    Ptr insert(Ptr& page, const K& key, const T& value, bool& added) {
        ++stats_->index_page_visits;
        own(page);
        if (page->leaf) {
            auto it = lower(page->entries, key);
            if (it != page->entries.end() && equal(it->first, key)) it->second = value;
            else { page->entries.insert(it, {key, value}); added = true; }
        } else {
            auto it = child_lower(page->children, key);
            if (it == page->children.end()) --it;
            const auto position = std::size_t(it - page->children.begin());
            auto right = insert(it->page, key, value, added);
            it->maximum = it->page->maximum();
            if (right) page->children.insert(page->children.begin() + position + 1,
                                             {right->maximum(), std::move(right)});
        }
        if (page->size() <= fanout) { page->recount(); return {}; }
        auto right = make(page->leaf);
        if (page->leaf) split(page->entries, right->entries);
        else split(page->children, right->children);
        page->recount(); right->recount();
        return right;
    }
    static void split(auto& left, auto& right) {
        auto middle = left.begin() + left.size() / 2;
        right.insert(right.end(), std::make_move_iterator(middle), std::make_move_iterator(left.end()));
        left.erase(middle, left.end());
    }
    void remove(Ptr& page, const K& key) {
        ++stats_->index_page_visits;
        own(page);
        if (page->leaf) { page->entries.erase(lower(page->entries, key)); page->recount(); return; }
        auto it = child_lower(page->children, key);
        const auto position = std::size_t(it - page->children.begin());
        remove(it->page, key);
        if (it->page->size() == 0) page->children.erase(it);
        else it->maximum = it->page->maximum();
        if (page->children.size() >= 2) {
            const auto left = std::min(position, page->children.size() - 2);
            auto& a = page->children[left];
            auto& b = page->children[left + 1];
            if (a.page->size() + b.page->size() <= fanout) {
                own(a.page);
                if (a.page->leaf)
                    a.page->entries.insert(a.page->entries.end(), b.page->entries.begin(), b.page->entries.end());
                else a.page->children.insert(a.page->children.end(), b.page->children.begin(), b.page->children.end());
                a.page->recount();
                a.maximum = a.page->maximum();
                page->children.erase(page->children.begin() + left + 1);
            }
        }
        page->recount();
    }
    template<class Fn> static void walk(const Page* page, const K& low, const K& high, Fn& fn) {
        if (!page) return;
        ++page->stats->index_page_visits;
        if (page->leaf) {
            for (auto it = lower(page->entries, low); it != page->entries.end() && !(high < it->first); ++it)
                fn(it->first, it->second);
        } else {
            for (auto it = child_lower(page->children, low); it != page->children.end(); ++it) {
                walk(it->page.get(), low, high, fn);
                if (!(it->maximum < high)) break;
            }
        }
    }
    template<class Fn> static void walk_positions(const Page* page, std::size_t start,
                                                 std::size_t count, Fn& fn) {
        ++page->stats->index_page_visits;
        if (page->leaf) {
            for (std::size_t i = start; i < start + count; ++i) fn(page->entries[i].first, page->entries[i].second);
            return;
        }
        for (const auto& child : page->children) {
            if (start >= child.page->count) { start -= child.page->count; continue; }
            const auto take = std::min(count, child.page->count - start);
            walk_positions(child.page.get(), start, take, fn);
            count -= take;
            if (!count) break;
            start = 0;
        }
    }
};

} // namespace flexeva::ras::detail
