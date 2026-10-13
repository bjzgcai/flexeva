#include "paper_dependencies_index.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>

namespace flexeva::maya::paper {
namespace {
namespace rt = core::runtime;
using Id = rt::Id;
constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
constexpr ras::SchemaId schema = 0x4d4159411021ULL;
struct StoredFootprint { Footprint footprint; Id resource = 0; };
struct Use { Id partition = 0, producer = 0; };
struct Interval { std::uint64_t first = 0, last = 0; };
struct EdgeCount { std::uint64_t before = 0, after = 0; };
struct PendingUse { Id resource = 0, sequence = 0, partition = 0; FootprintMode mode{}; };

void require(bool condition, const char* message) { if (!condition) throw ras::Error(message); }
template<class T> ras::Buffer write(rt::Context& context, std::span<const T> values) {
    auto writer = context.buffer(values.size_bytes(), {schema, sizeof(T), alignof(T)});
    if (!values.empty()) {
        auto output = writer.reserve(values.size_bytes());
        std::memcpy(output.bytes().data(), values.data(), values.size_bytes());
        output.commit();
    }
    return writer.finish();
}
template<class T> ras::Buffer write(rt::Context& context, const T& value) {
    return write(context, std::span<const T>(&value, 1));
}
template<class T> T scalar(const ras::DirectoryRecord& record) {
    const auto array = record.metadata.array<T>(schema);
    require(array.size() == 1, "Maya dependency index scalar extent mismatch");
    return array[0];
}
struct Directories {
    ras::DirectoryId footprints, definitions, previous_uses, next_uses, edges;
    explicit Directories(rt::Context& context)
        : footprints(context.evaluator_directory(dependency_index_slot_prefix)),
          definitions(context.evaluator_directory(dependency_index_slot_prefix + 1)),
          previous_uses(context.evaluator_directory(dependency_index_slot_prefix + 2)),
          next_uses(context.evaluator_directory(dependency_index_slot_prefix + 3)),
          edges(context.evaluator_directory(dependency_index_slot_prefix + 4)) {}
    ras::DirectoryId uses(FootprintMode mode) const {
        require(mode == FootprintMode::UsePrevious || mode == FootprintMode::UseNext,
                "Maya dependency index invalid use mode");
        return mode == FootprintMode::UsePrevious ? previous_uses : next_uses;
    }
};

class Repair {
public:
    rt::Context& context;
    Directories directories;
    std::map<DependencyResourceKey, Id> resources;
    std::map<Id, std::vector<std::uint64_t>> changed_definitions;
    std::map<Id, std::vector<Interval>> previous_intervals, next_intervals;
    std::vector<PendingUse> pending_uses;
    std::map<rt::Dependency, EdgeCount> edge_counts;

    explicit Repair(rt::Context& value) : context(value), directories(value) {}

    void adjust(Id producer, Id consumer, bool add) {
        if (!producer || producer == consumer) return;
        const rt::Dependency edge{producer, consumer};
        auto [position, inserted] = edge_counts.try_emplace(edge);
        auto& count = position->second;
        if (inserted) {
            if (const auto existing = context.metadata_find(directories.edges, {producer, consumer}))
                count.before = scalar<std::uint64_t>(*existing);
            count.after = count.before;
        }
        if (add) {
            require(count.after != maximum, "Maya dependency reference count overflow");
            ++count.after;
        } else {
            require(count.after != 0, "Maya dependency reference count underflow");
            --count.after;
        }
    }

    void remove(Id partition) {
        const auto entry = context.metadata_find(directories.footprints, {partition, 0});
        if (!entry) return;
        const auto rows = entry->metadata.array<StoredFootprint>(schema);
        for (const auto& row : rows.span()) {
            const auto& footprint = row.footprint;
            const ras::DirectoryKey key{row.resource, footprint.sequence};
            if (footprint.mode == FootprintMode::Define) {
                const auto existing = context.metadata_find(directories.definitions, key);
                require(existing && scalar<Id>(*existing) == partition,
                        "Maya dependency definition ownership mismatch");
                context.metadata_erase(directories.definitions, key);
                changed_definitions[row.resource].push_back(footprint.sequence);
            } else {
                const auto directory = directories.uses(footprint.mode);
                const auto existing = context.metadata_find(directory, key);
                require(bool(existing), "Maya dependency use absent from index");
                const auto use = scalar<Use>(*existing);
                require(use.partition == partition, "Maya dependency use ownership mismatch");
                adjust(use.producer, partition, false);
                context.metadata_erase(directory, key);
            }
        }
        context.metadata_erase(directories.footprints, {partition, 0});
    }

    void insert(Id partition, std::span<const Footprint> footprints) {
        std::vector<StoredFootprint> rows;
        rows.reserve(footprints.size());
        for (const auto& footprint : footprints) {
            auto [position, inserted] = resources.try_emplace(footprint.resource, 0);
            if (inserted)
                position->second = context.identity(dependency_index_slot_prefix + 16, footprint.resource);
            const auto resource = position->second;
            const ras::DirectoryKey key{resource, footprint.sequence};
            if (footprint.mode == FootprintMode::Define) {
                require(!context.metadata_find(directories.definitions, key),
                        "Maya dependency duplicate definition at one resource sequence");
                context.metadata_put(directories.definitions, key, {write(context, partition), {}, {}});
                changed_definitions[resource].push_back(footprint.sequence);
            } else {
                directories.uses(footprint.mode); // Validate before publishing footprint metadata.
                pending_uses.push_back({resource, footprint.sequence, partition, footprint.mode});
            }
            rows.push_back({footprint, resource});
        }
        if (!rows.empty())
            context.metadata_put(directories.footprints, {partition, 0},
                                 {write(context, std::span<const StoredFootprint>(rows)), {}, {}});
    }

    void build_intervals(bool previous_uses, bool next_uses) {
        if (!previous_uses && !next_uses) return;
        for (auto& [resource, sequences] : changed_definitions) {
            std::sort(sequences.begin(), sequences.end());
            sequences.erase(std::unique(sequences.begin(), sequences.end()), sequences.end());
            for (const auto sequence : sequences) {
                if (previous_uses && sequence != maximum) {
                    const auto next = context.metadata_next(directories.definitions, {resource, sequence + 1});
                    const auto last = next && next->first.first == resource ? next->first.second : maximum;
                    previous_intervals[resource].push_back({sequence + 1, last});
                }
                if (next_uses && sequence != 0) {
                    const auto previous = context.metadata_prev(directories.definitions, {resource, sequence});
                    const auto first = previous && previous->first.first == resource ? previous->first.second : 0;
                    next_intervals[resource].push_back({first, sequence - 1});
                }
            }
        }
    }

    Id producer(ras::DirectoryKey key, FootprintMode mode) {
        std::optional<std::pair<ras::DirectoryKey, ras::DirectoryRecord>> definition;
        if (mode == FootprintMode::UsePrevious)
            definition = context.metadata_prev(directories.definitions, key);
        else if (key.second != maximum)
            definition = context.metadata_next(directories.definitions, {key.first, key.second + 1});
        return definition && definition->first.first == key.first ? scalar<Id>(definition->second) : 0;
    }

    void resolve(ras::DirectoryKey key, const ras::DirectoryRecord& record,
                 FootprintMode mode, ras::DirectoryId uses) {
        auto use = scalar<Use>(record);
        const auto current_producer = producer(key, mode);
        if (current_producer == use.producer) return;
        adjust(use.producer, use.partition, false);
        adjust(current_producer, use.partition, true);
        use.producer = current_producer;
        context.metadata_put(uses, key, {write(context, use), {}, {}});
    }

    void repair(std::map<Id, std::vector<Interval>>& by_resource, FootprintMode mode) {
        const auto uses = directories.uses(mode);
        for (auto& [resource, intervals] : by_resource) {
            std::sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            std::vector<Interval> merged;
            for (const auto interval : intervals) {
                if (merged.empty() || (merged.back().last != maximum && interval.first > merged.back().last + 1))
                    merged.push_back(interval);
                else merged.back().last = std::max(merged.back().last, interval.last);
            }
            for (const auto interval : merged)
                context.metadata_range(uses, {resource, interval.first}, {resource, interval.last},
                    [&](ras::DirectoryKey key, const ras::DirectoryRecord& record) {
                        resolve(key, record, mode, uses);
                    });
        }
    }

    rt::DependencyDelta finish() {
        const bool previous_uses = bool(context.metadata_next(directories.previous_uses, {}));
        const bool next_uses = bool(context.metadata_next(directories.next_uses, {}));
        build_intervals(previous_uses, next_uses);
        repair(previous_intervals, FootprintMode::UsePrevious);
        repair(next_intervals, FootprintMode::UseNext);
        // New uses resolve against the completed definition delta. Never write
        // a placeholder and immediately read/rewrite it during initialization.
        for (const auto& pending : pending_uses) {
            const auto directory = directories.uses(pending.mode);
            const ras::DirectoryKey key{pending.resource, pending.sequence};
            if (const auto existing = context.metadata_find(directory, key))
                throw ras::Error("Maya dependency duplicate use at resource identity " +
                    std::to_string(pending.resource) + " sequence " + std::to_string(pending.sequence) +
                    " partitions " + std::to_string(scalar<Use>(*existing).partition) + "/" +
                    std::to_string(pending.partition));
            const auto source = producer(key, pending.mode);
            context.metadata_put(directory, key, {write(context, Use{pending.partition, source}), {}, {}});
            adjust(source, pending.partition, true);
        }
        rt::DependencyDelta delta;
        for (const auto& [edge, count] : edge_counts) {
            if (count.before == count.after) continue;
            if (!count.before && count.after) delta.upserts.push_back(edge);
            if (count.before && !count.after) delta.removed.push_back(edge);
            const ras::DirectoryKey key{edge.producer, edge.consumer};
            if (count.after) context.metadata_put(directories.edges, key, {write(context, count.after), {}, {}});
            else context.metadata_erase(directories.edges, key);
        }
        return delta;
    }
};
} // namespace

rt::DependencyDelta update_dependencies(rt::Context& context, std::span<const IndexUpdate> updates,
                                        std::span<const Id> removed) {
    if (updates.empty() && removed.empty()) return {};
    std::set<Id> changed;
    for (const auto& update : updates)
        require(update.partition && changed.insert(update.partition).second,
                "Maya dependency duplicate or null partition update");
    for (const auto partition : removed)
        require(partition && changed.insert(partition).second,
                "Maya dependency overlapping removal or null partition");
    Repair repair(context);
    std::vector<IndexUpdate> pending;
    for (const auto& update : updates) {
        auto footprints = update.footprints;
        std::sort(footprints.begin(), footprints.end());
        footprints.erase(std::unique(footprints.begin(), footprints.end()), footprints.end());
        const auto existing = context.metadata_find(repair.directories.footprints, {update.partition, 0});
        if (existing) {
            const auto rows = existing->metadata.array<StoredFootprint>(schema);
            if (rows.size() == footprints.size() && std::equal(rows.span().begin(), rows.span().end(), footprints.begin(),
                    [](const auto& row, const auto& footprint) { return row.footprint == footprint; })) continue;
        } else if (footprints.empty()) continue;
        repair.remove(update.partition);
        pending.push_back({update.partition, std::move(footprints)});
    }
    for (const auto partition : removed) repair.remove(partition);
    for (const auto& update : pending) repair.insert(update.partition, update.footprints);
    return repair.finish();
}

std::vector<Footprint> read_dependency_footprints(rt::Context& context, Id partition) {
    const auto directory = context.evaluator_directory(dependency_index_slot_prefix);
    const auto entry = context.metadata_find(directory, {partition, 0});
    if (!entry) return {};
    const auto rows = entry->metadata.array<StoredFootprint>(schema);
    std::vector<Footprint> result;
    result.reserve(rows.size());
    for (const auto& row : rows.span()) result.push_back(row.footprint);
    return result;
}

} // namespace flexeva::maya::paper
