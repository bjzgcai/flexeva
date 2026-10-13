#include <flexeva/ras/ras.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace flexeva::ras;
using Clock = std::chrono::steady_clock;

static void check(bool condition, const char* message) {
    if (!condition) throw Error(message);
}

static std::size_t payload_size(std::size_t index, std::uint32_t generation) {
    auto residue = index % 1000;
    std::size_t size = residue == 999 ? 1024 * 1024 : residue >= 900 ? 4096 : 64;
    return size * (1 + generation % 2);
}

static PartitionId emit(Transaction& tx, LayerId layer, LaneId lane, std::size_t index,
                        std::uint32_t generation, PartitionId replace = {}) {
    const auto size = payload_size(index, generation);
    auto writer = tx.partition(layer, size, {101, 1, 1}, replace);
    auto bytes = writer.payload().reserve(size);
    std::memset(bytes.bytes().data(), int((index + generation * 7) % 251), size);
    bytes.commit();
    LanePosition position{lane, index};
    writer.event(1, {0, size}, {&position, 1});
    return writer.finish();
}

static std::size_t tracked(const MemoryStats& stats) {
    return stats.capacity_bytes + stats.index_bytes + stats.metadata_bytes;
}

static void released(const MemoryStats& stats) {
    check(tracked(stats) == 0 && stats.payload_bytes == 0 && stats.blocks == 0 &&
          stats.versions == 0 && stats.pending_writers == 0 && stats.pending_reservations == 0,
          "RAS storage remains after the last view and version were released");
}

static void run(std::size_t count) {
    check(count >= 100 && count % 2 == 0, "partition count must be even and at least 100");
    constexpr std::size_t rounds = 1000, retain = 8, changes = 32;
    Store store;
    ByteView pin;
    MemoryStats initial, final;
    std::uint64_t generated = 0, emitted = 0, page_visits = 0;
    std::size_t peak_tracked = 0;
    std::vector<double> timings;
    auto start = Clock::now();
    {
        const auto pairs = count / 2;
        std::vector<PartitionId> upper(pairs), lower(pairs);
        std::vector<std::uint32_t> generations(pairs);
        auto tx = store.begin();
        auto up_layer = tx.add_layer(1), low_layer = tx.add_layer(2);
        auto up_lane = tx.add_lane(up_layer, 0), low_lane = tx.add_lane(low_layer, 0);
        std::size_t base_payload = 0;
        for (std::size_t i = 0; i < pairs; ++i) {
            upper[i] = emit(tx, up_layer, up_lane, i, 0);
            lower[i] = emit(tx, low_layer, low_lane, i, 0);
            tx.add_relation({{upper[i]}, {lower[i]}, RelationKind::Lineage});
            base_payload += 2 * payload_size(i, 0);
        }
        std::deque<Version> history;
        history.push_back(tx.publish());
        pin = history.back().partition(lower[0]).payload().bytes();
        initial = store.stats();
        peak_tracked = tracked(initial);
        std::mt19937_64 random(917);
        const auto capacity_bound = 3 * base_payload + (changes * 2 * (retain + 1) + 1) *
                                    std::size_t(2 * 1024 * 1024) + 12 * 65536;
        const auto depth = std::size_t(std::ceil(std::log(double(count)) / std::log(16.0))) + 1;
        const auto index_bound = initial.index_bytes + (retain + 1) * changes * 2 * depth * 8192;
        for (std::size_t round = 0; round < rounds; ++round) {
            std::array<std::size_t, changes> selected{};
            for (std::size_t n = 0; n < changes; ++n) {
                std::size_t candidate;
                do { candidate = random() % pairs; }
                while (std::find(selected.begin(), selected.begin() + n, candidate) != selected.begin() + n);
                selected[n] = candidate;
            }
            const auto before = store.stats();
            auto tick = Clock::now();
            auto update = store.begin(history.back());
            for (auto index : selected) {
                auto generation = ++generations[index];
                emit(update, up_layer, up_lane, index, generation, upper[index]);
                ++emitted; generated += payload_size(index, generation);
                std::size_t found = 0;
                history.back().outgoing(upper[index], [&](RelationId, const Relation& relation) {
                    check(relation.kind == RelationKind::Lineage && relation.target.partition == lower[index],
                          "AL lookup returned an unexpected partition");
                    emit(update, low_layer, low_lane, index, generation, relation.target.partition);
                    ++emitted; generated += payload_size(index, generation); ++found;
                });
                check(found == 1, "AL must locate exactly one affected lower partition");
            }
            history.push_back(update.publish());
            if (history.size() > retain) history.pop_front();
            timings.push_back(std::chrono::duration<double, std::micro>(Clock::now() - tick).count());
            const auto after = store.stats();
            check(after.copied_payload_bytes == before.copied_payload_bytes, "refresh copied payload");
            check(after.view_requests == before.view_requests, "refresh read an existing payload");
            check(after.live_payload_allocations <= count + changes * 2 * (retain - 1) + 1,
                  "old payload allocations accumulated beyond retained history");
            check(after.capacity_bytes <= capacity_bound, "payload pool exceeded its bounded working set");
            check(after.index_bytes <= index_bound, "index storage accumulated beyond retained paths");
            check(after.index_page_visits - before.index_page_visits <= changes * 2 * depth * 64,
                  "refresh visited more than the affected index paths");
            page_visits += after.index_page_visits - before.index_page_visits;
            peak_tracked = std::max(peak_tracked, tracked(after));
            // These reads validate results after the measured refresh window.
            const auto index = selected.front();
            const auto view = history.back().partition(lower[index]).payload().bytes();
            const auto expected = std::byte((index + generations[index] * 7) % 251);
            check(view.size() == payload_size(index, generations[index]) &&
                  view.span().front() == expected && view.span().back() == expected,
                  "generated payload is incorrect");
        }
        final = store.stats();
        check(final.reused_payload_slots > initial.reused_payload_slots, "released payload slots were not reused");
        check(emitted == rounds * changes * 2, "unexpected event generation outside selected partitions");
    }
    const auto pinned = store.stats();
    check(pinned.versions == 0 && pinned.index_bytes == 0 && pinned.capacity_bytes <= 65536 &&
          pinned.payload_bytes == 64 && pin.span().front() == std::byte{0},
          "one payload view retained unrelated state");
    pin = {};
    released(store.stats());
    std::sort(timings.begin(), timings.end());
    struct rusage usage{};
    check(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    std::cout << "PASS N=" << count << " rounds=" << rounds << " refreshed_per_round=" << changes * 2
              << " generated_bytes=" << generated << " payload_copy_bytes=" << final.copied_payload_bytes
              << " index_visits=" << page_visits << " p50_us=" << timings[timings.size() / 2]
              << " p95_us=" << timings[timings.size() * 95 / 100]
              << " initial_RAS_bytes=" << tracked(initial) << " peak_round_end_RAS_bytes=" << peak_tracked
              << " final_capacity_bytes=" << final.capacity_bytes
              << " slot_reuses=" << final.reused_payload_slots
              << " peak_process_RSS_KiB=" << usage.ru_maxrss
              << " elapsed_s=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
}

int main(int argc, char** argv) {
    try {
        if (argc == 2) run(std::stoull(argv[1]));
        else if (argc == 1) for (auto count : {10000, 100000, 1000000}) run(count);
        else throw Error("usage: refresh_storage_scaling [partition-count]");
    } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
