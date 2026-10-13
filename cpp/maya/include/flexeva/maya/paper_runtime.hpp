#pragma once

#include <flexeva/core/runtime.hpp>
#include <flexeva/code/layer.hpp>
#include <flexeva/maya/model.hpp>
#include <memory>
#include <span>
#include <string>

namespace flexeva::maya::paper {
using Id = core::runtime::Id;
using Evidence = core::runtime::Evidence;
enum class Mode { FullRank, FullCompact, Refresh };
using Region = code::Occurrence;
struct Communicator {
    Id id = 0;
    std::vector<std::uint32_t> members; // Communicator-local index -> global logical rank.
};
struct RankProjection {
    std::uint32_t rank = 0, representative = 0;
    // Representative membership ID -> logical membership ID. Local rank is
    // derived from membership; peer indices keep their declared logical role.
    std::vector<std::pair<Id, Id>> communicator_map;
};
class Capture {
public:
    Id identity(Id domain, std::span<const std::byte> coordinate);
    // Declare the dynamic occurrence before entering it. The entry evidence
    // decides Generate/Retain; final packets may add facts but cannot replace
    // that entry decision after native payloads have already been elided.
    void declare_region(Region, std::span<const Id> dependency_refs = {});
    void region(Id, const std::function<void()>&);
    void begin_region(Id);
    void end_region();
    bool recording() const;
    void check_native_failure() const;
    // Registers the native communicator handle against the frozen membership.
    void communicator(std::uint64_t handle, Id membership, std::uint32_t local_rank);
    core::runtime::GroundResolution ground(core::runtime::GroundRequest,
        std::span<const std::byte> inputs, ras::SchemaId input_schema);
    void code_dictionary(std::span<const std::byte> packet);
    void code_occurrence(Id region, std::span<const std::byte> packet,
        std::span<const Id> ground_facts = {}, std::span<const Id> dependency_refs = {});
private:
    struct Impl;
    explicit Capture(Impl&);
    Impl* impl_;
    friend class Session;
};
struct Input {
    std::vector<Region> regions;
    Evidence configuration;
    std::vector<core::runtime::Dependency> dependencies;
    std::vector<Communicator> communicators;
    // May execute required preparation regions in retain mode. It must enter
    // every selected region; repeated and nested scopes are supported.
    std::function<void(Capture&)> execute;
    std::vector<RankProjection> rank_projections; // Empty preserves independent rank execution.
    Evidence frontend_evidence; // Frozen plan/provenance, stored independently of local source selection keys.
    std::size_t capture_block_bytes = 0; // Physical packing hint; zero uses Options. Never changes source selection.
    bool dynamic_regions = false; // Real execution declares Code occurrences through Capture.
    Evidence window_semantic; // Complete frozen source/input/runtime identity when explicitly admitted.
    bool window_complete = false; // Enables exact-window reuse only without Ground or volatile regions.
};
// Native records have this POD header followed by kernel_name_size bytes and
// zero padding to an 8-byte boundary. Borrowed pointers never enter RAS.
struct NativeRecord {
    Operation operation{};
    std::uint32_t flags_v2 = 0, reserved = 0;
    std::uint32_t grid[3]{}, block[3]{};
    std::uint64_t shared_mem = 0, batch_count = 1;
    std::int64_t stride_a = 0, stride_b = 0, stride_c = 0, stride_d = 0;
    std::uint64_t d = 0, aux = 0, d_offset = 0, aux_offset = 0, kernel_name_size = 0;
    std::int32_t ldd = 0, d_type = 0, epilogue = 0, pointer_mode = 0, math_mode = 0;
    std::uint32_t api_kind = 0;
};
void for_each_native(const ras::Buffer&, const std::function<void(const NativeRecord&, std::string_view)>&);
void for_each_native(const core::runtime::LayerView&, const core::runtime::Partition&,
    const std::function<void(const NativeRecord&, std::string_view)>&);
struct Inspection {
    std::string normalized_digest, dependency_digest;
    std::uint64_t logical_events = 0, physical_records = 0, trace_bytes = 0;
};
struct Metrics {
    std::uint64_t native_calls = 0, retained_calls = 0, retained_trace_calls = 0, raw_records = 0;
    std::uint64_t raw_bytes = 0, copied_trace_bytes = 0, selected_regions = 0;
    std::uint64_t executed_regions = 0, preparation_regions = 0;
    std::uint64_t trace_partitions = 0, trace_partitions_read = 0;
    std::uint64_t cost_calls = 0, scheduled_events = 0;
    std::uint64_t replay_input_reads = 0, replay_graph_nodes = 0, replay_updated_nodes = 0;
    std::uint64_t replay_reused_nodes = 0, replay_graph_rebuilt = 0;
    bool diagnostics_enabled = false;
    std::uint64_t published_versions = 0, resource_records_read = 0;
    std::uint64_t retained_record_checks = 0;
    std::uint64_t normalization_raw_records = 0, normalized_output_adoptions = 0;
    std::uint64_t retained_normalized_body_reads = 0, publication_body_reads = 0;
    std::uint64_t publication_descriptor_visits = 0, feedback_body_reads = 0;
    // Subphases of capture_seconds; do not add them to capture again.
    double normalization_seconds = 0, retained_state_seconds = 0;
    double capture_seconds = 0, cost_seconds = 0, replay_seconds = 0, total_seconds = 0;
    // Disjoint coarse Capture callback phases; normalization/retained state nest in host execution.
    double capture_setup_seconds = 0, host_execution_seconds = 0, capture_finalize_seconds = 0;
    core::runtime::Counters core;
    ras::MemoryStats memory;
};
struct Snapshot {
    core::runtime::Anchor anchor;
    Feedback feedback;
    Metrics metrics;
};
struct Options {
    Mode mode = Mode::Refresh;
    bool diagnostics = false; // Optional per-record clocks and read audits; coarse phase times remain available.
    // Runtime scope/synchronization cuts often contain only one or two calls.
    // Bound unused capacity; a single larger record still gets one exact block.
    std::size_t chunk_bytes = 4096;
    Id identity = 0x4d41594150415001ULL;
    Evidence cost_identity{std::byte{1}};
    CostProvider costs;
    std::function<Cost(const FakeCudaRasRecordV2&)> native_costs;
    std::vector<core::runtime::GroundRegistration> grounds;
    std::function<Evidence(const FakeCudaRasRecordV2&)> native_cost_key;
};
class Session {
public:
    explicit Session(Options = {});
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Snapshot init(const Input&);
    Snapshot refresh(const Snapshot&, const Input&);
    Snapshot commit(const Snapshot&) const; // Reference promotion; no execution.
    core::runtime::LayerView layer(const Snapshot&, std::size_t) const;
    core::runtime::BoundaryView trace_outputs(const Snapshot&) const;
    const ras::Store& store() const noexcept;
    Inspection inspect(const Snapshot&) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace flexeva::maya::paper
