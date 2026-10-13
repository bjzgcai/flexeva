#pragma once

#include <flexeva/core/runtime.hpp>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace flexeva::code {
using Id = core::runtime::Id;
using Evidence = core::runtime::Evidence;
namespace rt = core::runtime;
inline constexpr ras::SchemaId occurrence_schema = 0x434f44450001ULL;
inline constexpr ras::SchemaId dependency_schema = 0x434f44450002ULL;
inline constexpr ras::SchemaId evidence_schema = 0x434f44450003ULL;
struct Occurrence {
    Id id = 0;
    std::uint32_t rank = 0;
    std::uint64_t revision = 0;
    Evidence semantic;
    bool produces_trace = true; // Whether the occurrence emits evaluator output; host-only setup remains Code state.
    Id execution_region = 0; // Actual representative scope; zero means this region itself.
    bool semantic_complete = false; // Evidence covers the precise consumed source/configuration inputs.
    bool reuse_allowed = true; // Unknown frontend rules must execute on every refresh.
    Id site_id = 0; // Stable rank-independent source site; zero uses conservative manifest order.
    Evidence site_identity; // Complete stable source-site coordinate, independent of intern allocation order.
    Evidence occurrence_identity; // Complete rank-independent dynamic occurrence coordinate.
    std::vector<Id> active_lanes; // Empty means {rank}; shared occurrences are stored once for this common lane set.
};

// Backend-independent Code metadata. Evaluator output bodies, generation IDs,
// resource projections and cost state belong to transform receipts, never here.
struct OccurrenceHeader {
    Id id=0; std::uint64_t revision=0; std::uint32_t rank=0,reserved=0;
    Id site=0,representative=0; std::uint64_t semantic_bytes=0,site_bytes=0,occurrence_bytes=0;
    Id evidence_version=0; std::uint64_t evidence_slices=0;
    std::uint32_t produces=1,complete=0,reusable=1,padding=0;
    std::uint64_t lane_count=0;
};
OccurrenceHeader occurrence_header(const ras::Buffer&);
Occurrence occurrence(const ras::Buffer&);
std::vector<rt::OutputSlice> evidence_slices(const ras::Buffer&);
struct EvidencePool { rt::BlockBundle owner; Id version=0; std::vector<rt::OutputSlice> slices; };
struct EvidenceRecord { std::uint64_t kind=0,bytes=0,facts=0; };
class EvidenceWriter {
public:
    explicit EvidenceWriter(rt::Context&, std::size_t block_bytes=65536);
    rt::OutputSlice append(std::uint64_t kind, std::span<const std::byte>, std::span<const Id> facts={});
    rt::BlockBundle finish();
    Id version() const noexcept;
private:
    rt::BlockWriter writer_;
    Id version_;
};
// Complete source certificates include the transitive explicit producer graph.
// The extra context is only used for conservative/incomplete source evidence.
class SourceGraph {
public:
    SourceGraph();
    ~SourceGraph();
    void reset(Evidence context, std::uint64_t round,
               std::function<void(Id)> before_invalidate={});
    void declare(const Occurrence&);
    void dependencies(Id consumer, std::span<const Id> producers);
    const std::map<Id,const Occurrence*>& regions() const;
    const std::vector<rt::Dependency>& dependencies() const;
    Evidence key(const Occurrence&, rt::Context&);
    bool same_source(const rt::Partition&, const Occurrence&, rt::Context&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
struct PartitionEvidence {
    Evidence semantic_suffix, state;
    EvidencePool code;
    std::vector<rt::BlockBundle> ground_values;
    bool ground_volatile=false;
};
rt::PartitionDelta partition(rt::Context&, SourceGraph&, std::span<const Occurrence>,
    const std::function<PartitionEvidence(const Occurrence&)>&);
rt::DependencyDelta dependency_delta(rt::Context&, std::span<const rt::Dependency>);
rt::State state(rt::Context&, std::span<const rt::Dependency>, Evidence semantic,
                std::span<const rt::BlockBundle> provenance={});
// A transform supplies only opaque generation evidence and whether an output
// exists. The shared builder binds each Code occurrence to its actual output;
// no logical lower partition is created merely to maintain this route.
struct OutputEvidence { bool exists=false; Evidence semantic, state; Id owner=0; };
rt::BoundaryDelta boundaries(const rt::LayerView&, rt::Context&,
    std::span<const Id> output_changes,
    const std::function<OutputEvidence(Id)>&);
} // namespace flexeva::code
