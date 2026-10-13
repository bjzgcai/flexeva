#pragma once

#include <flexeva/ras/ras.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace flexeva::core::runtime {
using Id = std::uint64_t;
using Evidence = std::vector<std::byte>; // Exact semantic evidence, not a pointer/hash identity.
namespace detail { struct Access; struct Runtime; struct ChunkOutputData; struct ChunkAccess; struct BlockBundleData; struct BundleAccess; struct GroundSession; }
class BlockBundle;
// Indexed RAS blocks belonging to one receipt. This handle holds only small
// metadata; reads require the current callback's receipt capability.
class ChunkOutput {
public:
    ChunkOutput() = default;
    explicit operator bool() const noexcept;
    std::size_t count() const noexcept;
    std::size_t size() const noexcept;
    BlockBundle storage() const;
    bool same_storage(const ChunkOutput&) const noexcept;
private:
    friend class ChunkWriter; friend struct detail::Runtime; friend struct detail::ChunkAccess;
    explicit ChunkOutput(std::shared_ptr<const detail::ChunkOutputData>);
    std::shared_ptr<const detail::ChunkOutputData> data_;
};
// Evidence declares complete semantic content, including external state consumed
// by the evaluator. Core compares it without reading opaque payload bytes.
// State with a payload requires nonempty semantic evidence; State{} is valid.
class BlockBundle {
public:
    BlockBundle() = default;
    explicit operator bool() const noexcept;
    std::size_t count() const noexcept;
    std::size_t size() const noexcept;
private:
    friend struct detail::BundleAccess;
    explicit BlockBundle(std::shared_ptr<const detail::BlockBundleData>);
    std::shared_ptr<const detail::BlockBundleData> data_;
};
struct State { ras::Buffer payload; Evidence semantic; std::vector<BlockBundle> bundles{}; };
struct RootInput { std::uint64_t revision = 0; std::span<const std::byte> bytes; };
using RootCapture = State;
struct PartitionHead {
    Id id=0, semantic_version=0, state_version=0, layout_version=0, topology_version=0;
    std::uint64_t event_count=0;
};
struct BoundaryHead {
    Id id=0, owner=0, generation_stamp=0, topology_version=0;
    std::size_t input_count=0;
    bool consumes_layer_state=false;
};
struct Partition {
    Id id = 0;
    ras::Buffer payload;
    std::vector<Id> lanes; // Nonempty canonical common active-laneset; lane label 0 is valid.
    Evidence semantic, state; // semantic is required; constant content uses an explicit marker.
    std::vector<Id> origins; // Incoming coarse receipt IDs; empty at the root.
    std::uint64_t event_count = 1;
    std::vector<Id> event_kinds{1}; // Canonical opaque labels; kind 0 is valid.
    Evidence layout; // Physical layout evidence; excluded from semantic generation keys.
    PartitionHead versions{}; // Read-only stamps assigned by Core at admission.
    std::vector<BlockBundle> bundles{}; // Explicit immutable auxiliary storage capabilities.
};
// Splice coordinates refer to the parent order; ranges must be disjoint.
struct OrderSplice { std::size_t start = 0, delete_count = 0; std::vector<Id> partitions; };
struct PartitionDelta { std::vector<Partition> upserts; std::vector<Id> removed; std::vector<OrderSplice> order; };
struct Dependency { Id producer = 0, consumer = 0; auto operator<=>(const Dependency&) const = default; };
struct DependencyDelta { std::vector<Dependency> upserts, removed; };
struct LayerDelta {
    std::vector<Id> added, changed, removed, layout_changed, state_changed, invalidated, order_affected;
    std::vector<Id> topology_changed; // Origins changed without necessarily changing content.
    bool order_changed = false, dependencies_changed = false, layer_state_changed = false;
    bool semantic_empty() const noexcept;
};
struct Boundary {
    Id id = 0, owner = 0;
    std::vector<Id> inputs; // Complete semantic inputs. owner is bookkeeping, never an implicit input.
    ras::Buffer payload;
    Evidence semantic, state; // semantic is required; constant content uses an explicit marker.
    bool consumes_layer_state = false;
    std::vector<Id> parent_receipts; // Explicit prior receipt capabilities for Apply.
    std::vector<BlockBundle> bundles{};
};
struct BoundaryPatch {
    Id id=0;
    std::optional<Boundary> header; // Optional replacement header; inputs must be empty.
    std::vector<Id> add_inputs, remove_inputs;
};
struct BoundaryDelta { std::vector<Boundary> upserts; std::vector<Id> removed; std::vector<BoundaryPatch> patches; };
struct Receipt {
    Id boundary = 0;
    ras::Buffer payload;
    std::uint64_t event_count = 0;
    State state;
    Evidence semantic; // Complete content evidence; required if consumed as a parent receipt input.
    ChunkOutput chunks{}; // Optional indexed blocks; payload may additionally hold a small header.
    Id semantic_version=0; // Core-managed complete receipt identity.
    std::vector<BlockBundle> bundles{};
};
// Operation-local transform output produced during Capture/Partition. It is
// not part of the upper layer and acquires persistent ownership only when its
// declared Apply Boundary returns the adopted receipt. Empty body is valid.
struct PreparedOutput { Receipt receipt; BlockBundle body; };
// Ordered, disjoint byte slices in one sealed output. Physical adjacency is
// optional; the evaluator must prove the selected events are consecutive in
// each common active lane before materializing a logical partition.
struct OutputSlice {
    std::size_t block=0, offset=0, bytes=0;
};
// A materialized range payload is this header followed by slice_count
// OutputSlice records. The partition's sole bundle is the original body root.
struct OutputRangeHeader {
    Id boundary=0, semantic_version=0;
    std::size_t slice_count=0;
};
using OutputRangeValidator = std::function<bool(const Receipt&, std::span<const OutputSlice>, const Partition&)>;
struct ApplyResult { std::vector<Receipt> generated; std::vector<Id> removed; };
enum class LineageKind { Boundary, Data };
struct Lineage {
    Id id = 0, boundary = 0, producer = 0, consumer = 0; // id is Core-managed; ignored on upsert.
    LineageKind kind = LineageKind::Data;
    auto operator<=>(const Lineage&) const = default;
};
struct LinkDelta { std::vector<Lineage> upserts; std::vector<Id> removed; };
// A factorized AL route. Owner is provenance, input is a semantic dependency;
// both roles can belong to the same route. The output is the whole receipt.
// Fine partition bindings, if demanded, are queried separately.
struct OutputTarget {
    Id boundary=0, semantic_version=0;
    bool owner=false, input=false, has_output=false;
};
struct PartitionRequest {
    bool initial = false;
    RootCapture root;
    std::vector<Id> changed_receipts, removed_receipts;
};
struct Counters {
    std::uint64_t payload_reads = 0, full_enumerations = 0, metadata_reads = 0;
    std::uint64_t selected_boundaries = 0, generated_receipts = 0, dependency_visits = 0;
    std::uint64_t partition_callbacks = 0, boundary_callbacks = 0, apply_callbacks = 0, link_callbacks = 0, state_callbacks = 0;
    double selection_seconds=0, partition_seconds=0, lineage_seconds=0;
    // Top-level evaluate interval ends after publication. Callback time is
    // inclusive of its nested work (including Ground), counted once; framework
    // time is the residual. Legacy phase timers above include callbacks.
    double evaluate_seconds=0, evaluator_callback_seconds=0, framework_seconds=0;
    std::uint64_t merge_compare_bytes=0;
    std::uint64_t header_reads=0;
    // Evidence bytes decoded while choosing Apply; callback admission and selected Apply reads are separate.
    std::uint64_t selection_evidence_bytes=0;
    std::uint64_t lineage_upserts=0, lineage_removals=0, lineage_retained=0, boundary_input_writes=0;
    std::uint64_t chunk_intern_hash_bytes = 0, chunk_intern_compare_bytes = 0;
    // Shared bytes are committed logical lengths, not allocator capacity saved.
    std::uint64_t chunk_intern_shared_bytes = 0, chunk_intern_reused_blocks = 0;
    std::uint64_t ground_calls=0, ground_operation_hits=0, ground_parent_hits=0;
    double ground_seconds=0;
    std::uint64_t prepared_adoptions=0, previous_output_adoptions=0;
    std::uint64_t shared_output_adoptions=0, output_root_bindings=0, output_block_writes=0;
    std::uint64_t route_upserts=0, route_removals=0, route_visits=0;
    std::uint64_t route_entries=0, materialized_output_bindings=0;
    // Real Buffer::bytes/array acquired extents in these scopes, not CPU loads.
    std::uint64_t admission_body_bytes_read=0, admission_metadata_bytes_read=0;
    std::uint64_t adoption_body_bytes_read=0, adopted_block_index_writes=0;
    // False means fine timing/read-audit fields were not measured, not zero
    // work. Algorithmic work counters and all correctness checks remain live.
    bool diagnostics_enabled=true;
};
class LayerView {
public:
    LayerView() = default;
    explicit operator bool() const noexcept;
    PartitionHead head(Id) const;
    void partitions_from(Id origin, const std::function<void(Id)>&) const;
    bool origin_contains(Id partition, Id origin) const;
    Partition partition(Id) const; // Metadata plus owning opaque payload capability; no decoding.
    ras::Buffer payload(Id) const; // Counted native-content access, scoped to this view.
    ras::Buffer block(const BlockBundle&, std::size_t ordinal) const; // Read an explicitly granted auxiliary block.
    // Return a nonempty aligned slice, sharing the explicitly granted owner.
    // Checks the block and byte bounds without borrowing or copying its body.
    ras::Buffer block(const BlockBundle&, const OutputSlice&) const;
    bool contains(Id) const;
    std::size_t size() const;
    Id at(std::size_t) const;
    std::size_t rank(Id) const;
    ras::OrderStats order_stats() const;
    void partitions(const std::function<void(const Partition&)>&) const; // Explicit full enumeration.
    void consumers(Id, const std::function<void(Id)>&) const;
    void producers(Id, const std::function<void(Id)>&) const;
    State state() const;
private:
    friend class Context; friend class Coordinator; friend struct detail::Runtime;
    explicit LayerView(std::shared_ptr<detail::Access> access);
    std::shared_ptr<detail::Access> access_;
};
class BoundaryView {
public:
    BoundaryView() = default;
    BoundaryHead head(Id) const;
    void inputs(Id boundary, const std::function<void(Id)>&) const;
    bool has_input(Id boundary, Id input) const;
    Boundary boundary(Id) const;
    bool contains(Id) const;
    void boundaries(const std::function<void(const Boundary&)>&) const; // Explicit full enumeration.
    void downstream(Id producer, const std::function<void(const OutputTarget&)>&) const;
    void partitions_from(Id boundary, const std::function<void(Id)>&) const;
    Receipt output(Id boundary) const;
    ras::Buffer output_chunk(Id boundary, std::size_t ordinal) const;
    // A read-only range view; validates addresses without borrowing body bytes.
    std::vector<ras::Buffer> output_range(Id boundary, std::span<const OutputSlice>) const;
    void outputs(const std::function<void(const Receipt&)>&) const; // Explicit output enumeration.
    // Diagnostic expansion only; relations have no persistent edge identity.
    void lineage_to(Id consumer, const std::function<void(const Lineage&)>&) const;
    std::optional<Lineage> lineage(Id boundary, Id producer, Id consumer, LineageKind kind) const;
    Evidence generation_key(Id) const; // Compatibility view of the fixed 8-byte generation stamp.
private:
    friend class Context; friend class Coordinator; friend struct detail::Runtime;
    explicit BoundaryView(std::shared_ptr<detail::Access> access);
    std::shared_ptr<detail::Access> access_;
};
class OutputWriter {
public:
    OutputWriter(OutputWriter&&) noexcept;
    OutputWriter& operator=(OutputWriter&&) noexcept;
    ~OutputWriter();
    ras::WriteSpan reserve(std::size_t bytes);
    std::size_t size() const;
    std::size_t capacity() const;
    ras::Buffer finish();
private:
    friend class Context;
    OutputWriter(ras::BufferWriter, std::shared_ptr<detail::Access>);
    ras::BufferWriter writer_;
    std::shared_ptr<detail::Access> access_;
};
// Unknown-length output in fixed contiguous RAS blocks. A reservation never
// crosses a block; a reservation larger than block_bytes gets a larger block.
// Committed bytes are sealed without copying or exposing unused capacity.
class ChunkWriter {
public:
    ChunkWriter(ChunkWriter&&) noexcept;
    ChunkWriter& operator=(ChunkWriter&&) noexcept;
    ~ChunkWriter();
    ras::WriteSpan reserve(std::size_t bytes);
    // Seal the previous block and reserve one complete block of a known exact size.
    // Commit the returned borrow before seal/finish; no existing bytes are copied.
    ras::WriteSpan reserve_exact(std::size_t bytes);
    std::size_t size() const;
    void append(const ras::Buffer&); // Reuse a granted immutable block without copying.
    void seal(); // End the current block, including at a lane-continuity boundary; empty is a no-op.
    ChunkOutput finish();
private:
    friend class Context;
    struct Impl;
    ChunkWriter(std::shared_ptr<detail::Access>, std::size_t, ras::Layout, bool intern);
    std::unique_ptr<Impl> impl_;
};
// Cooperative C++ capabilities, not a memory sandbox: evaluator code must not
// use retained native pointers or another Boundary's data as undeclared inputs.
class BlockWriter {
public:
    BlockWriter(BlockWriter&&) noexcept;
    BlockWriter& operator=(BlockWriter&&) noexcept;
    ~BlockWriter();
    ras::WriteSpan reserve(std::size_t bytes);
    // Capture the final block/byte address after any rollover. The address is
    // stable, but describes initialized immutable data only after full commit()
    // and finish(). A partial commit requires shrinking location.bytes; an
    // abandoned borrow invalidates the writer. Failure leaves location intact.
    // Locations do not create a bundle, directory, or independent capability.
    ras::WriteSpan reserve(std::size_t bytes, OutputSlice& location);
    ras::WriteSpan reserve_exact(std::size_t bytes);
    void append(const ras::Buffer&); // Retain an already granted immutable block, without copying.
    void seal();
    std::size_t size() const;
    BlockBundle finish();
private:
    friend class Context; friend class GroundContext;
    struct Impl;
    BlockWriter(std::shared_ptr<detail::Access>, std::size_t, ras::Layout, Id, Id, Evidence);
    std::unique_ptr<Impl> impl_;
};
struct GroundRequest {
    Id request_id=0, point_id=0;
    std::vector<Id> requested_lanes;
    Evidence query;
    State inputs;
};
struct GroundResult { Id request_id=0; State value; bool reusable=false; };
// fact_id is semantic evidence, not a dereference capability. Consumers retain
// value payload/bundles explicitly when they need the numerical fact later.
struct GroundResolution { Id request_id=0, fact_id=0; State value; bool reusable=false; };
class GroundContext {
public:
    const State& inputs() const;
    OutputWriter buffer(std::size_t bytes, ras::Layout layout = {});
    ras::Buffer copy(std::span<const std::byte>, ras::SchemaId schema = 0);
    BlockWriter blocks(std::size_t block_bytes=65536, ras::Layout layout={});
    ras::Buffer block(const BlockBundle&, std::size_t ordinal) const;
private:
    friend struct detail::GroundSession;
    GroundContext(std::shared_ptr<detail::Access>, State);
    std::shared_ptr<detail::Access> access_;
    State inputs_;
};
struct GroundRegistration {
    Id id=0, layer=0;
    Evidence identity;
    ras::SchemaId result_schema=0;
    std::function<bool(const GroundRequest&, const GroundContext&)> matches;
    std::function<GroundResult(const GroundRequest&, GroundContext&)> handler;
};
class Context {
public:
    OutputWriter buffer(std::size_t bytes, ras::Layout layout = {});
    // Apply only; one writer per Boundary. Opt-in interning shares byte-identical
    // sealed blocks of the same layout generated within this Apply callback.
    ChunkWriter chunks(std::size_t block_bytes = 65536, ras::Layout layout = {}, bool intern = false);
    ras::Buffer copy(std::span<const std::byte> bytes, ras::SchemaId schema = 0);
    std::vector<Id> input_order(Id lane) const;
    Context for_boundary(Id) const; // Isolated Apply input/receipt capabilities; same callback lifetime.
    LayerView previous() const;
    LayerView upper() const;
    BoundaryView boundaries() const;
    Receipt receipt(Id) const; // Partition: current indexed input. Apply: explicitly declared parent only.
    void receipts(const std::function<void(const Receipt&)>&) const; // Incoming outputs in Partition/State.
    ras::Buffer receipt_chunk(Id boundary, std::size_t ordinal) const;
    // Partition phase only. Header must have no payload, bundles, or origins.
    // The required validator proves event membership/common lanes/per-lane
    // consecutiveness from evaluator metadata. This returns one final upsert;
    // publication occurs only when the caller returns it in PartitionDelta.
    Partition materialize_range(Partition header, Id boundary, std::span<const OutputSlice>,
                                const OutputRangeValidator&);
    void retain_receipt_chunk(Id boundary, std::size_t ordinal, const ras::Buffer& equal_backing);
    void receipt_chunks(Id boundary, const std::function<void(std::size_t, const ras::Buffer&)>&) const;
    const LayerDelta& delta() const;
    const LayerDelta& upper_delta() const;
    std::span<const Id> changed_boundaries() const;
    std::span<const Id> removed_boundaries() const;
    // Evaluator-private indexed metadata shares the current RAS transaction.
    // Unavailable during Apply: evaluator indexing cannot expand its capabilities.
    ras::DirectoryId evaluator_directory(Id slot);
    std::optional<ras::DirectoryRecord> metadata_find(ras::DirectoryId, ras::DirectoryKey) const;
    void metadata_put(ras::DirectoryId, ras::DirectoryKey, const ras::DirectoryRecord&);
    void metadata_erase(ras::DirectoryId, ras::DirectoryKey);
    void metadata_range(ras::DirectoryId, ras::DirectoryKey first, ras::DirectoryKey last,
                        const std::function<void(ras::DirectoryKey,const ras::DirectoryRecord&)>&) const;
    std::optional<std::pair<ras::DirectoryKey,ras::DirectoryRecord>> metadata_next(ras::DirectoryId,ras::DirectoryKey) const;
    std::optional<std::pair<ras::DirectoryKey,ras::DirectoryRecord>> metadata_prev(ras::DirectoryId,ras::DirectoryKey) const;
    Id new_identity();
    Id identity(Id domain, std::span<const Id> coordinate); // Exact stable logical coordinate, never a hash identity.
    BlockWriter blocks(std::size_t block_bytes=65536, ras::Layout layout={});
    ras::Buffer block(const BlockBundle&, std::size_t ordinal) const;
    ras::Buffer block(const BlockBundle&, const OutputSlice&) const;
    // Capture/Partition stage output once; only its declared Apply Boundary may adopt it.
    BlockWriter prepare(Id transform, Id boundary, Evidence boundary_semantic,
                        std::size_t block_bytes=65536, ras::Layout layout={});
    // Only the current layer's registered outgoing transform and the parent
    // supplied to refresh are accessible. Capture/Partition only; no history
    // enumeration and no body reads are performed by the lookup itself.
    std::optional<Receipt> parent_output(Id transform, Id boundary) const;
    // Stage already granted immutable capabilities independently of Code/root
    // payloads. receipt.chunks must be empty; Apply binds body explicitly.
    // The key may be staged once per operation. Capture/Partition only.
    void stage_output(Id transform, Id boundary, Evidence boundary_semantic,
                      Receipt receipt, BlockBundle body={});
    // Inspect only this layer's outgoing transform during Capture/Partition/
    // Boundary. Apply must use the isolated, semantic-checked overload below.
    std::optional<PreparedOutput> prepared_output(Id transform, Id boundary) const;
    std::optional<PreparedOutput> prepared_output() const;
    ChunkOutput adopt_prepared(const BlockBundle&);
    // Bind an immutable body explicitly granted to this Apply Boundary. Sharing
    // evidence belongs to the evaluator; this does not grant another output.
    ChunkOutput adopt_output(const BlockBundle&);
    // Same-boundary old physical output is a cache, not a declared semantic input.
    std::optional<Receipt> previous_output() const;
    ras::Buffer previous_output_chunk(std::size_t ordinal) const;
    ChunkOutput adopt_previous_output();
    GroundResolution ground(const GroundRequest&);
    [[noreturn]] void ground() const; // Compatibility failure for the former placeholder.
private:
    friend struct detail::Runtime; friend class GroundContext;
    explicit Context(std::shared_ptr<detail::Access> access);
    std::shared_ptr<detail::Access> access_;
};
struct LayerRegistration {
    Id id = 0, identity = 0;
    ras::SchemaId payload_schema = 0;
    std::function<RootCapture(const RootInput&, Context&)> capture; // Required only on root.
    std::function<PartitionDelta(const PartitionRequest&, Context&)> partition;
    std::function<DependencyDelta(const LayerView&, const LayerDelta&, Context&)> data_dep;
    std::function<State(const LayerView&, Context&)> build_state; // Optional; after incoming Link. Root also receives current RootCapture grants.
};
struct TransformRegistration {
    Id id = 0, identity = 0, upper = 0, lower = 0;
    std::function<BoundaryDelta(const LayerView&, Context&)> boundary;
    std::function<ApplyResult(std::span<const Boundary>, Context&)> apply;
    // Optional diagnostic declaration. AL storage is derived from complete
    // Boundary inputs/owner and demanded Partition.origins, never edge products.
    std::function<LinkDelta(const LayerView& lower, const LayerView& upper, const BoundaryView&, Context&)> link;
};
struct Pipeline { Id identity = 0; std::vector<LayerRegistration> layers; std::vector<TransformRegistration> transforms; std::vector<GroundRegistration> grounds{}; };
struct Anchor { ras::Version version; ras::Buffer manifest; Id window = 0; ras::StateId manifest_state; };
struct Candidate { ras::Version version; ras::Buffer manifest; Id window = 0; Counters counters; ras::StateId manifest_state; };
struct CoordinatorOptions { bool diagnostics=true; };
class Coordinator {
public:
    explicit Coordinator(ras::Store&, CoordinatorOptions={});
    ~Coordinator();
    Coordinator(const Coordinator&) = delete;
    Coordinator& operator=(const Coordinator&) = delete;
    void register_pipeline(Pipeline); // Validates only; never executes evaluator code.
    Anchor init(const RootInput&);
    Candidate refresh(const Anchor&, const RootInput&);
    Anchor commit(const Candidate&) const; // Reference promotion only; no callback or publication.
    LayerView layer(const Anchor&, std::size_t) const;
    LayerView layer(const Candidate&, std::size_t) const;
    BoundaryView transform(const Candidate&, std::size_t) const;
    BoundaryView transform(const Anchor&, std::size_t) const;
    const Counters& last_counters() const;
private:
    std::unique_ptr<detail::Runtime> runtime_;
};
} // namespace flexeva::core::runtime
