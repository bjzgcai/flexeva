#pragma once
#include <flexeva/core/runtime.hpp>
#include <map>
#include <set>
#include <limits>
#include <unordered_map>

namespace flexeva::core::runtime::detail {
inline constexpr ras::SchemaId metadata_schema = 0x464c4558434f5245ULL;
inline constexpr Id maximum = std::numeric_limits<Id>::max();
extern thread_local Counters* current_selection_counter;
struct SelectionAudit {
    explicit SelectionAudit(Counters*) noexcept;
    ~SelectionAudit();
    SelectionAudit(const SelectionAudit&) = delete;
    SelectionAudit& operator=(const SelectionAudit&) = delete;
private:
    Counters* previous_;
};
struct AdmissionAudit {
    Counters* counters;
    bool adoption;
    Counters* previous_adoption;
    std::optional<ras::BufferReadAudit> read;
    AdmissionAudit(Counters*,bool adopt=false) noexcept;
    ~AdmissionAudit();
    static void observe(void*,const ras::BufferIdentity&,const ras::Layout&,std::size_t);
};
void count_block_write(Counters*);
struct LayerData {
    Id id=0, identity=0; ras::SchemaId schema=0; ras::LayerId layer;
    ras::DirectoryId parts, dependencies, producers, origins, states, evaluator;
    ras::OrderId order;
};
struct TransformData {
    Id id=0, identity=0;
    ras::DirectoryId boundaries, receipts, routes, members, consumed_receipts, generations, parent_inputs;
    ras::DirectoryId state_consumers, parent_consumers;
};
struct BoundaryTopology {
    Id id=0, old_owner=0, owner=0;
    std::vector<Id> added_inputs, removed_inputs;
    bool removed=false;
};
struct Manifest;
struct LinkContext { std::vector<BoundaryTopology> topology; };
inline constexpr Id route_owner=1, route_input=2;
void change_route(ras::Transaction&,const TransformData&,Id producer,Id boundary,Id role,bool add,Counters*);
void downstream(const ras::ReadView&,const TransformData&,Id producer,const std::function<void(const OutputTarget&)>&,Counters*,bool resolve_output=true);
std::optional<Lineage> lookup_lineage(const ras::ReadView&,const TransformData&,const LayerData&,const Lineage&,Counters*);
void admit_lineage(ras::Transaction&,const Manifest&,std::size_t,const LinkDelta&,const LayerDelta&,const LinkContext&,Counters*);
struct ChunkOutputData {
    Id boundary = 0;
    std::size_t count = 0, bytes = 0;
    bool finished = false;
    ras::Layout layout;
    BlockBundle storage;
};
struct ChunkAccess {
    static const ChunkOutputData* get(const ChunkOutput& value) { return value.data_.get(); }
    static ChunkOutput make(std::shared_ptr<const ChunkOutputData> value) { return ChunkOutput(std::move(value)); }
};
struct BlockBundleData {
    ras::DirectoryId directory;
    Id id=0;
    std::size_t count=0, bytes=0;
    ras::Layout layout;
    Id transform=0, boundary=0, operation=0;
    Evidence semantic;
};
struct BundleAccess {
    static const BlockBundleData* get(const BlockBundle& value) { return value.data_.get(); }
    static BlockBundle make(std::shared_ptr<const BlockBundleData> value) { return BlockBundle(std::move(value)); }
};
struct Manifest {
    Id window=0; ras::LayerId control_layer; ras::StateId state;
    std::vector<LayerData> layers;
    std::vector<TransformData> transforms;
    ras::DirectoryId ground_facts, bundle_owners;
    Evidence ground_policy;
};
using GrantKey=std::pair<std::uint64_t,std::uint64_t>;
using Grants=std::map<GrantKey,std::vector<ras::Range>>;
struct ChunkInternPool {
    std::unordered_map<std::size_t,std::vector<ras::Buffer>> buckets;
};
struct Scope {
    bool live=true;
    ras::Transaction* tx=nullptr;
    std::shared_ptr<Counters> counters;
    std::unique_ptr<ChunkInternPool> chunk_intern;
};
enum class Phase { Public, Capture, Partition, Dependency, Boundary, Apply, Link, State };
struct GroundSession;
struct BundleSession;
struct PreparedEntry { Evidence semantic; PreparedOutput output; };
struct PreparedSession { std::map<std::pair<Id,Id>,PreparedEntry> outputs; };
struct Access {
    std::shared_ptr<Scope> scope;
    std::shared_ptr<GroundSession> ground_session;
    std::shared_ptr<BundleSession> bundle_session;
    std::shared_ptr<PreparedSession> prepared_session;
    Id operation=0;
    std::shared_ptr<std::set<std::pair<ras::DirectoryId,Id>>> bundle_grants=std::make_shared<std::set<std::pair<ras::DirectoryId,Id>>>();
    std::shared_ptr<std::map<Id,std::shared_ptr<std::set<std::pair<ras::DirectoryId,Id>>>>> boundary_bundle_grants=std::make_shared<std::map<Id,std::shared_ptr<std::set<std::pair<ras::DirectoryId,Id>>>>>();
    ras::ReadView read, old;
    std::optional<LayerData> layer, previous_layer, upper_layer, lower_layer;
    std::optional<TransformData> transform, previous_transform;
    std::optional<TransformData> outgoing_transform, previous_outgoing_transform;
    Phase phase=Phase::Public;
    std::shared_ptr<const LayerDelta> delta, upper_delta;
    std::shared_ptr<const std::vector<Id>> changed_boundaries, removed_boundaries;
    std::shared_ptr<const std::vector<Boundary>> selected;
    std::optional<Boundary> isolated;
    bool restricted=false, allow_state=true;
    std::shared_ptr<std::set<ras::DirectoryId>> evaluator_directories=std::make_shared<std::set<ras::DirectoryId>>();
    std::shared_ptr<Grants> grants=std::make_shared<Grants>();
    std::shared_ptr<std::map<Id,std::shared_ptr<Grants>>> boundary_grants=std::make_shared<std::map<Id,std::shared_ptr<Grants>>>();
    std::shared_ptr<std::map<Id,std::shared_ptr<ChunkOutputData>>> chunk_outputs=std::make_shared<std::map<Id,std::shared_ptr<ChunkOutputData>>>();
};
void check(const Access&);
void grant(Access&,const ras::Buffer&);
void grant_fresh(Access&,const ras::Buffer&);
bool granted(const Access&,const ras::Buffer&);
void require_granted(const Access&,const ras::Buffer&);
void grant_bundle(Access&, const BlockBundle&);
void require_bundle(const Access&, const BlockBundle&);
void grant_bundles(Access&, const std::vector<BlockBundle>&);
void require_bundles(const Access&, const std::vector<BlockBundle>&);
ras::Buffer read_block(Access&, const BlockBundle&, std::size_t);
Evidence encode_bundles(const std::vector<BlockBundle>&);
std::vector<BlockBundle> decode_bundles(std::span<const std::byte>);
bool same_bundles(const std::vector<BlockBundle>&, const std::vector<BlockBundle>&);
std::shared_ptr<BundleSession> make_bundle_session(ras::Transaction&, ras::DirectoryId);
void track_bundle_directory(Access&, ras::DirectoryId);
void replace_bundle_refs(const std::shared_ptr<BundleSession>&,const std::vector<BlockBundle>&,const std::vector<BlockBundle>&);
void replace_bundle_refs(Access&,const std::vector<BlockBundle>&,const std::vector<BlockBundle>&);
void finish_bundle_session(const std::shared_ptr<BundleSession>&);
PartitionHead load_partition_head(const ras::ReadView&,const LayerData&,Id,Counters* =nullptr);
BoundaryHead load_boundary_head(const ras::ReadView&,const TransformData&,Id,Counters* =nullptr);
Id load_receipt_version(const ras::ReadView&,const TransformData&,Id,Counters* =nullptr);
Id load_receipt_generation(const ras::ReadView&,const TransformData&,Id,Counters* =nullptr);
Partition load_partition(const ras::ReadView&,const LayerData&,Id,Counters* =nullptr);
ras::PartitionId native_partition(const ras::ReadView&,const LayerData&,Id);
Boundary load_boundary(const ras::ReadView&,const TransformData&,Id,Counters* =nullptr,bool with_inputs=true);
Receipt load_receipt(const ras::ReadView&,const TransformData&,Id,Evidence* key=nullptr,Counters* =nullptr);
Receipt load_receipt_access(const ras::ReadView&,const TransformData&,Id,Counters* =nullptr);
std::vector<BlockBundle> receipt_bundles(const Receipt&);
State load_state(const ras::ReadView&,const LayerData&,Counters* =nullptr);
State load_state_record(const ras::DirectoryRecord&,Counters* =nullptr);
std::vector<Id> projected_inputs(const ras::ReadView&,const LayerData&,const Boundary&,Id);

ras::DirectoryRecord save_partition(ras::Transaction&,const Partition&,ras::PartitionId native={});
void bind_partition(ras::Transaction&,ras::DirectoryRecord&,ras::PartitionId);
ras::DirectoryRecord save_boundary(ras::Transaction&,const Boundary&,const BoundaryHead&);
ras::DirectoryRecord save_receipt(ras::Transaction&,const Receipt&,const Evidence& generation_key);
ras::DirectoryRecord save_state(ras::Transaction&,const State&);
ras::DirectoryRecord save_relation(ras::Transaction&,ras::RelationId);
ras::RelationId load_relation(const ras::DirectoryRecord&);
ras::Buffer save_manifest(ras::Transaction&,const Manifest&);
Manifest load_manifest(const ras::Buffer&);

} // namespace flexeva::core::runtime::detail
