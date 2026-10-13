#pragma once

#include <flexeva/maya/model.hpp>

#include <array>
#include <compare>
#include <cstdint>
#include <limits>

namespace flexeva::maya::paper {

// A raw call is stored once. Its host and optional device projection refer to
// that same call; these keys describe execution contexts, not duplicated work.
// RAS interns the complete key: normalized stream identities are already wider
// than 32 bits, so packing rank and stream into a single integer is not valid.
enum class LaneKind : std::uint32_t { Host = 1, Device = 2 };
struct LaneKey {
    std::uint32_t rank = 0;
    LaneKind kind = LaneKind::Host;
    std::uint64_t stream = 0;
    auto operator<=>(const LaneKey&) const = default;
};
struct LaneContext {
    bool device = false;
    std::uint64_t stream = 0;
    auto operator<=>(const LaneContext&) const = default;
};
struct LaneProjection {
    std::array<LaneKey, 2> keys{};
    std::uint32_t size = 1;
};
inline LaneProjection lane_projection(std::uint32_t rank, LaneContext context) {
    LaneProjection result{{LaneKey{rank, LaneKind::Host, 0}, LaneKey{rank, LaneKind::Device, context.stream}}, 1};
    if (context.device) result.size = 2;
    return result;
}

inline constexpr bool communication(std::uint32_t op) noexcept {
    return op == FAKECUDA_RAS_SEND || op == FAKECUDA_RAS_RECV || op == FAKECUDA_RAS_ALLREDUCE ||
        op == FAKECUDA_RAS_BROADCAST || op == FAKECUDA_RAS_ALLGATHER || op == FAKECUDA_RAS_REDUCESCATTER;
}
inline constexpr bool device_operation(std::uint32_t op) noexcept {
    return communication(op) || op == FAKECUDA_RAS_EVENT_RECORD || op == FAKECUDA_RAS_STREAM_WAIT_EVENT ||
        op == FAKECUDA_RAS_MEMCPY_ASYNC || op == FAKECUDA_RAS_MEMSET_ASYNC ||
        op == FAKECUDA_RAS_GEMM || op == FAKECUDA_RAS_KERNEL;
}

enum class Synchronization : std::uint32_t {
    None = 0, EventRecord, EventWait, EventHostWait, StreamHostWait, DeviceHostWait,
    GroupBegin, GroupCommit, Communication
};
struct LaneClassification {
    LaneContext context{};
    Synchronization synchronization = Synchronization::None;
    bool seal_before = false, seal_after = false;
    bool outer_group_commit = false;
    std::uint32_t group_depth = 0; // Depth after this call.
};

// One classifier per rank, shared by that rank's successive code regions.
// Retained calls must also advance it: record/retain changes storage policy,
// never NCCL nesting or the execution context of the next emitted call.
class LaneClassifier {
public:
    LaneClassification consume(const FakeCudaRasRecord& call) {
        if (call.op < FAKECUDA_RAS_ALLOC || call.op > FAKECUDA_RAS_REDUCESCATTER)
            throw ras::Error("Maya lane classification unsupported native operation");
        LaneClassification result;
        result.context = {device_operation(call.op), device_operation(call.op) ? call.stream : 0};
        result.seal_before = seen_ && previous_ != result.context;
        switch (call.op) {
        case FAKECUDA_RAS_EVENT_RECORD: result.synchronization = Synchronization::EventRecord; break;
        case FAKECUDA_RAS_STREAM_WAIT_EVENT: result.synchronization = Synchronization::EventWait; break;
        case FAKECUDA_RAS_EVENT_SYNC: result.synchronization = Synchronization::EventHostWait; break;
        case FAKECUDA_RAS_STREAM_SYNC: result.synchronization = Synchronization::StreamHostWait; break;
        case FAKECUDA_RAS_ALLOC: case FAKECUDA_RAS_FREE: case FAKECUDA_RAS_DEVICE_SYNC:
            result.synchronization = Synchronization::DeviceHostWait; break;
        case FAKECUDA_RAS_GROUP_START:
            if (depth_ == std::numeric_limits<std::uint32_t>::max())
                throw ras::Error("Maya lane classification NCCL group nesting overflow");
            if (!depth_++) result.synchronization = Synchronization::GroupBegin;
            break;
        case FAKECUDA_RAS_GROUP_END:
            if (!depth_) throw ras::Error("Maya lane classification unmatched NCCL GroupEnd");
            if (!--depth_) {
                result.synchronization = Synchronization::GroupCommit;
                result.outer_group_commit = true;
            }
            break;
        case FAKECUDA_RAS_SEND: case FAKECUDA_RAS_RECV: case FAKECUDA_RAS_ALLREDUCE:
        case FAKECUDA_RAS_BROADCAST: case FAKECUDA_RAS_ALLGATHER: case FAKECUDA_RAS_REDUCESCATTER:
            // A grouped transfer is submitted with all of its peers by the
            // outer GroupEnd. Same-stream transfers inside that group need no
            // extra cut; changing host/device context still seals above.
            if(!depth_)result.synchronization = Synchronization::Communication;
            break;
        default: break;
        }
        if (result.synchronization != Synchronization::None) result.seal_before = result.seal_after = true;
        result.group_depth = depth_;
        previous_ = result.context; seen_ = true;
        return result;
    }
    std::uint32_t group_depth() const noexcept { return depth_; }
    void finish() const {
        if (depth_) throw ras::Error("Maya lane classification incomplete NCCL group");
    }
private:
    LaneContext previous_{};
    std::uint32_t depth_ = 0;
    bool seen_ = false;
};

// DataDep must preserve the scheduler's full execution relation: each raw
// call's host predecessor; same-stream order and legacy default-stream joins;
// event record -> stream/event waits; stream/device completion -> host waits;
// communicator matching by directed peer and ordinal. Inside an NCCL group,
// P2P completion is deferred until the outer GroupEnd. Every pending transfer
// depends on the common readiness frontier (including intervening CUDA waits),
// and all its completion nodes join each participating stream afterward.
// A single partition may have host+device lanes, but must not turn these into
// a false serial chain between independent device streams or group transfers.

} // namespace flexeva::maya::paper
