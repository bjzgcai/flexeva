#include <flexeva/maya/paper_cost.hpp>
#include <ras_capture.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace flexeva::maya {
namespace {

std::uint64_t checked_product(std::uint64_t left, std::uint64_t right) {
    if (right && left > std::numeric_limits<std::uint64_t>::max() / right)
        throw ras::Error("Maya native cost feature overflows uint64");
    return left * right;
}

std::uint64_t dimension(std::int32_t value) {
    if (value < 0) throw ras::Error("negative Maya native GEMM dimension");
    // The artifact record_count decoder substitutes 1 for missing/zero values.
    return value ? static_cast<std::uint64_t>(value) : 1;
}

std::uint64_t nccl_width(std::int32_t datatype) {
    switch (datatype) {
    case 0: case 1: case 10: case 11: return 1; // int8, uint8, float8
    case 6: case 9: return 2; // float16, bfloat16
    case 2: case 3: case 7: return 4; // int32, uint32, float32
    case 4: case 5: case 8: return 8; // int64, uint64, float64
    default: throw ras::Error("unknown Maya native NCCL datatype");
    }
}

std::string_view gemm_api(std::uint32_t kind) {
    switch (kind) {
    case FAKECUDA_RAS_API_CUBLAS_SGEMM: return "cublasSgemm_v2";
    case FAKECUDA_RAS_API_CUBLAS_DGEMM: return "cublasDgemm_v2";
    case FAKECUDA_RAS_API_CUBLAS_CGEMM: return "cublasCgemm_v2";
    case FAKECUDA_RAS_API_CUBLAS_ZGEMM: return "cublasZgemm_v2";
    case FAKECUDA_RAS_API_CUBLAS_GEMM_EX: return "cublasGemmEx";
    case FAKECUDA_RAS_API_CUBLAS_SGEMM_EX: return "cublasSgemmEx";
    case FAKECUDA_RAS_API_CUBLAS_GEMM_BATCHED_EX: return "cublasGemmBatchedEx";
    case FAKECUDA_RAS_API_CUBLAS_GEMM_STRIDED_BATCHED_EX: return "cublasGemmStridedBatchedEx";
    case FAKECUDA_RAS_API_CUBLAS_SGEMM_STRIDED_BATCHED: return "cublasSgemmStridedBatched";
    case FAKECUDA_RAS_API_CUBLAS_DGEMM_STRIDED_BATCHED: return "cublasDgemmStridedBatched";
    case FAKECUDA_RAS_API_CUBLAS_CGEMM_STRIDED_BATCHED: return "cublasCgemmStridedBatched";
    case FAKECUDA_RAS_API_CUBLAS_ZGEMM_STRIDED_BATCHED: return "cublasZgemmStridedBatched";
    case FAKECUDA_RAS_API_CUBLASLT_MATMUL: return "cublasLtMatmul";
    default: throw ras::Error("missing or unknown Maya native GEMM API identity");
    }
}

constexpr std::uint64_t cache_schema = 0x4d415941434f5354ULL;
constexpr std::uint64_t cache_format = 1;
constexpr ras::DirectoryKey marker_key{0, 0};

struct CacheMarker {
    std::uint64_t schema;
    std::uint64_t format;
    std::uint64_t policy;
};

struct CacheEntry {
    std::uint64_t api_size;
    std::uint64_t kind_size;
    std::uint64_t stream;
    std::uint64_t bytes;
    std::uint64_t count;
    std::int64_t rank;
    std::uint64_t blocking;
    double cost;
};

std::uint64_t policy_number(PaperCostPolicy policy) {
    switch (policy) {
    case PaperCostPolicy::ArtifactCompatible: return 1;
    case PaperCostPolicy::DeterministicV1: return 2;
    }
    throw ras::Error("unknown Maya paper cost policy");
}

ras::Buffer write_marker(ras::Transaction& tx, PaperCostPolicy policy) {
    const CacheMarker marker{cache_schema, cache_format, policy_number(policy)};
    auto writer = tx.buffer(sizeof(marker));
    auto span = writer.reserve(sizeof(marker));
    std::memcpy(span.bytes().data(), &marker, sizeof(marker));
    span.commit();
    return writer.finish();
}

CacheEntry key_for(const EventCostInput& event, PaperCostPolicy policy) {
    const bool legacy = policy == PaperCostPolicy::ArtifactCompatible;
    return {event.api.size(), event.kind.size(), legacy ? event.stream % 1024 : event.stream,
            event.bytes, event.count, legacy ? 0 : event.rank,
            legacy ? 0ULL : static_cast<std::uint64_t>(event.blocking), 0};
}

std::uint64_t hash_key(const CacheEntry& key, const EventCostInput& event) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto byte = [&](unsigned char value) { hash = (hash ^ value) * 1099511628211ULL; };
    const auto number = [&](std::uint64_t value) {
        for (unsigned i = 0; i != 8; ++i) { byte(static_cast<unsigned char>(value)); value >>= 8; }
    };
    number(key.api_size);
    for (unsigned char value : event.api) byte(value);
    number(key.kind_size);
    for (unsigned char value : event.kind) byte(value);
    number(key.stream);
    number(key.bytes);
    number(key.count);
    number(static_cast<std::uint64_t>(key.rank));
    number(key.blocking);
    return hash;
}

bool matches(const ras::Buffer& buffer, const CacheEntry& key,
             const EventCostInput& event, double& cost) {
    const auto view = buffer.bytes();
    const auto bytes = view.span();
    if (bytes.size() < sizeof(CacheEntry)) throw ras::Error("invalid Maya cost cache entry");
    CacheEntry stored{};
    std::memcpy(&stored, bytes.data(), sizeof(stored));
    const auto trailing = bytes.size() - sizeof(CacheEntry);
    if (stored.api_size > trailing || stored.kind_size != trailing - stored.api_size)
        throw ras::Error("invalid Maya cost cache key length");
    if (stored.api_size != key.api_size || stored.kind_size != key.kind_size ||
        stored.stream != key.stream || stored.bytes != key.bytes || stored.count != key.count ||
        stored.rank != key.rank || stored.blocking != key.blocking) return false;
    const auto* strings = reinterpret_cast<const char*>(bytes.data() + sizeof(CacheEntry));
    if (std::string_view(strings, stored.api_size) != event.api ||
        std::string_view(strings + stored.api_size, stored.kind_size) != event.kind) return false;
    cost = stored.cost;
    return true;
}

ras::Buffer write_entry(ras::Transaction& tx, CacheEntry key, const EventCostInput& event,
                        double cost) {
    if (event.api.size() > std::numeric_limits<std::size_t>::max() - sizeof(CacheEntry) ||
        event.kind.size() > std::numeric_limits<std::size_t>::max() - sizeof(CacheEntry) - event.api.size())
        throw ras::Error("Maya cost cache entry is too large");
    key.cost = cost;
    const auto size = sizeof(CacheEntry) + event.api.size() + event.kind.size();
    auto writer = tx.buffer(size);
    auto span = writer.reserve(size);
    auto* bytes = span.bytes().data();
    std::memcpy(bytes, &key, sizeof(key));
    if (!event.api.empty()) std::memcpy(bytes + sizeof(key), event.api.data(), event.api.size());
    if (!event.kind.empty())
        std::memcpy(bytes + sizeof(key) + event.api.size(), event.kind.data(), event.kind.size());
    span.commit();
    return writer.finish();
}

// Python's initial state exceeds 64 bits for most of the 96 trees. Only the
// left-shift term is masked; preserve the high word through both right shifts.
struct ForestState {
    std::uint64_t low, high;
    explicit ForestState(std::uint64_t tree) {
        constexpr std::uint64_t multiplier = 0x9E3779B97F4A7C15ULL;
        const auto lower = (multiplier & 0xffffffffULL) * tree;
        const auto upper = (multiplier >> 32) * tree;
        low = lower + (upper << 32);
        high = (upper >> 32) + (low < lower ? 1 : 0);
    }
    void xor_right(unsigned bits) {
        low ^= (low >> bits) | (high << (64 - bits));
        high ^= high >> bits;
    }
    void advance() {
        xor_right(12);
        low ^= low << 25;
        xor_right(27);
    }
};

double forest_cost(const EventCostInput& event) {
    const std::array<double, 8> features{
        static_cast<double>(event.stream % 1024 + 1),
        static_cast<double>(event.bytes % 1000003 + 1),
        static_cast<double>(event.count % 65521 + 1),
        static_cast<double>(event.api.size() + 1),
        static_cast<double>(event.kind.size() + 1),
        static_cast<double>(event.bytes / 1024 + 1),
        static_cast<double>(event.count / 8 + 1),
        static_cast<double>(static_cast<std::uint64_t>(event.rank) + 1)};
    double total = 0;
    for (std::uint64_t tree = 0; tree != 96; ++tree) {
        ForestState state(tree + 1);
        double leaf = 0;
        for (std::uint64_t depth = 0; depth != 18; ++depth) {
            state.advance();
            const auto feature = state.low % 8;
            const auto threshold = static_cast<double>((state.low >> 8) % 4096) + 0.5;
            if (std::fmod(features[feature] + static_cast<double>(depth * 13), 4096.0) > threshold)
                leaf += 0.003 * static_cast<double>(depth + 1);
            else
                leaf += 0.0015 * static_cast<double>(tree + 1);
        }
        total += leaf;
    }
    return std::max(1.0, std::log1p(total / 96.0 + features[1] * 0.00001) * 8.0);
}

double fallback_cost(const EventCostInput& event) {
    if (event.kind == "nccl_collective") {
        // Convert before multiplication so count * 2 retains Python's range.
        const auto scale = event.bytes ? static_cast<double>(event.bytes) : static_cast<double>(event.count) * 2;
        return std::max(1.0, scale / 1.0e6);
    }
    if (event.kind == "mem_copy") return std::max(1.0, static_cast<double>(event.bytes) / 1.0e6);
    if (event.kind == "host_marker") return 0;
    return event.blocking ? 2.0 : 1.0;
}

} // namespace

EventCostInput native_event_cost_input(const FakeCudaRasRecordV2& native) {
    const auto& op = native.base;
    if (op.abi_version != 2 || native.struct_size < sizeof(native))
        throw ras::Error("Maya native cost requires a complete ABI-2 record");
    if (native.flags_v2 & FAKECUDA_RAS_PAYLOAD_ELIDED)
        throw ras::Error("Maya retained record must reuse its prior Execution cost");
    if (op.rank < 0) throw ras::Error("Maya event cost requires a nonnegative logical rank");
    EventCostInput event;
    event.stream = op.stream; event.bytes = op.bytes;
    event.count = std::max<std::uint64_t>(op.count, 1); event.rank = op.rank;
    event.kind = "stream_op";
    switch (op.op) {
    case FAKECUDA_RAS_KERNEL:
        if (native.api_kind == FAKECUDA_RAS_API_CUDA_LAUNCH_KERNEL) event.api = "cudaLaunchKernel";
        else if (native.api_kind == FAKECUDA_RAS_API_CUDA_LAUNCH_KERNEL_EX_C) event.api = "cudaLaunchKernelExC";
        else throw ras::Error("missing or unknown Maya native kernel API identity");
        event.kind = "kernel_launch";
        if (!op.count) {
            for (auto value : native.grid) event.count = checked_product(event.count, std::max(value, 1U));
            for (auto value : native.block) event.count = checked_product(event.count, std::max(value, 1U));
        }
        break;
    case FAKECUDA_RAS_GEMM:
        event.api = gemm_api(native.api_kind); event.kind = "blas_compute";
        if (!op.count) {
            event.count = checked_product(dimension(op.m), dimension(op.n));
            event.count = checked_product(event.count, dimension(op.k));
            event.count = checked_product(event.count, std::max<std::uint64_t>(native.batch_count, 1));
        }
        break;
    case FAKECUDA_RAS_SEND: event.api = "ncclSend"; event.kind = "nccl_collective"; break;
    case FAKECUDA_RAS_RECV: event.api = "ncclRecv"; event.kind = "nccl_collective"; break;
    case FAKECUDA_RAS_ALLREDUCE: event.api = "ncclAllReduce"; event.kind = "nccl_collective"; break;
    case FAKECUDA_RAS_BROADCAST: event.api = "ncclBroadcast"; event.kind = "nccl_collective"; break;
    case FAKECUDA_RAS_ALLGATHER: event.api = "ncclAllGather"; event.kind = "nccl_collective"; break;
    case FAKECUDA_RAS_REDUCESCATTER: event.api = "ncclReduceScatter"; event.kind = "nccl_collective"; break;
    case FAKECUDA_RAS_MEMCPY_ASYNC:
        event.api = op.flags & 1 ? "cudaMemcpy" : "cudaMemcpyAsync"; event.kind = "mem_copy"; break;
    case FAKECUDA_RAS_MEMSET_ASYNC:
        event.api = op.flags & 1 ? "cudaMemset" : "cudaMemsetAsync"; event.kind = "mem_copy"; break;
    case FAKECUDA_RAS_ALLOC: event.api = "cudaMalloc"; event.kind = "mem_alloc"; break;
    case FAKECUDA_RAS_FREE: event.api = "cudaFree"; event.kind = "mem_alloc"; break;
    case FAKECUDA_RAS_STREAM_CREATE: event.api = "cudaStreamCreate"; break;
    case FAKECUDA_RAS_STREAM_DESTROY: event.api = "cudaStreamDestroy"; break;
    case FAKECUDA_RAS_EVENT_CREATE: event.api = "cudaEventCreateWithFlags"; break;
    case FAKECUDA_RAS_EVENT_DESTROY: event.api = "cudaEventDestroy"; break;
    case FAKECUDA_RAS_EVENT_RECORD: event.api = "cudaEventRecord"; break;
    case FAKECUDA_RAS_STREAM_WAIT_EVENT: event.api = "cudaStreamWaitEvent"; break;
    case FAKECUDA_RAS_EVENT_SYNC:
        event.blocking = !(op.flags & 1);
        event.api = event.blocking ? "cudaEventSynchronize" : "cudaEventQuery"; break;
    case FAKECUDA_RAS_STREAM_SYNC:
        event.blocking = !(op.flags & 1);
        event.api = event.blocking ? "cudaStreamSynchronize" : "cudaStreamQuery"; break;
    case FAKECUDA_RAS_DEVICE_SYNC: event.api = "cudaDeviceSynchronize"; event.blocking = true; break;
    case FAKECUDA_RAS_BLAS_CREATE: event.api = "cublasCreate_v2"; event.kind = "context_op"; break;
    case FAKECUDA_RAS_BLAS_DESTROY: event.api = "cublasDestroy_v2"; event.kind = "context_op"; break;
    case FAKECUDA_RAS_BLAS_SET_STREAM: event.api = "cublasSetStream_v2"; break;
    case FAKECUDA_RAS_NCCL_UID: event.api = "ncclGetUniqueId"; event.kind = "other"; break;
    case FAKECUDA_RAS_COMM_INIT: event.api = "ncclCommInitRank"; event.kind = "other"; break;
    case FAKECUDA_RAS_COMM_DESTROY: event.api = "ncclCommDestroy"; event.kind = "other"; break;
    case FAKECUDA_RAS_GROUP_START: event.api = "ncclGroupStart"; event.kind = "other"; break;
    case FAKECUDA_RAS_GROUP_END: event.api = "ncclGroupEnd"; event.kind = "other"; break;
    default: throw ras::Error("unknown Maya native cost operation");
    }
    if (event.kind == "nccl_collective" && !event.bytes)
        event.bytes = checked_product(event.count, nccl_width(op.datatype));
    return event;
}

double predict_event_us(const EventCostInput& event) {
    if (event.rank < 0) throw ras::Error("Maya event cost requires a nonnegative logical rank");
    if (event.kind == "kernel_launch" || event.kind == "blas_compute") return forest_cost(event);
    return fallback_cost(event);
}

PaperCostModel::PaperCostModel(ras::Transaction& tx, PaperCostPolicy policy, ras::DirectoryId cache)
    : transaction_(&tx), policy_(policy), cache_(cache) {
    const auto expected_policy = policy_number(policy);
    if (!cache_) {
        cache_ = tx.add_directory();
        tx.directory_put(cache_, marker_key, {write_marker(tx, policy), {}, {}});
        return;
    }
    const auto record = tx.read().directory_find(cache_, marker_key);
    if (!record || record->metadata.size() != sizeof(CacheMarker))
        throw ras::Error("missing Maya cost cache policy marker");
    CacheMarker marker{};
    const auto view = record->metadata.bytes();
    std::memcpy(&marker, view.span().data(), sizeof(marker));
    if (marker.schema != cache_schema || marker.format != cache_format || marker.policy != expected_policy)
        throw ras::Error("Maya cost cache policy mismatch");
}

std::string_view PaperCostModel::identity() const noexcept {
    return policy_ == PaperCostPolicy::ArtifactCompatible ? "artifact-compatible-v1" : "deterministic-v1";
}

double PaperCostModel::predict_us(const EventCostInput& event) {
    if (event.rank < 0) throw ras::Error("Maya event cost requires a nonnegative logical rank");
    ++counters_.events;
    if (policy_ == PaperCostPolicy::ArtifactCompatible && event.duration_hint_us > 0) {
        ++counters_.hint_uses;
        return event.duration_hint_us;
    }
    if (event.kind == "kernel_launch" || event.kind == "blas_compute") {
        ++counters_.prediction_calls;
        const auto entry = key_for(event, policy_);
        ras::DirectoryKey key{hash_key(entry, event), 1};
        const auto read = transaction_->read();
        for (;;) {
            ++counters_.cache_probes;
            const auto record = read.directory_find(cache_, key);
            if (!record) break;
            double cost = 0;
            if (matches(record->metadata, entry, event, cost)) {
                ++counters_.cache_hits;
                return cost;
            }
            if (key.second == std::numeric_limits<std::uint64_t>::max())
                throw ras::Error("Maya cost cache collision chain overflow");
            ++key.second;
        }
        const auto cost = forest_cost(event);
        transaction_->directory_put(cache_, key, {write_entry(*transaction_, entry, event, cost), {}, {}});
        ++counters_.cache_misses;
        return cost;
    }
    return fallback_cost(event);
}

} // namespace flexeva::maya
