#include "fake_types.h"
#include "fake_runtime_api.hpp"
#include "nccl.hpp"
#include <cstring>
#include <cstdlib>
#include <exception>
#include <atomic>
#include <mutex>
#include <limits>
#include <unordered_set>
#include <unistd.h>
#include "utils.hpp"
#include "../../include/common/ras_capture.h"

namespace {
std::atomic<unsigned long long> g_nccl_uid_counter{1};
// Native handles remain live fake-runtime objects; semantic identity and
// communication/group sequence state belong to the installed RAS sink.
std::unordered_set<ncclComm_t> g_ras_nccl_comms;

ncclResult_t ras_nccl_error(const char* message, ncclResult_t result) noexcept {
    fakecuda_ras_fail(message);
    return result;
}

ncclResult_t ras_nccl_unsupported(const char* api) noexcept {
    return ras_nccl_error(api, ncclInvalidUsage);
}

template <class F>
ncclResult_t ras_nccl_call(F&& body) noexcept {
    try { return body(); }
    catch (const std::exception& error) {
        return ras_nccl_error(error.what(), ncclInternalError);
    } catch (...) {
        return ras_nccl_error("NCCL direct capture failed", ncclInternalError);
    }
}

bool ras_valid_comm(ncclComm_t comm) {
    return comm && g_ras_nccl_comms.find(comm) != g_ras_nccl_comms.end();
}

bool ras_valid_stream(cudaStream_t stream) {
    if (!stream) return true;
    std::lock_guard<std::mutex> lock(g_streamMutex);
    return g_validStreams.find(to_fake_stream(stream)) != g_validStreams.end();
}

ncclResult_t ras_comm_init(ncclComm_t* output, int nranks,
    const ncclUniqueId& uid, int rank, int blocking) noexcept {
    return ras_nccl_call([&]() -> ncclResult_t {
        if (!output || nranks < 1 || rank < 0 || rank >= nranks)
            return ras_nccl_error("Invalid NCCL communicator arguments", ncclInvalidArgument);
        *output = nullptr;
        auto comm = static_cast<ncclComm_t>(calloc(1, sizeof(ncclComm)));
        if (!comm) return ras_nccl_error("NCCL communicator allocation failed", ncclSystemError);
        comm->rank = rank;
        comm->nranks = nranks;
        comm->commId = uid;
        comm->blocking = blocking;
        comm->asyncError = ncclSuccess;
        try { g_ras_nccl_comms.insert(comm); }
        catch (...) { free(comm); throw; }
        auto record = fakecuda::ras::record(FAKECUDA_RAS_COMM_INIT);
        record.comm = fakecuda::ras::address(comm);
        record.rank = rank;
        record.nranks = nranks;
        record.flags = blocking;
        static_assert(sizeof(record.comm_uid) == sizeof(uid.internal));
        std::memcpy(record.comm_uid, uid.internal, sizeof(record.comm_uid));
        if (!fakecuda::ras::emit(record)) {
            g_ras_nccl_comms.erase(comm);
            free(comm);
            return ncclInternalError;
        }
        *output = comm;
        return ncclSuccess;
    });
}

ncclResult_t ras_nccl_transfer(FakeCudaRasOp operation, const void* sendbuff,
    void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t reduction,
    int peer, ncclComm_t comm, cudaStream_t stream) noexcept {
    return ras_nccl_call([&]() -> ncclResult_t {
        if (!ras_valid_comm(comm) || !ras_valid_stream(stream))
            return ras_nccl_error("Invalid NCCL communicator or stream", ncclInvalidArgument);
        if (static_cast<int>(datatype) < 0 || static_cast<int>(datatype) >= ncclNumTypes)
            return ras_nccl_error("Invalid NCCL datatype", ncclInvalidArgument);
        const size_t width = get_dtype_size(datatype);
        if (count > std::numeric_limits<size_t>::max() / width)
            return ras_nccl_error("NCCL byte count overflow", ncclInvalidArgument);
        if ((operation == FAKECUDA_RAS_ALLGATHER || operation == FAKECUDA_RAS_REDUCESCATTER) &&
            count > std::numeric_limits<size_t>::max() / width / static_cast<size_t>(comm->nranks))
            return ras_nccl_error("NCCL collective byte count overflow", ncclInvalidArgument);
        const bool needs_send = operation != FAKECUDA_RAS_RECV &&
            (operation != FAKECUDA_RAS_BROADCAST || comm->rank == peer);
        if (count && ((needs_send && !sendbuff) ||
                     (operation != FAKECUDA_RAS_SEND && !recvbuff)))
            return ras_nccl_error("Invalid NCCL buffer", ncclInvalidArgument);
        if ((operation == FAKECUDA_RAS_SEND || operation == FAKECUDA_RAS_RECV ||
             operation == FAKECUDA_RAS_BROADCAST) && (peer < 0 || peer >= comm->nranks))
            return ras_nccl_error("Invalid NCCL peer", ncclInvalidArgument);
        if ((operation == FAKECUDA_RAS_ALLREDUCE || operation == FAKECUDA_RAS_REDUCESCATTER) &&
            (static_cast<int>(reduction) < 0 || static_cast<int>(reduction) >= ncclNumOps))
            return ras_nccl_error("Unsupported NCCL reduction operation", ncclInvalidArgument);
        auto record = fakecuda::ras::record(operation);
        record.comm = fakecuda::ras::address(comm);
        record.stream = fakecuda::ras::address(stream);
        record.src = fakecuda::ras::address(sendbuff);
        record.dst = fakecuda::ras::address(recvbuff);
        record.rank = comm->rank;
        record.nranks = comm->nranks;
        record.peer = peer;
        record.count = count;
        record.bytes = count * width;
        record.datatype = static_cast<int>(datatype);
        record.reduction = static_cast<int>(reduction);
        // Trace-only execution never fabricates collective tensor values.
        // Host numerical consumers must obtain an explicit Ground result.
        fake_stream_mark_work_enqueued(stream);
        comm->asyncError = ncclSuccess;
        if (!fakecuda::ras::emit(record)) {
            comm->asyncError = ncclInternalError;
            return ncclInternalError;
        }
        return ncclSuccess;
    });
}

ncclResult_t nccl_comm_init_common(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    int blocking
) {
    if (fakecuda::ras::active())
        return ras_comm_init(comm, nranks, commId, rank, blocking);
    if (comm == nullptr || nranks < 1 || rank < 0 || rank >= nranks) {
        LOG_ERROR(NCCL, "comm pointer is NULL\n");
        return ncclInvalidArgument;
    }

    *comm = nullptr;
    *comm = (ncclComm_t)calloc(1, sizeof(struct ncclComm));
    if (*comm == nullptr) {
        LOG_ERROR(NCCL, "Failed to allocate comm\n");
        return ncclSystemError;
    }

    (*comm)->rank = rank;
    (*comm)->nranks = nranks;
    (*comm)->asyncError = ncclSuccess;
    (*comm)->commId = commId;
    (*comm)->blocking = blocking ? 1 : 0;
    // Native liveness is independent of whether this capture window records
    // the setup call. Cached communicators remain usable in later windows.
    try { g_ras_nccl_comms.insert(*comm); }
    catch (...) { free(*comm); *comm = nullptr; return ncclSystemError; }

    return ncclSuccess;
}
}  // namespace


API ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId){
    if (!uniqueId)
        return ras_nccl_error("Null NCCL unique-id output", ncclInvalidArgument);
    std::memset(uniqueId->internal, 0, sizeof(uniqueId->internal));
    const auto uid = g_nccl_uid_counter.fetch_add(1);
    // Binary identity is carried without string conversion or truncation.
    const auto pid = static_cast<uint64_t>(getpid());
    std::memcpy(uniqueId->internal, &pid, sizeof(pid));
    std::memcpy(uniqueId->internal + sizeof(pid), &uid, sizeof(uid));
    if (fakecuda::ras::active()) {
        auto record = fakecuda::ras::record(FAKECUDA_RAS_NCCL_UID);
        std::memcpy(record.comm_uid, uniqueId->internal, sizeof(record.comm_uid));
        return fakecuda::ras::emit(record) ? ncclSuccess : ncclInternalError;
    }
    return ncclSuccess;
}

API ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    if (fakecuda::ras::active())
        return ras_nccl_transfer(FAKECUDA_RAS_ALLREDUCE, sendbuff, recvbuff,
            count, datatype, op, -1, comm, stream);
    LOG_DEBUG(NCCL, "ncclAllReduce() called.");

    if (sendbuff == nullptr || recvbuff == nullptr || count == 0) {
        if (comm != nullptr) comm->asyncError = ncclSuccess;
        return sendbuff == nullptr || recvbuff == nullptr ? ncclInvalidArgument : ncclSuccess;
    }

    size_t byte_count = count * get_dtype_size(datatype);

    // 只处理 metadata
    if (byte_count <= METADATA_THRESHOLD) {
        memcpy(recvbuff, sendbuff, byte_count);
        LOG_INFO(NCCL, "ncclAllReduce: copied %zu bytes", byte_count);
    } else {
        LOG_INFO(NCCL, "ncclAllReduce: skipped %zu bytes", byte_count);
    }

    if (comm != nullptr) {
        comm->asyncError = ncclSuccess;
    }

    return ncclSuccess;
}

API ncclResult_t  ncclCommAbort(ncclComm_t comm){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommAbort is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclCommAbort() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclMemFree(void *ptr){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclMemFree is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclMemFree() called.");
    free(ptr);
    return ncclSuccess;
}

API ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank){
    if (fakecuda::ras::active()) {
        if (!ras_valid_comm(comm) || !rank)
            return ras_nccl_error("Invalid NCCL rank query", ncclInvalidArgument);
        *rank = comm->rank; return ncclSuccess;
    }
    LOG_DEBUG(NCCL, "ncclCommUserRank() called.");
    if (comm == nullptr || rank == nullptr) {
        return ncclInvalidArgument;
    }
    *rank = comm->rank;
    return ncclSuccess;
}

API ncclResult_t  ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommRegister is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclCommRegister() called.");
    if (comm == nullptr || buff == nullptr || handle == nullptr) {
        return ncclInvalidArgument;
    }
    *handle = buff;
    return ncclSuccess;
}

API ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclRedOpDestroy is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclRedOpDestroy() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclRedOpCreatePreMulSum is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclRedOpCreatePreMulSum() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclCommFinalize(ncclComm_t comm){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommFinalize is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclCommFinalize() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclCommDeregister(const ncclComm_t comm, void* handle){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommDeregister is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclCommDeregister() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclGroupSimulateEnd(ncclSimInfo_t* simInfo){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclGroupSimulateEnd is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclGroupSimulateEnd() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclGroupEnd(){
    if (fakecuda::ras::active()) {
        auto record = fakecuda::ras::record(FAKECUDA_RAS_GROUP_END);
        return fakecuda::ras::emit(record) ? ncclSuccess : ncclInternalError;
    }
    LOG_DEBUG(NCCL, "ncclGroupEnd() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream){
    if (fakecuda::ras::active())
        return ras_nccl_transfer(FAKECUDA_RAS_BROADCAST, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream);
    LOG_DEBUG(NCCL, "ncclBroadcast() called.");
    if (recvbuff == nullptr || count == 0) {
        return recvbuff == nullptr ? ncclInvalidArgument : ncclSuccess;
    }

    size_t byte_count = count * get_dtype_size(datatype);

    // 只处理 metadata
    if (byte_count <= METADATA_THRESHOLD && sendbuff != nullptr && sendbuff != recvbuff) {
        memcpy(recvbuff, sendbuff, byte_count);
        LOG_INFO(NCCL, "ncclBroadcast: copied %zu bytes", byte_count);
    } else {
        LOG_INFO(NCCL, "ncclBroadcast: skipped %zu bytes", byte_count);
    }

    return ncclSuccess;
}

API ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream){
    if (fakecuda::ras::active())
        return ras_nccl_transfer(FAKECUDA_RAS_BROADCAST, buff, buff, count, datatype, ncclSum, root, comm, stream);
    LOG_DEBUG(NCCL, "ncclBcast() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError){
    if (fakecuda::ras::active()) {
        if (!ras_valid_comm(comm) || !asyncError)
            return ras_nccl_error("Invalid NCCL error query", ncclInvalidArgument);
        *asyncError = comm->asyncError; return ncclSuccess;
    }
    LOG_DEBUG(NCCL, "ncclCommGetAsyncError() called.");
    if (asyncError != nullptr) {
        *asyncError = ncclSuccess;
    }
    return ncclSuccess;
}

API const char*  ncclGetLastError(ncclComm_t comm){
    LOG_DEBUG(NCCL, "ncclGetLastError() called.");
    return "fake ncclGetLastError";
}

API ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommWindowRegister is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclCommWindowRegister() called.");
    if (comm == nullptr || buff == nullptr || win == nullptr) {
        return ncclInvalidArgument;
    }
    *win = reinterpret_cast<ncclWindow_t>(buff);
    return ncclSuccess;
}

API ncclResult_t  ncclCommDestroy(ncclComm_t comm){
    if (fakecuda::ras::active()) return ras_nccl_call([&]() -> ncclResult_t {
        if (!ras_valid_comm(comm))
            return ras_nccl_error("Invalid NCCL communicator destruction", ncclInvalidArgument);
        auto record = fakecuda::ras::record(FAKECUDA_RAS_COMM_DESTROY);
        record.comm = fakecuda::ras::address(comm);
        record.rank = comm->rank;
        record.nranks = comm->nranks;
        g_ras_nccl_comms.erase(comm);
        free(comm);
        const bool success = fakecuda::ras::emit(record);
        return success ? ncclSuccess : ncclInternalError;
    });
    LOG_DEBUG(NCCL, "ncclCommDestroy() called.");
    if (comm) {
        g_ras_nccl_comms.erase(comm);
        free(comm);  // ⭐ 释放内存
    }
    return ncclSuccess;
}

API ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count){
    if (fakecuda::ras::active()) {
        if (!ras_valid_comm(comm) || !count)
            return ras_nccl_error("Invalid NCCL size query", ncclInvalidArgument);
        *count = comm->nranks; return ncclSuccess;
    }
    LOG_DEBUG(NCCL, "ncclCommCount() called.");
    if (comm == nullptr || count == nullptr) {
        return ncclInvalidArgument;
    }
    *count = comm->nranks;
    return ncclSuccess;
}

API ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommInitAll is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclCommInitAll() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config){
    if (fakecuda::ras::active())
        return ras_comm_init(comm, nranks, commId, rank, (config && config->blocking) ? 1 : 0);
    LOG_DEBUG(NCCL, "ncclCommInitRankConfig() called.");
    return nccl_comm_init_common(
        comm,
        nranks,
        commId,
        rank,
        (config && config->blocking) ? 1 : 0
    );
}

API ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream){
    if (fakecuda::ras::active())
        return ras_nccl_transfer(FAKECUDA_RAS_RECV, nullptr, recvbuff,
            count, datatype, ncclSum, peer, comm, stream);
    LOG_DEBUG(NCCL, "ncclRecv() called.");
    return ncclSuccess;
}

API ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommWindowDeregister is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclCommWindowDeregister() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank){
    if (fakecuda::ras::active())
        return ras_comm_init(comm, nranks, commId, rank, 0);
    LOG_DEBUG(NCCL, "ncclCommInitRank() called.");
    return nccl_comm_init_common(comm, nranks, commId, rank, 0);
}

API ncclResult_t  ncclGetVersion(int *version){
    if (fakecuda::ras::active()) {
        if (!version) return ras_nccl_error("Invalid NCCL version output", ncclInvalidArgument);
        *version = 22620; return ncclSuccess;
    }
    LOG_DEBUG(NCCL, "ncclGetVersion() called.");
    if (version) *version = 22620; // NCCL 2.26.2
    return ncclSuccess;
}

API const char*  ncclGetErrorString(ncclResult_t result){
    LOG_DEBUG(NCCL, "ncclGetErrorString() called.");
    return "fake ncclGetErrorString";
}

API ncclResult_t  ncclMemAlloc(void** ptr, size_t size){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclMemAlloc is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclMemAlloc() called.");
    if (ptr == nullptr) {
        return ncclInvalidArgument;
    }
    void* allocated = malloc(size > 0 ? size : 1);
    if (allocated == nullptr) {
        return ncclSystemError;
    }
    *ptr = allocated;
    return ncclSuccess;
}

API ncclResult_t  ncclGroupStart(){
    if (fakecuda::ras::active()) {
        auto record = fakecuda::ras::record(FAKECUDA_RAS_GROUP_START);
        return fakecuda::ras::emit(record) ? ncclSuccess : ncclInternalError;
    }
    LOG_DEBUG(NCCL, "ncclGroupStart() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream){
    if (fakecuda::ras::active())
        return ras_nccl_transfer(FAKECUDA_RAS_SEND, sendbuff, nullptr,
            count, datatype, ncclSum, peer, comm, stream);
    LOG_DEBUG(NCCL, "ncclSend() called.");
    return ncclSuccess;
}

API ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommInitRankScalable is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclCommInitRankScalable() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    if (fakecuda::ras::active())
        return ras_nccl_transfer(FAKECUDA_RAS_REDUCESCATTER, sendbuff, recvbuff, recvcount, datatype, op, -1, comm, stream);
    LOG_DEBUG(NCCL, "ncclReduceScatter() called.");

    if (sendbuff == nullptr || recvbuff == nullptr || recvcount == 0) {
        return sendbuff == nullptr || recvbuff == nullptr ? ncclInvalidArgument : ncclSuccess;
    }

    size_t recv_bytes = recvcount * get_dtype_size(datatype);

    // 只处理 metadata
    if (recv_bytes <= METADATA_THRESHOLD) {
        int rank = (comm != nullptr) ? comm->rank : 0;
        void* src = (char*)sendbuff + (rank * recv_bytes);
        memcpy(recvbuff, src, recv_bytes);
        LOG_INFO(NCCL, "ncclReduceScatter: copied %zu bytes", recv_bytes);
    } else {
        LOG_INFO(NCCL, "ncclReduceScatter: skipped %zu bytes", recv_bytes);
    }

    return ncclSuccess;
}

API ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream){
    if (fakecuda::ras::active())
        return ras_nccl_transfer(FAKECUDA_RAS_ALLGATHER, sendbuff, recvbuff, sendcount, datatype, ncclSum, -1, comm, stream);
    LOG_DEBUG(NCCL, "ncclAllGather() called.");
    if (sendbuff == nullptr || recvbuff == nullptr || sendcount == 0) {
        return sendbuff == nullptr || recvbuff == nullptr ? ncclInvalidArgument : ncclSuccess;
    }

    size_t dtype_size = get_dtype_size(datatype);
    size_t send_bytes = sendcount * dtype_size;

    // 只处理 metadata（小数据）
    if (send_bytes <= METADATA_THRESHOLD) {
        int rank = (comm != nullptr) ? comm->rank : 0;
        int nranks = (comm != nullptr) ? comm->nranks : 1;

        // 将 sendbuff 复制到 recvbuff 的所有位置（模拟所有 rank 数据一致）
        for (int i = 0; i < nranks; i++) {
            void* dst = (char*)recvbuff + (i * send_bytes);
            memcpy(dst, sendbuff, send_bytes);
        }
        LOG_INFO(NCCL, "ncclAllGather: copied %zu bytes", send_bytes);
    } else {
        LOG_INFO(NCCL, "ncclAllGather: skipped %zu bytes", send_bytes);
    }

    return ncclSuccess;
}

API ncclResult_t  ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config){
    if (newcomm) *newcomm = nullptr;
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclCommSplit is outside the direct RAS capture subset");
    return ncclInvalidUsage;
}

API ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream){
    if (fakecuda::ras::active())
        return ras_nccl_unsupported("ncclReduce is outside the direct RAS capture subset");
    LOG_DEBUG(NCCL, "ncclReduce() called.");
    return ncclSuccess;
}
