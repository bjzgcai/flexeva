#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Process-local synchronous ABI. Install/reset and intercepted calls must be
 * externally serialized. The callback borrows op only until it returns. Rank,
 * Boundary and record/retain policy belong to the callback's user context. */
typedef enum FakeCudaRasOp {
    FAKECUDA_RAS_ALLOC = 1, FAKECUDA_RAS_FREE,
    FAKECUDA_RAS_STREAM_CREATE, FAKECUDA_RAS_STREAM_DESTROY,
    FAKECUDA_RAS_EVENT_CREATE, FAKECUDA_RAS_EVENT_DESTROY,
    FAKECUDA_RAS_EVENT_RECORD, FAKECUDA_RAS_STREAM_WAIT_EVENT,
    FAKECUDA_RAS_EVENT_SYNC, FAKECUDA_RAS_STREAM_SYNC, FAKECUDA_RAS_DEVICE_SYNC,
    FAKECUDA_RAS_MEMCPY_ASYNC, FAKECUDA_RAS_MEMSET_ASYNC,
    FAKECUDA_RAS_BLAS_CREATE, FAKECUDA_RAS_BLAS_DESTROY,
    FAKECUDA_RAS_BLAS_SET_STREAM, FAKECUDA_RAS_GEMM,
    FAKECUDA_RAS_NCCL_UID, FAKECUDA_RAS_COMM_INIT, FAKECUDA_RAS_COMM_DESTROY,
    FAKECUDA_RAS_GROUP_START, FAKECUDA_RAS_GROUP_END,
    FAKECUDA_RAS_SEND, FAKECUDA_RAS_RECV, FAKECUDA_RAS_ALLREDUCE,
    FAKECUDA_RAS_KERNEL,
    /* Appended values preserve existing on-disk native operation identities.
     * BROADCAST.peer is its local root; ALLGATHER.count is sendcount;
     * REDUCESCATTER.count is recvcount. No numerical data is synthesized. */
    FAKECUDA_RAS_BROADCAST, FAKECUDA_RAS_ALLGATHER, FAKECUDA_RAS_REDUCESCATTER
} FakeCudaRasOp;

typedef struct FakeCudaRasRecord {
    uint32_t abi_version; /* 1 */
    uint32_t op;
    uint64_t stream, event, handle, comm;
    uint64_t src, dst, a, b, c;
    uint64_t bytes, count;
    int32_t flags, priority, value, copy_kind;
    int32_t m, n, k, lda, ldb, ldc;
    int32_t trans_a, trans_b, a_type, b_type, c_type, compute_type, algorithm;
    int32_t rank, nranks, peer, datatype, reduction;
    uint8_t comm_uid[128]; /* Exact ncclUniqueId bytes; UID and COMM_INIT only. */
} FakeCudaRasRecord;

/* ABI 2 extends rather than changes ABI 1. All pointers are borrowed only for
 * the synchronous callback. Sinks must copy names into their owned storage;
 * neither function addresses nor name pointers are persistent identities. */
enum {
    FAKECUDA_RAS_PAYLOAD_ELIDED = 1u,
    FAKECUDA_RAS_POINTER_ARRAY_BATCH = 2u
};
/* Exact public entrypoint identity is separate from operation class: cost
 * models may distinguish APIs that share dimensions and scheduling semantics.
 * Values are stable; typed GEMM identities name their exported _v2 entrypoint. */
typedef enum FakeCudaRasApiKind {
    FAKECUDA_RAS_API_UNSPECIFIED = 0,
    FAKECUDA_RAS_API_CUDA_LAUNCH_KERNEL = 1,
    FAKECUDA_RAS_API_CUDA_LAUNCH_KERNEL_EX_C = 2,
    FAKECUDA_RAS_API_CUBLAS_SGEMM = 10,
    FAKECUDA_RAS_API_CUBLAS_DGEMM = 11,
    FAKECUDA_RAS_API_CUBLAS_CGEMM = 12,
    FAKECUDA_RAS_API_CUBLAS_ZGEMM = 13,
    FAKECUDA_RAS_API_CUBLAS_GEMM_EX = 14,
    FAKECUDA_RAS_API_CUBLAS_SGEMM_EX = 15,
    FAKECUDA_RAS_API_CUBLAS_GEMM_BATCHED_EX = 16,
    FAKECUDA_RAS_API_CUBLAS_GEMM_STRIDED_BATCHED_EX = 17,
    FAKECUDA_RAS_API_CUBLAS_SGEMM_STRIDED_BATCHED = 18,
    FAKECUDA_RAS_API_CUBLAS_DGEMM_STRIDED_BATCHED = 19,
    FAKECUDA_RAS_API_CUBLAS_CGEMM_STRIDED_BATCHED = 20,
    FAKECUDA_RAS_API_CUBLAS_ZGEMM_STRIDED_BATCHED = 21,
    FAKECUDA_RAS_API_CUBLASLT_MATMUL = 22
} FakeCudaRasApiKind;
typedef struct FakeCudaRasRecordV2 {
    FakeCudaRasRecord base; /* base.abi_version == 2 */
    uint32_t struct_size, flags_v2;
    const char* kernel_name;
    uint64_t kernel_name_size;
    uint32_t grid[3], block[3];
    uint64_t shared_mem;
    uint64_t batch_count;
    int64_t stride_a, stride_b, stride_c, stride_d;
    uint64_t d, aux;
    int32_t ldd, d_type, epilogue, pointer_mode, math_mode;
    uint32_t api_kind; /* FakeCudaRasApiKind, including retained operations. */
} FakeCudaRasRecordV2;

/* Nonzero callback return or exception poisons the session. A poisoned session
 * remains active (legacy logging must stay disabled) until reset. */
typedef int (*FakeCudaRasSink)(void* user, const FakeCudaRasRecord* op);
int fakecuda_ras_install(void* user, FakeCudaRasSink sink);
typedef int (*FakeCudaRasSinkV2)(void* user, const FakeCudaRasRecordV2* op);
/* The policy is queried after native state changes and before trace-only
 * decoding. Return 1 to record, 0 to retain, or negative to fail. Retain still
 * calls the sink with PAYLOAD_ELIDED and cheap structural fields, including
 * GEMM operand addresses so resource identity advances in both modes. */
typedef int (*FakeCudaRasPolicy)(void* user, uint32_t op);
int fakecuda_ras_install_v2(void* user, FakeCudaRasSinkV2 sink,
                          FakeCudaRasPolicy policy);
int fakecuda_ras_wants_payload(uint32_t op);
int fakecuda_ras_emit_v2(const FakeCudaRasRecordV2* op);
/* Read-only live native allocation lookup. Returns 1 iff address is inside
 * [base, base + bytes), including allocations predating capture. Unknown
 * addresses return 0 and clear valid outputs; null outputs return 0. This
 * neither initializes the runtime nor emits a capture operation. */
int fakecuda_ras_allocation_range(uint64_t address, uint64_t* base, uint64_t* bytes);
void fakecuda_ras_reset(void);
int fakecuda_ras_active(void);
int fakecuda_ras_status(void); /* zero healthy/inactive, nonzero poisoned */
const char* fakecuda_ras_last_error(void);
int fakecuda_ras_emit(const FakeCudaRasRecord* op);
void fakecuda_ras_fail(const char* message);

#ifdef __cplusplus
}

namespace fakecuda::ras {
inline bool active() { return fakecuda_ras_active() != 0; }
inline FakeCudaRasRecord record(FakeCudaRasOp op) {
    FakeCudaRasRecord r{}; r.abi_version = 1; r.op = op; return r;
}
inline FakeCudaRasRecordV2 record_v2(FakeCudaRasOp op) {
    FakeCudaRasRecordV2 r{};
    r.base.abi_version = 2; r.base.op = op;
    r.struct_size = sizeof(r); r.batch_count = 1;
    return r;
}
inline bool wants_payload(FakeCudaRasOp op) {
    return fakecuda_ras_wants_payload(op) > 0;
}
template <class T> inline uint64_t address(T* p) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}
inline bool emit(const FakeCudaRasRecord& r) { return fakecuda_ras_emit(&r) == 0; }
inline bool emit(const FakeCudaRasRecordV2& r) { return fakecuda_ras_emit_v2(&r) == 0; }
}
#endif
