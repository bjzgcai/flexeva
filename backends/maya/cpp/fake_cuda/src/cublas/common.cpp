#include "fake_types.h"
#include <algorithm>
#include "fake_runtime_api.hpp"
#include "cublas.hpp"
#include "utils.hpp"
#include "../../include/common/ras_capture.h"

namespace {
// This descriptor is filled only after native stream state has advanced. The
// retain path emits a small routing record without full operation details.
cublasStatus_t direct_gemm(uint32_t api_kind, cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void* A, cudaDataType a_type, int lda, const void* B,
    cudaDataType b_type, int ldb, void* C, cudaDataType c_type, int ldc,
    cublasComputeType_t compute, cublasGemmAlgo_t algorithm,
    int batch_count = 1, int64_t stride_a = 0, int64_t stride_b = 0,
    int64_t stride_c = 0, bool pointer_array = false) {
    const bool trans_valid = transa >= CUBLAS_OP_N && transa <= CUBLAS_OP_C &&
                             transb >= CUBLAS_OP_N && transb <= CUBLAS_OP_C;
    if (!fake_valid_blas_handle(handle) || !fake_valid_stream(handle->stream) ||
        !trans_valid || m < 0 || n < 0 || k < 0 || batch_count < 0 ||
        lda < std::max(1, transa == CUBLAS_OP_N ? m : k) ||
        ldb < std::max(1, transb == CUBLAS_OP_N ? k : n) ||
        ldc < std::max(1, m) || (m && n && batch_count && (!A || !B || !C))) {
        fakecuda_ras_fail("Invalid cuBLAS GEMM arguments");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!m || !n || !batch_count) return CUBLAS_STATUS_SUCCESS;
    fake_stream_mark_work_enqueued(handle->stream);
    auto r = fakecuda::ras::record_v2(FAKECUDA_RAS_GEMM);
    r.api_kind = api_kind;
    r.base.handle = fakecuda::ras::address(handle);
    r.base.stream = fakecuda::ras::address(handle->stream);
    // Operand identities advance normalization state even when trace details
    // are retained. These are pointer values only, never operand dereferences.
    r.base.a = fakecuda::ras::address(A); r.base.b = fakecuda::ras::address(B);
    r.base.c = fakecuda::ras::address(C); r.d = r.base.c;
    if (pointer_array) r.flags_v2 |= FAKECUDA_RAS_POINTER_ARRAY_BATCH;
    if (!fakecuda::ras::wants_payload(FAKECUDA_RAS_GEMM)) {
        r.flags_v2 |= FAKECUDA_RAS_PAYLOAD_ELIDED;
        return fakecuda::ras::emit(r) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
    }
    r.base.m = m; r.base.n = n; r.base.k = k;
    r.base.lda = lda; r.base.ldb = ldb; r.base.ldc = ldc; r.ldd = ldc;
    r.base.trans_a = transa; r.base.trans_b = transb;
    r.base.a_type = a_type; r.base.b_type = b_type; r.base.c_type = c_type;
    r.d_type = c_type; r.base.compute_type = compute; r.base.algorithm = algorithm;
    r.batch_count = batch_count;
    r.stride_a = stride_a; r.stride_b = stride_b; r.stride_c = stride_c;
    r.stride_d = stride_c;
    auto* ctx = static_cast<FakeBlasContext*>(handle);
    r.math_mode = ctx->math_mode; r.pointer_mode = ctx->pointer_mode;
    return fakecuda::ras::emit(r) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
}

}

API cublasStatus_t cublasCreate_v2(cublasHandle_t* handle){
    if (!handle) {
        fakecuda_ras_fail("Invalid cublasCreate_v2 arguments");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    LOG_DEBUG(CUBLAS, "cublasCreate_v2() called.");
    // 分配内存
    auto* ctx = new (std::nothrow) FakeBlasContext{};
    if (!ctx) {
        if (fakecuda::ras::active()) {
            *handle = nullptr;
            fakecuda_ras_fail("cuBLAS handle allocation failed");
        }
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    // 初始化内部状态
    ctx->magic_number = 0xBEEF; // 标记，用于 debug
    ctx->stream = 0;            // 默认流 (cudaStreamDefault)
    {
        try { g_ras_blas_handles.insert(ctx); }
        catch (...) {
            delete ctx;
            *handle = nullptr;
            fakecuda_ras_fail("cuBLAS handle registration failed");
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
    }
    *handle = ctx; // 赋值给输出指针
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_BLAS_CREATE);
        r.handle = fakecuda::ras::address(ctx);
        const auto result = fakecuda::ras::emit(r) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
        if (result != CUBLAS_STATUS_SUCCESS) {
            g_ras_blas_handles.erase(ctx);
            delete ctx;
            *handle = nullptr;
        }
        return result;
    }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDestroy_v2(cublasHandle_t handle){
    if (!fake_valid_blas_handle(handle)) {
        fakecuda_ras_fail("Invalid cublasDestroy_v2 arguments");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    LOG_DEBUG(CUBLAS, "cublasDestroy_v2() called.");
    if (!handle || handle->magic_number != 0xBEEF) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_BLAS_DESTROY);
        r.handle = fakecuda::ras::address(handle);
        g_ras_blas_handles.erase(handle);
        delete static_cast<FakeBlasContext*>(handle);
        const auto result = fakecuda::ras::emit(r) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
        return result;
    }
    // 释放内存
    g_ras_blas_handles.erase(handle);
    delete static_cast<FakeBlasContext*>(handle);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_DGEMM, handle, transa, transb, m, n, k,
        A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb, C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
    LOG_DEBUG(CUBLAS, "cublasDgemm_v2() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t  cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_SGEMM_STRIDED_BATCHED, handle, transa, transb, m, n, k,
        A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT, batchCount, strideA, strideB, strideC);
    LOG_DEBUG(CUBLAS, "cublasSgemmStridedBatched() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    LOG_DEBUG(CUBLAS, "cublasDdot_v2() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_SGEMM, handle, transa, transb, m, n, k,
        A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    LOG_DEBUG(CUBLAS, "cublasSgemm_v2() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    LOG_DEBUG(CUBLAS, "cublasCdotu_v2() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_ZGEMM, handle, transa, transb, m, n, k,
        A, CUDA_C_64F, lda, B, CUDA_C_64F, ldb, C, CUDA_C_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode){
    if (!fake_valid_blas_handle(handle) || !mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = static_cast<FakeBlasContext*>(handle)->math_mode;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_CGEMM, handle, transa, transb, m, n, k,
        A, CUDA_C_32F, lda, B, CUDA_C_32F, ldb, C, CUDA_C_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_ZGEMM_STRIDED_BATCHED, handle, transa, transb, m, n, k,
        A, CUDA_C_64F, lda, B, CUDA_C_64F, ldb, C, CUDA_C_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT, batchCount, strideA, strideB, strideC);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx,  const cuDoubleComplex* beta, cuDoubleComplex* y, int incy){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode){
    if (!fake_valid_blas_handle(handle) || !mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = static_cast<FakeBlasContext*>(handle)->pointer_mode;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int* P, int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_CGEMM_STRIDED_BATCHED, handle, transa, transb, m, n, k,
        A, CUDA_C_32F, lda, B, CUDA_C_32F, ldb, C, CUDA_C_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT, batchCount, strideA, strideB, strideC);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int* P, int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_GEMM_EX, handle, transa, transb, m, n, k,
        A, Atype, lda, B, Btype, ldb, C, Ctype, ldc, computeType, algo);
    cudaStream_t stream = handle ? handle->stream : nullptr;
    fake_stream_mark_work_enqueued(stream);
    LOG_INFO(CUDART,
        "[cublasGemmEx] %c%c  M=%-7d N=%-7d K=%-7d"
        "  ldA=%-6d ldB=%-6d ldC=%-6d  A=%p B=%p C=%p  α=%.4f",
        transa==CUBLAS_OP_N?'N':'T', transb==CUBLAS_OP_N?'N':'T',
        m, n, k, lda, ldb, ldc, A, B, C,
        *reinterpret_cast<const float*>(alpha));
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes){
    if (!fake_valid_blas_handle(handle) || (!workspace && workspaceSizeInBytes)) {
        if (fakecuda::ras::active()) fakecuda_ras_fail("Invalid cuBLAS workspace");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    auto* ctx = static_cast<FakeBlasContext*>(handle);
    ctx->workspace = workspace;
    ctx->workspace_bytes = workspaceSizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_DGEMM_STRIDED_BATCHED, handle, transa, transb, m, n, k,
        A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb, C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT, batchCount, strideA, strideB, strideC);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const float* beta, void* C, cudaDataType Ctype, int ldc){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_SGEMM_EX, handle, transa, transb, m, n, k,
        A, Atype, lda, B, Btype, ldb, C, Ctype, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo){
    if (fakecuda::ras::active()) return direct_gemm(FAKECUDA_RAS_API_CUBLAS_GEMM_STRIDED_BATCHED_EX, handle, transa, transb, m, n, k,
        A, Atype, lda, B, Btype, ldb, C, Ctype, ldc, computeType, algo,
        batchCount, strideA, strideB, strideC);
    LOG_DEBUG(CUBLAS, "cublasGemmStridedBatchedEx() called.");
    cudaStream_t stream = handle ? handle->stream : nullptr;
    fake_stream_mark_work_enqueued(stream);
    LOG_INFO(CUDART,
        "[cublasGemmStridedBatchedEx] %c%c"
        "  M=%-7d N=%-7d K=%-7d  batch=%d"
        "  ldA=%-5d ldB=%-5d ldC=%-5d"
        "  strA=%-10lld strB=%-10lld strC=%-10lld"
        "  A=%p B=%p C=%p  α=%.4f",
        transa==CUBLAS_OP_N?'N':'T', transb==CUBLAS_OP_N?'N':'T',
        m, n, k, batchCount, lda, ldb, ldc, strideA, strideB, strideC, A, B, C,
        *reinterpret_cast<const float*>(alpha));
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const void* alpha,
    const void* const A[], cudaDataType a_type, int lda,
    const void* const B[], cudaDataType b_type, int ldb, const void* beta,
    void* const C[], cudaDataType c_type, int ldc, int batch_count,
    cublasComputeType_t compute, cublasGemmAlgo_t algorithm) {
    if (fakecuda::ras::active()) {
        // Device pointer arrays are borrowed opaque operands. The sink must
        // understand this flag; it must never dereference them as host arrays.
        return direct_gemm(FAKECUDA_RAS_API_CUBLAS_GEMM_BATCHED_EX, handle, transa, transb, m, n, k, A, a_type, lda,
            B, b_type, ldb, const_cast<void*>(static_cast<const void*>(C)),
            c_type, ldc, compute, algorithm, batch_count, 0, 0, 0, true);
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

API cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* stream) {
    if (!fake_valid_blas_handle(handle) || !stream) return CUBLAS_STATUS_INVALID_VALUE;
    *stream = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}
