#pragma once

#include <cstdlib>
#include <unordered_set>
#include <new>
#include "fake_types.h"

// The direct-capture ABI serializes intercepted calls externally. Keep native
// liveness separate from semantic identities owned by the RAS sink.
inline std::unordered_set<cublasHandle_t> g_ras_blas_handles;
inline bool fake_valid_blas_handle(cublasHandle_t handle) {
    return handle && g_ras_blas_handles.find(handle) != g_ras_blas_handles.end();
}

// API 可见性
#define API extern "C" __attribute__((visibility("default")))


struct FakeBlasContext : cublasContext {
    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
};

struct FakeLayout {
    cudaDataType type = CUDA_R_32F;
    uint64_t rows = 0, cols = 0;
    int64_t  ld   = 0;
    int32_t  batch  = 1;
    int64_t  stride = 0;
};
struct FakeDesc {
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cudaDataType_t    scaleType = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    uint32_t epilogue = 1; // CUBLASLT_EPILOGUE_DEFAULT
    int32_t pointer_mode = 0;
    const void* bias = nullptr;
    cudaDataType bias_type = CUDA_R_32F;
};
