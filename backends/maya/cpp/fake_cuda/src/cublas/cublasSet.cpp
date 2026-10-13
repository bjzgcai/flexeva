#include "fake_types.h"
#include "fake_runtime_api.hpp"
#include "cublas.hpp"
#include "utils.hpp"
#include "../../include/common/trace_log.hpp"

namespace {
std::uint64_t opaque_id(const void* value) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
}
}


API cublasStatus_t  cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId){
    if (!fake_valid_blas_handle(handle) || !fake_valid_stream(streamId)) {
        fakecuda_ras_fail("Invalid cublasSetStream_v2 arguments");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUBLAS, "cublasSetStream_v2() called.");
    if (handle != nullptr) {
        handle->stream = streamId;
    }
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_BLAS_SET_STREAM);
        r.handle = fakecuda::ras::address(handle);
        r.stream = fakecuda::ras::address(streamId);
        const auto result = fakecuda::ras::emit(r) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
        return result;
    }
    auto payload = make_trace_payload();
    if (payload) payload->add_uint64("handle_id", opaque_id(handle));
    if (payload) payload->add_uint64("stream_id", trace_stream_id(streamId));
    TRACE_API_EX(CUBLAS, "cublasSetStream_v2", "stream_op", payload);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode){
    fakecuda::capture::WrapperCall capture_call;
    const int value = static_cast<int>(mode);
    if (!fake_valid_blas_handle(handle) || (value & ~19) != 0) {
        if (fakecuda::ras::active()) fakecuda_ras_fail("Invalid cuBLAS math mode");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    static_cast<FakeBlasContext*>(handle)->math_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode){
    fakecuda::capture::WrapperCall capture_call;
    if (!fake_valid_blas_handle(handle) || (mode != CUBLAS_POINTER_MODE_HOST && mode != CUBLAS_POINTER_MODE_DEVICE)) {
        if (fakecuda::ras::active()) fakecuda_ras_fail("Invalid cuBLAS pointer mode");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    static_cast<FakeBlasContext*>(handle)->pointer_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUBLAS_STATUS_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    return CUBLAS_STATUS_SUCCESS;
}
