#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "ras_capture.h"


API cudaError_t cudaMemGetInfo ( size_t* free, size_t* total ){
    LOG_DEBUG(CUDART, "cudaMemGetInfo() called.");
    if(free != nullptr && total != nullptr){
        *total = fake_getDeviceTotalMem();
        *free = fake_getDeviceFreeMem();
        fake_setLastError(cudaSuccess);
        return cudaSuccess;
    }else{
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
}

API cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind){
    if (fakecuda::ras::active()) {
        if ((count && (!src || !dst)) || kind < cudaMemcpyHostToHost || kind > cudaMemcpyDeviceToDevice) {
            fakecuda_ras_fail("Unsupported or invalid cudaMemcpy arguments");
            return cudaErrorInvalidValue;
        }
        fake_stream_mark_work_enqueued(nullptr);
        if (count) std::memmove(dst, src, count);
        const bool host_visible = kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyHostToHost;
        if (host_visible)
            fake_stream_mark_completed_through(nullptr, fake_stream_enqueued_seq(nullptr));
        auto r = fakecuda::ras::record(FAKECUDA_RAS_MEMCPY_ASYNC);
        r.src = fakecuda::ras::address(src); r.dst = fakecuda::ras::address(dst);
        r.bytes = count; r.copy_kind = static_cast<int>(kind); r.flags = 1;
        if (!fakecuda::ras::emit(r)) return cudaErrorUnknown;
        if (host_visible) {
            auto sync = fakecuda::ras::record(FAKECUDA_RAS_STREAM_SYNC);
            if (!fakecuda::ras::emit(sync)) return cudaErrorUnknown;
        }
        fake_setLastError(cudaSuccess);
        return cudaSuccess;
    }
    LOG_DEBUG(CUDART, "cudaMemcpy() called.");
    if (dst == nullptr || src == nullptr || count == 0) {
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if (count <= METADATA_THRESHOLD) {
        memcpy(dst, src, count);
    }
    // if (kind == cudaMemcpyDeviceToHost) {
    //     memset(dst, 0, count);
    // }
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemcpy);

API cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaMemcpy2DAsync() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemcpy2DAsync);

API cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)){
    if (fakecuda::ras::active() && ((count && (!src || !dst)) || (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDeviceToDevice) || !fake_valid_stream(stream))) {
        fakecuda_ras_fail("Invalid cudaMemcpyAsync arguments");
        return cudaErrorInvalidValue;
    }
    LOG_DEBUG(CUDART, "cudaMemcpyAsync() called.");
    if (fakecuda::ras::active()) {
        fake_stream_mark_work_enqueued(stream);
        if (count) std::memmove(dst, src, count);
        auto r = fakecuda::ras::record(FAKECUDA_RAS_MEMCPY_ASYNC);
        r.src = fakecuda::ras::address(src);
        r.dst = fakecuda::ras::address(dst);
        r.bytes = count;
        r.copy_kind = static_cast<int>(kind);
        r.stream = fakecuda::ras::address(stream);
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        fake_setLastError(result);
        return result;
    }
    fake_stream_mark_work_enqueued(stream);
    memcpy(dst, src, count);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemcpyAsync);

API cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream __dv(0)){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaMemcpyPeerAsync() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemcpyPeerAsync);

API cudaError_t cudaMemset(void *devPtr, int value, size_t count){
    if (fakecuda::ras::active()) {
        if (count && !devPtr) { fakecuda_ras_fail("Invalid cudaMemset arguments"); return cudaErrorInvalidValue; }
        fake_stream_mark_work_enqueued(nullptr);
        if (count) std::memset(devPtr, value, count);
        auto r = fakecuda::ras::record(FAKECUDA_RAS_MEMSET_ASYNC);
        r.dst = fakecuda::ras::address(devPtr); r.bytes = count; r.value = value; r.flags = 1;
        if (!fakecuda::ras::emit(r)) return cudaErrorUnknown;
        fake_setLastError(cudaSuccess);
        return cudaSuccess;
    }
    LOG_DEBUG(CUDART, "cudaMemset() called.");
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemset);

API cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0)){
    if (fakecuda::ras::active() && ((count && !devPtr) || !fake_valid_stream(stream))) {
        fakecuda_ras_fail("Invalid cudaMemsetAsync arguments");
        return cudaErrorInvalidValue;
    }
    LOG_DEBUG(CUDART, "cudaMemsetAsync() called.");
    if (fakecuda::ras::active()) {
        fake_stream_mark_work_enqueued(stream);
        if (count) std::memset(devPtr, value, count);
        auto r = fakecuda::ras::record(FAKECUDA_RAS_MEMSET_ASYNC);
        r.dst = fakecuda::ras::address(devPtr);
        r.bytes = count;
        r.value = value;
        r.stream = fakecuda::ras::address(stream);
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        fake_setLastError(result);
        return result;
    }
    fake_stream_mark_work_enqueued(stream);
    if (devPtr && count > 0) {
        memset(devPtr, value, count);
    }
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemsetAsync);

API cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value ){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaMemPoolGetAttribute called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemPoolGetAttribute);

API cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaMemPoolSetAccess called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemPoolSetAccess);

API cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value ){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaMemPoolSetAttribute called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemPoolSetAttribute);

API cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaMemPoolTrimTo called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemPoolTrimTo);
