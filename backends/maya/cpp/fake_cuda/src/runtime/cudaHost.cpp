#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"


API cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags){
    fakecuda::capture::WrapperCall capture_call;
    LOG_DEBUG(CUDART, "cudaHostAlloc() called.");
    if (!pHost || !size || (flags & ~7U)) return cudaErrorInvalidValue;
    void *ptr = mmap(nullptr, size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                     -1, 0);
    if (ptr == MAP_FAILED) {
        return cudaErrorMemoryAllocation;
    }
    {
        std::lock_guard<std::mutex> lock(g_host_mutex);
        g_host_allocations[ptr] = size;
    }
    *pHost = ptr;
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaHostAlloc);

namespace {
std::map<void*, size_t> host_registrations;
}

API cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags){
    fakecuda::capture::WrapperCall capture_call;
    LOG_DEBUG(CUDART, "cudaHostRegister() called.");
    if (!ptr || !size || (flags & ~15U)) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g_host_mutex);
    if (!host_registrations.emplace(ptr, size).second) return cudaErrorHostMemoryAlreadyRegistered;
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaHostRegister);

API cudaError_t cudaHostUnregister(void *ptr){
    fakecuda::capture::WrapperCall capture_call;
    LOG_DEBUG(CUDART, "cudaHostUnregister() called.");
    std::lock_guard<std::mutex> lock(g_host_mutex);
    if (!host_registrations.erase(ptr)) return cudaErrorHostMemoryNotRegistered;
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaHostUnregister);

API cudaError_t cudaFreeHost(void *ptr){
    fakecuda::capture::WrapperCall capture_call;
    LOG_DEBUG(CUDART, "cudaFreeHost() called.");
    fake_setLastError(cudaSuccess);
    {
        std::lock_guard<std::mutex> lock(g_host_mutex);
        auto it = g_host_allocations.find(ptr);
        if (it == g_host_allocations.end()) {
            return cudaErrorInvalidValue;
        }
        size_t size = it->second;
        munmap(ptr, size);
        g_host_allocations.erase(it);
    }
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaFreeHost);
