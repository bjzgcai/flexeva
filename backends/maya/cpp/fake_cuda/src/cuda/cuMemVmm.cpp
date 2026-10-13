#include "utils.hpp"
#include "function_registry.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>

namespace {

std::atomic<std::uint64_t> g_fake_vmm_handle_counter{1};

const char* cu_error_string(CUresult error) {
    switch (error) {
        case CUDA_SUCCESS:
            return "CUDA_SUCCESS";
        case CUDA_ERROR_INVALID_VALUE:
            return "CUDA_ERROR_INVALID_VALUE";
        case CUDA_ERROR_OUT_OF_MEMORY:
            return "CUDA_ERROR_OUT_OF_MEMORY";
        case CUDA_ERROR_NOT_INITIALIZED:
            return "CUDA_ERROR_NOT_INITIALIZED";
        case CUDA_ERROR_INVALID_DEVICE:
            return "CUDA_ERROR_INVALID_DEVICE";
        case CUDA_ERROR_INVALID_HANDLE:
            return "CUDA_ERROR_INVALID_HANDLE";
        case CUDA_ERROR_NOT_PERMITTED:
            return "CUDA_ERROR_NOT_PERMITTED";
        case CUDA_ERROR_NOT_SUPPORTED:
            return "CUDA_ERROR_NOT_SUPPORTED";
        case CUDA_ERROR_UNKNOWN:
            return "CUDA_ERROR_UNKNOWN";
        default:
            return "CUDA_ERROR_UNKNOWN";
    }
}

}  // namespace

API CUresult cuMemAddressReserve(
    std::uint64_t* ptr,
    size_t size,
    size_t alignment,
    std::uint64_t addr,
    unsigned long long flags) {
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUDA_ERROR_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemAddressReserve() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (ptr == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *ptr = 0;
        if (size == 0) {
            return CUDA_SUCCESS;
        }
        (void)alignment;
        (void)addr;
        (void)flags;
        void* reserved = mmap(
            nullptr,
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
            -1,
            0);
        if (reserved == MAP_FAILED) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        *ptr = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(reserved));
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemAddressReserve: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemAddressReserve);

API CUresult cuMemAddressFree(std::uint64_t ptr, size_t size) {
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUDA_ERROR_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemAddressFree() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (ptr == 0 || size == 0) {
            return CUDA_SUCCESS;
        }
        void* address = reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr));
        if (munmap(address, size) != 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemAddressFree: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemAddressFree);

API CUresult cuMemCreate(
    std::uint64_t* handle,
    size_t size,
    const void* prop,
    unsigned long long flags) {
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUDA_ERROR_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemCreate() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (handle == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        (void)size;
        (void)prop;
        (void)flags;
        *handle = g_fake_vmm_handle_counter.fetch_add(1, std::memory_order_relaxed);
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemCreate: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemCreate);

API CUresult cuMemRelease(std::uint64_t handle) {
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUDA_ERROR_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemRelease() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        (void)handle;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemRelease: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemRelease);

API CUresult cuMemMap(
    std::uint64_t ptr,
    size_t size,
    size_t offset,
    std::uint64_t handle,
    unsigned long long flags) {
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUDA_ERROR_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemMap() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        (void)size;
        (void)offset;
        (void)handle;
        (void)flags;
        if (ptr == 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemMap: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemMap);

API CUresult cuMemUnmap(std::uint64_t ptr, size_t size) {
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUDA_ERROR_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemUnmap() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        (void)size;
        if (ptr == 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemUnmap: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemUnmap);

API CUresult cuMemSetAccess(
    std::uint64_t ptr,
    size_t size,
    const void* desc,
    size_t count) {
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return CUDA_ERROR_NOT_SUPPORTED; }
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemSetAccess() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        (void)size;
        (void)desc;
        (void)count;
        if (ptr == 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemSetAccess: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemSetAccess);

API CUresult cuMemGetAllocationGranularity(
    size_t* granularity,
    const void* prop,
    int option) {
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemGetAllocationGranularity() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (granularity == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        (void)prop;
        (void)option;
        *granularity = 64 * 1024;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemGetAllocationGranularity: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemGetAllocationGranularity);

API CUresult cuMemExportToShareableHandle(
    void* shareableHandle,
    std::uint64_t handle,
    int handleType,
    unsigned long long flags) {
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemExportToShareableHandle() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        (void)handle;
        (void)handleType;
        (void)flags;
        if (shareableHandle == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return CUDA_ERROR_NOT_SUPPORTED;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemExportToShareableHandle: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemExportToShareableHandle);

API CUresult cuMemImportFromShareableHandle(
    std::uint64_t* handle,
    void* osHandle,
    int shHandleType) {
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemImportFromShareableHandle() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        (void)osHandle;
        (void)shHandleType;
        if (handle == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *handle = 0;
        return CUDA_ERROR_NOT_SUPPORTED;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemImportFromShareableHandle: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemImportFromShareableHandle);

API CUresult cuMemsetD32Async(
    std::uint64_t dstDevice,
    unsigned int value,
    size_t count,
    cudaStream_t stream) {
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuMemsetD32Async() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        (void)stream;
        if (dstDevice == 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        std::uint32_t* dst =
            reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(dstDevice));
        for (size_t i = 0; i < count; ++i) {
            dst[i] = value;
        }
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuMemsetD32Async: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuMemsetD32Async);

API CUresult cuStreamWriteValue32(
    cudaStream_t stream,
    std::uint64_t addr,
    std::uint32_t value,
    unsigned int flags) {
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuStreamWriteValue32() called.");
        if (!fake_isInitialized()) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        (void)stream;
        (void)flags;
        if (addr == 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        auto* dst = reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(addr));
        *dst = value;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuStreamWriteValue32: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuStreamWriteValue32);

API CUresult cuGetErrorString(CUresult error, const char** pStr) {
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuGetErrorString() called.");
        if (pStr == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *pStr = cu_error_string(error);
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuGetErrorString: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuGetErrorString);
