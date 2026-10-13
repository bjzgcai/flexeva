#include "utils.hpp"
#include "function_registry.hpp"
#include "fake_device_core.h"


API CUresult cuDeviceGetName(char *name, int len, CUdevice dev){
    try{
        LOG_DEBUG(CUDA, "cuDeviceGetName() called.");
        if (!fake_isInitialized()) return CUDA_ERROR_NOT_INITIALIZED;
        int count = fake_getDeviceCount();
        if (name == nullptr || len <= 0 || dev <0 || dev >= count) return CUDA_ERROR_INVALID_VALUE;
        std::string device_name = fake_getDeviceName(dev);
        if (device_name.empty()) return CUDA_ERROR_INVALID_VALUE;
        snprintf(name, len, "%s", device_name.c_str());
        return CUDA_SUCCESS;
    }catch (const std::exception& e) {
        fprintf(stderr, "Error in cuDeviceGetName: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuDeviceGetName);

API CUresult cuDeviceGetCount(int *count){
    try {
        LOG_DEBUG(CUDA, "cuDeviceGetCount() called.");
        if (!fake_isInitialized()) return CUDA_ERROR_NOT_INITIALIZED;
        if(count != nullptr)
            *count = fake_getDeviceCount();
        else return CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuDeviceGetCount: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuDeviceGetCount);

API CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
    try {
        LOG_DEBUG(CUDA, "cuDeviceGet() called.");
        if (!fake_isInitialized()) return CUDA_ERROR_NOT_INITIALIZED;
        if (!device)
            return CUDA_ERROR_INVALID_VALUE;
        if (ordinal < 0 || ordinal >= fake_getDeviceCount()) return CUDA_ERROR_INVALID_DEVICE;
        *device = ordinal;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuDeviceGet: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuDeviceGet);

API CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev){
    try {
        LOG_DEBUG(CUDA, "cuDeviceGetAttribute() called.");
        if (!fake_isInitialized()) return CUDA_ERROR_NOT_INITIALIZED;
        if (pi == nullptr) return CUDA_ERROR_INVALID_VALUE;
        if (dev < 0 || dev >= fake_getDeviceCount()) return CUDA_ERROR_INVALID_DEVICE;
        *pi = fake_getDevProps(attrib, dev);
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuDeviceGetAttribute: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuDeviceGetAttribute);

API CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
    try {
        LOG_DEBUG(CUDA, "cuDeviceTotalMem() called.");
        if (!fake_isInitialized()) return CUDA_ERROR_NOT_INITIALIZED;
        if (bytes == nullptr) return CUDA_ERROR_INVALID_VALUE;
        if (dev < 0 || dev >= fake_getDeviceCount()) return CUDA_ERROR_INVALID_DEVICE;
        *bytes = fake_getDeviceTotalMem(dev);
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuDeviceTotalMem: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuDeviceTotalMem);

API CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active){
    try {
        LOG_DEBUG(CUDA, "cuDevicePrimaryCtxGetState() called.");
        if (!fake_isInitialized()) return CUDA_ERROR_NOT_INITIALIZED;
        if (dev < 0 || dev >= fake_getDeviceCount())
            return CUDA_ERROR_INVALID_DEVICE;
        if (flags == nullptr || active == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        int currentDevice;
        fake_getDevice(&currentDevice);
        if (dev == currentDevice)
            *active = 1;  // 活跃
        else
            *active = 0;  // 不活跃
        // 设置默认标志（调度策略）
        *flags = CU_CTX_SCHED_AUTO;  // 0x01，默认自动调度
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuDeviceTotalMem: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuDevicePrimaryCtxGetState);
