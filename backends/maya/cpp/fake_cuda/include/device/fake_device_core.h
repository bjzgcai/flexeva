#pragma once
#include <string>
#include "device_manager.hpp"

extern "C" {
    // 线程安全的查询接口
    bool fake_isInitialized();
    int fake_getDriverVersion();
    int fake_getDeviceCount();
    std::string fake_getDeviceName(int device_id);
    size_t fake_getDeviceTotalMem(int device_id = -1);
    size_t fake_getDeviceFreeMem(int device_id = -1);
    int fake_getDevProps(CUdevice_attribute attrib, int dev);
    cudaUUID_t fake_getDeviceUUID(int device_id);
    cudaError_t fake_getLastError();
    void fake_setLastError(cudaError_t error = cudaSuccess);
    CUcontext fake_getCurrentContext() ;
    CUcontext fake_getContextForDevice(int dev_id);

    // 设备管理接口
    bool fake_initialize();
    void fake_setDevice(int device_id);
    void fake_getDevice(int* device_id);
    cudaError_t fake_allocateMemory(void** devPtr, size_t size);
    cudaError_t fake_freeMemory(void* devPtr);
}