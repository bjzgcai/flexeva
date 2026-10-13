#include "fake_device_core.h"
#include "device_manager.hpp"
#include "fake_types.h"
#include "capture_scope.h"
#include "ras_capture.h"

DeviceManager& GPU_Devices = DeviceManager::getInstance();

extern "C" {
    int fakecuda_ras_allocation_range(uint64_t address, uint64_t* base, uint64_t* bytes) {
        if (!base || !bytes) return 0;
        return GPU_Devices.allocationRange(address, *base, *bytes) ? 1 : 0;
    }
    bool fake_isInitialized(){
        return GPU_Devices.isInitialized();
    }
    int fake_getDriverVersion(){
        return GPU_Devices.getDriverVersion();
    }
    int fake_getDeviceCount(){
        return GPU_Devices.getDeviceCount();
    }
    std::string fake_getDeviceName(int device_id){
        return GPU_Devices.getDeviceName(device_id);
    }
    size_t fake_getDeviceTotalMem(int device_id){
        return GPU_Devices.getDeviceTotalMem(device_id);
    }
    size_t fake_getDeviceFreeMem(int device_id){
        return GPU_Devices.getDeviceFreeMem(device_id);
    }
    int fake_getDevProps(CUdevice_attribute attrib, int dev){
        return GPU_Devices.getDevProps(attrib, dev);
    }
    bool fake_initialize(){
        return GPU_Devices.initialize();
    }
    cudaUUID_t fake_getDeviceUUID(int dev){
        return GPU_Devices.getDeviceUUID(dev);
    }
    void fake_setDevice(int device_id){
        GPU_Devices.setDevice(device_id);
    }
    void fake_getDevice(int* device_id){
        GPU_Devices.getDevice(device_id);
    }
    cudaError_t fake_getLastError(){
        return GPU_Devices.getLastError();
    }
    void fake_setLastError(cudaError_t error){
        GPU_Devices.setLastError(error);
    }
    CUcontext fake_getCurrentContext(){
        return GPU_Devices.getCurrentContext();
    }
    CUcontext fake_getContextForDevice(int dev_id){
        return GPU_Devices.getContextForDevice(dev_id);
    }


    cudaError_t fake_allocateMemory(void** devPtr, size_t size){
        if(!fake_isInitialized()){
            fake_initialize();
        }
        const bool legacy_tracked = !fakecuda::ras::active();
        const auto result = GPU_Devices.allocateMemory(devPtr, size, legacy_tracked);
        if (legacy_tracked && result == cudaSuccess && devPtr != nullptr && *devPtr != nullptr)
            fakecuda::capture::resource_created(FAKECUDA_ALLOCATION, reinterpret_cast<uintptr_t>(*devPtr), size);
        return result;
    }
    cudaError_t fake_freeMemory(void* devPtr){
        bool legacy_tracked = false;
        const auto result = GPU_Devices.freeMemory(devPtr, legacy_tracked);
        // Native lifetime can outlive the capture that created the allocation
        // (for example, a Python exception traceback retaining a Tensor).
        // Retire only the registry that owns this allocation's generation.
        if (result == cudaSuccess && devPtr != nullptr && legacy_tracked)
            fakecuda::capture::retire_legacy_resource(FAKECUDA_ALLOCATION, reinterpret_cast<uintptr_t>(devPtr));
        return result;
    }
}
