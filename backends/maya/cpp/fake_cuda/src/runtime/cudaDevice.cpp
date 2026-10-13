#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "ras_capture.h"


API cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice){
    LOG_DEBUG(CUDART, "cudaDeviceCanAccessPeer() called.");
    if (!canAccessPeer) return cudaErrorInvalidValue;
    if (device < 0 || peerDevice < 0 || device >= fake_getDeviceCount() || peerDevice >= fake_getDeviceCount()) return cudaErrorInvalidDevice;
    *canAccessPeer = device != peerDevice;
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceCanAccessPeer);


API cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaDeviceEnablePeerAccess() called.");
    if(flags != 0){
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if(peerDevice < 0 || peerDevice >= fake_getDeviceCount()){
        fake_setLastError(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }
    // fprintf(stderr, "[fakecuda] Peer access to device %d enabled.\n", peerDevice);
    int current_device;
    fake_getDevice(&current_device);
    // fprintf(stderr, "[fakecuda] now device %d .\n",current_device);
    return cudaSuccess;

}
REGISTER_CUDA_FUNCTION(cudaDeviceEnablePeerAccess);

API cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device){
    LOG_DEBUG(CUDART, "cudaDeviceGetAttribute() called.");
    if (!value) return cudaErrorInvalidValue;
    if (device < 0 || device >= fake_getDeviceCount()) return cudaErrorInvalidDevice;
    *value = fake_getDevProps(static_cast<CUdevice_attribute>(attr), device);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceGetAttribute);

API cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaDeviceGetDefaultMemPool() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceGetDefaultMemPool);

API cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device){
    LOG_DEBUG(CUDART, "cudaDeviceGetPCIBusId() called.");
    if (!pciBusId || len < 13) return cudaErrorInvalidValue;
    if (device < 0 || device >= fake_getDeviceCount()) return cudaErrorInvalidDevice;
    std::snprintf(pciBusId, static_cast<size_t>(len), "%04x:%02x:%02x.0",
        fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device),
        fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device),
        fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device));
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceGetPCIBusId);

API cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority){
    LOG_DEBUG(CUDART, "cudaDeviceGetStreamPriorityRange() called.");
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = -1;
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceGetStreamPriorityRange);

API cudaError_t cudaDeviceSynchronize(void){
    LOG_DEBUG(CUDART, "cudaDeviceSynchronize() called.");
    // Complete all outstanding virtual stream work before returning.
    fake_mark_all_streams_completed();
    fake_mark_ready_events_visible();
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_DEVICE_SYNC);

        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        fake_setLastError(result);
        return result;
    }
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceSynchronize);
