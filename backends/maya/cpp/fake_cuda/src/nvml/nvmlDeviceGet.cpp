#include "utils.hpp"
#include "fake_types.h"
#include "fake_device_core.h"

namespace {

void* make_nvml_device_handle(int dev) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(dev + 1));
}

std::string normalize_bus_id(const char* raw) {
    if (raw == nullptr) {
        return "";
    }
    std::string normalized(raw);
    for (char& ch : normalized) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return normalized;
}

std::string fake_nvml_bus_id(int dev) {
    char buffer[32];
    const int domain = fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev);
    const int bus = fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
    const int device = fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
    std::snprintf(buffer, sizeof(buffer), "%08x:%02x:%02x.0", domain, bus, device);
    return std::string(buffer);
}

}  // namespace

API nvmlReturn_t nvmlInit_v2() {
    fprintf(stderr, "[fakenvml] nvmlInit_v2() called.\n");
    if (!fake_isInitialized()) {
        fake_initialize();
    }
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount){
    fprintf(stderr, "[fakenvml] nvmlDeviceGetCount_v2() called.\n");
    if(!fake_isInitialized()){
        fake_initialize();
        fprintf(stderr, "[fakenvml] fake nvml initialized.\n");
    }
    if(deviceCount != nullptr){
        *deviceCount = fake_getDeviceCount();
        return NVML_SUCCESS;
    }
    else{
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return NVML_ERROR_UNKNOWN;
}
API nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount){
    return nvmlDeviceGetCount_v2(deviceCount);
}

API nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion) {
    fprintf(stderr, "[fakenvml] nvmlSystemGetCudaDriverVersion_v2() called.\n");
    if (!fake_isInitialized()) {
        fake_initialize();
    }
    if (cudaDriverVersion == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *cudaDriverVersion = fake_getDriverVersion();
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(
    const char* pciBusId,
    void** device) {
    fprintf(stderr, "[fakenvml] nvmlDeviceGetHandleByPciBusId_v2() called.\n");
    if (!fake_isInitialized()) {
        fake_initialize();
    }
    if (pciBusId == nullptr || device == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    const std::string requested = normalize_bus_id(pciBusId);
    const int count = fake_getDeviceCount();
    for (int dev = 0; dev < count; ++dev) {
        if (fake_nvml_bus_id(dev) == requested) {
            *device = make_nvml_device_handle(dev);
            return NVML_SUCCESS;
        }
    }
    if (count > 0) {
        *device = make_nvml_device_handle(0);
        return NVML_SUCCESS;
    }
    return NVML_ERROR_NOT_FOUND;
}

API nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(
    void* device,
    unsigned int link,
    int* nvLinkDeviceType) {
    fprintf(stderr, "[fakenvml] nvmlDeviceGetNvLinkRemoteDeviceType() called.\n");
    (void)device;
    (void)link;
    if (nvLinkDeviceType == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *nvLinkDeviceType = 0;
    return NVML_ERROR_NOT_SUPPORTED;
}

API nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(
    void* device,
    unsigned int link,
    void* pciInfo) {
    fprintf(stderr, "[fakenvml] nvmlDeviceGetNvLinkRemotePciInfo_v2() called.\n");
    (void)device;
    (void)link;
    (void)pciInfo;
    return NVML_ERROR_NOT_SUPPORTED;
}

API nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(
    void* device,
    unsigned int* infoCount,
    void* infos) {
    fprintf(stderr, "[fakenvml] nvmlDeviceGetComputeRunningProcesses() called.\n");
    (void)device;
    (void)infos;
    if (infoCount == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *infoCount = 0;
    return NVML_SUCCESS;
}
