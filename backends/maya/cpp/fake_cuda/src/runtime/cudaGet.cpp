#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "trace_log.hpp"

API cudaError_t cudaGetDeviceCount(int *count){
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaGetDeviceCount() called.");
    if(!fake_isInitialized()){
        fake_initialize();
        LOG_DEBUG(CUDART, "fake cuda initialized.");
    }
    if(count != nullptr){
        *count = fake_getDeviceCount();
        if (fakecuda::ras::active()) { fake_setLastError(cudaSuccess); return cudaSuccess; }
        auto payload = make_trace_payload();
        if (payload) payload->add_int("count", *count);
        TRACE_API_EX(CUDART, "cudaGetDeviceCount", "context_op", payload);
        fake_setLastError(cudaSuccess);
        return cudaSuccess;
    }
    else{
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    fake_setLastError(cudaErrorUnknown);
    return cudaErrorUnknown;
}
REGISTER_CUDA_FUNCTION(cudaGetDeviceCount);

API cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device ){
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaGetDeviceProperties() called.");
    if(prop == nullptr){
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if(!fake_isInitialized()){
        fake_initialize();
    }
    if(device < 0 || device >= fake_getDeviceCount()){
        fprintf(stderr, "[fakecuda] cudaGetDeviceProperties() device count %d\n", fake_getDeviceCount());
        fprintf(stderr, "[fakecuda] cudaGetDeviceProperties() invalid device %d\n", device);
        fake_setLastError(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    std::memset(prop, 0, sizeof(*prop));
    std::string deviceName = fake_getDeviceName(device);
    size_t copyLen = std::min(deviceName.size(), sizeof(prop->name) - 1);
    deviceName.copy(prop->name, copyLen);
    prop->name[copyLen] = '\0';  // 手动添加null终止符
    prop->uuid = fake_getDeviceUUID(device);
    prop->totalGlobalMem = fake_getDeviceTotalMem(device);
    prop->sharedMemPerBlock = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);
    prop->warpSize = fake_getDevProps(CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    prop->memPitch = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_PITCH, device);
    prop->maxThreadsPerBlock = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    for(int i = 0; i < 3; i++){
        prop->maxThreadsDim[i] = fake_getDevProps(static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X + i), device);
        prop->maxGridSize[i] = fake_getDevProps(static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i), device);
    }
    prop->clockRate = fake_getDevProps(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
    prop->totalConstMem = fake_getDevProps(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);
    prop->major = fake_getDevProps(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    prop->minor = fake_getDevProps(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    prop->textureAlignment = fake_getDevProps(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device);
    prop->texturePitchAlignment = fake_getDevProps(CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, device);
    prop->multiProcessorCount = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    prop->canMapHostMemory = fake_getDevProps(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device);
    for(int i = 0; i < 2; i++){
        prop->maxTexture2D[i] = fake_getDevProps(static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH + i), device);
        prop->maxSurface2D[i] = fake_getDevProps(static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH + i), device);
    }
    prop->surfaceAlignment = fake_getDevProps(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, device);
    prop->concurrentKernels = fake_getDevProps(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device);
    prop->ECCEnabled = fake_getDevProps(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device);
    prop->pciBusID = fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device);
    prop->pciDeviceID = fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
    prop->pciDomainID = fake_getDevProps(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);
    prop->tccDriver = fake_getDevProps(CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device);
    prop->asyncEngineCount = fake_getDevProps(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device);
    prop->unifiedAddressing = fake_getDevProps(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device);
    prop->memoryClockRate = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
    prop->memoryBusWidth = fake_getDevProps(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
    prop->l2CacheSize = fake_getDevProps(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device);
    prop->persistingL2CacheMaxSize = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, device);
    prop->maxThreadsPerMultiProcessor = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
    prop->streamPrioritiesSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, device);
    prop->globalL1CacheSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, device);
    prop->localL1CacheSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, device);
    prop->sharedMemPerMultiprocessor = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device);
    prop->regsPerMultiprocessor = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device);
    prop->managedMemory = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device);
    prop->hostNativeAtomicSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device);
    prop->singleToDoublePrecisionPerfRatio = fake_getDevProps(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, device);
    prop->pageableMemoryAccess = fake_getDevProps(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device);
    prop->concurrentManagedAccess = fake_getDevProps(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device);
    prop->computePreemptionSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, device);
    prop->canUseHostPointerForRegisteredMem = fake_getDevProps(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, device);
    prop->cooperativeLaunch = fake_getDevProps(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device);
    prop->cooperativeMultiDeviceLaunch = fake_getDevProps(CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, device);
    prop->sharedMemPerBlockOptin = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);
    prop->pageableMemoryAccessUsesHostPageTables = fake_getDevProps(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device);
    prop->directManagedMemAccessFromHost = fake_getDevProps(CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device);
    prop->maxBlocksPerMultiProcessor = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, device);
    prop->accessPolicyMaxWindowSize = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, device);
    prop->hostRegisterSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, device);
    prop->sparseCudaArraySupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, device);
    prop->hostRegisterReadOnlySupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED, device);
    prop->timelineSemaphoreInteropSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED, device);
    prop->memoryPoolsSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device);
    prop->gpuDirectRDMAFlushWritesOptions = fake_getDevProps(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, device);
    prop->gpuDirectRDMAWritesOrdering = fake_getDevProps(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, device);
    prop->memoryPoolSupportedHandleTypes = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES, device);
    prop->deferredMappingCudaArraySupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED, device);
    prop->ipcEventSupported = fake_getDevProps(CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, device);
    prop->clusterLaunch = fake_getDevProps(CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, device);
    prop->unifiedFunctionPointers = fake_getDevProps(CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS, device);
    if (fakecuda::ras::active()) { fake_setLastError(cudaSuccess); return cudaSuccess; }
    auto payload = make_trace_payload();
    if (payload) payload->add_int("device", device);
    if (payload) payload->add_string("name", prop->name);
    TRACE_API_EX(CUDART, "cudaGetDeviceProperties", "context_op", payload);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGetDeviceProperties);
API cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device ){
    fakecuda::capture::WrapperCall capture_call;
    return cudaGetDeviceProperties(prop, device);
}
REGISTER_CUDA_FUNCTION(cudaGetDeviceProperties_v2);

API cudaError_t cudaGetDevice (int* device){
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG_NOENTRY(CUDART, "cudaGetDevice() called.");
    if (!fakecuda::ras::active()) fakecuda::trace::maybe_record_wrapper_entry_api("cudaGetDevice");
    if(device == nullptr){
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    fake_getDevice(device);
    if (fakecuda::ras::active()) { fake_setLastError(cudaSuccess); return cudaSuccess; }
    // fprintf(stderr, "[fakecuda] cudaGetDevice() called, device is %d\n", *device);
    auto payload = make_trace_payload();
    if (payload) payload->add_int("device", *device);
    TRACE_API_LIGHT_EX(CUDART, "cudaGetDevice", "context_op", payload);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGetDevice);

API const char *cudaGetErrorName(cudaError_t error){
    fakecuda::capture::WrapperCall capture_call;
    LOG_DEBUG(CUDART, "cudaGetErrorName() called.");
    return "cudaSuccess";
}
REGISTER_CUDA_FUNCTION(cudaGetErrorName);

API const char *cudaGetErrorString(cudaError_t error){
    fakecuda::capture::WrapperCall capture_call;
    LOG_DEBUG(CUDART, "cudaGetErrorString() called.");
    static char buf[64];
    snprintf(buf, sizeof(buf), "cudaError_%d", error);
    return buf;
}
REGISTER_CUDA_FUNCTION(cudaGetErrorString);

API cudaError_t cudaGetLastError (){
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG_NOENTRY(CUDART, "cudaGetLastError() called.");
    cudaError_t last = fake_getLastError();
    if (fakecuda::ras::active()) { fake_setLastError(cudaSuccess); return last; }
    auto payload = make_trace_payload();
    if (payload) payload->add_int("error", static_cast<int>(last));
    TRACE_API_LIGHT_EX(CUDART, "cudaGetLastError", "other", payload);
    fake_setLastError(cudaSuccess); // Reset last error after reporting it.
    return last;
}
REGISTER_CUDA_FUNCTION(cudaGetLastError);

API cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, enum cudaDriverEntryPointQueryResult *driverStatus){
    fakecuda::capture::WrapperCall capture_call;
    (void)cudaVersion; (void)flags;
    if (!symbol || !funcPtr) return cudaErrorInvalidValue;
    *funcPtr = FunctionRegistry::instance().get_function(symbol);
    if (driverStatus) *driverStatus = *funcPtr ? cudaDriverEntryPointSuccess : cudaDriverEntryPointSymbolNotFound;
    const auto result = *funcPtr ? cudaSuccess : cudaErrorInvalidSymbol;
    fake_setLastError(result);
    return result;
}
REGISTER_CUDA_FUNCTION(cudaGetDriverEntryPointByVersion);

API cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, enum cudaDriverEntryPointQueryResult *driverStatus = NULL){
    return cudaGetDriverEntryPointByVersion(symbol, funcPtr, static_cast<unsigned int>(fake_getDriverVersion()), flags, driverStatus);
}
REGISTER_CUDA_FUNCTION(cudaGetDriverEntryPoint);
