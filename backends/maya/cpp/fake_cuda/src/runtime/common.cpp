#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "ras_capture.h"

API cudaError_t cudaDriverGetVersion ( int* driverVersion ){
    LOG_DEBUG(CUDART, "cudaDriverGetVersion() called.");
    if(driverVersion != nullptr){
        *driverVersion = fake_getDriverVersion();
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
REGISTER_CUDA_FUNCTION(cudaDriverGetVersion);

API cudaError_t cudaMalloc(void** devPtr, size_t size ){
    if (fakecuda::ras::active() && !devPtr) {
        fakecuda_ras_fail("Invalid cudaMalloc arguments");
        return cudaErrorInvalidValue;
    }
    // fprintf(stderr, "[fakecuda] cudaMalloc(%p, %zu) called.\n", devPtr, size);
    LOG_DEBUG(CUDART, "cudaMalloc() called.");
    if(devPtr == nullptr){
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    cudaError_t res = fake_allocateMemory(devPtr, size);
    if(res != cudaSuccess){
        fake_setLastError(res);
        return res;
    }
    // A zero-byte CUDA allocation succeeds with a null pointer. It owns no
    // native resource and therefore creates no RAS allocation event.
    if (size == 0) { fake_setLastError(cudaSuccess); return cudaSuccess; }
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_ALLOC);
        r.dst = fakecuda::ras::address(*devPtr);
        r.bytes = size;
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        if (result != cudaSuccess) {
            fake_freeMemory(*devPtr);
            *devPtr = nullptr;
        }
        fake_setLastError(result);
        return result;
    }
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMalloc);

API cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaMallocAsync() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMallocAsync);

API cudaError_t cudaFree(void* devPtr){
    if (devPtr == nullptr) { fake_setLastError(cudaSuccess); return cudaSuccess; }
    // fprintf(stderr, "[fakecuda] cudaFree(%p) called.\n", devPtr);
    LOG_DEBUG(CUDART, "cudaFree() called.");
    if(devPtr == nullptr){
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_FREE);
        r.src = fakecuda::ras::address(devPtr);
        const auto freed = fake_freeMemory(devPtr);
        if (freed != cudaSuccess) { fakecuda_ras_fail("cudaFree failed"); return freed; }
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        fake_setLastError(result);
        return result;
    }
    cudaError_t res = fake_freeMemory(devPtr);
    if(res != cudaSuccess){
        fake_setLastError(res);
        return res;
    }
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaFree);

API cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaFreeAsync() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaFreeAsync);



API cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func){
    LOG_DEBUG(CUDART, "cudaFuncGetAttributes() called.");
    if (!attr || !func) return cudaErrorInvalidValue;
    memset(attr, 0, sizeof(*attr));
    auto* ki = KernelRegistry::getInstance().FindByHost(func, true);  // ← true
    if (!ki) {
        // 未注册的函数，返回安全默认值
        attr->maxThreadsPerBlock = 1024;
        attr->numRegs = 32;
        attr->ptxVersion = 70;
        attr->binaryVersion = 70;
        attr->maxDynamicSharedSizeBytes = 49152;
        attr->preferredShmemCarveout = -1;
        return cudaSuccess;
    }
    attr->sharedSizeBytes           = ki->attributes.sharedSizeBytes;
    attr->constSizeBytes            = ki->attributes.constSizeBytes;
    attr->localSizeBytes            = ki->attributes.localSizeBytes;
    attr->maxThreadsPerBlock        = ki->attributes.maxThreadsPerBlock;
    attr->numRegs                   = ki->attributes.numRegs;
    attr->ptxVersion                = ki->attributes.ptxVersion;
    attr->binaryVersion             = ki->attributes.binaryVersion;
    attr->cacheModeCA               = ki->attributes.cacheModeCA;
    attr->maxDynamicSharedSizeBytes = ki->attributes.maxDynamicSharedSizeBytes;
    attr->preferredShmemCarveout    = ki->attributes.preferredShmemCarveout;
    LOG_INFO(CUDART, "cudaFuncGetAttributes %s → regs=%d maxThreads=%d shared=%zu const=%zu",
             ki->readableName.c_str(), attr->numRegs, attr->maxThreadsPerBlock,
             attr->sharedSizeBytes, attr->constSizeBytes);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaFuncGetAttributes);

API cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaFuncSetAttribute() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaFuncSetAttribute);

API cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags){
    LOG_DEBUG(CUDART, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() called.");
    if (!numBlocks || blockSize <= 0) return cudaErrorInvalidValue;
    auto* ki = KernelRegistry::getInstance().FindByHost(func, true);  // ← true
    int regs        = ki ? ki->attributes.numRegs : 32;
    int staticSmem  = ki ? (int)ki->attributes.sharedSizeBytes : 0;
    int totalSmem   = staticSmem + (int)dynamicSMemSize;
    // 从模拟设备查询 SM 参数
    int smMajor          = fake_getDevProps(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
    int maxThreadsPerSM  = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, 0);
    int maxBlocksPerSM   = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, 0);
    int regsPerSM        = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, 0);
    int smemPerSM        = fake_getDevProps(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, 0);
    int maxWarpsPerSM    = maxThreadsPerSM / 32;
    // 安全默认值
    if (maxThreadsPerSM == 0) maxThreadsPerSM = 2048;
    if (maxBlocksPerSM  == 0) maxBlocksPerSM  = 32;
    if (regsPerSM       == 0) regsPerSM       = 65536;
    if (smemPerSM       == 0) smemPerSM       = 49152;
    if (maxWarpsPerSM   == 0) maxWarpsPerSM   = 64;
    // 寄存器分配粒度（取决于架构）
    int regAllocUnit = (smMajor >= 8) ? 256 : 256;  // sm_70+ 都是 256
    int warpAllocGranularity = 4;                     // warp 分配粒度
    // ==================== 1. 线程数限制 ====================
    int warpsPerBlock = (blockSize + 31) / 32;
    int blocksByWarps = maxWarpsPerSM / warpsPerBlock;
    // ==================== 2. 寄存器限制 ====================
    int blocksByRegs = maxBlocksPerSM;
    if (regs > 0) {
        // 每个 warp 的寄存器分配 = ceil(regs * 32 / regAllocUnit) * regAllocUnit
        int regsPerWarp = ((regs * 32 + regAllocUnit - 1) / regAllocUnit) * regAllocUnit;
        int maxWarps = regsPerSM / regsPerWarp;
        // warp 数量向下对齐到分配粒度
        maxWarps = (maxWarps / warpAllocGranularity) * warpAllocGranularity;
        blocksByRegs = maxWarps / warpsPerBlock;
    }
    // ==================== 3. 共享内存限制 ====================
    int blocksBySmem = maxBlocksPerSM;
    if (totalSmem > 0) {
        // 共享内存分配粒度 = 256 bytes
        int smemAllocUnit = 256;
        int smemPerBlock = ((totalSmem + smemAllocUnit - 1) / smemAllocUnit) * smemAllocUnit;
        blocksBySmem = smemPerSM / smemPerBlock;
    }
    // ==================== 取最小值 ====================
    int result = blocksByWarps;
    if (blocksByRegs < result) result = blocksByRegs;
    if (blocksBySmem < result) result = blocksBySmem;
    if (result > maxBlocksPerSM) result = maxBlocksPerSM;
    if (result < 1) result = 1;
    *numBlocks = result;
    LOG_INFO(CUDART, "cudaOccupancyMaxActiveBlocks → %d "
             "(blockSize=%d warps=%d regs=%d smem=%d "
             "limitWarps=%d limitRegs=%d limitSmem=%d)",
             result, blockSize, warpsPerBlock, regs, totalSmem,
             blocksByWarps, blocksByRegs, blocksBySmem);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);

API cudaError_t cudaPeekAtLastError(void){
    LOG_DEBUG(CUDART, "cudaPeekAtLastError() called.");
    return fake_getLastError();
}
REGISTER_CUDA_FUNCTION(cudaPeekAtLastError);

API cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaPointerGetAttributes() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaPointerGetAttributes);

API cudaError_t cudaRuntimeGetVersion(int *runtimeVersion){
    LOG_DEBUG(CUDART, "cudaRuntimeGetVersion() called.");
    if (!runtimeVersion) return cudaErrorInvalidValue;
    *runtimeVersion = fake_getDriverVersion();
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaRuntimeGetVersion);

API cudaError_t cudaSetDevice (int device){
    LOG_DEBUG(CUDART, "cudaSetDevice() called.");
    if (device < 0 || device >= fake_getDeviceCount()){
        fake_setLastError(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }
    fake_setDevice(device);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaSetDevice);

API cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode){
    LOG_DEBUG(CUDART, "cudaThreadExchangeStreamCaptureMode() called.");
    static thread_local cudaStreamCaptureMode current_mode = cudaStreamCaptureModeGlobal;
    if (!mode || *mode < cudaStreamCaptureModeGlobal || *mode > cudaStreamCaptureModeRelaxed) return cudaErrorInvalidValue;
    const auto previous = current_mode;
    current_mode = *mode;
    *mode = previous;
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaThreadExchangeStreamCaptureMode);

// ============================================================================
// CUDA Registration Functions (Internal Runtime API)
// These are used by CUDA to register kernels, variables, etc.
// ============================================================================

// ============================================================================
// 函数: __cudaRegisterFatBinary
// 作用: 在程序启动时被编译器自动调用，注册包含所有 kernel 代码的 "fat binary"
//       (包含多种架构的 PTX/CUBIN 代码)
// 时机: 在 main() 之前的全局构造阶段
// ============================================================================
API void** __cudaRegisterFatBinary(void *fatCubin){
    LOG_DEBUG(CUDART, "__cudaRegisterFatBinary called.");
    void** handle = KernelRegistry::getInstance().RegisterFatbin(fatCubin);
    return handle;
}
REGISTER_CUDA_FUNCTION(__cudaRegisterFatBinary);

// ============================================================================
// 函数: __cudaRegisterFatBinaryEnd
// 作用: 标记 fat binary 注册完成（某些 CUDA 版本会调用）
// ============================================================================
API void __cudaRegisterFatBinaryEnd(void **fatCubinHandle){
    LOG_DEBUG(CUDART, "__cudaRegisterFatBinaryEnd called.");
}
REGISTER_CUDA_FUNCTION(__cudaRegisterFatBinaryEnd);

API void __cudaUnregisterFatBinary(void **fatCubinHandle){
    LOG_DEBUG(CUDART, "__cudaUnregisterFatBinary called.");
    KernelRegistry::getInstance().UnregisterFatbin(fatCubinHandle);
}
REGISTER_CUDA_FUNCTION(__cudaUnregisterFatBinary);

API void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global){
    LOG_DEBUG(CUDART, "__cudaRegisterVar called.");
    KernelRegistry::getInstance().RegisterVar(fatCubinHandle, hostVar, deviceName, size, constant, global);
}
REGISTER_CUDA_FUNCTION(__cudaRegisterVar);

// ============================================================================
// 函数: __cudaRegisterFunction
// 作用: 为每个 __global__ kernel 函数注册元数据，建立 host 和 device 函数的映射
// 参数:
//   - fatCubinHandle: fat binary 的句柄
//   - hostFun: 主机端的函数指针（用于查找对应的 kernel）
//   - deviceFun: 设备端函数名的 mangled name
//   - deviceName: 设备端函数的可读名称（如 "addVectors"）
//   - thread_limit: 线程数限制（-1 表示无限制）
//   - tid/bid/bDim/gDim: 线程/块的索引和维度信息（通常为 nullptr）
// ============================================================================
API void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize){
    LOG_DEBUG(CUDART, "__cudaRegisterFunction called.");
    KernelRegistry::getInstance().RegisterFunction(fatCubinHandle, (const void*)hostFun, deviceFun, deviceName, thread_limit);
}
REGISTER_CUDA_FUNCTION(__cudaRegisterFunction);

API void __cudaInitModule(void **fatCubinHandle){
    LOG_DEBUG(CUDART, "__cudaInitModule called.");
    // No-op
}
REGISTER_CUDA_FUNCTION(__cudaInitModule);

// ============================================================================
// 函数: __cudaPopCallConfiguration
// 作用: cudaLaunchKernel 调用时取出之前保存的配置
// 时机: 在实际 kernel 启动时
// ============================================================================
API cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void **stream){
    LOG_DEBUG(CUDART, "__cudaPopCallConfiguration() called.");
    if (!gridDim || !blockDim || !sharedMem || !stream) return cudaErrorInvalidValue;
    auto& config = KernelRegistry::getInstance().getLaunchConfig();
    *gridDim   = config.gridDim;
    *blockDim  = config.blockDim;
    *sharedMem = config.sharedMem;
    *stream    = config.stream;
    LOG_INFO(CUDART, "__cudaPopCallConfiguration grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu",
              gridDim->x, gridDim->y, gridDim->z,
              blockDim->x, blockDim->y, blockDim->z, *sharedMem);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(__cudaPopCallConfiguration);

// ============================================================================
// 函数: __cudaPushCallConfiguration
// 作用: kernel 启动语法 <<<gridDim, blockDim, sharedMem, stream>>> 时
//       编译器生成的代码会先调用此函数保存配置
// 时机: 在 kernel<<<>>>() 调用之前
// ============================================================================
API unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), void *stream __dv(0)){
    LOG_DEBUG(CUDART, "__cudaPushCallConfiguration() called.");
    auto& config = KernelRegistry::getInstance().getLaunchConfig();
    config.gridDim   = gridDim;
    config.blockDim  = blockDim;
    config.sharedMem = sharedMem;
    config.stream    = stream;
    LOG_INFO(CUDART, "__cudaPushCallConfiguration grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu",
              gridDim.x, gridDim.y, gridDim.z,
              blockDim.x, blockDim.y, blockDim.z, sharedMem);
    return cudaSuccess; // 返回 0 表示成功
}
REGISTER_CUDA_FUNCTION(__cudaPushCallConfiguration);

// ============================================================================
// CUDA Profiler Functions
// ============================================================================

API cudaError_t cudaProfilerStart(void){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaProfilerStart() called.");
    // No-op in fake implementation
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaProfilerStart);

API cudaError_t cudaProfilerStop(void){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    LOG_DEBUG(CUDART, "cudaProfilerStop() called.");
    // No-op in fake implementation
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaProfilerStop);
