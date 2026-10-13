#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "ras_capture.h"

namespace {
cudaError_t direct_kernel(const void* func, dim3 grid, dim3 block,
                          size_t shared_mem, cudaStream_t stream,
                          FakeCudaRasApiKind api_kind) {
    if (!func || !grid.x || !grid.y || !grid.z || !block.x || !block.y ||
        !block.z || !fake_valid_stream(stream)) {
        fakecuda_ras_fail("invalid direct kernel launch arguments");
        return cudaErrorInvalidValue;
    }
    fake_stream_mark_work_enqueued(stream);
    auto r = fakecuda::ras::record_v2(FAKECUDA_RAS_KERNEL);
    r.api_kind = api_kind;
    r.base.stream = fakecuda::ras::address(stream);
    r.grid[0] = grid.x; r.grid[1] = grid.y; r.grid[2] = grid.z;
    r.block[0] = block.x; r.block[1] = block.y; r.block[2] = block.z;
    r.shared_mem = shared_mem;
    if (fakecuda::ras::wants_payload(FAKECUDA_RAS_KERNEL)) {
        const auto* ki = KernelRegistry::getInstance().FindByHost(func);
        if (!ki || ki->readableName.empty()) {
            fakecuda_ras_fail("kernel launch has no registered semantic name");
            return cudaErrorInvalidDeviceFunction;
        }
        r.kernel_name = ki->readableName.data();
        r.kernel_name_size = ki->readableName.size();
    } else {
        r.flags_v2 |= FAKECUDA_RAS_PAYLOAD_ELIDED;
    }
    const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
    fake_setLastError(result);
    return result;
}
}

API cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fprintf(stderr, "[fakecuda] cudaLaunchHostFunc() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaLaunchHostFunc);

// ============================================================================
// 函数: cudaLaunchKernel
// 作用: 实际启动 kernel 的函数
// 参数:
//   - func: kernel 函数指针（对应之前注册的 hostFun）
//   - gridDim/blockDim: 网格和块的维度
//   - args: kernel 参数数组
//   - sharedMem: 动态共享内存大小
//   - stream: CUDA 流
// ============================================================================
API cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream){
    if (fakecuda::ras::active()) return direct_kernel(func, gridDim, blockDim, sharedMem, stream,
                                                     FAKECUDA_RAS_API_CUDA_LAUNCH_KERNEL);
    LOG_DEBUG(CUDART, "cudaLaunchKernel() called.");
    fake_stream_mark_work_enqueued(stream);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaLaunchKernel);

API cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args){
    if (fakecuda::ras::active()) {
        if (!config || (config->numAttrs && !config->attrs)) {
            fakecuda_ras_fail("invalid extended kernel launch configuration");
            return cudaErrorInvalidValue;
        }
        // Attributes with scheduling or event semantics need explicit replay
        // support; never discard them from the direct descriptor.
        for (unsigned i = 0; i < config->numAttrs; ++i) {
            if (config->attrs[i].id != cudaLaunchAttributeIgnore) {
                fakecuda_ras_fail("extended kernel launch attribute is unsupported");
                return cudaErrorNotSupported;
            }
        }
        return direct_kernel(func, config->gridDim, config->blockDim,
                             config->dynamicSmemBytes, config->stream,
                             FAKECUDA_RAS_API_CUDA_LAUNCH_KERNEL_EX_C);
    }
    LOG_DEBUG(CUDART, "cudaLaunchKernelExC() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaLaunchKernelExC);
