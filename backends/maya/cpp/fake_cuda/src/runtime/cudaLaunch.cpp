#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "kernel_arg_parsers.h"
#include "trace_log.hpp"

namespace {
cudaError_t direct_kernel(const void* func, dim3 grid, dim3 block,
                          size_t shared_mem, cudaStream_t stream,
                          FakeCudaRasApiKind api_kind) {
    if (!func || !grid.x || !grid.y || !grid.z || !block.x || !block.y ||
        !block.z || !fake_valid_stream(stream)) {
        fakecuda_ras_fail("invalid direct kernel launch arguments");
        return cudaErrorInvalidValue;
    }
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    // Native FakeCUDA execution updates stream progress in both modes. No
    // argument parser, JSON payload, or synthetic error-query record is used.
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
    fakecuda::capture::WrapperCall capture_call;
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
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaLaunchKernel() called.");
    fake_stream_mark_work_enqueued(stream);
    // The remaining body only produces trace/argument diagnostics; stream
    // progress above is the ordinary FakeCUDA execution effect in every mode.
    if (!fakecuda::capture::generates_trace()) return cudaSuccess;
    auto* ki = KernelRegistry::getInstance().FindByHost(func);
    std::string kernel_name = ki ? ki->readableName : std::string("(unknown)");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    auto payload = make_trace_payload();
    if (payload) payload->add_string("kernel", kernel_name);
    if (payload) payload->add_uint("grid_x", gridDim.x);
    if (payload) payload->add_uint("grid_y", gridDim.y);
    if (payload) payload->add_uint("grid_z", gridDim.z);
    if (payload) payload->add_uint("block_x", blockDim.x);
    if (payload) payload->add_uint("block_y", blockDim.y);
    if (payload) payload->add_uint("block_z", blockDim.z);
    if (payload) payload->add_size("shared_mem", sharedMem);
    if (payload) payload->add_uint64("stream_id", trace_stream_id(stream));
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(CUDART, "cudaLaunchKernel", "kernel_launch", payload);
        TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR("cudaLaunchKernel");
    }
    LOG_INFO(CUDART, "cudaLaunchKernel %s grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu stream=%p",
             kernel_name.c_str(),
             gridDim.x, gridDim.y, gridDim.z,
             blockDim.x, blockDim.y, blockDim.z,
             sharedMem, (void*)stream);
    dispatchKernelArgs(ki->readableName, args, gridDim, blockDim, sharedMem); // ← 新增这一行
    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(CUDART, "cudaLaunchKernel", "kernel_launch", payload);
        TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR("cudaLaunchKernel");
    }
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
    fakecuda::capture::WrapperCall capture_call;
    fprintf(stderr, "[fakecuda] cudaLaunchKernelExC() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaLaunchKernelExC);
