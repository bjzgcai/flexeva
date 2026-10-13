#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"


API cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fakecuda::capture::WrapperCall capture_call;
    fprintf(stderr, "[fakecuda] cudaGraphDebugDotPrint() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGraphDebugDotPrint);

API cudaError_t cudaGraphDestroy(cudaGraph_t graph){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fakecuda::capture::WrapperCall capture_call;
    fprintf(stderr, "[fakecuda] cudaGraphDestroy() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGraphDestroy);

API cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fakecuda::capture::WrapperCall capture_call;
    fprintf(stderr, "[fakecuda] cudaGraphExecDestroy() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGraphExecDestroy);

API cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fakecuda::capture::WrapperCall capture_call;
    fprintf(stderr, "[fakecuda] cudaGraphGetNodes() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGraphGetNodes);

API cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags __dv(0)){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fakecuda::capture::WrapperCall capture_call;
    fprintf(stderr, "[fakecuda] cudaGraphInstantiate() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGraphInstantiate);

API cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags __dv(0)){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fakecuda::capture::WrapperCall capture_call;
    fprintf(stderr, "[fakecuda] cudaGraphInstantiateWithFlags() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGraphInstantiateWithFlags);

API cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fakecuda::capture::WrapperCall capture_call;
    fprintf(stderr, "[fakecuda] cudaGraphLaunch() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaGraphLaunch);
