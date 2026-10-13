#include "utils.hpp"
#include "function_registry.hpp"
#include "fake_device_core.h"


API CUresult cuCtxGetCurrent(CUcontext *pctx){
    try{
        LOG_DEBUG(CUDA, "cuCtxGetCurrent() called.");
        if (!fake_isInitialized()) return CUDA_ERROR_NOT_INITIALIZED;
        if (pctx == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        CUcontext ctx = fake_getCurrentContext();
        *pctx = ctx;
        return CUDA_SUCCESS;
    }catch (const std::exception& e) {
        fprintf(stderr, "Error in cuCtxGetCurrent: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuCtxGetCurrent);
