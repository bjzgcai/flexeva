#include "utils.hpp"
#include <dlfcn.h>
#include "function_registry.hpp"
#include "fake_device_core.h"

// ==========================================
// Hook dlsym
// ==========================================
typedef void* (*real_dlsym_t)(void *, const char *);
API void *dlsym(void *handle, const char *symbol) {
    // 1. 优先查你的注册表
    // 这是拦截 PyTorch 入口的关键：PyTorch 调 dlsym 找 "cuInit" 或 "cuGetProcAddress" 时直接返回 fake 地址
    void* fake_func = FunctionRegistry::instance().get_function(symbol);
    if (fake_func) {
        return fake_func;
    }
    // 2. 获取系统原始 dlsym (仅用于非 CUDA 符号)
    static real_dlsym_t real_dlsym_func = nullptr;
    if (!real_dlsym_func) {
        real_dlsym_func = (real_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym_func) {
            fprintf(stderr, "Error: Failed to find real dlsym\n");
            return nullptr;
        }
    }
    // 3. 这里的 handle 可能是 PyTorch 尝试打开的 libcuda.so 的 handle
    // 但因为我们拦截了符号，它实际上会拿到我们的 fake 函数
    return real_dlsym_func(handle, symbol);
}


API CUresult cuInit(unsigned int Flags){
    fakecuda::capture::WrapperCall capture_call;
    try {
        LOG_DEBUG(CUDA, "cuInit() called.");
        if(fake_isInitialized()) {
            return CUDA_SUCCESS;
        }
        bool res = fake_initialize();
        if (!res) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuInit: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuInit);

API CUresult cuDriverGetVersion(int* driverVersion){
    fakecuda::capture::WrapperCall capture_call;
    try{
        LOG_DEBUG(CUDA, "cuDriverGetVersion() called.");
        // if (!fake_isInitialized()) return CUDA_ERROR_NOT_INITIALIZED;
        if(driverVersion != nullptr)
            *driverVersion = fake_getDriverVersion();
        else return CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in cuDriverGetVersion: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }

}
REGISTER_CUDA_FUNCTION(cuDriverGetVersion);

API CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus){
    fakecuda::capture::WrapperCall capture_call;
    (void)cudaVersion; (void)flags;
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
    *pfn = FunctionRegistry::instance().get_function(symbol);
    if (symbolStatus) *symbolStatus = *pfn ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return *pfn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}
REGISTER_CUDA_FUNCTION(cuGetProcAddress);

API CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId){
    fakecuda::capture::WrapperCall capture_call;
    try{
        LOG_DEBUG(CUDA, "cuGetExportTable() called.");
        if (!ppExportTable || !pExportTableId)
            return CUDA_ERROR_INVALID_VALUE;
        *ppExportTable = nullptr;
        return CUDA_ERROR_NOT_SUPPORTED;
    } catch (const std::exception& e){
        fprintf(stderr, "Error in cuGetExportTable: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}
REGISTER_CUDA_FUNCTION(cuGetExportTable);
