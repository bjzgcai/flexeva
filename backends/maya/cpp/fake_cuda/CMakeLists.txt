cmake_minimum_required(VERSION 3.10)
project(FakeCudaDriver)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/liboutput)
set(FAKECUDA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_definitions(PROJECT_ROOT_DIR="${FAKECUDA_SOURCE_DIR}")

# fake device manager library
set(DEVICE_LIB_SOURCES
    src/device/device_manager.cpp
    src/device/fake_device_core.cpp
    src/common/ras_capture.cpp
)
add_library(fake_cuda_core SHARED ${DEVICE_LIB_SOURCES})
target_include_directories(fake_cuda_core PUBLIC ${FAKECUDA_SOURCE_DIR}/include/device ${FAKECUDA_SOURCE_DIR}/include/common)


# fake cuda runtime library
set(RUNTIME_LIB_SOURCES
    src/runtime/common.cpp
    src/runtime/cudaDevice.cpp
    src/runtime/cudaEvent.cpp
    src/runtime/cudaGet.cpp
    src/runtime/cudaGraph.cpp
    src/runtime/cudaHost.cpp
    src/runtime/cudaIpc.cpp
    src/runtime/cudaLaunch.cpp
    src/runtime/cudaMem.cpp
    src/runtime/cudaStream.cpp

)
add_library(cudart SHARED ${RUNTIME_LIB_SOURCES})
target_link_libraries(cudart PRIVATE fake_cuda_core)
target_include_directories(cudart PUBLIC ${FAKECUDA_SOURCE_DIR}/include/runtime ${FAKECUDA_SOURCE_DIR}/include/common)

set_target_properties(cudart PROPERTIES
    POSITION_INDEPENDENT_CODE ON  #地址无关代码
    OUTPUT_NAME "cudart"        # 文件名变成 libcudart.so
    SOVERSION "12"             # SONAME 变成 libcudart.so.12
    LINK_FLAGS "-Wl,--version-script=${FAKECUDA_SOURCE_DIR}/version/libcudart.version"
)

# fake cuda library
set(CUDA_LIB_SOURCES
    src/cuda/common.cpp
    src/cuda/cuDevice.cpp
    src/cuda/cuCtx.cpp
    src/cuda/cuMemVmm.cpp
)
add_library(cuda SHARED ${CUDA_LIB_SOURCES})
target_link_libraries(cuda PRIVATE fake_cuda_core ${CMAKE_DL_LIBS})
target_include_directories(cuda PUBLIC ${FAKECUDA_SOURCE_DIR}/include/runtime ${FAKECUDA_SOURCE_DIR}/include/common)
set_target_properties(cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON  #地址无关代码
    OUTPUT_NAME "cuda"        # 文件名变成 libcuda.so
    SOVERSION "1"             # SONAME 变成 libcuda.so.1
    LINK_FLAGS "-Wl,--version-script=${FAKECUDA_SOURCE_DIR}/version/libcuda.version"
)

# fake nvml library
set(NVML_LIB_SOURCES
    src/nvml/nvmlDeviceGet.cpp
)
add_library(nvidia-ml SHARED ${NVML_LIB_SOURCES})
target_link_libraries(nvidia-ml PRIVATE fake_cuda_core ${CMAKE_DL_LIBS})
target_include_directories(nvidia-ml PUBLIC ${FAKECUDA_SOURCE_DIR}/include/runtime ${FAKECUDA_SOURCE_DIR}/include/common)
set_target_properties(nvidia-ml PROPERTIES POSITION_INDEPENDENT_CODE ON)


# fake cublas library
set(CUBLAS_LIB_SOURCES
    src/cublas/common.cpp
    src/cublas/cublasSet.cpp
)
add_library(cublas SHARED ${CUBLAS_LIB_SOURCES})
target_link_libraries(cublas PRIVATE fake_cuda_core ${CMAKE_DL_LIBS})
target_include_directories(cublas PUBLIC ${FAKECUDA_SOURCE_DIR}/include/common ${FAKECUDA_SOURCE_DIR}/include/runtime ${FAKECUDA_SOURCE_DIR}/include/cublas)
set_target_properties(cublas PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cublas PROPERTIES
    POSITION_INDEPENDENT_CODE ON  #地址无关代码
    OUTPUT_NAME "cublas"        # 文件名变成 libcublas.so
    SOVERSION "12"             # SONAME 变成 libcublas.so.12
    LINK_FLAGS "-Wl,--version-script=${FAKECUDA_SOURCE_DIR}/version/libcublas.version"
)


# fake cublasLt library
set(CUBLASLT_LIB_SOURCES
    src/cublasLt/common.cpp
)
add_library(cublasLt SHARED ${CUBLASLT_LIB_SOURCES})
target_link_libraries(cublasLt PRIVATE fake_cuda_core ${CMAKE_DL_LIBS})
target_include_directories(cublasLt PUBLIC ${FAKECUDA_SOURCE_DIR}/include/common ${FAKECUDA_SOURCE_DIR}/include/runtime ${FAKECUDA_SOURCE_DIR}/include/cublas)
set_target_properties(cublasLt PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cublasLt PROPERTIES
    POSITION_INDEPENDENT_CODE ON  #地址无关代码
    OUTPUT_NAME "cublasLt"        # 文件名变成 libcublas.so
    SOVERSION "12"             # SONAME 变成 libcublas.so.12
    LINK_FLAGS "-Wl,--version-script=${FAKECUDA_SOURCE_DIR}/version/libcublasLt.version"
)

# fake nccl library
set(NCCL_LIB_SOURCES
    src/nccl/common.cpp
)
add_library(nccl SHARED ${NCCL_LIB_SOURCES})
target_link_libraries(nccl PRIVATE fake_cuda_core ${CMAKE_DL_LIBS})
target_include_directories(nccl PUBLIC ${FAKECUDA_SOURCE_DIR}/include/common ${FAKECUDA_SOURCE_DIR}/include/runtime ${FAKECUDA_SOURCE_DIR}/include/nccl )
set_target_properties(nccl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(nccl PROPERTIES
    POSITION_INDEPENDENT_CODE ON  #地址无关代码
    OUTPUT_NAME "nccl"        # 文件名变成 libcublas.so
    SOVERSION "2"             # SONAME 变成 libcublas.so.12
    LINK_FLAGS "-Wl,--version-script=${FAKECUDA_SOURCE_DIR}/version/libnccl.version -Wl,--no-undefined"
)
