#pragma once
#include <string>
#include <atomic>
#include <vector>
#include <mutex>
#include <cstdint>
#include <map>
#include "json.hpp"
#include "fake_types.h"

struct DevProps{
    // 基础参数
    std::string name = "Fake NVIDIA A800-SXM4-40GB";
    cudaUUID_t uuid;
    int id = 0;
    size_t DeviceTotalMem = 42405855232;        // 全局内存总量（bytes）
    size_t DeviceUsedMem = 0;         // 使用的内存（bytes）
    int CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 1410000;            // 时钟频率（kHz）
    int CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 1215000;     // 内存时钟频率（kHz）
    int CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 5120;   // 显存位宽 （bit）
    int CU_DEVICE_ATTRIBUTE_MAX_PITCH = 2147483647;                    // pitch 是指在二维数组中，每一行数据的字节数（包括为了内存对齐而添加的填充字节）。

    // 计算能力
    int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 8;  // 计算能力主版本
    int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 0;  // 计算能力次版本

    // 性能与带宽相关
    int CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 2;  // 单精度与双精度性能比

    // 线程块配置
    int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1024;  // 每块最大线程数
    int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 1024;        // 块维度X
    int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 1024;        // 块维度Y
    int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 64;          // 块维度Z
    int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 2147483647;   // 网格维度X
    int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 65535;       // 网格维度Y
    int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 65535;       // 网格维度Z

    // Warp和SM配置
    int CU_DEVICE_ATTRIBUTE_WARP_SIZE = 32;             // Warp大小
    int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 108; // 多处理器数量
    int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 2048;   // 每多处理器最大线程数
    int CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 32;   // 每多处理器最大块数

    // 共享内存
    int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 49152;       // 每块最大共享内存（bytes）
    int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 167936; // 每SM共享内存（bytes）
    int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 166912; // 每个线程块在使用“opt-in”机制时可用的最大共享内存容量（bytes），在需要大量共享内存的高性能计算或深度学习算子中，开发者可以通过这个属性判断是否可以使用更大的共享内存，从而优化性能。

    // 常量内存和缓存
    int CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 65536;     // 常量内存大小（bytes）
    int CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 41943040;          // L2缓存大小（bytes）

    // 寄存器
    int CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 65536;   // 每块最大寄存器数
    int CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 65536; // 每SM最大寄存器数

    // 特殊功能支持
    int CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 0; // 指示当前设备是否支持通过 Tensor Map 访问内存
    int CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 0;  // 指示当前设备是否支持统一函数指针特性

    // 统一内存和托管内存
    int CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 1;          // 统一地址空间支持
    int CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 1;             // 托管内存支持
    int CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 1;  // 并发托管内存访问支持
    int CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 0; // 主机直接访问托管内存支持

    // 页面内存
    int CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 0;           // GPU是否可以直接访问主机上的可分页内存
    int CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 0;    // GPU 在访问主机可分页内存时，不再通过内部“页锁定影子缓冲区”，而是直接走主机的页表，像 CPU 一样做地址转换与换页，从而实现真正的“零拷贝”访问。

    // 虚拟内存
    int CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 1;      // 虚拟内存管理支持
    int CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 1;                   // 内存池支持

    // 并发与并行特性
    int CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 1;                 // 并发内核执行支持
    int CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 3;                 // 异步引擎数量
    int CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 1;            // 流优先级支持
    int CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 1;                     // 协同启动支持

    // 缓存配置
    int CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 1;        // 全局L1缓存支持
    int CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 1;         // 本地L1缓存支持
    int CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 26214400; // 最大持久L2缓存大小（bytes）

    // IPC和通信
    int CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 1;                // IPC事件支持
    int CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 9;     // 内存池支持的句柄类型
    int CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 1; // 支持通过 POSIX 文件描述符导出内存

    // 其他重要属性
    int CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 0;                       // 计算模式
    int CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 1;                         // ECC支持
    int CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 1;                 // 主机内存映射支持
    int CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 1;              // 主机内存注册支持

    // 纹理属性
    int CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 512;                     // 纹理对齐要求
    int CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 32;                // 纹理pitch对齐要求
    int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 131072;            // 最大2D纹理宽度
    int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 65536;            // 最大2D纹理高度

    //Surface属性
    int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 131072;            // 最大2D表面宽度
    int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 65536;            // 最大2D表面高度
    int CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 512;                  // 表面对齐要求

    //pcie属性
    int CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 1;                            // PCI总线ID
    int CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 0;                         // PCI设备ID
    int CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 0;                         // PCI域ID

    int CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 0;                           // TCC驱动支持
    int CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 0;          // 主机本地原子操作支持
    int CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 1;         // 计算抢占支持
    int CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 1; // 已注册内存主机指针使用支持
    int CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 1;        // 多设备协同启动支持
    int CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 134213632; // 最大访问策略窗口大小（bytes）
    int CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 1;            // 稀疏CUDA数组支持
    int CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 1;     // 只读主机内存注册支持
    int CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 1; // 时间线信号量互操作支持
    int CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 1; // GPU直接RDMA写入刷新选项支持
    int CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 0;      // GPU直接RDMA写入排序支持
    int CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 1; // 延迟映射CUDA数组支持
    int CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 0;                        // 集群启动支持
};


class DeviceManager{
    private:
        int driver_version = 13000;        // 驱动版本
        int device_count = 0;              // 设备数量
        static thread_local int currentDevice;  // 每个线程独立的当前设备
        static thread_local cudaError_t lastError;  // 每个线程独立的最后错误状态

        // 存储每个设备的“主Context”
        std::vector<CUctx_st*> fake_contexts;
        mutable std::mutex ctx_mutex;

        struct AllocInfo {
            size_t size;
            size_t reserved;
            int device_id;
        };
        struct DeviceArena {
            void* base = nullptr;
            size_t bytes = 0;
            // Page-aligned offset -> free span length, ordered by address.
            std::map<size_t, size_t> free_ranges;
        };
        std::vector<DeviceArena> arenas;
        // 用于追踪已分配的指针，key是分配的地址
        std::map<void*, AllocInfo> allocation_map;
        mutable std::mutex alloc_mutex; // 保护 allocation_map 和 DeviceUsedMem 的修改

        std::vector<DevProps> devices;

        std::atomic<bool> initialized{false};
        mutable std::mutex init_mutex;

        DeviceManager() = default;            // 禁止外部构造
        ~DeviceManager();                  // 禁止外部析构
        bool doInitialize(); // 实际初始化逻辑
    public:
        // 删除拷贝
        DeviceManager(const DeviceManager&) = delete;
        DeviceManager& operator=(const DeviceManager&) = delete;

        // 唯一访问点
        static DeviceManager& getInstance(); // C++11 起线程安全

        // 线程安全的查询接口
        bool isInitialized() const noexcept;
        int getDriverVersion() const noexcept;
        int getDeviceCount() const noexcept;
        std::string getDeviceName(int device_id) const noexcept;
        size_t getDeviceTotalMem(int device_id = -1) const noexcept;
        size_t getDeviceFreeMem(int device_id = -1) const noexcept;
        int getDevProps(CUdevice_attribute attrib, int dev) const noexcept;
        cudaUUID_t getDeviceUUID(int device_id) const noexcept;
        cudaError_t getLastError() const noexcept;
        void setLastError(cudaError_t error)const noexcept;
        CUcontext getCurrentContext() const noexcept;
        CUcontext getContextForDevice(int dev_id) const noexcept;

        // 设备管理接口
        bool initialize();
        void setDevice(int device_id) const noexcept;
        void getDevice(int* device_id) const noexcept;
        cudaError_t allocateMemory(void** devPtr, size_t size) noexcept;
        cudaError_t freeMemory(void* devPtr) noexcept;
        bool allocationRange(uint64_t address, uint64_t& base, uint64_t& bytes) const noexcept;
};
