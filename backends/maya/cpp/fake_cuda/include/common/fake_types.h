#pragma once
#include <cstdint>
#include <cstddef>


// 定义内部使用的 Stream 结构
struct CUstream_st {
    int id;
    // 如果需要支持 Event 同步，这里可能需要加个 std::atomic<int> 或者 std::mutex
    // 但如果只是为了跑通流程，空的或者只存个 ID 就够了
};
using cudaStream_t = CUstream_st*;

// 定义内部使用的 Context 结构
struct CUctx_st {
    int id;
    int magic; // 用于校验是否是有效的 Context
};
typedef struct CUctx_st *CUcontext;
// 全局默认流（Default Stream），ID 设为 0
// CUDA 标准规定默认流是 0 (NULL)，但有些库可能会检查指针有效性
// 这里我们保留 NULL 作为默认流的语义，或者定义一个特殊的静态对象
static CUstream_st default_stream_obj = {0};

using cudaHostFn_t = void(*)(void*);

/**
 * CUDA graph
 */
typedef struct CUgraph_st *cudaGraph_t;

/**
 * CUDA graph node.
 */
typedef struct CUgraphNode_st *cudaGraphNode_t;

/**
 * CUDA event types
 */
typedef struct CUevent_st *cudaEvent_t;

/**
 * Event record node parameters
 */
struct cudaEventRecordNodeParams {
    cudaEvent_t event; /**< The event to record when the node executes */
};

/**
 * CUDA memory pool
 */
typedef struct CUmemPoolHandle_st *cudaMemPool_t;

/**
 * CUDA device node handle for device-side node update
 */
typedef struct CUgraphDeviceUpdatableNode_st* cudaGraphDeviceNode_t;

/**
 * CUDA executable (launchable) graph
 */
typedef struct CUgraphExec_st* cudaGraphExec_t;

#define __dv(v) \
        = v

struct uint3
{
    unsigned int x, y, z;
};
typedef struct uint3 uint3;
struct dim3
{
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
        constexpr operator uint3(void) const { return uint3{x, y, z}; }
};

/**
 * CUDA error types
 */
enum cudaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
    cudaSuccess                           =      0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    cudaErrorInvalidValue                 =     1,

    /**
     * The API call failed because it was unable to allocate enough memory or
     * other resources to perform the requested operation.
     */
    cudaErrorMemoryAllocation             =      2,

    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    cudaErrorInitializationError          =      3,

    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    cudaErrorCudartUnloading              =     4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    cudaErrorProfilerDisabled             =     5,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
     * ::cudaProfilerStop without initialization.
     */
    cudaErrorProfilerNotInitialized       =     6,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    cudaErrorProfilerAlreadyStarted       =     7,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
     cudaErrorProfilerAlreadyStopped       =    8,
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::cudaDeviceProp for more
     * device limitations.
     */
    cudaErrorInvalidConfiguration         =      9,

    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    cudaErrorInvalidPitchValue            =     12,

    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    cudaErrorInvalidSymbol                =     13,

    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidHostPointer           =     16,

    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidDevicePointer         =     17,
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    cudaErrorInvalidTexture               =     18,

    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
     */
    cudaErrorInvalidTextureBinding        =     19,

    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    cudaErrorInvalidChannelDescriptor     =     20,

    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    cudaErrorInvalidMemcpyDirection       =     21,

    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::cudaGetSymbolAddress().
     */
    cudaErrorAddressOfConstant            =     22,

    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureFetchFailed           =     23,

    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureNotBound              =     24,

    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorSynchronizationError         =     25,
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    cudaErrorInvalidFilterSetting         =     26,

    /**
     * This indicates that an attempt was made to read an unsupported data type as a
     * normalized float. This is not supported by CUDA.
     */
    cudaErrorInvalidNormSetting           =     27,

    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMixedDeviceExecution         =     28,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorNotYetImplemented            =     31,

    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMemoryValueTooLarge          =     32,
    /**
     * This indicates that the CUDA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in CUDA API returning this error.
     */
    cudaErrorStubLibrary                  =     34,

    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    cudaErrorInsufficientDriver           =     35,

    /**
     * This indicates that the API call requires a newer CUDA driver than the one
     * currently installed. Users should install an updated NVIDIA CUDA driver
     * to allow the API call to succeed.
     */
    cudaErrorCallRequiresNewerDriver      =     36,

    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    cudaErrorInvalidSurface               =     37,

    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    cudaErrorDuplicateVariableName        =     43,

    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateTextureName         =     44,

    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateSurfaceName         =     45,

    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeProhibited, ::cudaComputeModeExclusiveProcess, or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    cudaErrorDevicesUnavailable           =     46,

    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version
     * of the API, because the Runtime API call expects a primary driver
     * context and the Driver context is not primary, or because the Driver
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions
     * with the CUDA Driver API" for more information.
     */
    cudaErrorIncompatibleDriverContext    =     49,

    /**
     * The device function being invoked (usually via ::cudaLaunchKernel()) was not
     * previously configured via the ::cudaConfigureCall() function.
     */
    cudaErrorMissingConfiguration         =      52,

    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorPriorLaunchFailure           =      53,
    /**
     * This error indicates that a device runtime grid launch did not occur
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches.
     */
    cudaErrorLaunchMaxDepthExceeded       =     65,

    /**
     * This error indicates that a grid launch did not occur because the kernel
     * uses file-scoped textures which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support textures created with
     * the Texture Object API's.
     */
    cudaErrorLaunchFileScopedTex          =     66,

    /**
     * This error indicates that a grid launch did not occur because the kernel
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    cudaErrorLaunchFileScopedSurf         =     67,

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on
     * launched grids at a greater depth successfully, the maximum nested
     * depth at which ::cudaDeviceSynchronize will be called must be specified
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime.
     * Keep in mind that additional levels of sync depth require the runtime
     * to reserve large amounts of device memory that cannot be used for
     * user allocations. Note that ::cudaDeviceSynchronize made from device
     * runtime is only supported on devices of compute capability < 9.0.
     */
    cudaErrorSyncDepthExceeded            =     68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    cudaErrorLaunchPendingCountExceeded   =     69,

    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    cudaErrorInvalidDeviceFunction        =      98,

    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    cudaErrorNoDevice                     =     100,

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device or that the action requested is
     * invalid for the specified device.
     */
    cudaErrorInvalidDevice                =     101,

    /**
     * This indicates that the device doesn't have a valid Grid License.
     */
    cudaErrorDeviceNotLicensed            =     102,

   /**
    * By default, the CUDA runtime may perform a minimal set of self-tests,
    * as well as CUDA driver tests, to establish the validity of both.
    * Introduced in CUDA 11.2, this error return indicates that at least one
    * of these tests has failed and the validity of either the runtime
    * or the driver could not be established.
    */
   cudaErrorSoftwareValidityNotEstablished  =     103,

    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    cudaErrorStartupFailure               =    127,

    /**
     * This indicates that the device kernel image is invalid.
     */
    cudaErrorInvalidKernelImage           =     200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    cudaErrorDeviceUninitialized          =     201,

    /**
     * This indicates that the buffer object could not be mapped.
     */
    cudaErrorMapBufferObjectFailed        =     205,

    /**
     * This indicates that the buffer object could not be unmapped.
     */
    cudaErrorUnmapBufferObjectFailed      =     206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    cudaErrorArrayIsMapped                =     207,

    /**
     * This indicates that the resource is already mapped.
     */
    cudaErrorAlreadyMapped                =     208,

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    cudaErrorNoKernelImageForDevice       =     209,

    /**
     * This indicates that a resource has already been acquired.
     */
    cudaErrorAlreadyAcquired              =     210,

    /**
     * This indicates that a resource is not mapped.
     */
    cudaErrorNotMapped                    =     211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    cudaErrorNotMappedAsArray             =     212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    cudaErrorNotMappedAsPointer           =     213,

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    cudaErrorECCUncorrectable             =     214,

    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    cudaErrorUnsupportedLimit             =     215,

    /**
     * This indicates that a call tried to access an exclusive-thread device that
     * is already in use by a different thread.
     */
    cudaErrorDeviceAlreadyInUse           =     216,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    cudaErrorPeerAccessUnsupported        =     217,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    cudaErrorInvalidPtx                   =     218,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    cudaErrorInvalidGraphicsContext       =     219,

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    cudaErrorNvlinkUncorrectable          =     220,

    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    cudaErrorJitCompilerNotFound          =     221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     * The most common reason for this, is the PTX was generated by a compiler newer
     * than what is supported by the CUDA driver and PTX JIT compiler.
     */
    cudaErrorUnsupportedPtxVersion        =     222,

    /**
     * This indicates that the JIT compilation was disabled. The JIT compilation compiles
     * PTX. The runtime may fall back to compiling PTX if an application does not contain
     * a suitable binary for the current device.
     */
    cudaErrorJitCompilationDisabled       =     223,

    /**
     * This indicates that the provided execution affinity is not supported by the device.
     */
    cudaErrorUnsupportedExecAffinity      =     224,

    /**
     * This indicates that the code to be compiled by the PTX JIT contains
     * unsupported call to cudaDeviceSynchronize.
     */
    cudaErrorUnsupportedDevSideSync       =     225,

    /**
     * This indicates that an exception occurred on the device that is now
     * contained by the GPU's error containment capability. Common causes are -
     * a. Certain types of invalid accesses of peer GPU memory over nvlink
     * b. Certain classes of hardware errors
     * This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must
     * be terminated and relaunched.
     */
    cudaErrorContained                    =     226,

    /**
     * This indicates that the device kernel source is invalid.
     */
    cudaErrorInvalidSource                =     300,

    /**
     * This indicates that the file specified was not found.
     */
    cudaErrorFileNotFound                 =     301,

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    cudaErrorSharedObjectSymbolNotFound   =     302,

    /**
     * This indicates that initialization of a shared object failed.
     */
    cudaErrorSharedObjectInitFailed       =     303,

    /**
     * This error indicates that an OS call failed.
     */
    cudaErrorOperatingSystem              =     304,

    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    cudaErrorInvalidResourceHandle        =     400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    cudaErrorIllegalState                 =     401,

    /**
     * This indicates an attempt was made to introspect an object in a way that
     * would discard semantically important information. This is either due to
     * the object using funtionality newer than the API version used to
     * introspect it or omission of optional return arguments.
     */
    cudaErrorLossyQuery                   =     402,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    cudaErrorSymbolNotFound               =     500,

    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
     */
    cudaErrorNotReady                     =     600,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalAddress               =     700,

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    cudaErrorLaunchOutOfResources         =      701,

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchTimeout                =      702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    cudaErrorLaunchIncompatibleTexturing  =     703,

    /**
     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    cudaErrorPeerAccessAlreadyEnabled     =     704,

    /**
     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to
     * disable peer addressing which has not been enabled yet via
     * ::cudaDeviceEnablePeerAccess().
     */
    cudaErrorPeerAccessNotEnabled         =     705,

    /**
     * This indicates that the user has called ::cudaSetValidDevices(),
     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    cudaErrorSetOnActiveProcess           =     708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    cudaErrorContextIsDestroyed           =     709,

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again. All existing allocations are invalid. To continue
     * using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorAssert                        =    710,

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cudaEnablePeerAccess().
     */
    cudaErrorTooManyPeers                 =     711,

    /**
     * This error indicates that the memory range passed to ::cudaHostRegister()
     * has already been registered.
     */
    cudaErrorHostMemoryAlreadyRegistered  =     712,

    /**
     * This error indicates that the pointer passed to ::cudaHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    cudaErrorHostMemoryNotRegistered      =     713,

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorHardwareStackError           =     714,

    /**
     * The device encountered an illegal instruction during kernel execution
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalInstruction           =     715,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorMisalignedAddress            =     716,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidAddressSpace          =     717,

    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidPc                    =     718,

    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchFailure                =      719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
     */
    cudaErrorCooperativeLaunchTooLarge    =     720,

    /**
     * An exception occurred on the device while exiting a kernel using tensor memory: the
     * tensor memory was not completely deallocated. This leaves the process in an inconsistent
     * state and any further CUDA work will return the same error. To continue using CUDA, the
     * process must be terminated and relaunched.
     */
    cudaErrorTensorMemoryLeak             =     721,

    /**
     * This error indicates the attempted operation is not permitted.
     */
    cudaErrorNotPermitted                 =     800,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    cudaErrorNotSupported                 =     801,

    /**
     * This error indicates that the system is not yet ready to start any CUDA
     * work.  To continue using CUDA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    cudaErrorSystemNotReady               =     802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the CUDA driver. Refer to the compatibility documentation
     * for supported versions.
     */
    cudaErrorSystemDriverMismatch         =     803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by CUDA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
     * environment variable.
     */
    cudaErrorCompatNotSupportedOnDevice   =     804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    cudaErrorMpsConnectionFailed          =     805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    cudaErrorMpsRpcFailure                =     806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    cudaErrorMpsServerNotReady            =     807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    cudaErrorMpsMaxClientsReached         =     808,

    /**
     * This error indicates the the hardware resources required to device connections have been exhausted.
     */
    cudaErrorMpsMaxConnectionsReached     =     809,

    /**
     * This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorMpsClientTerminated          =     810,

    /**
     * This error indicates, that the program is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.
     */
    cudaErrorCdpNotSupported              =     811,

    /**
     * This error indicates, that the program contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.
     */
    cudaErrorCdpVersionMismatch           =     812,

    /**
     * The operation is not permitted when the stream is capturing.
     */
    cudaErrorStreamCaptureUnsupported     =    900,

    /**
     * The current capture sequence on the stream has been invalidated due to
     * a previous error.
     */
    cudaErrorStreamCaptureInvalidated     =    901,

    /**
     * The operation would have resulted in a merge of two independent capture
     * sequences.
     */
    cudaErrorStreamCaptureMerge           =    902,

    /**
     * The capture was not initiated in this stream.
     */
    cudaErrorStreamCaptureUnmatched       =    903,

    /**
     * The capture sequence contains a fork that was not joined to the primary
     * stream.
     */
    cudaErrorStreamCaptureUnjoined        =    904,

    /**
     * A dependency would have been created which crosses the capture sequence
     * boundary. Only implicit in-stream ordering dependencies are allowed to
     * cross the boundary.
     */
    cudaErrorStreamCaptureIsolation       =    905,

    /**
     * The operation would have resulted in a disallowed implicit dependency on
     * a current capture sequence from cudaStreamLegacy.
     */
    cudaErrorStreamCaptureImplicit        =    906,

    /**
     * The operation is not permitted on an event which was last recorded in a
     * capturing stream.
     */
    cudaErrorCapturedEvent                =    907,

    /**
     * A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed
     * argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a
     * different thread.
     */
    cudaErrorStreamCaptureWrongThread     =    908,

    /**
     * This indicates that the wait operation has timed out.
     */
    cudaErrorTimeout                      =    909,

    /**
     * This error indicates that the graph update was not performed because it included
     * changes which violated constraints specific to instantiated graph update.
     */
    cudaErrorGraphExecUpdateFailure       =    910,

    /**
     * This indicates that an async error has occurred in a device outside of CUDA.
     * If CUDA was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must be
     * terminated and relaunched.
     */
    cudaErrorExternalDevice               =    911,

    /**
     * This indicates that a kernel launch error has occurred due to cluster
     * misconfiguration.
     */
    cudaErrorInvalidClusterSize           =    912,

    /**
     * Indiciates a function handle is not loaded when calling an API that requires
     * a loaded function.
     */
    cudaErrorFunctionNotLoaded            =    913,

    /**
     * This error indicates one or more resources passed in are not valid resource
     * types for the operation.
     */
    cudaErrorInvalidResourceType          =    914,

    /**
     * This error indicates one or more resources are insufficient or non-applicable for
     * the operation.
     */
    cudaErrorInvalidResourceConfiguration =    915,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    cudaErrorUnknown                      =    999

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    , cudaErrorApiFailureBase               =  10000
};
typedef enum cudaError cudaError_t;


using cuuint64_t = uint64_t;
using CUdevice = int;                                     /**< CUDA device */
/**
 * CUDA UUID types
 */

struct CUuuid_st {     /**< CUDA definition of UUID */
    char bytes[16];
};
using CUuuid = CUuuid_st;


#define cuDeviceTotalMem                    cuDeviceTotalMem_v2
#define cuGetProcAddress                    cuGetProcAddress_v2
#define cuCtxCreate                         cuCtxCreate_v2
#define cuCtxCreate_v3                      cuCtxCreate_v3
#define cuCtxCreate_v4                      cuCtxCreate_v4
#define cuModuleGetGlobal                   cuModuleGetGlobal_v2
#define cuMemGetInfo                        cuMemGetInfo_v2
#define cuMemAlloc                          cuMemAlloc_v2
#define cuMemAllocPitch                     cuMemAllocPitch_v2
#define cuMemFree                           cuMemFree_v2
#define cuMemGetAddressRange                cuMemGetAddressRange_v2
#define cuMemAllocHost                      cuMemAllocHost_v2
#define cuMemHostGetDevicePointer           cuMemHostGetDevicePointer_v2
#define cuArrayCreate                       cuArrayCreate_v2
#define cuArrayGetDescriptor                cuArrayGetDescriptor_v2
#define cuArray3DCreate                     cuArray3DCreate_v2
#define cuArray3DGetDescriptor              cuArray3DGetDescriptor_v2
#define cuCtxDestroy                        cuCtxDestroy_v2
#define cuCtxPopCurrent                     cuCtxPopCurrent_v2
#define cuCtxPushCurrent                    cuCtxPushCurrent_v2
#define cuStreamDestroy                     cuStreamDestroy_v2
#define cuEventDestroy                      cuEventDestroy_v2
#define cuLinkCreate                        cuLinkCreate_v2
#define cuLinkAddData                       cuLinkAddData_v2
#define cuLinkAddFile                       cuLinkAddFile_v2
#define cuMemHostRegister                   cuMemHostRegister_v2
#define cuGraphicsResourceSetMapFlags       cuGraphicsResourceSetMapFlags_v2
#define cuDevicePrimaryCtxRelease           cuDevicePrimaryCtxRelease_v2
#define cuDevicePrimaryCtxReset             cuDevicePrimaryCtxReset_v2
#define cuDevicePrimaryCtxSetFlags          cuDevicePrimaryCtxSetFlags_v2
#define cuDeviceGetUuid_v2                  cuDeviceGetUuid_v2
#define cuIpcOpenMemHandle                  cuIpcOpenMemHandle_v2

/**
 * Error codes
 */
typedef enum cudaError_enum {
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    CUDA_SUCCESS                              = 0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    CUDA_ERROR_INVALID_VALUE                  = 1,

    /**
     * The API call failed because it was unable to allocate enough memory or
     * other resources to perform the requested operation.
     */
    CUDA_ERROR_OUT_OF_MEMORY                  = 2,

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    CUDA_ERROR_NOT_INITIALIZED                = 3,

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    CUDA_ERROR_DEINITIALIZED                  = 4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    CUDA_ERROR_PROFILER_DISABLED              = 5,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cuProfilerStart or
     * ::cuProfilerStop without initialization.
     */
    CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cuProfilerStart() when profiling is already enabled.
     */
    CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cuProfilerStop() when profiling is already disabled.
     */
    CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,

    /**
     * This indicates that the CUDA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in CUDA API returning this error.
     */
    CUDA_ERROR_STUB_LIBRARY                   = 34,

    /**
     * This indicates that requested CUDA device is unavailable at the current
     * time. Devices are often unavailable due to use of
     * ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS or ::CU_COMPUTEMODE_PROHIBITED.
     */
    CUDA_ERROR_DEVICE_UNAVAILABLE            = 46,

    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    CUDA_ERROR_NO_DEVICE                      = 100,

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device or that the action requested is
     * invalid for the specified device.
     */
    CUDA_ERROR_INVALID_DEVICE                 = 101,

    /**
     * This error indicates that the Grid license is not applied.
     */
    CUDA_ERROR_DEVICE_NOT_LICENSED            = 102,

    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    CUDA_ERROR_INVALID_IMAGE                  = 200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     * This can also be returned if the green context passed to an API call
     * was not converted to a ::CUcontext using ::cuCtxFromGreenCtx API.
     */
    CUDA_ERROR_INVALID_CONTEXT                = 201,

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,

    /**
     * This indicates that a map or register operation has failed.
     */
    CUDA_ERROR_MAP_FAILED                     = 205,

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    CUDA_ERROR_UNMAP_FAILED                   = 206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    CUDA_ERROR_ARRAY_IS_MAPPED                = 207,

    /**
     * This indicates that the resource is already mapped.
     */
    CUDA_ERROR_ALREADY_MAPPED                 = 208,

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,

    /**
     * This indicates that a resource has already been acquired.
     */
    CUDA_ERROR_ALREADY_ACQUIRED               = 210,

    /**
     * This indicates that a resource is not mapped.
     */
    CUDA_ERROR_NOT_MAPPED                     = 211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    CUDA_ERROR_ECC_UNCORRECTABLE              = 214,

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

    /**
     * This indicates that peer access is not supported across the given
     * devices.
     */
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,

    /**
     * This indicates that a PTX JIT compilation failed.
     */
    CUDA_ERROR_INVALID_PTX                    = 218,

    /**
     * This indicates an error with OpenGL or DirectX context.
     */
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = 219,

    /**
    * This indicates that an uncorrectable NVLink error was detected during the
    * execution.
    */
    CUDA_ERROR_NVLINK_UNCORRECTABLE           = 220,

    /**
    * This indicates that the PTX JIT compiler library was not found.
    */
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND         = 221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     */

    CUDA_ERROR_UNSUPPORTED_PTX_VERSION        = 222,

    /**
     * This indicates that the PTX JIT compilation was disabled.
     */
    CUDA_ERROR_JIT_COMPILATION_DISABLED       = 223,

    /**
     * This indicates that the ::CUexecAffinityType passed to the API call is not
     * supported by the active device.
     */
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY      = 224,

    /**
     * This indicates that the code to be compiled by the PTX JIT contains
     * unsupported call to cudaDeviceSynchronize.
     */
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC       = 225,

    /**
     * This indicates that an exception occurred on the device that is now
     * contained by the GPU's error containment capability. Common causes are -
     * a. Certain types of invalid accesses of peer GPU memory over nvlink
     * b. Certain classes of hardware errors
     * This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must
     * be terminated and relaunched.
     */
    CUDA_ERROR_CONTAINED                      = 226,

    /**
     * This indicates that the device kernel source is invalid. This includes
     * compilation/linker errors encountered in device code or user error.
     */
    CUDA_ERROR_INVALID_SOURCE                 = 300,

    /**
     * This indicates that the file specified was not found.
     */
    CUDA_ERROR_FILE_NOT_FOUND                 = 301,

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

    /**
     * This indicates that initialization of a shared object failed.
     */
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

    /**
     * This indicates that an OS call failed.
     */
    CUDA_ERROR_OPERATING_SYSTEM               = 304,

    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    CUDA_ERROR_INVALID_HANDLE                 = 400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    CUDA_ERROR_ILLEGAL_STATE                  = 401,

    /**
     * This indicates an attempt was made to introspect an object in a way that
     * would discard semantically important information. This is either due to
     * the object using funtionality newer than the API version used to
     * introspect it or omission of optional return arguments.
     */
    CUDA_ERROR_LOSSY_QUERY                    = 402,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    CUDA_ERROR_NOT_FOUND                      = 500,

    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    CUDA_ERROR_NOT_READY                      = 600,

    /**
     * While executing a kernel, the device encountered a
     * load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    CUDA_ERROR_ILLEGAL_ADDRESS                = 700,

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,

    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is
     * trying to disable peer access which has not been enabled yet
     * via ::cuCtxEnablePeerAccess().
     */
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    CUDA_ERROR_ASSERT                         = 710,

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cuCtxEnablePeerAccess().
     */
    CUDA_ERROR_TOO_MANY_PEERS                 = 711,

    /**
     * This error indicates that the memory range passed to ::cuMemHostRegister()
     * has already been registered.
     */
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

    /**
     * This error indicates that the pointer passed to ::cuMemHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,

    /**
     * While executing a kernel, the device encountered a stack error.
     * This can be due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    CUDA_ERROR_HARDWARE_STACK_ERROR           = 714,

    /**
     * While executing a kernel, the device encountered an illegal instruction.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    CUDA_ERROR_ILLEGAL_INSTRUCTION            = 715,

    /**
     * While executing a kernel, the device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    CUDA_ERROR_MISALIGNED_ADDRESS             = 716,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    CUDA_ERROR_INVALID_ADDRESS_SPACE          = 717,

    /**
     * While executing a kernel, the device program counter wrapped its address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    CUDA_ERROR_INVALID_PC                     = 718,

    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    CUDA_ERROR_LAUNCH_FAILED                  = 719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
     */
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720,

    /**
     * An exception occurred on the device while exiting a kernel using tensor memory: the
     * tensor memory was not completely deallocated. This leaves the process in an inconsistent
     * state and any further CUDA work will return the same error. To continue using CUDA, the
     * process must be terminated and relaunched.
     */
    CUDA_ERROR_TENSOR_MEMORY_LEAK             = 721,

    /**
     * This error indicates that the attempted operation is not permitted.
     */
    CUDA_ERROR_NOT_PERMITTED                  = 800,

    /**
     * This error indicates that the attempted operation is not supported
     * on the current system or device.
     */
    CUDA_ERROR_NOT_SUPPORTED                  = 801,

    /**
     * This error indicates that the system is not yet ready to start any CUDA
     * work.  To continue using CUDA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    CUDA_ERROR_SYSTEM_NOT_READY               = 802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the CUDA driver. Refer to the compatibility documentation
     * for supported versions.
     */
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH         = 803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by CUDA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
     * environment variable.
     */
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    CUDA_ERROR_MPS_CONNECTION_FAILED          = 805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    CUDA_ERROR_MPS_RPC_FAILURE                = 806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    CUDA_ERROR_MPS_SERVER_NOT_READY           = 807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED        = 808,

    /**
     * This error indicates the the hardware resources required to support device connections have been exhausted.
     */
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED    = 809,

    /**
     * This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.
     */
    CUDA_ERROR_MPS_CLIENT_TERMINATED          = 810,

    /**
     * This error indicates that the module is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.
     */
    CUDA_ERROR_CDP_NOT_SUPPORTED              = 811,

    /**
     * This error indicates that a module contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.
     */
    CUDA_ERROR_CDP_VERSION_MISMATCH           = 812,

    /**
     * This error indicates that the operation is not permitted when
     * the stream is capturing.
     */
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED     = 900,

    /**
     * This error indicates that the current capture sequence on the stream
     * has been invalidated due to a previous error.
     */
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED     = 901,

    /**
     * This error indicates that the operation would have resulted in a merge
     * of two independent capture sequences.
     */
    CUDA_ERROR_STREAM_CAPTURE_MERGE           = 902,

    /**
     * This error indicates that the capture was not initiated in this stream.
     */
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED       = 903,

    /**
     * This error indicates that the capture sequence contains a fork that was
     * not joined to the primary stream.
     */
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED        = 904,

    /**
     * This error indicates that a dependency would have been created which
     * crosses the capture sequence boundary. Only implicit in-stream ordering
     * dependencies are allowed to cross the boundary.
     */
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION       = 905,

    /**
     * This error indicates a disallowed implicit dependency on a current capture
     * sequence from cudaStreamLegacy.
     */
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT        = 906,

    /**
     * This error indicates that the operation is not permitted on an event which
     * was last recorded in a capturing stream.
     */
    CUDA_ERROR_CAPTURED_EVENT                 = 907,

    /**
     * A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED
     * argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a
     * different thread.
     */
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD    = 908,

    /**
     * This error indicates that the timeout specified for the wait operation has lapsed.
     */
    CUDA_ERROR_TIMEOUT                        = 909,

    /**
     * This error indicates that the graph update was not performed because it included
     * changes which violated constraints specific to instantiated graph update.
     */
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE      = 910,

    /**
     * This indicates that an async error has occurred in a device outside of CUDA.
     * If CUDA was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must be
     * terminated and relaunched.
     */
    CUDA_ERROR_EXTERNAL_DEVICE               = 911,

    /**
     * Indicates a kernel launch error due to cluster misconfiguration.
     */
    CUDA_ERROR_INVALID_CLUSTER_SIZE           = 912,

    /**
     * Indiciates a function handle is not loaded when calling an API that requires
     * a loaded function.
    */
    CUDA_ERROR_FUNCTION_NOT_LOADED            = 913,

    /**
     * This error indicates one or more resources passed in are not valid resource
     * types for the operation.
    */
    CUDA_ERROR_INVALID_RESOURCE_TYPE          = 914,

    /**
     * This error indicates one or more resources are insufficient or non-applicable for
     * the operation.
    */
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,

    /**
     * This error indicates that an error happened during the key rotation
     * sequence.
    */
    CUDA_ERROR_KEY_ROTATION                   = 916,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    CUDA_ERROR_UNKNOWN                        = 999
} CUresult;

// callback type that needs cudaError_t (defined above)
using cudaStreamCallback_t = void(*)(cudaStream_t, cudaError_t, void*);

/**
 * Flags to indicate search status. For more details see ::cuGetProcAddress
 */
typedef enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS                = 0,  /**< Symbol was succesfully found */
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND       = 1,  /**< Symbol was not found in search */
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2   /**< Symbol was found but version supplied was not sufficient */
}  CUdriverProcAddressQueryResult;


/**
 * Device properties
 */
typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,                          /**< Maximum number of threads per block */
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,                                /**< Maximum block dimension X */
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,                                /**< Maximum block dimension Y */
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,                                /**< Maximum block dimension Z */
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,                                 /**< Maximum grid dimension X */
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,                                 /**< Maximum grid dimension Y */
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,                                 /**< Maximum grid dimension Z */
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,                    /**< Maximum shared memory available per block in bytes */
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,                        /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,                          /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,                                     /**< Warp size in threads */
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                                     /**< Maximum pitch in bytes allowed by memory copies */
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,                       /**< Maximum number of 32-bit registers available per block */
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,                           /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                                    /**< Typical clock frequency in kilohertz */
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                             /**< Alignment requirement for textures */
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                                   /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,                          /**< Number of multiprocessors on device */
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,                           /**< Specifies whether there is a run time limit on kernels */
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,                                    /**< Device is integrated with host memory */
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,                           /**< Device can map host memory into CUDA address space */
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                                  /**< Compute mode (See ::CUcomputemode for details) */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,                       /**< Maximum 1D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,                       /**< Maximum 2D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,                      /**< Maximum 2D texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,                       /**< Maximum 3D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,                      /**< Maximum 3D texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,                       /**< Maximum 3D texture depth */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,               /**< Maximum 2D layered texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,              /**< Maximum 2D layered texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,              /**< Maximum layers in a 2D layered texture */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,                 /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,                /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,             /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                             /**< Alignment requirement for surfaces */
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,                            /**< Device can possibly execute multiple kernels concurrently */
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                                   /**< Device has ECC support enabled */
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                                    /**< PCI bus ID of the device */
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                                 /**< PCI device ID of the device */
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                                    /**< Device is using TCC driver model */
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                             /**< Peak memory clock frequency in kilohertz */
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,                       /**< Global memory bus width in bits */
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                                 /**< Size of L2 cache in bytes */
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,                /**< Maximum resident threads per multiprocessor */
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                            /**< Number of asynchronous engines */
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                            /**< Device shares a unified address space with the host */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,               /**< Maximum 1D layered texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,              /**< Maximum layers in a 1D layered texture */
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,                              /**< Deprecated, do not use. */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,                /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,               /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,             /**< Alternate maximum 3D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,            /**< Alternate maximum 3D texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,             /**< Alternate maximum 3D texture depth */
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                                 /**< PCI domain ID of the device */
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,                       /**< Pitch alignment requirement for textures */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,                  /**< Maximum cubemap texture width/height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,          /**< Maximum cubemap layered texture width/height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,         /**< Maximum layers in a cubemap layered texture */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,                       /**< Maximum 1D surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,                       /**< Maximum 2D surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,                      /**< Maximum 2D surface height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,                       /**< Maximum 3D surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,                      /**< Maximum 3D surface height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,                       /**< Maximum 3D surface depth */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,               /**< Maximum 1D layered surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,              /**< Maximum layers in a 1D layered surface */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,               /**< Maximum 2D layered surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,              /**< Maximum 2D layered surface height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,              /**< Maximum layers in a 2D layered surface */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,                  /**< Maximum cubemap surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,          /**< Maximum cubemap layered surface width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,         /**< Maximum layers in a cubemap layered surface */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,                /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,                /**< Maximum 2D linear texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,               /**< Maximum 2D linear texture height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,                /**< Maximum 2D linear texture pitch in bytes */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,             /**< Maximum mipmapped 2D texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,            /**< Maximum mipmapped 2D texture height */
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,                      /**< Major compute capability version number */
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,                      /**< Minor compute capability version number */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,             /**< Maximum mipmapped 1D texture width */
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,                   /**< Device supports stream priorities */
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,                     /**< Device supports caching globals in L1 */
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,                      /**< Device supports caching locals in L1 */
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,          /**< Maximum shared memory available per multiprocessor in bytes */
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,              /**< Maximum number of 32-bit registers available per multiprocessor */
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,                                /**< Device can allocate managed memory on this system */
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,                               /**< Device is on a multi-GPU board */
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,                      /**< Unique id for a group of devices on the same multi-GPU board */
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,                  /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,         /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,                        /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,                     /**< Device can coherently access managed memory concurrently with the CPU */
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,                  /**< Device supports compute preemption. */
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,       /**< Device can access host registered memory at the same virtual address as the CPU */
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,                     /**< Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related APIs are supported. */
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,              /**< Deprecated, along with v1 MemOps API, 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. */
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,              /**< Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is supported. */
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,                            /**< Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel */
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,               /**< Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated. */
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,             /**< Maximum optin shared memory per block */
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,                       /**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details. */
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,                       /**< Device supports host memory registration via ::cudaHostRegister. */
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100, /**< Device accesses pageable memory via the host's page tables. */
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,          /**< The host can directly access managed memory on the device without migration. */
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,         /**< Deprecated, Use CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED*/
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,         /**< Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs */
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,  /**< Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,           /**< Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,       /**< Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,                /**< Maximum number of blocks per multiprocessor */
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,                /**< Device supports compression of memory */
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,                 /**< Maximum L2 persisting lines capacity setting in bytes. */
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,                /**< Maximum value of CUaccessPolicyWindow::num_bytes. */
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,      /**< Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate */
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,             /**< Shared memory reserved by CUDA driver per block in bytes */
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,                  /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,            /**< Device supports using the ::cuMemHostRegister flag ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU */
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,         /**< External timeline semaphore interop is supported on the device */
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,                       /**< Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs */
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,                    /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,         /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum */
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,              /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,               /**< Handle types supported with mempool based IPC */
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,                               /**< Indicates device supports cluster launch */
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,        /**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,                /**< 64-bit operations are supported in ::cuStreamBatchMemOp and related MemOp APIs. */
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,                /**< ::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs. */
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,                            /**< Device supports buffer sharing with dma_buf mechanism. */
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,                          /**< Device supports IPC Events. */
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,                        /**< Number of memory domains the device supports. */
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,                  /**< Device supports accessing memory using Tensor Map. */
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,                 /**< Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or requested with cuMemCreate() */
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,                    /**< Device supports unified function pointers. */
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,                                  /**< NUMA configuration of a device: value is of type ::CUdeviceNumaConfig enum */
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,                                      /**< NUMA node ID of the GPU memory */
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,                          /**< Device supports switch multicast and reduction operations. */
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,                                  /**< Indicates if contexts created on this device will be shared via MPS */
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,                                 /**< NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA. */
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,                          /**< Device supports CIG with D3D12. */
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK = 136,                /**< The returned valued shall be interpreted as a bitmask, where the individual bits are described by the ::CUmemDecompressAlgorithm enum. */
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH = 137,                /**< The returned valued is the maximum length in bytes of a single decompress operation that is allowed. */
    CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID    = 139, /**< The combined 16-bit PCI device ID and 16-bit PCI vendor ID. */
    CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID = 140, /**< The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID. */
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143,             /**< Device supports HOST_NUMA location IPC between nodes in a multi-node system. */
    CU_DEVICE_ATTRIBUTE_MAX
} CUdevice_attribute;

/**
 * CUDA device attributes
 */
enum cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
    cudaDevAttrMaxBlockDimX                   = 2,  /**< Maximum block dimension X */
    cudaDevAttrMaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
    cudaDevAttrMaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
    cudaDevAttrMaxGridDimX                    = 5,  /**< Maximum grid dimension X */
    cudaDevAttrMaxGridDimY                    = 6,  /**< Maximum grid dimension Y */
    cudaDevAttrMaxGridDimZ                    = 7,  /**< Maximum grid dimension Z */
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
    cudaDevAttrTotalConstantMemory            = 9,  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    cudaDevAttrWarpSize                       = 10, /**< Warp size in threads */
    cudaDevAttrMaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
    cudaDevAttrMaxRegistersPerBlock           = 12, /**< Maximum number of 32-bit registers available per block */
    cudaDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    cudaDevAttrTextureAlignment               = 14, /**< Alignment requirement for textures */
    cudaDevAttrGpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
    cudaDevAttrMultiProcessorCount            = 16, /**< Number of multiprocessors on device */
    cudaDevAttrKernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
    cudaDevAttrIntegrated                     = 18, /**< Device is integrated with host memory */
    cudaDevAttrCanMapHostMemory               = 19, /**< Device can map host memory into CUDA address space */
    cudaDevAttrComputeMode                    = 20, /**< Compute mode (See ::cudaComputeMode for details) */
    cudaDevAttrMaxTexture1DWidth              = 21, /**< Maximum 1D texture width */
    cudaDevAttrMaxTexture2DWidth              = 22, /**< Maximum 2D texture width */
    cudaDevAttrMaxTexture2DHeight             = 23, /**< Maximum 2D texture height */
    cudaDevAttrMaxTexture3DWidth              = 24, /**< Maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeight             = 25, /**< Maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepth              = 26, /**< Maximum 3D texture depth */
    cudaDevAttrMaxTexture2DLayeredWidth       = 27, /**< Maximum 2D layered texture width */
    cudaDevAttrMaxTexture2DLayeredHeight      = 28, /**< Maximum 2D layered texture height */
    cudaDevAttrMaxTexture2DLayeredLayers      = 29, /**< Maximum layers in a 2D layered texture */
    cudaDevAttrSurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
    cudaDevAttrConcurrentKernels              = 31, /**< Device can possibly execute multiple kernels concurrently */
    cudaDevAttrEccEnabled                     = 32, /**< Device has ECC support enabled */
    cudaDevAttrPciBusId                       = 33, /**< PCI bus ID of the device */
    cudaDevAttrPciDeviceId                    = 34, /**< PCI device ID of the device */
    cudaDevAttrTccDriver                      = 35, /**< Device is using TCC driver model */
    cudaDevAttrMemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
    cudaDevAttrGlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
    cudaDevAttrL2CacheSize                    = 38, /**< Size of L2 cache in bytes */
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
    cudaDevAttrAsyncEngineCount               = 40, /**< Number of asynchronous engines */
    cudaDevAttrUnifiedAddressing              = 41, /**< Device shares a unified address space with the host */
    cudaDevAttrMaxTexture1DLayeredWidth       = 42, /**< Maximum 1D layered texture width */
    cudaDevAttrMaxTexture1DLayeredLayers      = 43, /**< Maximum layers in a 1D layered texture */
    cudaDevAttrMaxTexture2DGatherWidth        = 45, /**< Maximum 2D texture width if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherHeight       = 46, /**< Maximum 2D texture height if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture3DWidthAlt           = 47, /**< Alternate maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeightAlt          = 48, /**< Alternate maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepthAlt           = 49, /**< Alternate maximum 3D texture depth */
    cudaDevAttrPciDomainId                    = 50, /**< PCI domain ID of the device */
    cudaDevAttrTexturePitchAlignment          = 51, /**< Pitch alignment requirement for textures */
    cudaDevAttrMaxTextureCubemapWidth         = 52, /**< Maximum cubemap texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53, /**< Maximum cubemap layered texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
    cudaDevAttrMaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
    cudaDevAttrMaxSurface2DWidth              = 56, /**< Maximum 2D surface width */
    cudaDevAttrMaxSurface2DHeight             = 57, /**< Maximum 2D surface height */
    cudaDevAttrMaxSurface3DWidth              = 58, /**< Maximum 3D surface width */
    cudaDevAttrMaxSurface3DHeight             = 59, /**< Maximum 3D surface height */
    cudaDevAttrMaxSurface3DDepth              = 60, /**< Maximum 3D surface depth */
    cudaDevAttrMaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
    cudaDevAttrMaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
    cudaDevAttrMaxSurface2DLayeredWidth       = 63, /**< Maximum 2D layered surface width */
    cudaDevAttrMaxSurface2DLayeredHeight      = 64, /**< Maximum 2D layered surface height */
    cudaDevAttrMaxSurface2DLayeredLayers      = 65, /**< Maximum layers in a 2D layered surface */
    cudaDevAttrMaxSurfaceCubemapWidth         = 66, /**< Maximum cubemap surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, /**< Maximum cubemap layered surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
    cudaDevAttrMaxTexture1DLinearWidth        = 69, /**< Maximum 1D linear texture width */
    cudaDevAttrMaxTexture2DLinearWidth        = 70, /**< Maximum 2D linear texture width */
    cudaDevAttrMaxTexture2DLinearHeight       = 71, /**< Maximum 2D linear texture height */
    cudaDevAttrMaxTexture2DLinearPitch        = 72, /**< Maximum 2D linear texture pitch in bytes */
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73, /**< Maximum mipmapped 2D texture width */
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74, /**< Maximum mipmapped 2D texture height */
    cudaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */
    cudaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77, /**< Maximum mipmapped 1D texture width */
    cudaDevAttrStreamPrioritiesSupported      = 78, /**< Device supports stream priorities */
    cudaDevAttrGlobalL1CacheSupported         = 79, /**< Device supports caching globals in L1 */
    cudaDevAttrLocalL1CacheSupported          = 80, /**< Device supports caching locals in L1 */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
    cudaDevAttrManagedMemory                  = 83, /**< Device can allocate managed memory on this system */
    cudaDevAttrIsMultiGpuBoard                = 84, /**< Device is on a multi-GPU board */
    cudaDevAttrMultiGpuBoardGroupID           = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
    cudaDevAttrHostNativeAtomicSupported      = 86, /**< Link between the device and the host supports native atomic operations */
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    cudaDevAttrPageableMemoryAccess           = 88, /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    cudaDevAttrConcurrentManagedAccess        = 89, /**< Device can coherently access managed memory concurrently with the CPU */
    cudaDevAttrComputePreemptionSupported     = 90, /**< Device supports Compute Preemption */
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    cudaDevAttrReserved92                     = 92,
    cudaDevAttrReserved93                     = 93,
    cudaDevAttrReserved94                     = 94,
    cudaDevAttrCooperativeLaunch              = 95, /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel*/
    cudaDevAttrCooperativeMultiDeviceLaunch   = 96, /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97, /**< The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute */
    cudaDevAttrCanFlushRemoteWrites           = 98, /**< Device supports flushing of outstanding remote writes. */
    cudaDevAttrHostRegisterSupported          = 99, /**< Device supports host memory registration via ::cudaHostRegister. */
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100, /**< Device accesses pageable memory via the host's page tables. */
    cudaDevAttrDirectManagedMemAccessFromHost = 101, /**< Host can directly access managed memory on the device without migration. */
    cudaDevAttrMaxBlocksPerMultiprocessor     = 106, /**< Maximum number of blocks per multiprocessor */
    cudaDevAttrMaxPersistingL2CacheSize       = 108, /**< Maximum L2 persisting lines capacity setting in bytes. */
    cudaDevAttrMaxAccessPolicyWindowSize      = 109, /**< Maximum value of cudaAccessPolicyWindow::num_bytes. */
    cudaDevAttrReservedSharedMemoryPerBlock   = 111, /**< Shared memory reserved by CUDA driver per block in bytes */
    cudaDevAttrSparseCudaArraySupported       = 112, /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
    cudaDevAttrHostRegisterReadOnlySupported  = 113,  /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,  /**< External timeline semaphore interop is supported on the device */
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114,  /**< Deprecated, External timeline semaphore interop is supported on the device */
    cudaDevAttrMemoryPoolsSupported           = 115, /**< Device supports using the ::cudaMallocAsync and ::cudaMemPool family of APIs */
    cudaDevAttrGPUDirectRDMASupported         = 116, /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117, /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    cudaDevAttrGPUDirectRDMAWritesOrdering    = 118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::cudaGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119, /**< Handle types supported with mempool based IPC */
    cudaDevAttrClusterLaunch                  = 120, /**< Indicates device supports cluster launch */
    cudaDevAttrDeferredMappingCudaArraySupported = 121, /**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    cudaDevAttrReserved122                    = 122,
    cudaDevAttrReserved123                    = 123,
    cudaDevAttrReserved124                    = 124,
    cudaDevAttrIpcEventSupport                = 125, /**< Device supports IPC Events. */
    cudaDevAttrMemSyncDomainCount             = 126, /**< Number of memory synchronization domains the device supports. */
    cudaDevAttrReserved127                    = 127,
    cudaDevAttrReserved128                    = 128,
    cudaDevAttrReserved129                    = 129,
    cudaDevAttrNumaConfig                     = 130, /**< NUMA configuration of a device: value is of type ::cudaDeviceNumaConfig enum */
    cudaDevAttrNumaId                         = 131, /**< NUMA node ID of the GPU memory */
    cudaDevAttrReserved132                    = 132,
    cudaDevAttrMpsEnabled                     = 133, /**< Contexts created on this device will be shared via MPS */
    cudaDevAttrHostNumaId                     = 134, /**< NUMA ID of the host node closest to the device or -1 when system does not support NUMA */
    cudaDevAttrD3D12CigSupported              = 135, /**< Device supports CIG with D3D12. */
    cudaDevAttrGpuPciDeviceId                 = 139, /**< The combined 16-bit PCI device ID and 16-bit PCI vendor ID. */
    cudaDevAttrGpuPciSubsystemId              = 140, /**< The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID. */
    cudaDevAttrHostNumaMultinodeIpcSupported  = 143, /**< Device supports HostNuma location IPC between nodes in a multi-node system. */
    cudaDevAttrMax
};

using cudaUUID_t = CUuuid_st;
struct cudaDeviceProp
{
    char         name[256];                  /**< ASCII string identifying device */
    cudaUUID_t   uuid;                       /**< 16-byte unique identifier */
    char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int          regsPerBlock;               /**< 32-bit registers available per block */
    int          warpSize;                   /**< Warp size in threads */
    size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int          clockRate;                  /**< Deprecated, Clock frequency in kilohertz */
    size_t       totalConstMem;              /**< Constant memory available on device in bytes */
    int          major;                      /**< Major compute capability */
    int          minor;                      /**< Minor compute capability */
    size_t       textureAlignment;           /**< Alignment requirement for textures */
    size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int          deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int          multiProcessorCount;        /**< Number of multiprocessors on device */
    int          kernelExecTimeoutEnabled;   /**< Deprecated, Specified whether there is a run time limit on kernels */
    int          integrated;                 /**< Device is integrated as opposed to discrete */
    int          canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int          computeMode;                /**< Deprecated, Compute mode (See ::cudaComputeMode) */
    int          maxTexture1D;               /**< Maximum 1D texture size */
    int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int          maxTexture1DLinear;         /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int          maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int          maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int          maxSurface1D;               /**< Maximum 1D surface size */
    int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int          maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int          maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
    int          concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int          ECCEnabled;                 /**< Device has ECC support enabled */
    int          pciBusID;                   /**< PCI bus ID of the device */
    int          pciDeviceID;                /**< PCI device ID of the device */
    int          pciDomainID;                /**< PCI domain ID of the device */
    int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int          asyncEngineCount;           /**< Number of asynchronous engines */
    int          unifiedAddressing;          /**< Device shares a unified address space with the host */
    int          memoryClockRate;            /**< Deprecated, Peak memory clock frequency in kilohertz */
    int          memoryBusWidth;             /**< Global memory bus width in bits */
    int          l2CacheSize;                /**< Size of L2 cache in bytes */
    int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int          streamPrioritiesSupported;  /**< Device supports stream priorities */
    int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int          managedMemory;              /**< Device supports allocating managed memory on this system */
    int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int          singleToDoublePrecisionPerfRatio; /**< Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int          concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
    int          computePreemptionSupported; /**< Device supports Compute Preemption */
    int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    int          cooperativeMultiDeviceLaunch; /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
    int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
    int          accessPolicyMaxWindowSize;  /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
    size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */
    int          hostRegisterSupported;      /**< Device supports host memory registration via ::cudaHostRegister. */
    int          sparseCudaArraySupported;   /**< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise */
    int          hostRegisterReadOnlySupported; /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    int          timelineSemaphoreInteropSupported; /**< External timeline semaphore interop is supported on the device */
    int          memoryPoolsSupported;       /**< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise */
    int          gpuDirectRDMASupported;     /**< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
    unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    int          gpuDirectRDMAWritesOrdering;/**< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */
    unsigned int memoryPoolSupportedHandleTypes; /**< Bitmask of handle types supported with mempool-based IPC */
    int          deferredMappingCudaArraySupported; /**< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    int          ipcEventSupported;          /**< Device supports IPC Events. */
    int          clusterLaunch;              /**< Indicates device supports cluster launch */
    int          unifiedFunctionPointers;    /**< Indicates device supports unified pointers */
    int          reserved[63];               /**< Reserved for future use */
};

typedef enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO          = 0x00, /**< Automatic scheduling */
    CU_CTX_SCHED_SPIN          = 0x01, /**< Set spin as default scheduling */
    CU_CTX_SCHED_YIELD         = 0x02, /**< Set yield as default scheduling */
    CU_CTX_SCHED_BLOCKING_SYNC = 0x04, /**< Set blocking synchronization as default scheduling */
    CU_CTX_BLOCKING_SYNC       = 0x04, /**< Set blocking synchronization as default scheduling
                                         *  \deprecated This flag was deprecated as of CUDA 4.0
                                         *  and was replaced with ::CU_CTX_SCHED_BLOCKING_SYNC. */
    CU_CTX_SCHED_MASK          = 0x07,
    CU_CTX_MAP_HOST            = 0x08, /**< \deprecated This flag was deprecated as of CUDA 11.0
                                         *  and it no longer has any effect. All contexts
                                         *  as of CUDA 3.2 behave as though the flag is enabled. */
    CU_CTX_LMEM_RESIZE_TO_MAX  = 0x10, /**< Keep local memory allocation after launch */
    CU_CTX_COREDUMP_ENABLE     = 0x20, /**< Trigger coredumps from exceptions in this context */
    CU_CTX_USER_COREDUMP_ENABLE= 0x40, /**< Enable user pipe to trigger coredumps in this context */
    CU_CTX_SYNC_MEMOPS         = 0x80, /**< Ensure synchronous memory operations on this context will synchronize */
    CU_CTX_FLAGS_MASK          = 0xFF
} CUctx_flags;

enum cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess             = 0,  /**< Search for symbol found a match */
    cudaDriverEntryPointSymbolNotFound      = 1,  /**< Search for symbol was not found */
    cudaDriverEntryPointVersionNotSufficent = 2   /**< Search for symbol was found but version wasn't great enough */
};

typedef enum cudaDataType_t
{
    CUDA_R_16F  =  2, /* real as a half */
    CUDA_C_16F  =  6, /* complex as a pair of half numbers */
    CUDA_R_16BF = 14, /* real as a nv_bfloat16 */
    CUDA_C_16BF = 15, /* complex as a pair of nv_bfloat16 numbers */
    CUDA_R_32F  =  0, /* real as a float */
    CUDA_C_32F  =  4, /* complex as a pair of float numbers */
    CUDA_R_64F  =  1, /* real as a double */
    CUDA_C_64F  =  5, /* complex as a pair of double numbers */
    CUDA_R_4I   = 16, /* real as a signed 4-bit int */
    CUDA_C_4I   = 17, /* complex as a pair of signed 4-bit int numbers */
    CUDA_R_4U   = 18, /* real as a unsigned 4-bit int */
    CUDA_C_4U   = 19, /* complex as a pair of unsigned 4-bit int numbers */
    CUDA_R_8I   =  3, /* real as a signed 8-bit int */
    CUDA_C_8I   =  7, /* complex as a pair of signed 8-bit int numbers */
    CUDA_R_8U   =  8, /* real as a unsigned 8-bit int */
    CUDA_C_8U   =  9, /* complex as a pair of unsigned 8-bit int numbers */
    CUDA_R_16I  = 20, /* real as a signed 16-bit int */
    CUDA_C_16I  = 21, /* complex as a pair of signed 16-bit int numbers */
    CUDA_R_16U  = 22, /* real as a unsigned 16-bit int */
    CUDA_C_16U  = 23, /* complex as a pair of unsigned 16-bit int numbers */
    CUDA_R_32I  = 10, /* real as a signed 32-bit int */
    CUDA_C_32I  = 11, /* complex as a pair of signed 32-bit int numbers */
    CUDA_R_32U  = 12, /* real as a unsigned 32-bit int */
    CUDA_C_32U  = 13, /* complex as a pair of unsigned 32-bit int numbers */
    CUDA_R_64I  = 24, /* real as a signed 64-bit int */
    CUDA_C_64I  = 25, /* complex as a pair of signed 64-bit int numbers */
    CUDA_R_64U  = 26, /* real as a unsigned 64-bit int */
    CUDA_C_64U  = 27, /* complex as a pair of unsigned 64-bit int numbers */
    CUDA_R_8F_E4M3 = 28, /* real as a nv_fp8_e4m3 */
    CUDA_R_8F_UE4M3 = CUDA_R_8F_E4M3, /* real as an unsigned nv_fp8_e4m3 */
    CUDA_R_8F_E5M2 = 29, /* real as a nv_fp8_e5m2 */
    CUDA_R_8F_UE8M0 = 30,  /* real as an exponent-only unsigned nv_fp8_e8m0 */
    CUDA_R_6F_E2M3  = 31,  /* real as a nv_fp6_e2m3 */
    CUDA_R_6F_E3M2  = 32,  /* real as a nv_fp6_e3m2 */
    CUDA_R_4F_E2M1  = 33,  /* real as a nv_fp4_e2m1 */
} cudaDataType;

/**
 * CUDA memory copy types
 */
enum cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

/**
 * Possible modes for stream capture thread interactions. For more details see
 * ::cudaStreamBeginCapture and ::cudaThreadExchangeStreamCaptureMode
 */
enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal      = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed     = 2
};

/**
 * Possible stream capture statuses returned by ::cudaStreamIsCapturing
 */
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone        = 0, /**< Stream is not capturing */
    cudaStreamCaptureStatusActive      = 1, /**< Stream is actively capturing */
    cudaStreamCaptureStatusInvalidated = 2  /**< Stream is part of a capture sequence that has been invalidated */
};

/**
 * CUDA memory pool attributes
 */
enum cudaMemPoolAttr
{
    /**
     * (value type = int)
     * Allow cuMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * Cuda events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    cudaMemPoolReuseFollowEventDependencies   = 0x1,

    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    cudaMemPoolReuseAllowOpportunistic        = 0x2,

    /**
     * (value type = int)
     * Allow cuMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     */
    cudaMemPoolReuseAllowInternalDependencies = 0x3,


    /**
     * (value type = cuuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    cudaMemPoolAttrReleaseThreshold           = 0x4,

    /**
     * (value type = cuuint64_t)
     * Amount of backing memory currently allocated for the mempool.
     */
    cudaMemPoolAttrReservedMemCurrent         = 0x5,

    /**
     * (value type = cuuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    cudaMemPoolAttrReservedMemHigh            = 0x6,

    /**
     * (value type = cuuint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     */
    cudaMemPoolAttrUsedMemCurrent             = 0x7,

    /**
     * (value type = cuuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    cudaMemPoolAttrUsedMemHigh                = 0x8
};

/**
 * Launch attributes enum; used as id field of ::cudaLaunchAttribute
 */
typedef enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore                = 0 /**< Ignored entry, for convenient composition */
  , cudaLaunchAttributeAccessPolicyWindow    = 1 /**< Valid for streams, graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::accessPolicyWindow. */
  , cudaLaunchAttributeCooperative           = 2 /**< Valid for graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::cooperative. */
  , cudaLaunchAttributeSynchronizationPolicy = 3 /**< Valid for streams. See ::cudaLaunchAttributeValue::syncPolicy. */
  , cudaLaunchAttributeClusterDimension                  = 4 /**< Valid for graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::clusterDim. */
  , cudaLaunchAttributeClusterSchedulingPolicyPreference = 5 /**< Valid for graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::clusterSchedulingPolicyPreference. */
  , cudaLaunchAttributeProgrammaticStreamSerialization   = 6 /**< Valid for launches. Setting
                                                                  ::cudaLaunchAttributeValue::programmaticStreamSerializationAllowed
                                                                  to non-0 signals that the kernel will use programmatic
                                                                  means to resolve its stream dependency, so that the
                                                                  CUDA runtime should opportunistically allow the grid's
                                                                  execution to overlap with the previous kernel in the
                                                                  stream, if that kernel requests the overlap. The
                                                                  dependent launches can choose to wait on the
                                                                  dependency using the programmatic sync
                                                                  (cudaGridDependencySynchronize() or equivalent PTX
                                                                  instructions). */
  , cudaLaunchAttributeProgrammaticEvent                 = 7 /**< Valid for launches. Set
                                                                  ::cudaLaunchAttributeValue::programmaticEvent to
                                                                  record the event. Event recorded through this launch
                                                                  attribute is guaranteed to only trigger after all
                                                                  block in the associated kernel trigger the event.  A
                                                                  block can trigger the event programmatically in a
                                                                  future CUDA release. A trigger can also be inserted at
                                                                  the beginning of each block's execution if
                                                                  triggerAtBlockStart is set to non-0. The dependent
                                                                  launches can choose to wait on the dependency using
                                                                  the programmatic sync (cudaGridDependencySynchronize()
                                                                  or equivalent PTX instructions). Note that dependents
                                                                  (including the CPU thread calling
                                                                  cudaEventSynchronize()) are not guaranteed to observe
                                                                  the release precisely when it is released. For
                                                                  example, cudaEventSynchronize() may only observe the
                                                                  event trigger long after the associated kernel has
                                                                  completed. This recording type is primarily meant for
                                                                  establishing programmatic dependency between device
                                                                  tasks. Note also this type of dependency allows, but
                                                                  does not guarantee, concurrent execution of tasks.
                                                                  <br>
                                                                  The event supplied must not be an interprocess or
                                                                  interop event. The event must disable timing (i.e.
                                                                  must be created with the ::cudaEventDisableTiming flag
                                                                  set). */
  , cudaLaunchAttributePriority              = 8 /**< Valid for streams, graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::priority. */
  , cudaLaunchAttributeMemSyncDomainMap                  = 9 /**< Valid for streams, graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::memSyncDomainMap. */
  , cudaLaunchAttributeMemSyncDomain                    = 10 /**< Valid for streams, graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::memSyncDomain. */
  , cudaLaunchAttributePreferredClusterDimension = 11 /**< Valid for graph nodes and launches. Set
                                                           ::cudaLaunchAttributeValue::preferredClusterDim
                                                           to allow the kernel launch to specify a preferred substitute
                                                           cluster dimension. Blocks may be grouped according to either
                                                           the dimensions specified with this attribute (grouped into a
                                                           "preferred substitute cluster"), or the one specified with
                                                           ::cudaLaunchAttributeClusterDimension attribute (grouped
                                                           into a "regular cluster"). The cluster dimensions of a
                                                           "preferred substitute cluster" shall be an integer multiple
                                                           greater than zero of the regular cluster dimensions. The
                                                           device will attempt - on a best-effort basis - to group
                                                           thread blocks into preferred clusters over grouping them
                                                           into regular clusters. When it deems necessary (primarily
                                                           when the device temporarily runs out of physical resources
                                                           to launch the larger preferred clusters), the device may
                                                           switch to launch the regular clusters instead to attempt to
                                                           utilize as much of the physical device resources as possible.
                                                           <br>
                                                           Each type of cluster will have its enumeration / coordinate
                                                           setup as if the grid consists solely of its type of cluster.
                                                           For example, if the preferred substitute cluster dimensions
                                                           double the regular cluster dimensions, there might be
                                                           simultaneously a regular cluster indexed at (1,0,0), and a
                                                           preferred cluster indexed at (1,0,0). In this example, the
                                                           preferred substitute cluster (1,0,0) replaces regular
                                                           clusters (2,0,0) and (3,0,0) and groups their blocks.
                                                           <br>
                                                           This attribute will only take effect when a regular cluster
                                                           dimension has been specified. The preferred substitute cluster
                                                           dimension must be an integer multiple greater than zero of the
                                                           regular cluster dimension and must divide the grid. It must
                                                           also be no more than `maxBlocksPerCluster`, if it is set in
                                                           the kernel's `__launch_bounds__`. Otherwise it must be less
                                                           than the maximum value the driver can support. Otherwise,
                                                           setting this attribute to a value physically unable to fit on
                                                           any particular device is permitted. */
  , cudaLaunchAttributeLaunchCompletionEvent = 12 /**< Valid for launches. Set
                                                       ::cudaLaunchAttributeValue::launchCompletionEvent to record the
                                                       event.
                                                       <br>
                                                       Nominally, the event is triggered once all blocks of the kernel
                                                       have begun execution. Currently this is a best effort. If a kernel
                                                       B has a launch completion dependency on a kernel A, B may wait
                                                       until A is complete. Alternatively, blocks of B may begin before
                                                       all blocks of A have begun, for example if B can claim execution
                                                       resources unavailable to A (e.g. they run on different GPUs) or
                                                       if B is a higher priority than A.
                                                       Exercise caution if such an ordering inversion could lead
                                                       to deadlock.
                                                       <br>
                                                       A launch completion event is nominally similar to a programmatic
                                                       event with \c triggerAtBlockStart set except that it is not
                                                       visible to \c cudaGridDependencySynchronize() and can be used with
                                                       compute capability less than 9.0.
                                                       <br>
                                                       The event supplied must not be an interprocess or interop event.
                                                       The event must disable timing (i.e. must be created with the
                                                       ::cudaEventDisableTiming flag set). */
  , cudaLaunchAttributeDeviceUpdatableKernelNode = 13 /**< Valid for graph nodes, launches. This attribute is graphs-only,
                                                           and passing it to a launch in a non-capturing stream will result
                                                           in an error.
                                                           <br>
                                                           :cudaLaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable can
                                                           only be set to 0 or 1. Setting the field to 1 indicates that the
                                                           corresponding kernel node should be device-updatable. On success, a handle
                                                           will be returned via
                                                           ::cudaLaunchAttributeValue::deviceUpdatableKernelNode::devNode which can be
                                                           passed to the various device-side update functions to update the node's
                                                           kernel parameters from within another kernel. For more information on the
                                                           types of device updates that can be made, as well as the relevant limitations
                                                           thereof, see ::cudaGraphKernelNodeUpdatesApply.
                                                           <br>
                                                           Nodes which are device-updatable have additional restrictions compared to
                                                           regular kernel nodes. Firstly, device-updatable nodes cannot be removed
                                                           from their graph via ::cudaGraphDestroyNode. Additionally, once opted-in
                                                           to this functionality, a node cannot opt out, and any attempt to set the
                                                           deviceUpdatable attribute to 0 will result in an error. Device-updatable
                                                           kernel nodes also cannot have their attributes copied to/from another kernel
                                                           node via ::cudaGraphKernelNodeCopyAttributes. Graphs containing one or more
                                                           device-updatable nodes also do not allow multiple instantiation, and neither
                                                           the graph nor its instantiated version can be passed to ::cudaGraphExecUpdate.
                                                           <br>
                                                           If a graph contains device-updatable nodes and updates those nodes from the device
                                                           from within the graph, the graph must be uploaded with ::cuGraphUpload before it
                                                           is launched. For such a graph, if host-side executable graph updates are made to the
                                                           device-updatable nodes, the graph must be uploaded before it is launched again. */
  , cudaLaunchAttributePreferredSharedMemoryCarveout = 14 /**< Valid for launches. On devices where the L1 cache and shared memory use the
                                                               same hardware resources, setting ::cudaLaunchAttributeValue::sharedMemCarveout
                                                               to a percentage between 0-100 signals sets the shared memory carveout
                                                               preference in percent of the total shared memory for that kernel launch.
                                                               This attribute takes precedence over ::cudaFuncAttributePreferredSharedMemoryCarveout.
                                                               This is only a hint, and the driver can choose a different configuration if
                                                               required for the launch.*/
} cudaLaunchAttributeID;

/**
 * Specifies performance hint with ::cudaAccessPolicyWindow for hitProp and missProp members.
 */
enum cudaAccessProperty {
    cudaAccessPropertyNormal = 0,       /**< Normal cache persistence. */
    cudaAccessPropertyStreaming = 1,    /**< Streaming access is less likely to persit from cache. */
    cudaAccessPropertyPersisting = 2    /**< Persisting access is more likely to persist in cache.*/
};

/**
 * Specifies an access policy for a window, a contiguous extent of memory
 * beginning at base_ptr and ending at base_ptr + num_bytes.
 * Partition into many segments and assign segments such that.
 * sum of "hit segments" / window == approx. ratio.
 * sum of "miss segments" / window == approx 1-ratio.
 * Segments and ratio specifications are fitted to the capabilities of
 * the architecture.
 * Accesses in a hit segment apply the hitProp access policy.
 * Accesses in a miss segment apply the missProp access policy.
 */
struct cudaAccessPolicyWindow {
    void *base_ptr;                     /**< Starting address of the access policy window. CUDA driver may align it. */
    size_t num_bytes;                   /**< Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment. */
    float hitRatio;                     /**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. */
    enum cudaAccessProperty hitProp;    /**< ::CUaccessProperty set for hit. */
    enum cudaAccessProperty missProp;   /**< ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING. */
};

enum cudaSynchronizationPolicy {
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4
};

/**
 * Cluster scheduling policies. These may be passed to ::cudaFuncSetAttribute
 */
enum cudaClusterSchedulingPolicy {
    cudaClusterSchedulingPolicyDefault       = 0, /**< the default policy */
    cudaClusterSchedulingPolicySpread        = 1, /**< spread the blocks within a cluster to the SMs */
    cudaClusterSchedulingPolicyLoadBalancing = 2  /**< allow the hardware to load-balance the blocks in a cluster to the SMs */
};

/**
 * Memory Synchronization Domain map
 *
 * See ::cudaLaunchMemSyncDomain.
 *
 * By default, kernels are launched in domain 0. Kernel launched with ::cudaLaunchMemSyncDomainRemote will have a
 * different domain ID. User may also alter the domain ID with ::cudaLaunchMemSyncDomainMap for a specific stream /
 * graph node / kernel launch. See ::cudaLaunchAttributeMemSyncDomainMap.
 *
 * Domain ID range is available through ::cudaDevAttrMemSyncDomainCount.
 */
typedef struct cudaLaunchMemSyncDomainMap_st {
    unsigned char default_;                /**< The default domain ID to use for designated kernels */
    unsigned char remote;                  /**< The remote domain ID to use for designated kernels */
} cudaLaunchMemSyncDomainMap;

/**
 * Memory Synchronization Domain
 *
 * A kernel can be launched in a specified memory synchronization domain that affects all memory operations issued by
 * that kernel. A memory barrier issued in one domain will only order memory operations in that domain, thus eliminating
 * latency increase from memory barriers ordering unrelated traffic.
 *
 * By default, kernels are launched in domain 0. Kernel launched with ::cudaLaunchMemSyncDomainRemote will have a
 * different domain ID. User may also alter the domain ID with ::cudaLaunchMemSyncDomainMap for a specific stream /
 * graph node / kernel launch. See ::cudaLaunchAttributeMemSyncDomain, ::cudaStreamSetAttribute, ::cudaLaunchKernelEx,
 * ::cudaGraphKernelNodeSetAttribute.
 *
 * Memory operations done in kernels launched in different domains are considered system-scope distanced. In other
 * words, a GPU scoped memory synchronization is not sufficient for memory order to be observed by kernels in another
 * memory synchronization domain even if they are on the same GPU.
 */
typedef enum cudaLaunchMemSyncDomain {
    cudaLaunchMemSyncDomainDefault = 0,    /**< Launch kernels in the default domain */
    cudaLaunchMemSyncDomainRemote  = 1     /**< Launch kernels in the remote domain */
} cudaLaunchMemSyncDomain;

/**
 * Launch attributes union; used as value field of ::cudaLaunchAttribute
 */
typedef union cudaLaunchAttributeValue {
    char pad[64]; /* Pad to 64 bytes */
    struct cudaAccessPolicyWindow accessPolicyWindow; /**< Value of launch attribute ::cudaLaunchAttributeAccessPolicyWindow. */
    int cooperative; /**< Value of launch attribute ::cudaLaunchAttributeCooperative. Nonzero indicates a cooperative
                        kernel (see ::cudaLaunchCooperativeKernel). */
    enum cudaSynchronizationPolicy syncPolicy; /**< Value of launch attribute
                                                  ::cudaLaunchAttributeSynchronizationPolicy. ::cudaSynchronizationPolicy
                                                  for work queued up in this stream. */
    /**
     * Value of launch attribute ::cudaLaunchAttributeClusterDimension that
     * represents the desired cluster dimensions for the kernel. Opaque type
     * with the following fields:
     *     - \p x - The X dimension of the cluster, in blocks. Must be a divisor
     *              of the grid X dimension.
     *     - \p y - The Y dimension of the cluster, in blocks. Must be a divisor
     *              of the grid Y dimension.
     *     - \p z - The Z dimension of the cluster, in blocks. Must be a divisor
     *              of the grid Z dimension.
     */
    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    } clusterDim;
    enum cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; /**< Value of launch attribute
                                                                           ::cudaLaunchAttributeClusterSchedulingPolicyPreference. Cluster
                                                                           scheduling policy preference for the kernel. */
    int programmaticStreamSerializationAllowed; /**< Value of launch attribute
                                                   ::cudaLaunchAttributeProgrammaticStreamSerialization. */

    /**
     * Value of launch attribute ::cudaLaunchAttributeProgrammaticEvent
     * with the following fields:
     *     - \p cudaEvent_t event - Event to fire when all blocks trigger it.
     *     - \p int flags;        - Event record flags, see ::cudaEventRecordWithFlags. Does not accept
     *                               ::cudaEventRecordExternal.
     *     - \p int triggerAtBlockStart - If this is set to non-0, each block launch will automatically trigger the event.
     */
    struct {
        cudaEvent_t event;
        int flags;
        int triggerAtBlockStart;
    } programmaticEvent;
    int priority; /**< Value of launch attribute ::cudaLaunchAttributePriority. Execution priority of the kernel. */
    cudaLaunchMemSyncDomainMap memSyncDomainMap; /**< Value of launch attribute
                                                    ::cudaLaunchAttributeMemSyncDomainMap. See
                                                    ::cudaLaunchMemSyncDomainMap. */
    cudaLaunchMemSyncDomain memSyncDomain;       /**< Value of launch attribute ::cudaLaunchAttributeMemSyncDomain. See
                                                    ::cudaLaunchMemSyncDomain. */
    /**
     * Value of launch attribute ::cudaLaunchAttributePreferredClusterDimension
     * that represents the desired preferred cluster dimensions for the kernel.
     * Opaque type with the following fields:
     *     - \p x - The X dimension of the preferred cluster, in blocks. Must be
     *              a divisor of the grid X dimension, and must be a multiple of
     *              the \p x field of ::cudaLaunchAttributeValue::clusterDim.
     *     - \p y - The Y dimension of the preferred cluster, in blocks. Must be
     *              a divisor of the grid Y dimension, and must be a multiple of
     *              the \p y field of ::cudaLaunchAttributeValue::clusterDim.
     *     - \p z - The Z dimension of the preferred cluster, in blocks. Must be
     *              equal to the \p z field of ::cudaLaunchAttributeValue::clusterDim.
     */
    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    } preferredClusterDim;

    /**
     * Value of launch attribute ::cudaLaunchAttributeLaunchCompletionEvent
     * with the following fields:
     *     - \p cudaEvent_t event - Event to fire when the last block launches.
     *     - \p int flags - Event record flags, see ::cudaEventRecordWithFlags. Does not accept
     *                   ::cudaEventRecordExternal.
     */
    struct {
        cudaEvent_t event;
        int flags;
    } launchCompletionEvent;

    /**
     * Value of launch attribute ::cudaLaunchAttributeDeviceUpdatableKernelNode
     * with the following fields:
     *    - \p int deviceUpdatable - Whether or not the resulting kernel node should be device-updatable.
     *    - \p cudaGraphDeviceNode_t devNode - Returns a handle to pass to the various device-side update functions.
     */
    struct {
        int deviceUpdatable;
        cudaGraphDeviceNode_t devNode;
    } deviceUpdatableKernelNode;
    unsigned int sharedMemCarveout; /**< Value of launch attribute ::cudaLaunchAttributePreferredSharedMemoryCarveout. */
} cudaLaunchAttributeValue;

/**
 * Launch attribute
 */
typedef struct cudaLaunchAttribute_st {
    cudaLaunchAttributeID id; /**< Attribute to set */
    char pad[8 - sizeof(cudaLaunchAttributeID)];
    cudaLaunchAttributeValue val; /**< Value of the attribute */
} cudaLaunchAttribute;

typedef struct cudaLaunchConfig_st {
    dim3 gridDim;               /**< Grid dimensions */
    dim3 blockDim;              /**< Block dimensions */
    size_t dynamicSmemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    cudaStream_t stream;        /**< Stream identifier */
    cudaLaunchAttribute *attrs; /**< List of attributes; nullable if ::cudaLaunchConfig_t::numAttrs == 0 */
    unsigned int numAttrs;      /**< Number of attributes populated in ::cudaLaunchConfig_t::attrs */
} cudaLaunchConfig_t;

/**
 * CUDA IPC Handle Size
 */
#define CUDA_IPC_HANDLE_SIZE 64

/**
 * CUDA IPC event handle
 */
typedef struct cudaIpcEventHandle_st
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcEventHandle_t;

/**
 * CUDA IPC memory handle
 */
typedef struct cudaIpcMemHandle_st
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcMemHandle_t;

/**
 * CUDA memory types
 */
enum cudaMemoryType
{
    cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
    cudaMemoryTypeHost         = 1, /**< Host memory */
    cudaMemoryTypeDevice       = 2, /**< Device memory */
    cudaMemoryTypeManaged      = 3  /**< Managed memory */
};

/**
 * CUDA function attributes that can be set using ::cudaFuncSetAttribute
 */
enum cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8, /**< Maximum dynamic shared memory size */
    cudaFuncAttributePreferredSharedMemoryCarveout = 9, /**< Preferred shared memory-L1 cache split */
    cudaFuncAttributeClusterDimMustBeSet = 10, /**< Indicator to enforce valid cluster dimension specification on kernel launch */
    cudaFuncAttributeRequiredClusterWidth = 11, /**< Required cluster width */
    cudaFuncAttributeRequiredClusterHeight = 12, /**< Required cluster height */
    cudaFuncAttributeRequiredClusterDepth = 13, /**< Required cluster depth */
    cudaFuncAttributeNonPortableClusterSizeAllowed = 14, /**< Whether non-portable cluster scheduling policy is supported */
    cudaFuncAttributeClusterSchedulingPolicyPreference = 15, /**< Required cluster scheduling policy preference */
    cudaFuncAttributeMax
};

/**
 * CUDA function attributes
 */
struct cudaFuncAttributes
{
    size_t sharedSizeBytes;        /**< Size of shared memory in bytes */
    size_t constSizeBytes;         /**< Size of constant memory in bytes */
    size_t localSizeBytes;         /**< Size of local memory in bytes */
    int maxThreadsPerBlock;        /**< Maximum number of threads per block */
    int numRegs;                   /**< Number of registers used */
    int ptxVersion;                /**< PTX version */
    int binaryVersion;             /**< Binary version */
    int cacheModeCA;               /**< Cache mode for const args */
    int maxDynamicSharedSizeBytes; /**< Maximum size of dynamically-allocated shared memory */
    int preferredShmemCarveout;    /**< Preferred shared memory-L1 cache split */
};

/**
 * CUDA pointer attributes
 */
struct cudaPointerAttributes
{
    enum cudaMemoryType type;      /**< Memory type */
    int device;                    /**< Device ordinal */
    void *devicePointer;           /**< Device pointer */
    void *hostPointer;             /**< Host pointer */
    int isManaged;                 /**< Is managed memory */
    unsigned int allocationFlags;  /**< Allocation flags */
};


/**
 * Return values for NVML API calls.
 */
typedef enum nvmlReturn_enum
{
    // cppcheck-suppress *
    NVML_SUCCESS = 0,                          //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,              //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,           //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,              //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,              //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5,        //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,                  //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,          //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,         //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,          //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,                   //!< User provided timeout passed
    NVML_ERROR_IRQ_ISSUE = 11,                 //!< NVIDIA Kernel detected an interrupt issue with a GPU
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,         //!< NVML Shared Library couldn't be found or loaded
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,        //!< Local version of NVML doesn't implement this function
    NVML_ERROR_CORRUPTED_INFOROM = 14,         //!< infoROM is corrupted
    NVML_ERROR_GPU_IS_LOST = 15,               //!< The GPU has fallen off the bus or has otherwise become inaccessible
    NVML_ERROR_RESET_REQUIRED = 16,            //!< The GPU requires a reset before it can be used again
    NVML_ERROR_OPERATING_SYSTEM = 17,          //!< The GPU control device has been blocked by the operating system/cgroups
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    NVML_ERROR_IN_USE = 19,                    //!< An operation cannot be performed because the GPU is currently in use
    NVML_ERROR_MEMORY = 20,                    //!< Insufficient memory
    NVML_ERROR_NO_DATA = 21,                   //!< No data
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,    //!< The requested vgpu operation is not available on target device, becasue ECC is enabled
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,    //!< Ran out of critical resources, other than memory
    NVML_ERROR_FREQ_NOT_SUPPORTED = 24,        //!< Ran out of critical resources, other than memory
    NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25, //!< The provided version is invalid/unsupported
    NVML_ERROR_DEPRECATED  = 26,               //!< The requested functionality has been deprecated
    NVML_ERROR_NOT_READY = 27,                 //!< The system is not ready for the request
    NVML_ERROR_GPU_NOT_FOUND = 28,             //!< No GPUs were found
    NVML_ERROR_INVALID_STATE = 29,             //!< Resource not in correct state to perform requested operation
    NVML_ERROR_UNKNOWN = 999                   //!< An internal driver error occurred
} nvmlReturn_t;


// cublas 相关定义

/* CUBLAS status type returns */
typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
  CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

struct cublasContext {
    int magic_number;
    cudaStream_t stream;
};
typedef struct cublasContext* cublasHandle_t;

typedef enum {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1,
  CUBLAS_OP_C = 2,
  CUBLAS_OP_HERMITAN = 2, /* synonym if CUBLAS_OP_C */
  CUBLAS_OP_CONJG = 3     /* conjugate, placeholder - not supported in the current release */
} cublasOperation_t;

typedef enum { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 } cublasSideMode_t;

typedef enum { CUBLAS_FILL_MODE_LOWER = 0, CUBLAS_FILL_MODE_UPPER = 1, CUBLAS_FILL_MODE_FULL = 2 } cublasFillMode_t;

typedef enum { CUBLAS_DIAG_NON_UNIT = 0, CUBLAS_DIAG_UNIT = 1 } cublasDiagType_t;

/*Enum for default math mode/tensor operation*/
typedef enum {
  CUBLAS_DEFAULT_MATH = 0,

  /* deprecated, same effect as using CUBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release */
  CUBLAS_TENSOR_OP_MATH = 1,

  /* same as using matching _PEDANTIC compute type when using cublas<T>routine calls or cublasEx() calls with
     cudaDataType as compute type */
  CUBLAS_PEDANTIC_MATH = 2,

  /* allow accelerating single precision routines using TF32 tensor cores */
  CUBLAS_TF32_TENSOR_OP_MATH = 3,

  /* flag to force any reductons to use the accumulator type and not output type in case of mixed precision routines
     with lower size output type */
  CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
} cublasMath_t;

/* Enum for compute type
 *
 * - default types provide best available performance using all available hardware features
 *   and guarantee internal storage precision with at least the same precision and range;
 * - _PEDANTIC types ensure standard arithmetic and exact specified internal storage format;
 * - _FAST types allow for some loss of precision to enable higher throughput arithmetic.
 */
typedef enum {
  CUBLAS_COMPUTE_16F = 64,           /* half - default */
  CUBLAS_COMPUTE_16F_PEDANTIC = 65,  /* half - pedantic */
  CUBLAS_COMPUTE_32F = 68,           /* float - default */
  CUBLAS_COMPUTE_32F_PEDANTIC = 69,  /* float - pedantic */
  CUBLAS_COMPUTE_32F_FAST_16F = 74,  /* float - fast, allows down-converting inputs to half or TF32 */
  CUBLAS_COMPUTE_32F_FAST_16BF = 75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
  CUBLAS_COMPUTE_32F_FAST_TF32 = 77, /* float - fast, allows down-converting inputs to TF32 */
  CUBLAS_COMPUTE_64F = 70,           /* double - default */
  CUBLAS_COMPUTE_64F_PEDANTIC = 71,  /* double - pedantic */
  CUBLAS_COMPUTE_32I = 72,           /* signed 32-bit int - default */
  CUBLAS_COMPUTE_32I_PEDANTIC = 73,  /* signed 32-bit int - pedantic */
} cublasComputeType_t;

/*For different GEMM algorithm */
typedef enum {
  CUBLAS_GEMM_DFALT = -1,
  CUBLAS_GEMM_DEFAULT = -1,
  CUBLAS_GEMM_ALGO0 = 0,
  CUBLAS_GEMM_ALGO1 = 1,
  CUBLAS_GEMM_ALGO2 = 2,
  CUBLAS_GEMM_ALGO3 = 3,
  CUBLAS_GEMM_ALGO4 = 4,
  CUBLAS_GEMM_ALGO5 = 5,
  CUBLAS_GEMM_ALGO6 = 6,
  CUBLAS_GEMM_ALGO7 = 7,
  CUBLAS_GEMM_ALGO8 = 8,
  CUBLAS_GEMM_ALGO9 = 9,
  CUBLAS_GEMM_ALGO10 = 10,
  CUBLAS_GEMM_ALGO11 = 11,
  CUBLAS_GEMM_ALGO12 = 12,
  CUBLAS_GEMM_ALGO13 = 13,
  CUBLAS_GEMM_ALGO14 = 14,
  CUBLAS_GEMM_ALGO15 = 15,
  CUBLAS_GEMM_ALGO16 = 16,
  CUBLAS_GEMM_ALGO17 = 17,
  CUBLAS_GEMM_ALGO18 = 18,  // sliced 32x32
  CUBLAS_GEMM_ALGO19 = 19,  // sliced 64x32
  CUBLAS_GEMM_ALGO20 = 20,  // sliced 128x32
  CUBLAS_GEMM_ALGO21 = 21,  // sliced 32x32  -splitK
  CUBLAS_GEMM_ALGO22 = 22,  // sliced 64x32  -splitK
  CUBLAS_GEMM_ALGO23 = 23,  // sliced 128x32 -splitK
  CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
  CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
  CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
  CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
  CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
  CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
  CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
  CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
  CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
  CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
  CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
  CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
  CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
  CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
  CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
  CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
  CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
  CUBLAS_GEMM_ALGO15_TENSOR_OP = 115
} cublasGemmAlgo_t;

typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;

struct float2{
    float x; float y;
};
typedef float2 cuFloatComplex;
typedef cuFloatComplex cuComplex;

struct double2{
    double x, y;
};
typedef double2 cuDoubleComplex;

/* For backward compatibility purposes */
typedef cudaDataType cublasDataType_t;


// libcublasLt

struct cublasLtContext {
    int magic_number;
    cudaStream_t stream;
};
typedef struct cublasLtContext* cublasLtHandle_t;

/** Semi-opaque descriptor for cublasLtMatmul() operation details
 */
typedef struct {
  uint64_t data[32];
} cublasLtMatmulDescOpaque_t;

/** Opaque descriptor for cublasLtMatmul() operation details
 */
typedef cublasLtMatmulDescOpaque_t* cublasLtMatmulDesc_t;

/** Semi-opaque descriptor for matrix memory layout
 */
typedef struct {
  uint64_t data[8];
} cublasLtMatrixLayoutOpaque_t;

/** Opaque descriptor for matrix memory layout
 */
typedef cublasLtMatrixLayoutOpaque_t* cublasLtMatrixLayout_t;

/** Semi-opaque descriptor for cublasLtMatmulPreference() operation details
 */
typedef struct {
  uint64_t data[8];
} cublasLtMatmulPreferenceOpaque_t;

/** Opaque descriptor for cublasLtMatmulAlgoGetHeuristic() configuration
 */
typedef cublasLtMatmulPreferenceOpaque_t* cublasLtMatmulPreference_t;

/** Semi-opaque algorithm descriptor (to avoid complicated alloc/free schemes)
 *
 * This structure can be trivially serialized and later restored for use with the same version of cuBLAS library to save
 * on selecting the right configuration again.
 */
typedef struct {
  uint64_t data[8];
} cublasLtMatmulAlgo_t;

/** Results structure used by cublasLtMatmulAlgoGetHeuristic
 *
 * Holds returned configured algo descriptor and its runtime properties.
 */
typedef struct {
  /** Matmul algorithm descriptor.
   *
   * Must be initialized with cublasLtMatmulAlgoInit() if preferences' CUBLASLT_MATMUL_PERF_SEARCH_MODE is set to
   * CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID
   */
  cublasLtMatmulAlgo_t algo;

  /** Actual size of workspace memory required.
   */
  size_t workspaceSize;

  /** Result status, other fields are only valid if after call to cublasLtMatmulAlgoGetHeuristic() this member is set to
   * CUBLAS_STATUS_SUCCESS.
   */
  cublasStatus_t state;

  /** Waves count - a device utilization metric.
   *
   * wavesCount value of 1.0f suggests that when kernel is launched it will fully occupy the GPU.
   */
  float wavesCount;

  int reserved[4];
} cublasLtMatmulHeuristicResult_t;

/** Matmul descriptor attributes to define details of the operation. */
typedef enum {
  /** Compute type, see cudaDataType. Defines data type used for multiply and accumulate operations and the
   * accumulator during matrix multiplication.
   *
   * int32_t
   */
  CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,

  /** Scale type, see cudaDataType. Defines data type of alpha and beta. Accumulator and value from matrix C are
   * typically converted to scale type before final scaling. Value is then converted from scale type to type of matrix
   * D before being stored in memory.
   *
   * int32_t, default: same as CUBLASLT_MATMUL_DESC_COMPUTE_TYPE
   */
  CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,

  /** Pointer mode of alpha and beta, see cublasLtPointerMode_t. When CUBLASLT_POINTER_MODE_DEVICE_VECTOR is in use,
   * alpha/beta vector lenghts must match number of output matrix rows.
   *
   * int32_t, default: CUBLASLT_POINTER_MODE_HOST
   */
  CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,

  /** Transform of matrix A, see cublasOperation_t.
   *
   * int32_t, default: CUBLAS_OP_N
   */
  CUBLASLT_MATMUL_DESC_TRANSA = 3,

  /** Transform of matrix B, see cublasOperation_t.
   *
   * int32_t, default: CUBLAS_OP_N
   */
  CUBLASLT_MATMUL_DESC_TRANSB = 4,

  /** Transform of matrix C, see cublasOperation_t.
   *
   * Currently only CUBLAS_OP_N is supported.
   *
   * int32_t, default: CUBLAS_OP_N
   */
  CUBLASLT_MATMUL_DESC_TRANSC = 5,

  /** Matrix fill mode, see cublasFillMode_t.
   *
   * int32_t, default: CUBLAS_FILL_MODE_FULL
   */
  CUBLASLT_MATMUL_DESC_FILL_MODE = 6,

  /** Epilogue function, see cublasLtEpilogue_t.
   *
   * uint32_t, default: CUBLASLT_EPILOGUE_DEFAULT
   */
  CUBLASLT_MATMUL_DESC_EPILOGUE = 7,

  /** Bias or bias gradient vector pointer in the device memory.
   *
   * Bias case. See CUBLASLT_EPILOGUE_BIAS.
   * For bias data type see CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE.
   *
   * Bias vector length must match matrix D rows count.
   *
   * Bias gradient case. See CUBLASLT_EPILOGUE_DRELU_BGRAD and CUBLASLT_EPILOGUE_DGELU_BGRAD.
   * Bias gradient vector elements are the same type as the output elements
   * (Ctype) with the exception of IMMA kernels (see above).
   *
   * Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
   * depend on its value to determine expected pointer alignment.
   *
   * Bias case: const void *, default: NULL
   * Bias gradient case: void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,

  /** Batch stride for bias or bias gradient vector.
   *
   * Used together with CUBLASLT_MATMUL_DESC_BIAS_POINTER when matrix D's CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT > 1.
   *
   * int64_t, default: 0
   */
  CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,

  /** Pointer for epilogue auxiliary buffer.
   *
   * - Output vector for ReLu bit-mask in forward pass when CUBLASLT_EPILOGUE_RELU_AUX
   *   or CUBLASLT_EPILOGUE_RELU_AUX_BIAS epilogue is used.
   * - Input vector for ReLu bit-mask in backward pass when
   *   CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is used.
   *
   * - Output of GELU input matrix in forward pass when
   *   CUBLASLT_EPILOGUE_GELU_AUX_BIAS epilogue is used.
   * - Input of GELU input matrix for backward pass when
   *   CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue is used.
   *
   * For aux data type see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE.
   *
   * Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
   * depend on its value to determine expected pointer alignment.
   *
   * Requires setting CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD attribute.
   *
   * Forward pass: void *, default: NULL
   * Backward pass: const void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,

  /** Leading dimension for epilogue auxiliary buffer.
   *
   * - ReLu bit-mask matrix leading dimension in elements (i.e. bits)
   *   when CUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX_BIAS or CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is
   * used. Must be divisible by 128 and be no less than the number of rows in the output matrix.
   *
   * - GELU input matrix leading dimension in elements
   *   when CUBLASLT_EPILOGUE_GELU_AUX_BIAS or CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue used.
   *   Must be divisible by 8 and be no less than the number of rows in the output matrix.
   *
   * int64_t, default: 0
   */
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,

  /** Batch stride for epilogue auxiliary buffer.
   *
   * - ReLu bit-mask matrix batch stride in elements (i.e. bits)
   *   when CUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX_BIAS or CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is
   * used. Must be divisible by 128.
   *
   * - GELU input matrix batch stride in elements
   *   when CUBLASLT_EPILOGUE_GELU_AUX_BIAS or CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue used.
   *   Must be divisible by 8.
   *
   * int64_t, default: 0
   */
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,

  /** Batch stride for alpha vector.
   *
   * Used together with CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST when matrix D's
   * CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT > 1. If CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO is set then
   * CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE must be set to 0 as this mode doesnt supported batched alpha vector.
   *
   * int64_t, default: 0
   */
  CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,

  /** Number of SMs to target for parallel execution. Optimizes heuristics for execution on a different number of SMs
   *  when user expects a concurrent stream to be using some of the device resources.
   *
   *  int32_t, default: 0 - use the number reported by the device.
   */
  CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15,

  /** Device pointer to the scale factor value that converts data in matrix A to the compute data type range.
   *
   *  The scaling factor value must have the same type as the compute type.
   *
   *  If not specified, or set to NULL, the scaling factor is assumed to be 1.
   *
   *  If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
   *  will return CUBLAS_INVALID_VALUE.
   *
   *  const void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,

  /** Device pointer to the scale factor value to convert data in matrix B to compute data type range.
   *
   *  The scaling factor value must have the same type as the compute type.
   *
   *  If not specified, or set to NULL, the scaling factor is assumed to be 1.
   *
   *  If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
   *  will return CUBLAS_INVALID_VALUE.
   *
   *  const void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,

  /** Device pointer to the scale factor value to convert data in matrix C to compute data type range.
   *
   *  The scaling factor value must have the same type as the compute type.
   *
   *  If not specified, or set to NULL, the scaling factor is assumed to be 1.
   *
   *  If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
   *  will return CUBLAS_INVALID_VALUE.
   *
   *  const void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,

  /** Device pointer to the scale factor value to convert data in matrix D to compute data type range.
   *
   *  The scaling factor value must have the same type as the compute type.
   *
   *  If not specified, or set to NULL, the scaling factor is assumed to be 1.
   *
   *  If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
   *  will return CUBLAS_INVALID_VALUE.
   *
   *  const void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,

  /** Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
   *  output matrix.
   *
   *  The computed value has the same type as the compute type.
   *
   *  If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
   *  data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.
   *
   *  void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21,

  /** Type of the data to be stored to the memory pointed to by CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
   *
   *  If unset, the data type defaults to the type of elements of the output matrix with some exceptions, see details
   * below.
   *
   *  ReLu uses a bit-mask.
   *
   *  GELU input matrix elements type is the same as the type of elements of
   *  the output matrix with some exceptions, see details below.
   *
   *  For fp8 kernels with output type CUDA_R_8F_E4M3 the aux data type can be CUDA_R_8F_E4M3 or CUDA_R_16F with some
   *  restrictions.  See https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulDescAttributes_t for more details.
   *
   *  If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
   *  will return CUBLAS_INVALID_VALUE.
   *
   *  int32_t based on cudaDataType, default: -1
   */
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,

  /** Device pointer to the scaling factor value to convert results from compute type data range to storage
   *  data range in the auxiliary matrix that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
   *
   *  The scaling factor value must have the same type as the compute type.
   *
   *  If not specified, or set to NULL, the scaling factor is assumed to be 1. If set for an unsupported matrix data,
   *  scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.
   *
   *  void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23,

  /** Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
   *  buffer that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
   *
   *  The computed value has the same type as the compute type.
   *
   *  If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
   *  data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.
   *
   *  void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24,

  /** Flag for managing fp8 fast accumulation mode.
   *  When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate results
   *  will not periodically be promoted to a higher precision.
   *
   *  int8_t, default: 0 - fast accumulation mode is disabled.
   */
  CUBLASLT_MATMUL_DESC_FAST_ACCUM = 25,

  /** Type of bias or bias gradient vector in the device memory.
   *
   * Bias case: see CUBLASLT_EPILOGUE_BIAS.
   *
   * Bias vector elements are the same type as the elements of output matrix (Dtype) with the following exceptions:
   * - IMMA kernels with computeType=CUDA_R_32I and Ctype=CUDA_R_8I where the bias vector elements
   *   are the same type as alpha, beta (CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F)
   * - fp8 kernels with an output type of CUDA_R_32F, CUDA_R_8F_E4M3 or CUDA_R_8F_E5M2, See
   *   https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul for details.
   *
   * int32_t based on cudaDataType, default: -1
   */
  CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26,

  /** EXPERIMENTAL, DEPRECATED: Number of atomic synchronization chunks in the row dimension of the output matrix D.
   *
   * int32_t, default 0 (atomic synchronization disabled)
   */
  CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = 27,

  /** EXPERIMENTAL, DEPRECATED: Number of atomic synchronization chunks in the column dimension of the output matrix D.
   *
   * int32_t, default 0 (atomic synchronization disabled)
   */
  CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = 28,

  /** EXPERIMENTAL: Pointer to a device array of input atomic counters consumed by a matmul.
   *
   * int32_t *, default: NULL
   * */
  CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = 29,

  /** EXPERIMENTAL: Pointer to a device array of output atomic counters produced by a matmul.
   *
   * int32_t *, default: NULL
   * */
  CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = 30,

  /** Scaling mode that defines how the matrix scaling factor for matrix A is interpreted
   *
   * int32_t, default: 0 */
  CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 31,

  /** Scaling mode that defines how the matrix scaling factor for matrix B is interpreted
   *
   * int32_t, default: 0 */
  CUBLASLT_MATMUL_DESC_B_SCALE_MODE = 32,

  /** Scaling mode that defines how the matrix scaling factor for matrix C is interpreted
   *
   * int32_t, default: 0 */
  CUBLASLT_MATMUL_DESC_C_SCALE_MODE = 33,

  /** Scaling mode that defines how the matrix scaling factor for matrix D is interpreted
   *
   * int32_t, default: 0 */
  CUBLASLT_MATMUL_DESC_D_SCALE_MODE = 34,

  /** Scaling mode that defines how the matrix scaling factor for the auxiliary matrix is interpreted
   *
   * int32_t, default: 0 */
  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE = 35,

  /** Device pointer to the scale factors that are used to convert data in matrix D to the compute data type range.
   *
   *  The scaling factor value type is defined by the scaling mode (see CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE)
   *
   *  If set for an unsupported matrix data, scale, scale mode, and compute type combination, calling cublasLtMatmul()
   *  will return CUBLAS_INVALID_VALUE.
   *
   *  void *, default: NULL
   */
  CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER = 36,

  /** Scaling mode that defines how the output matrix scaling factor for matrix D is interpreted
   *
   * int32_t, default: 0 */
  CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE = 37,
} cublasLtMatmulDescAttributes_t;

/** Algo search preference to fine tune the heuristic function. */
typedef enum {
  /** Search mode, see cublasLtMatmulSearch_t.
   *
   * uint32_t, default: CUBLASLT_SEARCH_BEST_FIT
   */
  CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,

  /** Maximum allowed workspace size in bytes.
   *
   * uint64_t, default: 0 - no workspace allowed
   */
  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,

  /** Reduction scheme mask, see cublasLtReductionScheme_t. Filters heuristic result to only include algo configs that
   * use one of the required modes.
   *
   * E.g. mask value of 0x03 will allow only INPLACE and COMPUTE_TYPE reduction schemes.
   *
   * uint32_t, default: CUBLASLT_REDUCTION_SCHEME_MASK (allows all reduction schemes)
   */
  CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,

  /** Minimum buffer alignment for matrix A (in bytes).
   *
   * Selecting a smaller value will exclude algorithms that can not work with matrix A that is not as strictly aligned
   * as they need.
   *
   * uint32_t, default: 256
   */
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,

  /** Minimum buffer alignment for matrix B (in bytes).
   *
   * Selecting a smaller value will exclude algorithms that can not work with matrix B that is not as strictly aligned
   * as they need.
   *
   * uint32_t, default: 256
   */
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,

  /** Minimum buffer alignment for matrix C (in bytes).
   *
   * Selecting a smaller value will exclude algorithms that can not work with matrix C that is not as strictly aligned
   * as they need.
   *
   * uint32_t, default: 256
   */
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,

  /** Minimum buffer alignment for matrix D (in bytes).
   *
   * Selecting a smaller value will exclude algorithms that can not work with matrix D that is not as strictly aligned
   * as they need.
   *
   * uint32_t, default: 256
   */
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,

  /** Maximum wave count.
   *
   * See cublasLtMatmulHeuristicResult_t::wavesCount.
   *
   * Selecting a non-zero value will exclude algorithms that report device utilization higher than specified.
   *
   * float, default: 0.0f
   */
  CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 9,

  /** Numerical implementation details mask, see cublasLtNumericalImplFlags_t. Filters heuristic result to only include
   * algorithms that use the allowed implementations.
   *
   * uint64_t, default: uint64_t(-1) (allow everything)
   */
  CUBLASLT_MATMUL_PREF_IMPL_MASK = 12,
} cublasLtMatmulPreferenceAttributes_t;

/** Attributes of memory layout */
typedef enum {
  /** Data type, see cudaDataType.
   *
   * uint32_t
   */
  CUBLASLT_MATRIX_LAYOUT_TYPE = 0,

  /** Memory order of the data, see cublasLtOrder_t.
   *
   * int32_t, default: CUBLASLT_ORDER_COL
   */
  CUBLASLT_MATRIX_LAYOUT_ORDER = 1,

  /** Number of rows.
   *
   * Usually only values that can be expressed as int32_t are supported.
   *
   * uint64_t
   */
  CUBLASLT_MATRIX_LAYOUT_ROWS = 2,

  /** Number of columns.
   *
   * Usually only values that can be expressed as int32_t are supported.
   *
   * uint64_t
   */
  CUBLASLT_MATRIX_LAYOUT_COLS = 3,

  /** Matrix leading dimension.
   *
   * For CUBLASLT_ORDER_COL this is stride (in elements) of matrix column, for more details and documentation for
   * other memory orders see documentation for cublasLtOrder_t values.
   *
   * Currently only non-negative values are supported, must be large enough so that matrix memory locations are not
   * overlapping (e.g. greater or equal to CUBLASLT_MATRIX_LAYOUT_ROWS in case of CUBLASLT_ORDER_COL).
   *
   * int64_t;
   */
  CUBLASLT_MATRIX_LAYOUT_LD = 4,

  /** Number of matmul operations to perform in the batch.
   *
   * See also CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT
   *
   * int32_t, default: 1
   */
  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,

  /** Stride (in elements) to the next matrix for strided batch operation.
   *
   * When matrix type is planar-complex (CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET != 0), batch stride
   * is interpreted by cublasLtMatmul() in number of real valued sub-elements. E.g. for data of type CUDA_C_16F,
   * offset of 1024B is encoded as a stride of value 512 (since each element of the real and imaginary matrices
   * is a 2B (16bit) floating point type).
   *
   * NOTE: A bug in cublasLtMatrixTransform() causes it to interpret the batch stride for a planar-complex matrix
   * as if it was specified in number of complex elements. Therefore an offset of 1024B must be encoded as stride
   * value 256 when calling cublasLtMatrixTransform() (each complex element is 4B with real and imaginary values 2B
   * each). This behavior is expected to be corrected in the next major cuBLAS version.
   *
   * int64_t, default: 0
   */
  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,

  /** Stride (in bytes) to the imaginary plane for planar complex layout.
   *
   * int64_t, default: 0 - 0 means that layout is regular (real and imaginary parts of complex numbers are interleaved
   * in memory in each element)
   */
  CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7,
} cublasLtMatrixLayoutAttribute_t;


// NCCL

/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclRemoteError             =  6,
               ncclInProgress              =  7,
               ncclNumResults              =  8
} ncclResult_t;

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
    char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

/* Data types */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
               ncclBfloat16   = 9,
               ncclFloat8e4m3 = 10,
               ncclFloat8e5m2 = 11,
               ncclNumTypes   = 12
} ncclDataType_t;

/* Reduction operation selector */
typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;

typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               ncclAvg        = 4,
               /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
                * serves as the least possible value for dynamic ncclRedOp_t's
                * as constructed by ncclRedOpCreate*** functions. */
               ncclNumOps     = 5,
               /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
                * It is defined to be the largest signed value (since compilers
                * are permitted to use signed enums) that won't grow
                * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
                * maintain ABI compatibility. */
               ncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(ncclRedOp_dummy_t))
} ncclRedOp_t;

/* Opaque handle to communicator */
struct ncclComm {
    int rank;
    int nranks;
    ncclResult_t asyncError;
    ncclUniqueId commId;
    int blocking;
};
typedef struct ncclComm* ncclComm_t;

/* ncclScalarResidence_t: Location and dereferencing logic for scalar arguments. */
typedef enum {
  /* ncclScalarDevice: The scalar is in device-visible memory and will be
   * dereferenced while the collective is running. */
  ncclScalarDevice = 0,

  /* ncclScalarHostImmediate: The scalar is in host-visible memory and will be
   * dereferenced before the ncclRedOpCreate***() function returns. */
  ncclScalarHostImmediate = 1
} ncclScalarResidence_t;

/* This struct will be used by ncclGroupSimulateEnd() API to query information about simulation. */
typedef struct ncclSimInfo_v22200 {
    size_t size;
    unsigned int magic;
    unsigned int version;
    float estimatedTime;
} ncclSimInfo_t;

/* Communicator configuration. Users can assign value to attributes to specify the
 * behavior of a communicator. */
typedef struct ncclConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* attributes that users are able to customize. */
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char *netName;
  int splitShare;
  int trafficClass;
} ncclConfig_t;

typedef struct ncclWindow* ncclWindow_t;
