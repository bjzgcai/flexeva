#ifndef FLEXEVA_CODE_RUNTIME_H
#define FLEXEVA_CODE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLEXEVA_CODE_ABI_VERSION 1u
#define FLEXEVA_CODE_OBSERVER_ABI_VERSION 2u

enum FlexevaCodeStatus {
    FLEXEVA_CODE_OK = 0,
    FLEXEVA_CODE_INVALID_ARGUMENT = 1,
    FLEXEVA_CODE_CALLBACK_FAILED = 2,
    FLEXEVA_CODE_EXECUTION_FAILED = 3
};

enum FlexevaCodeScalarKind {
    FLEXEVA_CODE_BOOL = 1,
    FLEXEVA_CODE_I32 = 2,
    FLEXEVA_CODE_U32 = 3,
    FLEXEVA_CODE_I64 = 4,
    FLEXEVA_CODE_U64 = 5,
    FLEXEVA_CODE_F32 = 6,
    FLEXEVA_CODE_F64 = 7
};

typedef struct FlexevaCodeScalarV1 {
    uint32_t kind;
    uint32_t reserved;
    union {
        uint8_t boolean;
        int32_t i32;
        uint32_t u32;
        int64_t i64;
        uint64_t u64;
        float f32;
        double f64;
    } value;
} FlexevaCodeScalarV1;

typedef struct FlexevaCodeInvocationV1 {
    uint32_t abi_version;
    uint32_t rank;
    uint32_t world_size;
    uint32_t reserved;
    const FlexevaCodeScalarV1* arguments;
    size_t size;
    /* Written by the generated entry shim for a non-void entry. A caller must
     * only consume this value when invoke returns FLEXEVA_CODE_OK. */
    FlexevaCodeScalarV1* result;
} FlexevaCodeInvocationV1;

enum FlexevaCodeScopeKind {
    FLEXEVA_CODE_ENTRY = 1,
    FLEXEVA_CODE_BLOCK = 2,
    FLEXEVA_CODE_FUNCTION = 3,
    FLEXEVA_CODE_LOOP = 4,
    FLEXEVA_CODE_EMISSION = 5,
    FLEXEVA_CODE_BRANCH = 6,
    FLEXEVA_CODE_DEFINITION = 7,
    FLEXEVA_CODE_GROUND = 8
};

typedef struct FlexevaCodePathFrameV1 {
    uint64_t site;
    uint64_t ordinal;
    uint32_t kind;
    uint32_t reserved;
} FlexevaCodePathFrameV1;

/* All arrays and strings passed to callbacks are borrowed for that call only.
 * Path includes the current occurrence as its final frame. Ordinals are local
 * to one parent occurrence and source site, independent of other source sites.
 */
typedef struct FlexevaCodeOccurrenceV1 {
    uint64_t site;
    uint64_t ordinal;
    uint64_t order;
    uint32_t kind;
    uint32_t rank;
    const FlexevaCodePathFrameV1* path;
    size_t path_size;
} FlexevaCodeOccurrenceV1;

/* Callbacks return zero on success. Nonzero results poison the invocation even
 * if workload code catches the resulting C++ exception. begin/end/fail are
 * required; other callbacks are required when their corresponding hook runs.
 * A failed begin must roll back its own partial callback state; it has no token.
 */
typedef struct FlexevaCodeCallbacksV1 {
    uint32_t abi_version;
    uint32_t reserved;
    int (*begin)(void* user, const FlexevaCodeOccurrenceV1*,
                 const FlexevaCodeScalarV1* inputs, size_t size, uint64_t* token);
    int (*end)(void* user, uint64_t token);
    int (*branch)(void* user, uint64_t site, int taken);
    int (*definition)(void* user, uint64_t variable, FlexevaCodeScalarV1 value);
    int (*ground)(void* user, uint64_t site, const FlexevaCodeScalarV1* inputs,
                  size_t size, FlexevaCodeScalarV1* result);
    int (*communicator)(void* user, uint64_t handle, uint64_t membership,
                        uint32_t local_rank);
    int (*fail)(void* user, const char* message);
} FlexevaCodeCallbacksV1;

enum FlexevaCodeValueKindV2 {
    FLEXEVA_CODE_VALUE_METADATA = 1,
    FLEXEVA_CODE_VALUE_SCALAR = 2,
    FLEXEVA_CODE_VALUE_TENSOR = 3
};

/* A borrowed value description, never a Tensor/object address. metadata contains
 * caller-defined canonical bytes (for example dtype, shape and strides), valid
 * only during the callback. A consumer retaining evidence must copy the bytes.
 * flags is reserved and must be zero. */
typedef struct FlexevaCodeValueV2 {
    uint32_t kind;
    uint32_t flags;
    const void* metadata;
    size_t size;
} FlexevaCodeValueV2;

/* V2 adds optional observation hooks without changing any V1 layout. base must
 * be a valid V1 callback table. metadata is required only when a metadata hook
 * runs; it observes the containing active occurrence without creating a scope. */
typedef struct FlexevaCodeCallbacksV2 {
    uint32_t abi_version;
    uint32_t reserved;
    FlexevaCodeCallbacksV1 base;
    int (*metadata)(void* user, const FlexevaCodeOccurrenceV1*, uint64_t site,
                    const FlexevaCodeValueV2*);
} FlexevaCodeCallbacksV2;

typedef int (*FlexevaCodeInvokeV1)(const FlexevaCodeInvocationV1*,
                                 const FlexevaCodeCallbacksV1*, void* user);
typedef int (*FlexevaCodeInvokeV2)(const FlexevaCodeInvocationV1*,
                                 const FlexevaCodeCallbacksV2*, void* user);
typedef const char* (*FlexevaCodePlanV1)(void);
typedef int (*FlexevaCodePrepareV1)(void);

/* Attach one context to this thread before entering an instrumented extension.
 * The invocation's arguments/result and user must remain valid until detach.
 * Callbacks are copied. Helpers join this context without starting a Session.
 * Reentry/concurrent attachment fails. Detach must use the returned handle on
 * the attaching thread and always releases a matching context, even on failure.
 * If extension execution fails, call fail_v1 before detach to abort its capture.
 */
int flexeva_code_attach_v1(const FlexevaCodeInvocationV1*,
                           const FlexevaCodeCallbacksV1*, void* user, void** handle);
int flexeva_code_attach_v2(const FlexevaCodeInvocationV1*,
                           const FlexevaCodeCallbacksV2*, void* user, void** handle);
int flexeva_code_fail_v1(void* handle, const char* message);
int flexeva_code_detach_v1(void* handle);
int flexeva_code_is_active_v1(void);
/* Borrowed until the next runtime hook. Null outside a healthy occurrence. */
#ifdef __cplusplus
const FlexevaCodeOccurrenceV1* flexeva_code_current_occurrence_v1(void) noexcept;
#else
const FlexevaCodeOccurrenceV1* flexeva_code_current_occurrence_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
