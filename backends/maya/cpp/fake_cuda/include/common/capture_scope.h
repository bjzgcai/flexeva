#pragma once

// Opaque capture ownership. No source language, evaluator, or RAS identifiers
// are interpreted here. Status-returning functions use 0 for success and -1
// for misuse. Tokens must be unique for the lifetime of the native process.
// Ownership faults suppress further traces and make scope_end fail after
// closing the scope; an outstanding wrapper/protocol instead leaves it open.
#include <stddef.h>
#include <stdint.h>
#include "ras_capture.h"

#ifdef __cplusplus
extern "C" {
#endif

enum FakeCudaCaptureDecision { FAKECUDA_RECORD = 0, FAKECUDA_RETAIN = 1 };
enum FakeCudaResourceKind {
    FAKECUDA_ALLOCATION = 1, FAKECUDA_STREAM = 2, FAKECUDA_EVENT = 3,
    FAKECUDA_COMMUNICATOR = 4, FAKECUDA_BLAS_HANDLE = 5
};
enum FakeCudaResourceAction { FAKECUDA_CREATED = 1, FAKECUDA_DESTROYED = 2 };

// Counters use the same order as capture::Counter, schema version 1. They
// describe work, not a semantic hash or a proof that arbitrary reuse is valid.
struct FakeCudaCaptureReceipt {
    uint64_t schema_version;
    uint64_t token;
    uint64_t parent_token;
    uint64_t decision;
    uint64_t wrapper_calls;
    uint64_t completed_calls;
    uint64_t counters[13];
    uint64_t resource_count;
};

// Sparse lifetime changes, not per-call trace records. (creator_token, ordinal,
// kind) is a logical identity; native_handle/generation are process evidence.
// Owner 0 identifies resources created outside a capture scope. An adapter must
// validate their input binding; this API does not assume cross-run equivalence.
struct FakeCudaResourceBinding {
    uint64_t kind;
    uint64_t action;
    uint64_t native_handle;
    uint64_t creator_token;
    uint64_t ordinal;
    uint64_t generation;
    uint64_t extent_bytes;  // allocation range; zero for non-memory resources
    uint64_t api_handle;  // API pointer can differ from the trace/registry handle
};

int fakecuda_capture_scope_begin(uint64_t token, int decision);
int fakecuda_capture_scope_end(uint64_t token, struct FakeCudaCaptureReceipt* receipt);
int fakecuda_capture_scope_bind(uint64_t token);
int fakecuda_capture_scope_unbind(uint64_t token);
int fakecuda_capture_scope_resource(uint64_t token, size_t index,
                                   struct FakeCudaResourceBinding* binding);
int fakecuda_capture_scope_release(uint64_t token);
// Read-only direct wrapper count, used to locate native window-tail work.
int fakecuda_capture_scope_calls(uint64_t token, uint64_t* calls);
// Read-only activity for host-call guards; includes marker APIs without wrappers.
int fakecuda_capture_scope_activity(uint64_t token, uint64_t* calls, uint64_t* apis);
// Optional backend task-local transport. The provider must be native, noexcept,
// nonblocking, and stay loaded until uninstalled. It returns 0 when no task
// owns this thread. The registry validates every nonzero token at wrapper entry;
// closed/unknown owners poison the execution. Install only with no open scopes.
typedef uint64_t (*FakeCudaOwnerProvider)(void);
int fakecuda_capture_set_owner_provider(FakeCudaOwnerProvider provider);
// Call after joining relevant workers and closing all scopes, before accepting
// an execution. This does not synchronize or guess ownership of future work.
int fakecuda_capture_check(void);
int fakecuda_capture_resource_lookup(uint64_t kind, uint64_t native_handle,
                                    struct FakeCudaResourceBinding* binding);
// Read live entry resources with no open scopes. A null output queries count;
// a short buffer fails instead of returning a partial snapshot.
int fakecuda_capture_resource_snapshot(struct FakeCudaResourceBinding* output,
                                      size_t capacity, size_t* count);
const char* fakecuda_capture_last_error(void);

#ifdef __cplusplus
}

namespace fakecuda::capture {

// Intrusive stack frame: no per-wrapper heap allocation or global mutex. Its
// owner/decision is fixed until the wrapper returns, including callback nesting.
struct CallFrame {
    void* owner = nullptr;
    CallFrame* previous = nullptr;
};
void enter_call(CallFrame& frame);
void leave_call(CallFrame& frame);
int scoped_decision();  // -1 outside capture; 0 record; 1 retain
uint64_t scoped_token();
void scope_count(size_t counter, uint64_t amount);
void resource_created(uint64_t kind, uint64_t native_handle, uint64_t extent_bytes = 0,
                      uint64_t api_handle = 0);
void resource_destroyed(uint64_t kind, uint64_t native_handle);
// Native ownership proves that this generation has a legacy binding, even if
// destruction occurs during a later direct RAS capture. Missing bindings remain
// errors; this is not an unchecked cleanup operation.
void retire_legacy_resource(uint64_t kind, uint64_t native_handle);
enum class PendingKind { LaunchConfiguration = 0, NcclGroup = 1 };
void pending_begin(PendingKind kind);
void pending_end(PendingKind kind);

class WrapperCall {
public:
    WrapperCall() : direct_(fakecuda_ras_active() != 0) { if (!direct_) enter_call(frame_); }
    ~WrapperCall() { if (!direct_) leave_call(frame_); }
    WrapperCall(const WrapperCall&) = delete;
    WrapperCall& operator=(const WrapperCall&) = delete;
private:
    CallFrame frame_;
    bool direct_;
};

}  // namespace fakecuda::capture
#endif
