#include "ras_capture.h"

#include <cstdio>
#include <exception>
#include <type_traits>

static_assert(std::is_trivially_copyable<FakeCudaRasRecord>::value,
              "direct capture descriptor must remain POD");
static_assert(std::is_trivially_copyable<FakeCudaRasRecordV2>::value,
              "extended direct capture descriptor must remain POD");

namespace {
struct SinkState {
    void* user = nullptr;
    FakeCudaRasSink callback = nullptr;
    FakeCudaRasSinkV2 callback_v2 = nullptr;
    FakeCudaRasPolicy policy = nullptr;
    bool failed = false;
    bool in_callback = false;
    char error[256]{};
};
// Exactly one state lives in fake_cuda_core, shared by all intercepted DSOs.
SinkState state;
bool installed() { return state.callback || state.callback_v2; }

template <class Callback> int invoke(Callback callback) {
    if (state.in_callback) {
        fakecuda_ras_fail("recursive direct capture callback");
        return -1;
    }
    state.in_callback = true;
    try {
        if (callback() != 0)
            fakecuda_ras_fail("direct capture callback rejected operation");
    } catch (const std::exception& e) {
        fakecuda_ras_fail(e.what());
    } catch (...) {
        fakecuda_ras_fail("direct capture callback threw an unknown exception");
    }
    state.in_callback = false;
    return fakecuda_ras_status();
}
}

extern "C" int fakecuda_ras_install(void* user, FakeCudaRasSink sink) {
    if (installed() || !sink) return -1;
    state = {};
    state.user = user;
    state.callback = sink;
    return 0;
}
extern "C" int fakecuda_ras_install_v2(void* user, FakeCudaRasSinkV2 sink,
                                      FakeCudaRasPolicy policy) {
    if (installed() || !sink) return -1;
    state = {};
    state.user = user;
    state.callback_v2 = sink;
    state.policy = policy;
    return 0;
}
extern "C" void fakecuda_ras_reset(void) {
    if (state.in_callback) {
        fakecuda_ras_fail("cannot reset direct capture from its callback");
        return;
    }
    state = {};
}
extern "C" int fakecuda_ras_active(void) { return installed(); }
extern "C" int fakecuda_ras_status(void) { return state.failed ? -1 : 0; }
extern "C" const char* fakecuda_ras_last_error(void) { return state.error; }
extern "C" void fakecuda_ras_fail(const char* message) {
    if (!installed() || state.failed) return;
    state.failed = true;
    std::snprintf(state.error, sizeof(state.error), "%s", message ? message : "direct capture failed");
}
extern "C" int fakecuda_ras_emit(const FakeCudaRasRecord* op) {
    if (!installed() || state.failed) return -1;
    if (!op || op->abi_version != 1 || op->op < FAKECUDA_RAS_ALLOC ||
        op->op > FAKECUDA_RAS_REDUCESCATTER || op->op == FAKECUDA_RAS_KERNEL) {
        fakecuda_ras_fail("invalid direct capture descriptor");
        return -1;
    }
    if (state.callback_v2) {
        auto extended = fakecuda::ras::record_v2(static_cast<FakeCudaRasOp>(op->op));
        extended.base = *op;
        extended.base.abi_version = 2;
        return invoke([&] { return state.callback_v2(state.user, &extended); });
    }
    return invoke([&] { return state.callback(state.user, op); });
}

extern "C" int fakecuda_ras_wants_payload(uint32_t op) {
    if (!installed() || state.failed) return -1;
    if (!state.policy) return 1;
    int decision = -1;
    const auto status = invoke([&] {
        decision = state.policy(state.user, op);
        return decision < 0 ? -1 : 0;
    });
    return status == 0 ? (decision != 0) : -1;
}

extern "C" int fakecuda_ras_emit_v2(const FakeCudaRasRecordV2* op) {
    if (!installed() || state.failed) return -1;
    if (!op || op->base.abi_version != 2 || op->struct_size != sizeof(*op) ||
        op->base.op < FAKECUDA_RAS_ALLOC || op->base.op > FAKECUDA_RAS_REDUCESCATTER ||
        (op->kernel_name_size && !op->kernel_name)) {
        fakecuda_ras_fail("invalid extended direct capture descriptor");
        return -1;
    }
    if (state.callback_v2)
        return invoke([&] { return state.callback_v2(state.user, op); });
    // ABI 1 cannot represent these operations. Preserve its old behavior and
    // reject extension loss rather than quietly truncating semantic fields.
    if (op->base.op == FAKECUDA_RAS_KERNEL || op->flags_v2 || op->batch_count != 1 ||
        op->stride_a || op->stride_b || op->stride_c || op->stride_d ||
        (op->d && op->d != op->base.c) ||
        (op->ldd && op->ldd != op->base.ldc) ||
        (op->d && op->d_type != op->base.c_type) ||
        op->aux || op->epilogue || op->pointer_mode || op->math_mode) {
        fakecuda_ras_fail("operation requires direct capture ABI 2");
        return -1;
    }
    auto legacy = op->base;
    legacy.abi_version = 1;
    return invoke([&] { return state.callback(state.user, &legacy); });
}
