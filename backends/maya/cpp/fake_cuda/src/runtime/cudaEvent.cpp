#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "trace_log.hpp"

API cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);

API cudaError_t cudaEventCreate(cudaEvent_t *event){
    fakecuda::capture::WrapperCall capture_call;
    LOG_DEBUG(CUDART, "cudaEventCreate() called.");
    return cudaEventCreateWithFlags(event, 0);
}
REGISTER_CUDA_FUNCTION(cudaEventCreate);

API cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags){
    if (fakecuda::ras::active() && (!event)) {
        fakecuda_ras_fail("Invalid cudaEventCreateWithFlags arguments");
        return cudaErrorInvalidValue;
    }
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaEventCreateWithFlags() called.");
    if (event == nullptr) {
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    CUevent_st* fake_event = nullptr;
    auto allocate = [&] {
        fake_event = new CUevent_st{
            g_eventIdCounter++, flags, 0ULL, nullptr, 0ULL, 0ULL, false,
        };
        std::lock_guard<std::mutex> lock(g_eventMutex);
        g_validEvents.insert(fake_event);
    };
    if (fakecuda::ras::active()) {
        *event = nullptr;
        try { allocate(); }
        catch (...) {
            delete fake_event;
            fakecuda_ras_fail("CUDA event allocation or registration failed");
            fake_setLastError(cudaErrorMemoryAllocation);
            return cudaErrorMemoryAllocation;
        }
    } else {
        allocate();
    }
    *event = to_cuda_event(fake_event);
    fakecuda::capture::resource_created(FAKECUDA_EVENT, fake_event->id, 0,
        reinterpret_cast<uintptr_t>(*event));
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_EVENT_CREATE);
        r.event = fakecuda::ras::address(*event);
        r.flags = static_cast<int>(flags);
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        if (result != cudaSuccess) {
            { std::lock_guard<std::mutex> lock(g_eventMutex); g_validEvents.erase(fake_event); }
            delete fake_event;
            *event = nullptr;
        }
        fake_setLastError(result);
        return result;
    }
    auto payload = make_trace_payload();
    if (payload) payload->add_uint64("event_id", fake_event->id);
    if (payload) payload->add_uint("flags", flags);
    TRACE_API_EX(CUDART, "cudaEventCreateWithFlags", "stream_op", payload);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaEventCreateWithFlags);

API cudaError_t cudaEventDestroy(cudaEvent_t event){
    if (fakecuda::ras::active() && (!fake_valid_event(event))) {
        fakecuda_ras_fail("Invalid cudaEventDestroy arguments");
        return cudaErrorInvalidValue;
    }
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaEventDestroy() called.");
    CUevent_st* fake_event = to_fake_event(event);
    if (fake_event == nullptr) {
        fake_setLastError(cudaErrorInvalidResourceHandle);
        return cudaErrorInvalidResourceHandle;
    }
    {
        std::lock_guard<std::mutex> lock(g_eventMutex);
        auto it = g_validEvents.find(fake_event);
        if (it == g_validEvents.end()) {
            fake_setLastError(cudaErrorInvalidResourceHandle);
            return cudaErrorInvalidResourceHandle;
        }
        g_validEvents.erase(it);
    }
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_EVENT_DESTROY);
        r.event = fakecuda::ras::address(event);
        delete fake_event;
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        fake_setLastError(result);
        return result;
    }
    auto payload = make_trace_payload();
    if (payload) payload->add_uint64("event_id", fake_event->id);
    TRACE_API_EX(CUDART, "cudaEventDestroy", "stream_op", payload);
    fakecuda::capture::resource_destroyed(FAKECUDA_EVENT, fake_event->id);
    delete fake_event;
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaEventDestroy);

API cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end){
    if (fakecuda::ras::active()) { fakecuda_ras_fail(__func__); return cudaErrorNotSupported; }
    fakecuda::capture::WrapperCall capture_call;
    LOG_DEBUG(CUDART, "cudaEventElapsedTime() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaEventElapsedTime);

API cudaError_t cudaEventQuery(cudaEvent_t event){
    if (fakecuda::ras::active() && !fake_valid_event(event)) { fakecuda_ras_fail("Invalid cudaEventQuery handle"); return cudaErrorInvalidResourceHandle; }
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaEventQuery() called.");
    CUevent_st* fake_event = to_fake_event(event);
    if (fake_event == nullptr) {
        fake_setLastError(cudaErrorInvalidResourceHandle);
        return cudaErrorInvalidResourceHandle;
    }

    cudaError_t result = cudaSuccess;
    bool ready = true;
    if (!fake_event->recorded) {
        // An event that has never been recorded represents no pending work.
        ready = true;
        result = cudaSuccess;
    } else if (!fake_stream_sequence_ready(
                   fake_event->last_stream_handle,
                   fake_event->recorded_stream_seq)) {
        ready = false;
        result = cudaErrorNotReady;
        fake_stream_mark_one_progress(fake_event->last_stream_handle);
    } else if (fake_event->pending_query_polls > 0ULL) {
        // CUDA events become visible only after prior stream work reaches the
        // event marker.  Without executing kernels, fake-cuda advances this
        // virtual stream frontier one polling observation at a time so that
        // framework event-poll loops are preserved without sleeping.
        ready = false;
        result = cudaErrorNotReady;
        fake_event->pending_query_polls -= 1ULL;
    }

    if (fakecuda::ras::active()) {
        if (ready && fake_event->recorded) {
            auto r = fakecuda::ras::record(FAKECUDA_RAS_EVENT_SYNC);
            r.event = fakecuda::ras::address(event); r.flags = 1;
            if (!fakecuda::ras::emit(r)) return cudaErrorUnknown;
        }
        fake_setLastError(result);
        return result;
    }
    auto payload = make_trace_payload();
    if (payload) payload->add_uint64("event_id", fake_event->id);
    if (payload) payload->add_uint64("stream_id", fake_event->last_stream_id);
    if (payload) payload->add_uint64("recorded_stream_seq", fake_event->recorded_stream_seq);
    if (payload) payload->add_uint64("completed_stream_seq", fake_stream_completed_seq(fake_event->last_stream_handle));
    if (payload) payload->add_uint64("pending_query_polls", fake_event->pending_query_polls);
    if (payload) payload->add_int("result", static_cast<int>(result));
    if (payload) payload->add_bool("ready", ready);
    TRACE_API_EX(CUDART, "cudaEventQuery", "stream_op", payload);
    fake_setLastError(result);
    return result;
}
REGISTER_CUDA_FUNCTION(cudaEventQuery);

static unsigned long long event_query_visibility_poll_budget(cudaStream_t stream, unsigned long long seq) {
    static_cast<void>(stream);
    static_cast<void>(seq);
    // Maya's direct-measure line should not invent extra host-side polling
    // after the stream frontier has already advanced to the recorded event.
    // Preserve only the stream-sequence readiness contract; any observed poll
    // loop must come from the framework's real query behavior, not a synthetic
    // fake-cuda visibility budget.
    return 0ULL;
}

static void record_event_on_stream(CUevent_st* fake_event, cudaStream_t stream) {
    if (fake_event == nullptr) {
        return;
    }
    const unsigned long long seq = fake_stream_mark_work_enqueued(stream);
    fake_event->last_stream_id = trace_stream_id(stream);
    fake_event->last_stream_handle = stream;
    fake_event->recorded_stream_seq = seq;
    fake_event->pending_query_polls = event_query_visibility_poll_budget(stream, seq);
    fake_event->recorded = true;
}

API cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0)){
    if (fakecuda::ras::active() && (!fake_valid_event(event) || !fake_valid_stream(stream))) {
        fakecuda_ras_fail("Invalid cudaEventRecord arguments");
        return cudaErrorInvalidValue;
    }
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaEventRecord() called.");
    CUevent_st* fake_event = to_fake_event(event);
    record_event_on_stream(fake_event, stream);
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_EVENT_RECORD);
        r.event = fakecuda::ras::address(event);
        r.stream = fakecuda::ras::address(stream);
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        fake_setLastError(result);
        return result;
    }
    auto payload = make_trace_payload();
    if (payload) payload->add_uint64("event_id", trace_event_id(event));
    if (payload) payload->add_uint64("stream_id", trace_stream_id(stream));
    TRACE_API_EX(CUDART, "cudaEventRecord", "stream_op", payload);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaEventRecord);

API cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream __dv(0), unsigned int flags __dv(0)){
    if (fakecuda::ras::active() && (!fake_valid_event(event) || !fake_valid_stream(stream) || flags != 0)) {
        fakecuda_ras_fail("Invalid cudaEventRecordWithFlags arguments");
        return cudaErrorInvalidValue;
    }
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaEventRecordWithFlags() called.");
    CUevent_st* fake_event = to_fake_event(event);
    record_event_on_stream(fake_event, stream);
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_EVENT_RECORD);
        r.event = fakecuda::ras::address(event);
        r.stream = fakecuda::ras::address(stream);
        r.flags = static_cast<int>(flags);
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        fake_setLastError(result);
        return result;
    }
    auto payload = make_trace_payload();
    if (payload) payload->add_uint64("event_id", trace_event_id(event));
    if (payload) payload->add_uint64("stream_id", trace_stream_id(stream));
    if (payload) payload->add_uint("flags", flags);
    TRACE_API_EX(CUDART, "cudaEventRecordWithFlags", "stream_op", payload);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaEventRecordWithFlags);

API cudaError_t cudaEventSynchronize(cudaEvent_t event){
    if (fakecuda::ras::active() && (!fake_valid_event(event))) {
        fakecuda_ras_fail("Invalid cudaEventSynchronize arguments");
        return cudaErrorInvalidValue;
    }
    fakecuda::capture::WrapperCall capture_call;
    fakecuda::capture::count(fakecuda::capture::Counter::ApiCalls);
    LOG_DEBUG(CUDART, "cudaEventSynchronize() called.");
    CUevent_st* fake_event = to_fake_event(event);
    if (fake_event != nullptr && fake_event->recorded) {
        fake_stream_mark_completed_through(
            fake_event->last_stream_handle,
            fake_event->recorded_stream_seq
        );
        fake_event->pending_query_polls = 0ULL;
    }
    if (fakecuda::ras::active()) {
        auto r = fakecuda::ras::record(FAKECUDA_RAS_EVENT_SYNC);
        r.event = fakecuda::ras::address(event);
        const auto result = fakecuda::ras::emit(r) ? cudaSuccess : cudaErrorUnknown;
        fake_setLastError(result);
        return result;
    }
    auto payload = make_trace_payload();
    if (payload) payload->add_uint64("event_id", trace_event_id(event));
    TRACE_API_EX(CUDART, "cudaEventSynchronize", "stream_op", payload);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaEventSynchronize);
