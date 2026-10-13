#pragma once
#include <sys/mman.h>
#include <map>
#include <mutex>
#include <atomic>
#include <unordered_set>
#include <cstdint>
#include "fake_types.h"
// 为什么这样设计：
// 1. 用堆分配的指针直接作为handle，天然唯一，不需要额外映射
// 2. 记录capture状态是必须的，PyTorch会查询
// 3. priority需要记录，因为有GetPriority查询
struct FakeStream {
    unsigned long long id;                          // 用于日志识别
    unsigned int flags;                             // 创建时的flags
    int priority;                                   // 创建时的priority
    std::atomic<unsigned long long> enqueued_seq{0};
    std::atomic<unsigned long long> completed_seq{0};

    // Capture相关（PyTorch Graph模式必须）
    bool isCapturing = false;
    cudaStreamCaptureMode captureMode = cudaStreamCaptureModeGlobal;
    unsigned long long captureId = 0;               // 每次capture唯一ID
};

struct CUevent_st {
    unsigned long long id;
    unsigned int flags;
    cudaStream_t last_stream_handle;
    unsigned long long recorded_stream_seq;
    unsigned long long pending_query_polls;
    bool recorded;
};

// 全局状态
inline std::atomic<unsigned long long> g_streamIdCounter{1};
inline std::atomic<unsigned long long> g_captureIdCounter{1};
inline std::atomic<unsigned long long> g_eventIdCounter{1};
// 用于有效性检查，防止野指针
inline std::unordered_set<FakeStream*> g_validStreams;
inline std::mutex g_streamMutex;
inline std::unordered_set<CUevent_st*> g_validEvents;
inline std::mutex g_eventMutex;
// nullptr代表default stream，不在这里管理
static inline FakeStream* to_fake_stream(cudaStream_t stream) {
    if (stream == nullptr) return nullptr;
    return reinterpret_cast<FakeStream*>(stream);
}
static inline cudaStream_t to_cuda_stream(FakeStream* stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

static inline CUevent_st* to_fake_event(cudaEvent_t event) {
    if (event == nullptr) return nullptr;
    return reinterpret_cast<CUevent_st*>(event);
}

static inline cudaEvent_t to_cuda_event(CUevent_st* event) {
    return reinterpret_cast<cudaEvent_t>(event);
}

static inline bool fake_valid_stream(cudaStream_t stream) {
    if (!stream) return true;
    std::lock_guard<std::mutex> lock(g_streamMutex);
    return g_validStreams.find(to_fake_stream(stream)) != g_validStreams.end();
}

static inline bool fake_valid_event(cudaEvent_t event) {
    std::lock_guard<std::mutex> lock(g_eventMutex);
    return event && g_validEvents.find(to_fake_event(event)) != g_validEvents.end();
}

static inline unsigned long long fake_stream_mark_work_enqueued(
    cudaStream_t stream,
    unsigned long long work_units = 1ULL
) {
    FakeStream* fake_stream = to_fake_stream(stream);
    if (fake_stream == nullptr) {
        return 1ULL;
    }
    const unsigned long long units = work_units == 0ULL ? 1ULL : work_units;
    return fake_stream->enqueued_seq.fetch_add(units, std::memory_order_relaxed) + units;
}

static inline unsigned long long fake_stream_enqueued_seq(cudaStream_t stream) {
    FakeStream* fake_stream = to_fake_stream(stream);
    return fake_stream ? fake_stream->enqueued_seq.load(std::memory_order_relaxed) : 1ULL;
}

static inline unsigned long long fake_stream_completed_seq(cudaStream_t stream) {
    FakeStream* fake_stream = to_fake_stream(stream);
    return fake_stream ? fake_stream->completed_seq.load(std::memory_order_relaxed) : 0ULL;
}

static inline void fake_stream_mark_completed_through(cudaStream_t stream, unsigned long long seq) {
    FakeStream* fake_stream = to_fake_stream(stream);
    if (fake_stream == nullptr) {
        return;
    }
    unsigned long long observed = fake_stream->completed_seq.load(std::memory_order_relaxed);
    while (observed < seq &&
           !fake_stream->completed_seq.compare_exchange_weak(
               observed,
               seq,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

static inline void fake_stream_mark_one_progress(cudaStream_t stream) {
    FakeStream* fake_stream = to_fake_stream(stream);
    if (fake_stream == nullptr) {
        return;
    }
    unsigned long long completed = fake_stream->completed_seq.load(std::memory_order_relaxed);
    const unsigned long long enqueued = fake_stream->enqueued_seq.load(std::memory_order_relaxed);
    while (completed < enqueued &&
           !fake_stream->completed_seq.compare_exchange_weak(
               completed,
               completed + 1ULL,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

static inline void fake_mark_all_streams_completed() {
    std::lock_guard<std::mutex> lock(g_streamMutex);
    for (FakeStream* fake_stream : g_validStreams) {
        if (fake_stream == nullptr) {
            continue;
        }
        const unsigned long long enqueued =
            fake_stream->enqueued_seq.load(std::memory_order_relaxed);
        unsigned long long observed =
            fake_stream->completed_seq.load(std::memory_order_relaxed);
        while (observed < enqueued &&
               !fake_stream->completed_seq.compare_exchange_weak(
                   observed,
                   enqueued,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
    }
}

static inline bool fake_stream_sequence_ready(cudaStream_t stream, unsigned long long seq) {
    if (stream == nullptr) {
        return seq <= 1ULL;
    }
    return fake_stream_completed_seq(stream) >= seq;
}

static inline unsigned long long fake_stream_pending_depth(cudaStream_t stream, unsigned long long seq) {
    const unsigned long long completed = fake_stream_completed_seq(stream);
    if (seq <= completed) {
        return 0ULL;
    }
    return seq - completed;
}

static inline void fake_mark_ready_events_visible(cudaStream_t stream_filter = nullptr) {
    std::lock_guard<std::mutex> lock(g_eventMutex);
    for (CUevent_st* fake_event : g_validEvents) {
        if (fake_event == nullptr || !fake_event->recorded) {
            continue;
        }
        if (stream_filter != nullptr && fake_event->last_stream_handle != stream_filter) {
            continue;
        }
        if (fake_stream_sequence_ready(
                fake_event->last_stream_handle,
                fake_event->recorded_stream_seq)) {
            fake_event->pending_query_polls = 0ULL;
        }
    }
}
