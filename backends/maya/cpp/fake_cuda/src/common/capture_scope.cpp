#include "capture_scope.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
struct Scope {
    uint64_t token;
    uint64_t parent;
    int decision;
    bool closed = false;
    uint64_t bindings = 0;  // registry mutex
    uint64_t ordinal = 0;  // registry mutex
    std::atomic<uint64_t> entered{0}, completed{0};
    std::atomic<uint64_t> pending{0};
    std::array<std::atomic<uint64_t>, 13> counters{};
    std::vector<FakeCudaResourceBinding> resources;  // registry mutex
};
struct Binding { Scope* scope; bool borrowed; };
struct ResourceKey {
    uint64_t kind, handle;
    bool operator==(const ResourceKey& other) const {
        return kind == other.kind && handle == other.handle;
    }
};
struct ResourceHash {
    size_t operator()(const ResourceKey& key) const {
        return std::hash<uint64_t>{}(key.handle) ^ (key.kind * 0x9e3779b9U);
    }
};
struct Registry {
    std::mutex mutex;
    std::atomic<uint64_t> open{0};
    std::atomic<bool> failed{false};
    std::string failure;
    std::unordered_map<uint64_t, std::unique_ptr<Scope>> scopes;
    std::unordered_set<uint64_t> seen_tokens;
    std::unordered_map<ResourceKey, FakeCudaResourceBinding, ResourceHash> resources;
    std::unordered_map<ResourceKey, uint64_t, ResourceHash> generations;
    uint64_t unscoped_ordinal = 0;
};
Registry& registry() {
    // Process lifetime: thread-local wrapper destructors may run during exit.
    static Registry* instance = new Registry();
    return *instance;
}
thread_local std::vector<Binding> owners;
thread_local fakecuda::capture::CallFrame* current_call = nullptr;
thread_local std::array<std::vector<Scope*>, 2> pending_owners;
thread_local std::string error;
std::atomic<FakeCudaOwnerProvider> owner_provider{nullptr};

int fail(const char* message) { error = message; return -1; }
void poison_locked(const char* message) {
    auto& registry_ref = registry();
    if (!registry_ref.failed.load(std::memory_order_relaxed)) {
        registry_ref.failure = message;
        registry_ref.failed.store(true, std::memory_order_release);
        std::fprintf(stderr, "[fakecuda-capture-scope] ERROR: %s\n", message);
        void* frames[24];
        const int count = backtrace(frames, 24);
        backtrace_symbols_fd(frames, count, 2);
    }
}
void poison(const char* message) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    poison_locked(message);
}
Scope* current_owner() {
    if (current_call != nullptr) return static_cast<Scope*>(current_call->owner);
    return owners.empty() ? nullptr : owners.back().scope;
}
Scope* find_scope(uint64_t token) {
    auto& records = registry().scopes;
    auto it = records.find(token);
    return it == records.end() ? nullptr : it->second.get();
}
bool unfinished(const Scope& scope) {
    return scope.entered.load(std::memory_order_acquire) !=
           scope.completed.load(std::memory_order_acquire);
}
}  // namespace

extern "C" int fakecuda_capture_scope_begin(uint64_t token, int decision) {
    if (token == 0) return fail("capture token 0 is reserved for unscoped execution");
    if (decision != FAKECUDA_RECORD && decision != FAKECUDA_RETAIN)
        return fail("capture decision must be record (0) or retain (1)");
    const char* timing = std::getenv("FAKECUDA_HOST_TIMING_MODE");
    if (decision == FAKECUDA_RETAIN && timing != nullptr && std::string(timing) == "sleep")
        return fail("retain capture does not support host timing sleep mode");
    auto& registry_ref = registry();
    std::lock_guard<std::mutex> lock(registry_ref.mutex);
    if (registry_ref.failed.load(std::memory_order_acquire))
        return fail(registry_ref.failure.c_str());
    if (!registry_ref.seen_tokens.insert(token).second)
        return fail("capture token already used in this process");
    auto scope = std::make_unique<Scope>();
    scope->token = token;
    // Python callbacks on a task thread may open a nested source interface
    // while an outer native wrapper still owns the task's propagated token.
    Scope* parent = current_owner();
    if (parent == nullptr) {
        auto provider = owner_provider.load(std::memory_order_acquire);
        const uint64_t parent_token = provider == nullptr ? 0 : provider();
        if (parent_token != 0) {
            parent = find_scope(parent_token);
            if (parent == nullptr || parent->closed) {
                poison_locked("task provider supplied a missing or closed parent scope");
                return fail(registry_ref.failure.c_str());
            }
        }
    }
    scope->parent = parent == nullptr ? 0 : parent->token;
    scope->decision = decision;
    owners.push_back({scope.get(), false});
    registry_ref.scopes.emplace(token, std::move(scope));
    registry_ref.open.fetch_add(1, std::memory_order_release);
    error.clear();
    return 0;
}

extern "C" int fakecuda_capture_scope_end(uint64_t token, FakeCudaCaptureReceipt* receipt) {
    auto& registry_ref = registry();
    std::lock_guard<std::mutex> lock(registry_ref.mutex);
    if (receipt == nullptr) return fail("capture receipt pointer is required");
    if (owners.empty() || owners.back().scope->token != token || owners.back().borrowed)
        return fail("capture scopes must end on their owning thread in nesting order");
    Scope& scope = *owners.back().scope;
    if (unfinished(scope) || scope.bindings != 0 || scope.pending.load(std::memory_order_acquire) != 0)
        return fail("cannot close capture scope with unfinished wrappers, protocols, or helper bindings");
    *receipt = {};
    receipt->schema_version = 3;
    receipt->token = scope.token;
    receipt->parent_token = scope.parent;
    receipt->decision = scope.decision;
    receipt->wrapper_calls = scope.entered.load(std::memory_order_relaxed);
    receipt->completed_calls = scope.completed.load(std::memory_order_relaxed);
    for (size_t i = 0; i < scope.counters.size(); ++i)
        receipt->counters[i] = scope.counters[i].load(std::memory_order_relaxed);
    receipt->resource_count = scope.resources.size();
    scope.closed = true;
    owners.pop_back();
    registry_ref.open.fetch_sub(1, std::memory_order_release);
    if (registry_ref.failed.load(std::memory_order_acquire))
        return fail(registry_ref.failure.c_str());
    error.clear();
    return 0;
}

extern "C" int fakecuda_capture_scope_calls(uint64_t token, uint64_t* calls) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    Scope* scope = find_scope(token);
    if (scope == nullptr || scope->closed || calls == nullptr)
        return fail("cannot inspect a missing or closed capture scope");
    *calls = scope->entered.load(std::memory_order_acquire);
    return 0;
}

extern "C" int fakecuda_capture_scope_activity(uint64_t token, uint64_t* calls, uint64_t* apis) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    Scope* scope = find_scope(token);
    if (scope == nullptr || scope->closed || calls == nullptr || apis == nullptr)
        return fail("cannot inspect a missing or closed capture scope");
    *calls = scope->entered.load(std::memory_order_acquire);
    *apis = scope->counters[0].load(std::memory_order_acquire);
    return 0;
}

extern "C" int fakecuda_capture_scope_bind(uint64_t token) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    Scope* scope = find_scope(token);
    if (scope == nullptr || scope->closed) return fail("cannot bind a missing or closed capture scope");
    if (current_call != nullptr) return fail("bind helper ownership before entering native wrappers");
    owners.push_back({scope, true});
    ++scope->bindings;
    error.clear();
    return 0;
}

extern "C" int fakecuda_capture_scope_unbind(uint64_t token) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    if (owners.empty() || owners.back().scope->token != token || !owners.back().borrowed)
        return fail("helper bindings must unbind in nesting order");
    if (current_call != nullptr) return fail("cannot unbind helper with unfinished native wrappers");
    --owners.back().scope->bindings;
    owners.pop_back();
    error.clear();
    return 0;
}

extern "C" int fakecuda_capture_scope_resource(uint64_t token, size_t index,
                                                FakeCudaResourceBinding* binding) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    Scope* scope = find_scope(token);
    if (scope == nullptr || !scope->closed || binding == nullptr || index >= scope->resources.size())
        return fail("resource receipt requires a closed scope and valid index");
    *binding = scope->resources[index];
    return 0;
}

extern "C" int fakecuda_capture_scope_release(uint64_t token) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    Scope* scope = find_scope(token);
    if (scope == nullptr || !scope->closed) return fail("only a closed capture receipt can be released");
    registry().scopes.erase(token);
    return 0;
}

extern "C" int fakecuda_capture_resource_lookup(uint64_t kind, uint64_t handle,
                                                 FakeCudaResourceBinding* binding) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    auto it = registry().resources.find({kind, handle});
    if (it == registry().resources.end() || binding == nullptr)
        return fail("native resource has no live logical binding");
    *binding = it->second;
    return 0;
}

extern "C" const char* fakecuda_capture_last_error() { return error.c_str(); }

extern "C" int fakecuda_capture_resource_snapshot(FakeCudaResourceBinding* output,
                                                    size_t capacity, size_t* count) {
    auto& registry_ref = registry();
    std::lock_guard<std::mutex> lock(registry_ref.mutex);
    if (count == nullptr) return fail("resource snapshot requires a count pointer");
    if (registry_ref.failed.load(std::memory_order_acquire))
        return fail(registry_ref.failure.c_str());
    if (registry_ref.open.load(std::memory_order_acquire) != 0)
        return fail("resource snapshot requires no open capture scopes");
    *count = registry_ref.resources.size();
    if (output != nullptr) {
        if (capacity < *count) return fail("resource snapshot buffer is too small");
        size_t index = 0;
        for (const auto& item : registry_ref.resources) output[index++] = item.second;
    }
    error.clear();
    return 0;
}

extern "C" int fakecuda_capture_set_owner_provider(FakeCudaOwnerProvider provider) {
    std::lock_guard<std::mutex> lock(registry().mutex);
    if (registry().open.load(std::memory_order_acquire) != 0)
        return fail("owner provider must be installed/uninstalled outside capture scopes");
    if (provider != nullptr && owner_provider.load(std::memory_order_acquire) != nullptr)
        return fail("a task owner provider is already installed");
    owner_provider.store(provider, std::memory_order_release);
    return 0;
}

extern "C" int fakecuda_capture_check() {
    auto& registry_ref = registry();
    std::lock_guard<std::mutex> lock(registry_ref.mutex);
    if (registry_ref.failed.load(std::memory_order_acquire))
        return fail(registry_ref.failure.c_str());
    if (registry_ref.open.load(std::memory_order_acquire) != 0)
        return fail("capture execution still has open scopes");
    error.clear();
    return 0;
}

namespace fakecuda::capture {
void enter_call(CallFrame& frame) {
    frame.previous = current_call;
    if (!owners.empty()) {
        frame.owner = owners.back().scope;
    } else if (current_call != nullptr && current_call->owner != nullptr) {
        frame.owner = current_call->owner;
    } else {
        auto provider = owner_provider.load(std::memory_order_acquire);
        const uint64_t token = provider == nullptr ? 0 : provider();
        if (token != 0) {
            // Pin the wrapper count under the same lock used by scope_end and
            // release. A task snapshot alone does not keep a closed scope live.
            std::lock_guard<std::mutex> lock(registry().mutex);
            Scope* scope = find_scope(token);
            if (scope == nullptr || scope->closed) {
                poison_locked("task provider supplied a missing or closed capture scope");
            } else {
                frame.owner = scope;
                scope->entered.fetch_add(1, std::memory_order_relaxed);
                current_call = &frame;
                return;
            }
        } else if (registry().open.load(std::memory_order_acquire) != 0) {
            poison("native wrapper on unowned helper thread; explicitly bind a capture token");
        }
    }
    current_call = &frame;
    if (frame.owner != nullptr)
        static_cast<Scope*>(frame.owner)->entered.fetch_add(1, std::memory_order_relaxed);
}

void leave_call(CallFrame& frame) {
    if (current_call != &frame) poison("native wrapper stack is not nested");
    if (frame.owner != nullptr)
        static_cast<Scope*>(frame.owner)->completed.fetch_add(1, std::memory_order_release);
    current_call = frame.previous;
}

int scoped_decision() {
    if (registry().failed.load(std::memory_order_acquire)) return FAKECUDA_RETAIN;
    Scope* scope = current_owner();
    return scope == nullptr ? -1 : scope->decision;
}
uint64_t scoped_token() { Scope* scope = current_owner(); return scope == nullptr ? 0 : scope->token; }
void scope_count(size_t counter, uint64_t amount) {
    Scope* scope = current_owner();
    if (scope != nullptr && counter < scope->counters.size())
        scope->counters[counter].fetch_add(amount, std::memory_order_relaxed);
}

void resource_created(uint64_t kind, uint64_t handle, uint64_t extent_bytes, uint64_t api_handle) {
    if (fakecuda_ras_active()) return;
    if (handle == 0) return;
    auto& registry_ref = registry();
    std::lock_guard<std::mutex> lock(registry_ref.mutex);
    const ResourceKey key{kind, handle};
    if (registry_ref.resources.find(key) != registry_ref.resources.end()) {
        poison_locked("native resource handle created twice without destruction");
        return;
    }
    Scope* scope = current_owner();
    FakeCudaResourceBinding binding{kind, FAKECUDA_CREATED, handle,
        scope == nullptr ? 0 : scope->token,
        scope == nullptr ? ++registry_ref.unscoped_ordinal : ++scope->ordinal,
        ++registry_ref.generations[key], extent_bytes, api_handle == 0 ? handle : api_handle};
    registry_ref.resources.emplace(key, binding);
    if (scope != nullptr) scope->resources.push_back(binding);
}

void resource_destroyed(uint64_t kind, uint64_t handle) {
    if (fakecuda_ras_active()) return;
    retire_legacy_resource(kind, handle);
}

void retire_legacy_resource(uint64_t kind, uint64_t handle) {
    if (handle == 0) return;
    auto& registry_ref = registry();
    std::lock_guard<std::mutex> lock(registry_ref.mutex);
    auto it = registry_ref.resources.find({kind, handle});
    if (it == registry_ref.resources.end()) {
        poison_locked("native resource destroyed without a live binding");
        return;
    }
    auto binding = it->second;
    binding.action = FAKECUDA_DESTROYED;
    Scope* scope = fakecuda_ras_active() ? nullptr : current_owner();
    if (scope != nullptr) scope->resources.push_back(binding);
    registry_ref.resources.erase(it);
}

void pending_begin(PendingKind kind) {
    if (fakecuda_ras_active()) return;
    Scope* scope = current_owner();
    pending_owners[static_cast<size_t>(kind)].push_back(scope);
    if (scope != nullptr) scope->pending.fetch_add(1, std::memory_order_relaxed);
}

void pending_end(PendingKind kind) {
    if (fakecuda_ras_active()) return;
    Scope* scope = current_owner();
    auto& stack = pending_owners[static_cast<size_t>(kind)];
    if (stack.empty()) {
        if (scope != nullptr) poison("native multi-call protocol ended without its scoped beginning");
        return;  // Preserve unscoped legacy behavior.
    }
    Scope* producer = stack.back();
    stack.pop_back();
    if (producer != nullptr) producer->pending.fetch_sub(1, std::memory_order_release);
    if (producer != scope) poison("native multi-call protocol crossed capture ownership");
}
}  // namespace fakecuda::capture
