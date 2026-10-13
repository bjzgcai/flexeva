#include <flexeva/code/runtime.hpp>

#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace flexeva::code {
namespace {
struct SiteKey {
    std::uint64_t site;
    std::uint32_t kind;
    bool operator==(const SiteKey&) const = default;
};
struct SiteHash {
    std::size_t operator()(const SiteKey& key) const noexcept {
        return std::hash<std::uint64_t>{}(key.site) ^ (std::size_t(key.kind) << 1);
    }
};
using Positions = std::unordered_map<SiteKey, std::uint64_t, SiteHash>;
struct Frame {
    std::uint64_t id = 0, token = 0, order = 0;
    bool emitting = false;
    Positions children;
};
struct State {
    InvocationV1 invocation{};
    CallbacksV1 callbacks{};
    decltype(CallbacksV2::metadata) metadata = nullptr;
    void* user = nullptr;
    std::vector<Frame> frames;
    std::vector<PathFrameV1> path;
    std::unordered_set<std::uint64_t> detached_paths;
    Positions roots;
    OccurrenceV1 occurrence{};
    std::uint64_t next_frame = 1, next_order = 0;
    bool failed = false, reported = false, in_callback = false, finished = false;
    std::string failure;
};

thread_local State* active = nullptr;
std::mutex invocation_mutex;
thread_local std::unique_ptr<RunScope> attached_owner;

const CallbacksV1* validate_v2(const CallbacksV2* callbacks) {
    if (!callbacks || callbacks->abi_version != FLEXEVA_CODE_OBSERVER_ABI_VERSION || callbacks->reserved)
        throw Error("invalid Code V2 callbacks", FLEXEVA_CODE_INVALID_ARGUMENT);
    return &callbacks->base;
}

void poison(State& state, const char* message) noexcept {
    if (state.invocation.result) *state.invocation.result = {};
    if (!state.failed) {
        state.failed = true;
        try { state.failure = message; } catch (...) {}
    }
    if (!state.reported) {
        state.reported = true;
        detail::report_failure(&state.callbacks, state.user, message);
    }
}

State& state(bool require_healthy = true) {
    if (!active) throw Error("Code hook used outside an invocation");
    if (active->finished) throw Error("Code hook used after invocation finish");
    if (require_healthy && active->failed)
        throw Error(active->failure.empty() ? "Code invocation already failed" : active->failure,
                    FLEXEVA_CODE_CALLBACK_FAILED);
    return *active;
}

void update_occurrence(State& state) noexcept {
    if (state.frames.empty()) {
        state.occurrence = {};
        state.occurrence.rank = state.invocation.rank;
        return;
    }
    const auto& frame = state.path.back();
    state.occurrence = {frame.site, frame.ordinal, state.frames.back().order,
        frame.kind, state.invocation.rank, state.path.data(), state.path.size()};
}

template<class Function> void callback(State& state, const char* message, Function&& function) {
    if (state.in_callback) {
        poison(state, "Code runtime hook called recursively from a callback");
        throw Error("Code runtime hook called recursively from a callback", FLEXEVA_CODE_CALLBACK_FAILED);
    }
    struct Guard {
        bool& value;
        explicit Guard(bool& flag) : value(flag) { value = true; }
        ~Guard() { value = false; }
    } guard(state.in_callback);
    int result;
    try { result = function(); }
    catch (const std::exception& error) {
        poison(state, error.what());
        throw Error(error.what(), FLEXEVA_CODE_CALLBACK_FAILED);
    } catch (...) {
        poison(state, message);
        throw Error(message, FLEXEVA_CODE_CALLBACK_FAILED);
    }
    if (result != FLEXEVA_CODE_OK) {
        poison(state, message);
        throw Error(message, FLEXEVA_CODE_CALLBACK_FAILED);
    }
}

std::uint64_t push(std::uint64_t site, std::uint32_t kind,
                   std::optional<std::uint64_t> explicit_ordinal = {}) {
    auto& run = state();
    if (run.in_callback) {
        poison(run, "Code runtime hook called recursively from a callback");
        throw Error("Code runtime hook called recursively from a callback", FLEXEVA_CODE_CALLBACK_FAILED);
    }
    if (kind < FLEXEVA_CODE_ENTRY || kind > FLEXEVA_CODE_GROUND)
        throw Error("invalid Code scope kind", FLEXEVA_CODE_INVALID_ARGUMENT);
    auto& positions = run.frames.empty() ? run.roots : run.frames.back().children;
    auto& next = positions[{site, kind}];
    const auto ordinal = explicit_ordinal.value_or(next);
    if (ordinal == std::numeric_limits<std::uint64_t>::max() ||
        run.next_frame == std::numeric_limits<std::uint64_t>::max() ||
        run.next_order == std::numeric_limits<std::uint64_t>::max())
        throw Error("Code occurrence counter overflow");
    if (explicit_ordinal && ordinal < next)
        throw Error("Code path ordinal must increase within a parent occurrence");
    next = ordinal + 1;
    const auto id = run.next_frame++;
    Frame frame;
    frame.id = id;
    frame.order = run.next_order++;
    run.frames.push_back(std::move(frame));
    try { run.path.push_back({site, ordinal, kind, 0}); }
    catch (...) { run.frames.pop_back(); throw; }
    update_occurrence(run);
    return id;
}

void pop(State& run) noexcept {
    run.frames.pop_back();
    run.path.pop_back();
    update_occurrence(run);
}

void close_frame(std::uint64_t id) {
    auto& run = state(false);
    if (run.frames.empty() || run.frames.back().id != id) {
        poison(run, "Code scopes must close in reverse order");
        throw Error("Code scopes must close in reverse order");
    }
    const auto token = run.frames.back().token;
    const bool emitting = run.frames.back().emitting;
    try {
        if (emitting) callback(run, "Code end callback failed", [&] { return run.callbacks.end(run.user, token); });
    } catch (...) { pop(run); throw; }
    pop(run);
}

void close_noexcept(std::uint64_t id) noexcept {
    try { close_frame(id); }
    catch (const std::exception& error) { if (active) poison(*active, error.what()); }
    catch (...) { if (active) poison(*active, "unknown error closing Code scope"); }
}

void close_path_frame(std::uint64_t id) {
    auto& run = state(false);
    if (run.detached_paths.erase(id)) return;
    close_frame(id);
}

void close_path_noexcept(std::uint64_t id) noexcept {
    try { close_path_frame(id); }
    catch (const std::exception& error) { if (active) poison(*active, error.what()); }
    catch (...) { if (active) poison(*active, "unknown error closing Code path scope"); }
}

struct Event {
    std::uint64_t id;
    Event(std::uint64_t site, std::uint32_t kind) : id(push(site, kind)) {}
    ~Event() noexcept { close_noexcept(id); }
};

void validate_inputs(std::span<const ScalarV1> inputs) {
    for (const auto& value : inputs) detail::validate_scalar(value);
}
} // namespace

namespace detail {
void record_failure(const char* message) noexcept { if (active) poison(*active, message); }

void report_failure(const CallbacksV1* callbacks, void* user, const char* message) noexcept {
    if (!callbacks || callbacks->abi_version != FLEXEVA_CODE_ABI_VERSION || !callbacks->fail) return;
    try { (void)callbacks->fail(user, message); } catch (...) {}
}

void validate_scalar(const ScalarV1& value) {
    if (value.reserved || value.kind < FLEXEVA_CODE_BOOL || value.kind > FLEXEVA_CODE_F64 ||
        (value.kind == FLEXEVA_CODE_BOOL && value.value.boolean > 1))
        throw Error("invalid Code ABI scalar", FLEXEVA_CODE_INVALID_ARGUMENT);
}

void definition(std::uint64_t site, ScalarV1 value) {
    validate_scalar(value);
    Event event(site, FLEXEVA_CODE_DEFINITION);
    auto& run = state();
    callback(run, "Code definition callback failed or is missing", [&] {
        return run.callbacks.definition ? run.callbacks.definition(run.user, site, value) : -1;
    });
}

void metadata(std::uint64_t site, std::span<const std::byte> bytes,
              std::uint32_t kind, std::uint32_t flags) {
    auto& run = state();
    if (kind < FLEXEVA_CODE_VALUE_METADATA || kind > FLEXEVA_CODE_VALUE_TENSOR || flags ||
        (bytes.size() && !bytes.data())) {
        poison(run, "invalid Code value metadata");
        throw Error("invalid Code value metadata", FLEXEVA_CODE_INVALID_ARGUMENT);
    }
    bool has_scope = false;
    for (const auto& frame : run.frames) has_scope |= frame.emitting;
    if (!has_scope) {
        poison(run, "Code value metadata requires an active scope");
        throw Error("Code value metadata requires an active scope", FLEXEVA_CODE_INVALID_ARGUMENT);
    }
    const ValueV2 value{kind, flags, bytes.data(), bytes.size()};
    callback(run, "Code metadata callback failed or is missing", [&] {
        return run.metadata ? run.metadata(run.user, &run.occurrence, site, &value) : -1;
    });
}

void finish_paths_for_return() {
    auto& run = state();
    if (run.in_callback) {
        poison(run, "Code runtime hook called recursively from a callback");
        throw Error("Code runtime hook called recursively from a callback", FLEXEVA_CODE_CALLBACK_FAILED);
    }
    while (!run.frames.empty() && !run.frames.back().emitting && run.path.back().kind == FLEXEVA_CODE_LOOP) {
        // Insert before changing the path: allocation failure cannot orphan the
        // current PathScope. At most one entry is held per live detached scope.
        run.detached_paths.insert(run.frames.back().id);
        pop(run);
    }
}

ScalarV1 resolve_ground(std::uint64_t site, std::span<const ScalarV1> inputs, std::uint32_t expected_kind) {
    validate_inputs(inputs);
    Event event(site, FLEXEVA_CODE_GROUND);
    auto& run = state();
    ScalarV1 result{};
    callback(run, "Code Ground callback failed or is missing", [&] {
        return run.callbacks.ground ? run.callbacks.ground(run.user, site, inputs.data(), inputs.size(), &result) : -1;
    });
    try {
        validate_scalar(result);
        if (result.kind != expected_kind)
            throw Error("Code Ground callback returned a different scalar type", FLEXEVA_CODE_CALLBACK_FAILED);
    }
    catch (...) { poison(run, "Code Ground callback returned an invalid scalar"); throw; }
    return result;
}
} // namespace detail

struct RunScope::Impl { State state; };

RunScope::RunScope(const InvocationV1* invocation, const CallbacksV1* callbacks, void* user) {
    if (!invocation || !callbacks || invocation->abi_version != FLEXEVA_CODE_ABI_VERSION ||
        callbacks->abi_version != FLEXEVA_CODE_ABI_VERSION || invocation->reserved || callbacks->reserved ||
        !invocation->world_size || invocation->rank >= invocation->world_size ||
        (invocation->size && !invocation->arguments) || !callbacks->begin || !callbacks->end || !callbacks->fail)
        throw Error("invalid Code invocation or callbacks", FLEXEVA_CODE_INVALID_ARGUMENT);
    validate_inputs({invocation->arguments, invocation->size});
    if (active) {
        poison(*active, "nested Code invocations are unsupported");
        throw Error("nested Code invocations are unsupported");
    }
    impl_ = std::make_unique<Impl>();
    impl_->state.invocation = *invocation;
    impl_->state.callbacks = *callbacks;
    impl_->state.user = user;
    if (!invocation_mutex.try_lock())
        throw Error("concurrent Code invocations are unsupported");
    active = &impl_->state;
    if (invocation->result) *invocation->result = {};
}

RunScope::RunScope(const InvocationV1* invocation, const CallbacksV2* callbacks, void* user)
    : RunScope(invocation, validate_v2(callbacks), user) {
    impl_->state.metadata = callbacks->metadata;
}

RunScope::~RunScope() noexcept {
    if (!impl_) return;
    auto& run = impl_->state;
    if (!run.finished && !run.failed) poison(run, "Code invocation ended without finish");
    while (!run.frames.empty()) close_noexcept(run.frames.back().id);
    active = nullptr;
    invocation_mutex.unlock();
}

void RunScope::finish() {
    auto& run = impl_->state;
    if (active != &run || run.finished) throw Error("Code invocation is not active");
    if (!run.frames.empty() || !run.detached_paths.empty()) {
        poison(run, "Code invocation finished with open scopes");
        throw Error("Code invocation finished with open scopes");
    }
    if (run.failed) throw Error(run.failure.empty() ? "Code invocation failed" : run.failure,
                                FLEXEVA_CODE_CALLBACK_FAILED);
    run.finished = true;
}

void RunScope::fail(const char* message) noexcept { poison(impl_->state, message); }

Scope::Scope(std::uint64_t site, std::uint32_t kind, std::span<const ScalarV1> inputs) {
    validate_inputs(inputs);
    frame_ = push(site, kind);
    auto& run = state();
    std::uint64_t token = 0;
    try {
        callback(run, "Code begin callback failed", [&] {
            return run.callbacks.begin(run.user, &run.occurrence, inputs.data(), inputs.size(), &token);
        });
    } catch (...) { pop(run); throw; }
    run.frames.back().token = token;
    run.frames.back().emitting = true;
    active_ = true;
}

Scope::~Scope() noexcept { if (active_) close_noexcept(frame_); }
void Scope::close() {
    if (!active_) return;
    active_ = false;
    close_frame(frame_);
}

PathScope::PathScope(std::uint64_t site) : frame_(push(site, FLEXEVA_CODE_LOOP)), active_(true) {}
PathScope::PathScope(std::uint64_t site, std::uint64_t ordinal)
    : frame_(push(site, FLEXEVA_CODE_LOOP, ordinal)), active_(true) {}
PathScope::~PathScope() noexcept { if (active_) close_path_noexcept(frame_); }
void PathScope::close() {
    if (!active_) return;
    active_ = false;
    close_path_frame(frame_);
}

const InvocationV1& current_invocation() { return state(false).invocation; }
bool is_active() noexcept { return active != nullptr; }
const OccurrenceV1& current_occurrence() {
    auto& run = state(false);
    if (run.frames.empty()) throw Error("Code invocation has no active occurrence");
    return run.occurrence;
}

bool branch(std::uint64_t site, bool taken) {
    Event event(site, FLEXEVA_CODE_BRANCH);
    auto& run = state();
    callback(run, "Code branch callback failed or is missing", [&] {
        return run.callbacks.branch ? run.callbacks.branch(run.user, site, taken ? 1 : 0) : -1;
    });
    return taken;
}

void communicator(std::uint64_t handle, std::uint64_t membership, std::uint32_t local_rank) {
    auto& run = state();
    callback(run, "Code communicator callback failed or is missing", [&] {
        return run.callbacks.communicator ? run.callbacks.communicator(run.user, handle, membership, local_rank) : -1;
    });
}

Module::Module(const std::string& path) {
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) throw Error(std::string("cannot load Code module: ") + dlerror());
    try {
        (void)dlerror();
        void* symbol = dlsym(handle_, "flexeva_code_invoke_v1");
        const char* error = dlerror();
        if (error || !symbol)
            throw Error(std::string("cannot resolve Code module entry: ") +
                        (error ? error : "missing flexeva_code_invoke_v1"));
        static_assert(sizeof(invoke_) == sizeof(symbol));
        std::memcpy(&invoke_, &symbol, sizeof(invoke_));
        // A V1-only module remains loadable; callers requesting V2 receive an
        // explicit unsupported-entry error rather than losing metadata hooks.
        (void)dlerror();
        void* symbol_v2 = dlsym(handle_, "flexeva_code_invoke_v2");
        if (!dlerror() && symbol_v2) {
            static_assert(sizeof(invoke_v2_) == sizeof(symbol_v2));
            std::memcpy(&invoke_v2_, &symbol_v2, sizeof(invoke_v2_));
        }
        (void)dlerror();
        void* preparation = dlsym(handle_, "flexeva_code_prepare_v1");
        if (!dlerror() && preparation) {
            static_assert(sizeof(prepare_) == sizeof(preparation));
            std::memcpy(&prepare_, &preparation, sizeof(prepare_));
        }
        (void)dlerror();
        void* metadata = dlsym(handle_, "flexeva_code_plan_v1");
        const char* metadata_error = dlerror();
        if (metadata_error || !metadata)
            throw Error(std::string("cannot resolve Code module plan: ") +
                        (metadata_error ? metadata_error : "missing flexeva_code_plan_v1"));
        FlexevaCodePlanV1 read_plan = nullptr;
        static_assert(sizeof(read_plan) == sizeof(metadata));
        std::memcpy(&read_plan, &metadata, sizeof(read_plan));
        const char* plan = read_plan();
        if (!plan || !*plan) throw Error("Code module returned an empty plan");
        plan_ = plan;
    } catch (...) {
        dlclose(handle_);
        handle_ = nullptr;
        invoke_ = nullptr;
        invoke_v2_ = nullptr;
        prepare_ = nullptr;
        throw;
    }
}
Module::~Module() noexcept { if (handle_) dlclose(handle_); }
Module::Module(Module&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), invoke_(std::exchange(other.invoke_, nullptr)),
      invoke_v2_(std::exchange(other.invoke_v2_, nullptr)),
      prepare_(std::exchange(other.prepare_, nullptr)),
      plan_(std::move(other.plan_)) {}
Module& Module::operator=(Module&& other) noexcept {
    if (this != &other) {
        if (handle_) dlclose(handle_);
        handle_ = std::exchange(other.handle_, nullptr);
        invoke_ = std::exchange(other.invoke_, nullptr);
        invoke_v2_ = std::exchange(other.invoke_v2_, nullptr);
        prepare_ = std::exchange(other.prepare_, nullptr);
        plan_ = std::move(other.plan_);
    }
    return *this;
}
int Module::invoke(const InvocationV1& invocation, const CallbacksV1& callbacks, void* user) const noexcept {
    if (!invoke_) {
        detail::report_failure(&callbacks, user, "Code module has been moved from");
        return FLEXEVA_CODE_INVALID_ARGUMENT;
    }
    try { return invoke_(&invocation, &callbacks, user); }
    catch (const std::exception& error) { detail::report_failure(&callbacks, user, error.what()); }
    catch (...) { detail::report_failure(&callbacks, user, "Code module exception crossed the C ABI"); }
    return FLEXEVA_CODE_EXECUTION_FAILED;
}

int Module::invoke_v2(const InvocationV1& invocation, const CallbacksV2& callbacks, void* user) const noexcept {
    if (!invoke_v2_) {
        detail::report_failure(&callbacks.base, user, handle_ ?
            "Code module does not export flexeva_code_invoke_v2" : "Code module has been moved from");
        return FLEXEVA_CODE_INVALID_ARGUMENT;
    }
    try { return invoke_v2_(&invocation, &callbacks, user); }
    catch (const std::exception& error) { detail::report_failure(&callbacks.base, user, error.what()); }
    catch (...) { detail::report_failure(&callbacks.base, user, "Code module exception crossed the C ABI"); }
    return FLEXEVA_CODE_EXECUTION_FAILED;
}

void Module::prepare() const {
    if (!handle_) throw Error("Code module has been moved from", FLEXEVA_CODE_INVALID_ARGUMENT);
    if (!prepare_) return;
    int status;
    try { status = prepare_(); }
    catch (const std::exception& error) { throw Error(std::string("Code preparation failed: ") + error.what()); }
    catch (...) { throw Error("Code preparation exception crossed the C ABI"); }
    if (status != FLEXEVA_CODE_OK)
        throw Error("Code preparation failed with status " + std::to_string(status));
}

namespace {
template<class Callbacks>
int attach(const InvocationV1* invocation, const Callbacks* callbacks, void* user, void** handle) noexcept {
    const CallbacksV1* base;
    if constexpr (std::is_same_v<Callbacks, CallbacksV1>) base = callbacks;
    else base = callbacks ? &callbacks->base : nullptr;
    if (!handle) {
        detail::report_failure(base, user, "Code attachment requires a handle slot");
        return FLEXEVA_CODE_INVALID_ARGUMENT;
    }
    *handle = nullptr;
    try {
        auto owner = std::make_unique<RunScope>(invocation, callbacks, user);
        *handle = owner.get();
        attached_owner = std::move(owner);
        return FLEXEVA_CODE_OK;
    } catch (const Error& error) {
        detail::report_failure(base, user, error.what());
        return error.status();
    } catch (const std::exception& error) {
        detail::report_failure(base, user, error.what());
    } catch (...) { detail::report_failure(base, user, "unknown exception attaching Code runtime"); }
    return FLEXEVA_CODE_EXECUTION_FAILED;
}
} // namespace
} // namespace flexeva::code

extern "C" int flexeva_code_attach_v1(const FlexevaCodeInvocationV1* invocation,
                                       const FlexevaCodeCallbacksV1* callbacks,
                                       void* user, void** handle) {
    return flexeva::code::attach(invocation, callbacks, user, handle);
}
extern "C" int flexeva_code_attach_v2(const FlexevaCodeInvocationV1* invocation,
                                       const FlexevaCodeCallbacksV2* callbacks,
                                       void* user, void** handle) {
    return flexeva::code::attach(invocation, callbacks, user, handle);
}
extern "C" int flexeva_code_fail_v1(void* handle, const char* message) {
    auto& owner = flexeva::code::attached_owner;
    if (!handle || owner.get() != handle || !message) return FLEXEVA_CODE_INVALID_ARGUMENT;
    owner->fail(message);
    return FLEXEVA_CODE_OK;
}
extern "C" int flexeva_code_detach_v1(void* handle) {
    auto& owner = flexeva::code::attached_owner;
    if (!handle || owner.get() != handle) return FLEXEVA_CODE_INVALID_ARGUMENT;
    int result = FLEXEVA_CODE_OK;
    try { owner->finish(); }
    catch (const flexeva::code::Error& error) { owner->fail(error.what()); result = error.status(); }
    catch (const std::exception& error) { owner->fail(error.what()); result = FLEXEVA_CODE_EXECUTION_FAILED; }
    catch (...) { owner->fail("unknown exception detaching Code runtime"); result = FLEXEVA_CODE_EXECUTION_FAILED; }
    owner.reset();
    return result;
}
extern "C" int flexeva_code_is_active_v1(void) { return flexeva::code::is_active() ? 1 : 0; }
extern "C" const FlexevaCodeOccurrenceV1* flexeva_code_current_occurrence_v1(void) noexcept {
    try {
        auto* run = flexeva::code::active;
        if (!run || run->failed || run->finished || run->frames.empty()) return nullptr;
        return &run->occurrence;
    } catch (...) { return nullptr; }
}
