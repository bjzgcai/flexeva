#pragma once

#include <flexeva/code/runtime.h>

#include <functional>
#include <initializer_list>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace flexeva::code {
using ScalarV1 = FlexevaCodeScalarV1;
using InvocationV1 = FlexevaCodeInvocationV1;
using CallbacksV1 = FlexevaCodeCallbacksV1;
using CallbacksV2 = FlexevaCodeCallbacksV2;
using ValueV2 = FlexevaCodeValueV2;
using OccurrenceV1 = FlexevaCodeOccurrenceV1;
using PathFrameV1 = FlexevaCodePathFrameV1;

class Error : public std::runtime_error {
public:
    explicit Error(std::string message, int status = FLEXEVA_CODE_EXECUTION_FAILED)
        : std::runtime_error(std::move(message)), status_(status) {}
    int status() const noexcept { return status_; }
private:
    int status_;
};

namespace detail {
void report_failure(const CallbacksV1*, void*, const char*) noexcept;
void record_failure(const char*) noexcept;
void validate_scalar(const ScalarV1&);
void definition(std::uint64_t site, ScalarV1);
void metadata(std::uint64_t site, std::span<const std::byte>,
              std::uint32_t kind = FLEXEVA_CODE_VALUE_METADATA, std::uint32_t flags = 0);
// Retire trailing path-only frames before an early-return cleanup Scope opens.
// Their owning PathScope objects still close during ordinary C++ unwinding.
// Emitting scopes and callback events are never retired by this operation.
void finish_paths_for_return();
ScalarV1 resolve_ground(std::uint64_t site, std::span<const ScalarV1>, std::uint32_t expected_kind);
template<class> inline constexpr bool unsupported_scalar = false;
template<class Target, class Source> Target checked_scalar_cast(Source value) {
    if constexpr (sizeof(Target) < sizeof(Source)) {
        if (value < static_cast<Source>(std::numeric_limits<Target>::min()) ||
            value > static_cast<Source>(std::numeric_limits<Target>::max()))
            throw Error("Code ABI scalar value is out of range", FLEXEVA_CODE_INVALID_ARGUMENT);
    }
    return static_cast<Target>(value);
}
} // namespace detail

template<class T> ScalarV1 scalar(T value) {
    using Value = std::remove_cv_t<T>;
    ScalarV1 result{};
    if constexpr (std::is_enum_v<Value>) {
        return scalar(static_cast<std::underlying_type_t<Value>>(value));
    } else if constexpr (std::is_same_v<Value, bool>) {
        result.kind = FLEXEVA_CODE_BOOL;
        result.value.boolean = value ? 1 : 0;
    } else if constexpr (std::is_integral_v<Value> && sizeof(Value) <= 4 && std::is_signed_v<Value>) {
        result.kind = FLEXEVA_CODE_I32;
        result.value.i32 = value;
    } else if constexpr (std::is_integral_v<Value> && sizeof(Value) <= 4 && std::is_unsigned_v<Value>) {
        result.kind = FLEXEVA_CODE_U32;
        result.value.u32 = value;
    } else if constexpr (std::is_integral_v<Value> && sizeof(Value) == 8 && std::is_signed_v<Value>) {
        result.kind = FLEXEVA_CODE_I64;
        result.value.i64 = value;
    } else if constexpr (std::is_integral_v<Value> && sizeof(Value) == 8 && std::is_unsigned_v<Value>) {
        result.kind = FLEXEVA_CODE_U64;
        result.value.u64 = value;
    } else if constexpr (std::is_same_v<Value, float>) {
        result.kind = FLEXEVA_CODE_F32;
        result.value.f32 = value;
    } else if constexpr (std::is_same_v<Value, double>) {
        result.kind = FLEXEVA_CODE_F64;
        result.value.f64 = value;
    } else {
        static_assert(detail::unsupported_scalar<Value>, "unsupported Code ABI scalar type");
    }
    return result;
}

template<class T> T scalar_cast(const ScalarV1& value) {
    static_assert(!std::is_reference_v<T>, "Code ABI arguments are passed by value");
    detail::validate_scalar(value);
    if (value.kind != scalar(T{}).kind)
        throw Error("Code ABI scalar type mismatch", FLEXEVA_CODE_INVALID_ARGUMENT);
    if constexpr (std::is_enum_v<T>) return static_cast<T>(scalar_cast<std::underlying_type_t<T>>(value));
    else if constexpr (std::is_same_v<std::remove_cv_t<T>, bool>) return value.value.boolean != 0;
    else if constexpr (std::is_integral_v<T> && sizeof(T) <= 4 && std::is_signed_v<T>) return detail::checked_scalar_cast<T>(value.value.i32);
    else if constexpr (std::is_integral_v<T> && sizeof(T) <= 4 && std::is_unsigned_v<T>) return detail::checked_scalar_cast<T>(value.value.u32);
    else if constexpr (std::is_integral_v<T> && sizeof(T) == 8 && std::is_signed_v<T>) return value.value.i64;
    else if constexpr (std::is_integral_v<T> && sizeof(T) == 8 && std::is_unsigned_v<T>) return value.value.u64;
    else if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) return value.value.f32;
    else if constexpr (std::is_same_v<std::remove_cv_t<T>, double>) return value.value.f64;
}

template<class T> T argument(const InvocationV1& invocation, std::size_t index) {
    if (index >= invocation.size || !invocation.arguments)
        throw Error("Code ABI argument index is out of range", FLEXEVA_CODE_INVALID_ARGUMENT);
    return scalar_cast<T>(invocation.arguments[index]);
}

// An invocation exclusively owns the process-local capture context. Callbacks
// run synchronously on this thread; reentry and concurrent invocations fail.
class RunScope {
public:
    RunScope(const InvocationV1*, const CallbacksV1*, void* user);
    RunScope(const InvocationV1*, const CallbacksV2*, void* user);
    ~RunScope() noexcept;
    RunScope(const RunScope&) = delete;
    RunScope& operator=(const RunScope&) = delete;
    void finish();
    void fail(const char*) noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Scope closes normally with close(), and also closes during stack unwinding.
// A failing destructor records a sticky error for RunScope::finish().
class Scope {
public:
    explicit Scope(std::uint64_t site, std::uint32_t kind = FLEXEVA_CODE_BLOCK,
                   std::span<const ScalarV1> inputs = {});
    Scope(std::uint64_t site, std::uint32_t kind, std::initializer_list<ScalarV1> inputs)
        : Scope(site, kind, std::span<const ScalarV1>(inputs.begin(), inputs.size())) {}
    ~Scope() noexcept;
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    void close();
private:
    std::uint64_t frame_ = 0;
    bool active_ = false;
};

// Adds an activation/iteration to the dynamic path without requesting a new
// output scope. The automatic form increments only this site's local ordinal.
class PathScope {
public:
    explicit PathScope(std::uint64_t site);
    PathScope(std::uint64_t site, std::uint64_t ordinal);
    ~PathScope() noexcept;
    PathScope(const PathScope&) = delete;
    PathScope& operator=(const PathScope&) = delete;
    void close();
private:
    std::uint64_t frame_ = 0;
    bool active_ = false;
};

const InvocationV1& current_invocation();
const OccurrenceV1& current_occurrence();
bool is_active() noexcept;
bool branch(std::uint64_t site, bool taken);
void communicator(std::uint64_t handle, std::uint64_t membership, std::uint32_t local_rank);

template<class T> T define(std::uint64_t site, T value) {
    detail::definition(site, scalar(value));
    return value;
}

// Observe a direct scalar mutation after the original expression executes once.
// Postfix operators return their saved old value; prefix and compound assignment
// expressions retain their original lvalue result and aliasing behavior.
template<class T, class Function>
decltype(auto) update(std::uint64_t site, T& target, Function&& operation) {
    using Result = std::invoke_result_t<Function>;
    using Value = std::remove_cv_t<std::remove_reference_t<Result>>;
    static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>, "Code update requires a scalar target");
    static_assert(std::is_arithmetic_v<Value> || std::is_enum_v<Value>, "Code update requires a scalar result");
    static_assert(!std::is_rvalue_reference_v<Result>, "Code update does not accept an rvalue reference result");
    if constexpr (std::is_lvalue_reference_v<Result>) {
        auto& result = std::invoke(std::forward<Function>(operation));
        detail::definition(site, scalar(target));
        return result;
    } else {
        Result result = std::invoke(std::forward<Function>(operation));
        detail::definition(site, scalar(target));
        return result;
    }
}

template<class T> T ground(std::uint64_t site, std::span<const ScalarV1> inputs = {}) {
    const auto result = detail::resolve_ground(site, inputs, scalar(T{}).kind);
    try { return scalar_cast<T>(result); }
    catch (...) { detail::record_failure("Code Ground result is outside the requested scalar range"); throw; }
}

template<class T> T ground(std::uint64_t site, std::initializer_list<ScalarV1> inputs) {
    return ground<T>(site, std::span<const ScalarV1>(inputs.begin(), inputs.size()));
}

template<class Function>
decltype(auto) call(std::uint64_t site, Function&& function, std::span<const ScalarV1> inputs = {}) {
    Scope scope(site, FLEXEVA_CODE_EMISSION, inputs);
    // Return the original expression directly: preserves references and permits
    // guaranteed copy elision. Scope destruction runs before the caller resumes;
    // a failed end callback poisons the invocation even if the value is returned.
    return std::invoke(std::forward<Function>(function));
}

// Instrumented helpers may also be called by ordinary, uncaptured host code.
// A failed attached invocation remains active: hooks then propagate its sticky
// error instead of silently executing without observation.
template<class Function>
decltype(auto) passive_call(std::uint64_t site, Function&& function,
                           std::span<const ScalarV1> inputs = {}) {
    if (!is_active()) return std::invoke(std::forward<Function>(function));
    return call(site, std::forward<Function>(function), inputs);
}

template<class Function>
int invoke_guarded(const InvocationV1* invocation, const CallbacksV1* callbacks,
                   void* user, Function&& function) noexcept {
    std::optional<RunScope> run;
    try {
        run.emplace(invocation, callbacks, user);
        std::invoke(std::forward<Function>(function));
        run->finish();
        return FLEXEVA_CODE_OK;
    } catch (const Error& error) {
        if (run) run->fail(error.what());
        else detail::report_failure(callbacks, user, error.what());
        return error.status();
    } catch (const std::exception& error) {
        if (run) run->fail(error.what());
        else detail::report_failure(callbacks, user, error.what());
        return FLEXEVA_CODE_EXECUTION_FAILED;
    } catch (...) {
        constexpr auto message = "unknown exception in Code invocation";
        if (run) run->fail(message);
        else detail::report_failure(callbacks, user, message);
        return FLEXEVA_CODE_EXECUTION_FAILED;
    }
}

template<class Function>
int invoke_guarded_v2(const InvocationV1* invocation, const CallbacksV2* callbacks,
                      void* user, Function&& function) noexcept {
    std::optional<RunScope> run;
    try {
        run.emplace(invocation, callbacks, user);
        std::invoke(std::forward<Function>(function));
        run->finish();
        return FLEXEVA_CODE_OK;
    } catch (const Error& error) {
        if (run) run->fail(error.what());
        else detail::report_failure(callbacks ? &callbacks->base : nullptr, user, error.what());
        return error.status();
    } catch (const std::exception& error) {
        if (run) run->fail(error.what());
        else detail::report_failure(callbacks ? &callbacks->base : nullptr, user, error.what());
        return FLEXEVA_CODE_EXECUTION_FAILED;
    } catch (...) {
        constexpr auto message = "unknown exception in Code invocation";
        if (run) run->fail(message);
        else detail::report_failure(callbacks ? &callbacks->base : nullptr, user, message);
        return FLEXEVA_CODE_EXECUTION_FAILED;
    }
}

class Module {
public:
    explicit Module(const std::string& path);
    ~Module() noexcept;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) noexcept;
    Module& operator=(Module&&) noexcept;
    int invoke(const InvocationV1&, const CallbacksV1&, void* user) const noexcept;
    int invoke_v2(const InvocationV1&, const CallbacksV2&, void* user) const noexcept;
    // Optional fixed backend preparation. The evaluator chooses its capture
    // scope and timing; this never invokes the mutable training entry.
    void prepare() const;
    const std::string& plan_json() const noexcept { return plan_; }
private:
    void* handle_ = nullptr;
    FlexevaCodeInvokeV1 invoke_ = nullptr;
    FlexevaCodeInvokeV2 invoke_v2_ = nullptr;
    FlexevaCodePrepareV1 prepare_ = nullptr;
    std::string plan_;
};
} // namespace flexeva::code
