#pragma once

#include <flexeva/code/runtime.hpp>
#include <ATen/ATen.h>
#include <c10/core/AutogradState.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <vector>

namespace flexeva::code {
namespace detail {
// Shape-only descriptions are canonical values. Tensor addresses, storage
// identities and device contents never enter historical Code/RAS state.
inline void torch_value(std::vector<ScalarV1>& into, const at::Tensor& value) {
    into.push_back(scalar(std::int32_t(100)));
    into.push_back(scalar(value.defined()));
    if(!value.defined())return;
    into.push_back(scalar(std::int32_t(value.scalar_type())));
    into.push_back(scalar(std::int32_t(value.device().type())));
    into.push_back(scalar(value.requires_grad()));
    into.push_back(scalar(std::int64_t(value.dim())));
    for(auto size:value.sizes())into.push_back(scalar(size));
    if(value.layout()==at::kStrided)for(auto stride:value.strides())into.push_back(scalar(stride));
}
template<class T> void torch_value(std::vector<ScalarV1>& into, const T& value) {
    if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T>) {
        into.push_back(scalar(std::int32_t(101)));into.push_back(scalar(value));
    } else if constexpr (requires {value->parameters();}) {
        into.push_back(scalar(std::int32_t(102)));
        if constexpr(requires {value->is_training();})into.push_back(scalar(value->is_training()));
        const auto parameters=value->parameters();into.push_back(scalar(std::uint64_t(parameters.size())));
        for(const auto& parameter:parameters)torch_value(into,parameter);
    } else if constexpr (requires {value.param_groups();}) {
        into.push_back(scalar(std::int32_t(106)));
        into.push_back(scalar(std::uint64_t(value.param_groups().size())));
        for(const auto& group:value.param_groups()) {
            into.push_back(scalar(group.options().get_lr()));
            into.push_back(scalar(std::uint64_t(group.params().size())));
            for(const auto& parameter:group.params())torch_value(into,parameter);
        }
    } else if constexpr (requires {value.parameters();}) {
        into.push_back(scalar(std::int32_t(103)));
        if constexpr(requires {value.is_training();})into.push_back(scalar(value.is_training()));
        const auto parameters=value.parameters();into.push_back(scalar(std::uint64_t(parameters.size())));
        for(const auto& parameter:parameters)torch_value(into,parameter);
    } else if constexpr (requires {value.begin();value.end();value.size();} &&
                         !std::is_convertible_v<T,std::string>) {
        into.push_back(scalar(std::int32_t(104)));into.push_back(scalar(std::uint64_t(value.size())));
        for(const auto& element:value)torch_value(into,element);
    } else {
        // Opaque values are deliberately marked. The frontend admits their
        // effects conservatively; this marker is not a completeness proof.
        into.push_back(scalar(std::int32_t(105)));
    }
}
}
template<class... T> std::vector<ScalarV1> torch_values(const T&... values) {
    std::vector<ScalarV1> result;(detail::torch_value(result,values),...);return result;
}
template<class Function>
decltype(auto) torch_call(std::uint64_t site,Function&& function,const std::vector<ScalarV1>& inputs) {
    return passive_call(site,std::forward<Function>(function),inputs);
}
inline void torch_definition(std::uint64_t site,const std::vector<ScalarV1>& inputs) {
    if(is_active()){Scope scope(site,FLEXEVA_CODE_BLOCK,inputs);scope.close();}
}
class TorchPathScope {
public:
    explicit TorchPathScope(std::uint64_t site) {if(is_active())scope_.emplace(site);}
private:
    std::optional<PathScope> scope_;
};
inline bool torch_branch(std::uint64_t site,bool value) {return is_active()?branch(site,value):value;}
template<class T> T torch_define(std::uint64_t site,T value) {return is_active()?define(site,value):value;}
template<class T,class Function>
decltype(auto) torch_update(std::uint64_t site,T& target,Function&& function) {
    if(!is_active())return std::invoke(std::forward<Function>(function));
    return update(site,target,std::forward<Function>(function));
}
// Capture runs synchronously on the host thread, including autograd work.
// Restores the caller's setting even on exceptions and does not open a Session.
inline void prepare_torch() {
    const auto previous=c10::AutogradState::get_tls_state().get_multithreading_enabled();
    c10::AutogradState::get_tls_state().set_multithreading_enabled(false);
    try {
        // A fixed FakeCUDA evaluator context: allocator pool history must not
        // change the modeled native trace of a representative rank.
        c10::cuda::CUDACachingAllocator::enable(false);
        auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
        auto input=at::ones({8,8},options);input.set_requires_grad(true);
        auto output=at::mm(input,input);output.square().sum().backward();
    } catch (...) {c10::AutogradState::get_tls_state().set_multithreading_enabled(previous);throw;}
    c10::AutogradState::get_tls_state().set_multithreading_enabled(previous);
}
class TorchWindow;
inline thread_local TorchWindow* active_torch_window=nullptr;
class TorchWindow {
public:
    TorchWindow(std::uint64_t site,std::uint64_t cleanup_site,const std::vector<ScalarV1>& inputs)
        : cleanup_site_(cleanup_site), parent_(active_torch_window), previous_(c10::AutogradState::get_tls_state().get_multithreading_enabled()) {
        c10::AutogradState::get_tls_state().set_multithreading_enabled(false);
        try {if(is_active())scope_.emplace(site,FLEXEVA_CODE_ENTRY,inputs);}
        catch (...) {c10::AutogradState::get_tls_state().set_multithreading_enabled(previous_);throw;}
        active_torch_window=this;
    }
    void cleanup() {if(scope_&&!cleanup_){detail::finish_paths_for_return();cleanup_.emplace(cleanup_site_,FLEXEVA_CODE_BLOCK);}}
    ~TorchWindow() {cleanup_.reset();scope_.reset();active_torch_window=parent_;c10::AutogradState::get_tls_state().set_multithreading_enabled(previous_);}
private:
    std::uint64_t cleanup_site_;
    TorchWindow* parent_;
    bool previous_;
    std::optional<Scope> scope_,cleanup_;
};
inline void torch_cleanup() {if(active_torch_window)active_torch_window->cleanup();}
// A temporary at the return full-expression closes its trace only after the
// result has been evaluated, before automatic local destruction starts.
struct TorchReturnGuard {
    ~TorchReturnGuard() noexcept {
        try {torch_cleanup();}catch(const std::exception& error){detail::record_failure(error.what());}
        catch(...){detail::record_failure("Torch return cleanup failed");}
    }
};
} // namespace flexeva::code
