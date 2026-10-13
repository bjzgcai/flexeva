#pragma once
#include "utils.hpp"
#include <unordered_map>

// 自动注册函数地址
class FunctionRegistry {
private:
    std::unordered_map<std::string, void*> function_map_;
    mutable std::mutex mu_;
    FunctionRegistry() = default;

public:
    static FunctionRegistry& instance() {
        static FunctionRegistry instance;
        return instance;
    }

    void register_function(std::string_view name, void* address) {
        std::lock_guard lk(mu_);
        function_map_[std::string{name}] = address;
    }

    void* get_function(std::string_view name) const {
        std::lock_guard lk(mu_);
        auto it = function_map_.find(std::string{name});
        return it != function_map_.end() ? it->second : nullptr;
    }
};

// 宏：注册 CUDA 函数到函数注册表
#define REGISTER_CUDA_FUNCTION(func) \
    static inline void* reg_##func = ([]{ \
        FunctionRegistry::instance().register_function(#func, reinterpret_cast<void*>(&func)); \
    }(), nullptr)
