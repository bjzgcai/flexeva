#include "utils.hpp"

// // 辅助函数：demangle C++ 符号
// std::string demangle(const char* mangled_name) {
//     if (!mangled_name) return "";

//     int status = 0;
//     char* demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

//     if (status == 0 && demangled) {
//         std::string result(demangled);
//         free(demangled);
//         return result;
//     }

//     return mangled_name;  // 失败时返回原始名称
// }
