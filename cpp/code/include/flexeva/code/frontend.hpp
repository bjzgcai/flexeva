#pragma once

#include <string>
#include <vector>

namespace flexeva::code {

// Analysis never loads or executes the workload. Compilation inputs, including
// transitively included files, participate in the resulting source identity.
struct CppSource {
    std::string path;
    std::string entry;
    std::string rank_parameter = "rank";
    std::string world_parameter = "world_size";
    std::string compilation_database;
    std::vector<std::string> arguments;
    std::vector<std::string> link_arguments;
    std::vector<std::string> ground_functions;
    // Restricted Tensor/Module effects. Unknown calls execute conservatively.
    bool torch = false;
    bool extension = false;
    std::vector<std::string> mutable_headers;
    // Restricted sources are analyzed as one owned closure. Internal symbols
    // must be unambiguous across this closure. External libraries stay intact.
    std::vector<std::string> extra_sources;
};
struct CppAnalysis {
    std::string plan_json;
    std::string instrumented_source;
    std::vector<std::string> compile_arguments;
};
CppAnalysis analyze_cpp(const CppSource&);

} // namespace flexeva::code
