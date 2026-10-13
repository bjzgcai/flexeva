#pragma once

#include <flexeva/code/runtime.hpp>
#include <flexeva/maya/paper_runtime.hpp>
#include <memory>
#include <vector>

namespace flexeva::code {

struct CppInvocation {
    std::uint32_t world_size = 1;
    std::vector<std::uint32_t> ranks; // Empty means every rank in [0, world_size).
    std::vector<ScalarV1> arguments;
    bool share_ranks = true;
};
struct CppExecutionStatistics {
    std::uint64_t invocations = 0, occurrences = 0, ground_calls = 0;
    std::vector<std::uint32_t> representatives;
    std::vector<ScalarV1> returns;
};

// A compiled frontend artifact owns its immutable source plan and native code.
// Each input() creates isolated callback state; historical RAS versions never
// refer to transient workload pointers or to these execution statistics.
class CppWorkload {
public:
    struct Impl;
    explicit CppWorkload(const std::string& shared_library);
    maya::paper::Input input(const CppInvocation&,
                            std::shared_ptr<CppExecutionStatistics> statistics = {}) const;
    const std::string& plan_json() const noexcept;
private:
    std::shared_ptr<Impl> impl_;
};

// Ground callbacks exchange immutable arrays of ScalarV1. Handlers registered
// with Maya use these schemas and must supply an exact ScalarV1 result.
inline constexpr ras::SchemaId cpp_ground_inputs_schema = 0x4658435047494e01ULL;
inline constexpr ras::SchemaId cpp_ground_result_schema = 0x46584350474f5501ULL;

} // namespace flexeva::code
