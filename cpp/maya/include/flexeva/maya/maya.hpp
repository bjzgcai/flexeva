#pragma once

#include <flexeva/maya/model.hpp>
#include <memory>
#include <optional>

namespace flexeva::maya {

struct Changes {
    // Explicit, complete input changes supplied by the caller. This first
    // C++ entrypoint does not infer changes to arbitrary native application code.
    std::vector<std::uint64_t> boundaries;
    std::vector<std::uint64_t> cost_boundaries;
    // Required when boundaries or native call counts/order are added, removed,
    // or reordered. This requests full capture and downstream reconstruction;
    // callbacks are never automatically rerun after discovering a mismatch.
    // Changes to wait/lane/communication topology within the same call sequence
    // are detected in declared dirty boundaries and rebuild downstream layers.
    bool structure_changed = false;
};

struct RunOptions {
    Policies policies;
    std::uint64_t context = 1;
};

class Capture {
public:
    // Keys are nonzero, unique per invocation, and stable across candidates.
    // The callback always advances native state when this workload is invoked;
    // capture chooses record/retain before constructing a RAS record.
    void boundary(std::uint64_t key, std::uint32_t rank, std::uint64_t revision,
                  const std::function<void()>& callback);
private:
    struct Impl;
    explicit Capture(Impl& impl) : impl_(&impl) {}
    Impl* impl_;
    friend class Evaluator;
};
using Workload = std::function<void(Capture&)>;

struct Evaluation {
    Candidate candidate;
    Statistics statistics;
};

class Evaluator {
public:
    Evaluation run(const Workload&, const CostProvider&, RunOptions = {});
    Evaluation refresh(const Candidate&, const Workload&, const CostProvider&,
                       const Changes&, std::optional<RunOptions> = {});
    ras::Store& store() noexcept { return store_; }
private:
    ras::Store store_;
};

} // namespace flexeva::maya
