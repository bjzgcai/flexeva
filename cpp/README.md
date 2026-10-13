# C++ implementation and workload integration

FlexEva provides a C++20 storage layer and evaluation Core, a backend-independent Code layer, and the Maya simulation backend. The primary Maya entry point is [`flexeva::maya::paper::Session`](maya/include/flexeva/maya/paper_runtime.hpp). Python and C++ frontends execute instrumented host code in the same process as the evaluator.

## Architecture and interfaces

Maya registers two logical RAS layers:

```text
Python / C++ mutable source region
             |
    source analysis + host execution
             |
     Code execution evidence
             |
   CUDA / cuBLAS / NCCL interception
             |
      normalized Trace in RAS
             |
   cost annotations + replay + feedback
```

Normalized Trace is the final trace representation. Interception callbacks consume raw descriptors and write directly to its immutable backing, maintaining metadata during generation. Raw records, cost annotations, and scheduling state do not introduce additional RAS layers.

| Interface | Responsibility |
| --- | --- |
| [`ras/`](ras/) / `FlexEva::ras` | In-memory transactions, versions, persistent indexes, immutable buffers, and direct views. |
| [`core/runtime.hpp`](core/include/flexeva/core/runtime.hpp) / `FlexEva::core` | Layer/transform registration, partitions, dependencies, Boundaries, output Receipts, and Ground. |
| [`code/layer.hpp`](code/include/flexeva/code/layer.hpp) / `FlexEva::code_layer` | Source evidence, dynamic occurrences, active lanes, dependencies, and Code partition construction. |
| [`code/runtime.h`](code/include/flexeva/code/runtime.h) / `FlexEva::code_runtime` | Versioned C ABI, synchronous callbacks, and thread-local attachment for instrumented native workloads. |
| [`code/maya.hpp`](code/include/flexeva/code/maya.hpp) / `FlexEva::code_maya` | `CppWorkload` adapter between a generated workload DSO and Maya. |
| [`maya/paper_runtime.hpp`](maya/include/flexeva/maya/paper_runtime.hpp) / `FlexEva::maya` | `Input`, `Capture`, `Options`, `Snapshot`, and `Session`. |
| [`flexeva_maya_paper.frontend`](maya/python/flexeva_maya_paper/frontend.py) | Python workload registration, source analysis, execution, and Ground integration. |
| [`FakeCUDA`](../backends/maya/cpp/fake_cuda/) | Shared capture state and intercepted accelerator APIs. |

Core and RAS do not interpret CUDA operations or Tensor payloads. An evaluator supplies its own layer layouts, transforms, reuse evidence, and feedback rules. Link `FlexEva::code_layer` when source execution evidence is also needed; this target does not link Maya, Torch, or FakeCUDA. The older direct `maya::Evaluator` API remains in the native library for compatibility; new frontend integrations use `paper::Session`.

## Maya and frontend setup

Run these commands from the repository root. The native build requires Linux, CMake 3.20 or newer, and a C++20 compiler. Maya also requires OpenSSL development files. The optional C++ frontend requires LLVM/Clang 14 development packages; on Debian/Ubuntu these are `clang-14`, `llvm-14-dev`, `libclang-14-dev`, and `libclang-cpp14-dev`.

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release \
  -DFLEXEVA_CODE_BUILD_CPP=ON \
  -DFLEXEVA_MAYA_BUILD_PYTHON=ON \
  -DFLEXEVA_RAS_BUILD_TESTS=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build/cpp --parallel
ctest --test-dir build/cpp/ras --output-on-failure
export PYTHONPATH="$PWD/cpp/code/python:$PWD/cpp/maya/python:$PWD/src:$PWD/workload-runtime/src:$PWD/build/cpp/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import flexeva_code, flexeva_maya_paper, _flexeva_maya_paper'
```

Use the same Python interpreter and matching development headers as the workload environment. CMake builds the native extension under `build/cpp/python`; Python source packages are used from the checkout. The import check verifies package discovery and native linking, not a PyTorch simulation.

| Build option | Effect |
| --- | --- |
| `FLEXEVA_BUILD_MAYA=OFF` | Build RAS, Core, and the neutral Code layer/runtime without FakeCUDA or Maya. |
| `FLEXEVA_BUILD_CODE=OFF` with `FLEXEVA_BUILD_MAYA=OFF` | Build RAS and Core only. |
| `FLEXEVA_CODE_BUILD_CPP=ON` | Build the Clang frontend and `flexeva-code-cpp`; with Maya enabled, also build `FlexEva::code_maya`. |
| `FLEXEVA_MAYA_BUILD_PYTHON=ON` | Build the `_flexeva_maya_paper` Python extension. |
| `FLEXEVA_RAS_BUILD_TESTS=ON` | Build the RAS contract test and an opt-in storage scaling executable. |

The RAS contract test is registered under `build/cpp/ras`. The separate `refresh_storage_scaling` executable creates a large state and is not run by CTest. These tests cover storage contracts, not all frontend or evaluator behavior.

### Configure interception before running a workload

PyTorch simulation needs a CUDA-enabled PyTorch installation; LibTorch simulation needs compatible Torch headers and libraries. FakeCUDA libraries are built under `build/cpp/liboutput`. The process must load these libraries in place of the CUDA driver/runtime, cuBLAS, and NCCL libraries used by the workload.

A library search path can suffice for a native executable linked directly against the interception libraries. PyTorch distributions can load bundled libraries through absolute paths or package-specific search paths, so `LD_LIBRARY_PATH` or `LD_PRELOAD` alone is not a general solution. Use an environment that redirects those actual paths, for example PRoot file bindings, and verify the loaded paths with the dynamic loader's diagnostics or the process memory map. Configure this before importing Torch or loading the workload DSO.

For CUDA 12 Python wheels, [`tools/run_fake_cuda.py`](../tools/run_fake_cuda.py) discovers the selected interpreter's NVIDIA/Torch library directories and binds the interception libraries over the installed CUDA paths. Install PRoot and make `proot` available on `PATH`, or pass `--proot /path/to/proot`. After setting `PYTHONPATH` above, save the Python example below as `workload.py` and run:

```bash
python3 tools/run_fake_cuda.py --lib-dir build/cpp/liboutput \
  --python "$(command -v python3)" -- workload.py
```

Use `--library-dir PATH` for additional CUDA library locations. The helper requires all six FakeCUDA shared libraries and an environment without `LD_PRELOAD`. It configures library redirection; it does not install PyTorch or validate unsupported Torch/CUDA ABI combinations. The current workload path has been exercised with PyTorch 2.8.0 and CUDA 12.8 wheels.

Capture state is process-local. Execute one Session/workload capture at a time within a process. Maya's PyTorch/LibTorch preparation establishes the fixed `uncached-process-context-v1` policy: it disables Torch's CUDA caching allocator and prepares CUDA/BLAS/autograd state. This policy remains active in the simulation process and participates in reuse evidence. Preparation executes no user model or training step; its elapsed time is included in simulation execution.

## Python workloads

[`PythonWorkload`](maya/python/flexeva_maya_paper/frontend.py) takes frozen source files, an entry point, input arguments, and a registration. Source analysis builds control/data dependencies; instrumentation reports the dynamic occurrences that actually execute. The following example assumes interception has already been configured:

```python
import _flexeva_maya_paper as native
from flexeva_maya_paper.frontend import (
    MayaRules, PythonWorkload, WorkloadRegistration,
)

sources = {"toy.py": """
import torch
def entry(n):
    a = torch.zeros((n, n), device="cuda:0")
    return torch.mm(a, a)
"""}
registration = WorkloadRegistration(MayaRules())
workload = PythonWorkload(
    sources, "toy.entry", arguments=lambda rank: {"n": 8},
    registration=registration,
)
session = registration.new_session(native, mode="refresh")
first = native.init(session, workload.describe(), workload.execute)
changed = PythonWorkload(
    sources, "toy.entry", arguments=lambda rank: {"n": 16},
    registration=registration,
)
result = native.refresh(session, first, changed.describe(), changed.execute)
print(result["feedback"])
```

This API example demonstrates evaluation and refresh; its cost configuration is not a calibrated device model. Configure a cost provider for the target system before interpreting simulated timings as hardware predictions.

The restricted frontend supports ordinary `nn.Module` execution, Tensor geometry, forward/backward, and SGD/Adam/AdamW effects. Evidence includes source, shape, stride, dtype, device, producer relationships, and relevant state versions. It does not copy the complete model at every host statement. Unknown effects that can execute correctly weaken reuse evidence and cause conservative generation; unsupported numerical decisions require Ground.

`PythonWorkload.from_script()` extracts a window from executable top-level statements and a main guard. Select the training window separately from data loading, launchers, file I/O, and other external effects. It is not a general application slicer. `CallRule` extends foreign-call semantics, while `WorkloadRegistration` binds rules and context identity before the Session is created.

Representative-rank planning happens before model and optimizer initialization. The frontend uses restricted source analysis or explicitly supplied `RankEquivalenceEvidence`; without a complete proof, it executes each rank. Shared Code partitions carry a common `active_lanes` set, and Maya projects their outputs to all participating logical ranks. `last_execution` and `rank_execution_plan` expose the actual execution choices.

## C++ and LibTorch workloads

[`flexeva-code-cpp`](code/src/tool.cpp) uses LLVM/Clang 14 to analyze control flow, reaching definitions, and rank observations, then rewrites the selected source. With `--build`, it produces `plan.json`, `instrumented.cpp`, and `workload.so`. The DSO carries both the frozen plan and the entry point.

For a minimal host workload, save the following as `workload.cpp`:

```cpp
#include <cuda_runtime_api.h>

void train(unsigned rank, unsigned world_size, int n) {
    void* data;
    cudaMalloc(&data, n * sizeof(float));
    cudaMemsetAsync(data, 0, n * sizeof(float), nullptr);
    cudaFree(data);
}
```

Set `CUDA_INCLUDE_DIR` to the CUDA headers used by the workload, then build it:

```bash
build/cpp/code/flexeva-code-cpp \
  --source workload.cpp --entry train --output-dir build/cpp/workload --build \
  --link-arg "-L$PWD/build/cpp/liboutput" \
  --link-arg "-Wl,-rpath,$PWD/build/cpp/liboutput" --link-arg -lcudart \
  -- -std=c++20 "-I$CUDA_INCLUDE_DIR"
```

The standalone entry point must have named rank and world-size scalar parameters, by default `rank` and `world_size`. Remaining scalar arguments are supplied in declaration order. A caller linked to `FlexEva::code_maya` can evaluate the DSO as follows:

```cpp
#include <flexeva/code/maya.hpp>

namespace code = flexeva::code;
namespace paper = flexeva::maya::paper;

int main() {
    code::CppWorkload workload("build/cpp/workload/workload.so");
    code::CppInvocation invocation;
    invocation.world_size = 4;
    invocation.arguments = {code::scalar(128)};

    paper::Options options;
    options.costs = [](flexeva::ras::EventId, const flexeva::maya::Operation& op) {
        flexeva::maya::Cost cost{1, 0}; // Illustrative host/device durations.
        if (op.call.op == FAKECUDA_RAS_MEMSET_ASYNC) cost.device = 10;
        return cost;
    };
    paper::Session session(options);
    auto first = session.init(workload.input(invocation));
    invocation.arguments[0] = code::scalar(256);
    auto changed = session.refresh(first, workload.input(invocation));
}
```

`CppInvocation.ranks` defaults to every logical rank in `[0, world_size)`. Set `share_ranks=false` to force their separate execution. `CppExecutionStatistics` records the actual invocation count and representatives. Representative selection uses the source plan before invoking the DSO.

`--torch` enables the restricted LibTorch contract for Tensor/Module geometry, RAII, forward/backward, and optimizers. Supply the installation's Torch include/library paths and matching `_GLIBCXX_USE_CXX11_ABI`. Use `--mutable-header PATH` and `--extra-source PATH` to include owned source in the analyzed closure. Library headers remain fixed context. `--compile-commands DIR` supplies a compilation database; `--compiler` currently requires an executable named `clang++-14`. Generated DSOs use `-O2` by default; pass additional compile flags after `--` and keep those flags fixed when comparing candidates.

The scalar frontend accepts local state, supported control flow, direct nonrecursive helpers, and registered accelerator APIs. It rejects unsupported pointer operations, indirect calls, mutable global state, and unknown external effects. Torch mode supplies additional object-effect rules. Complex nested emission or Ground expressions should be split into local variables. The [frontend implementation](code/src/frontend.cpp) defines the accepted syntax and errors.

### Python-to-C++ extensions

Build an extension with `--extension --torch`, then register its callable and generated artifact through [`NativeObserverFactory`](code/python/flexeva_code/native.py) as the registration's `interface_observer_factory`. The extension keeps its normal Python/native call entry, while runtime callbacks attach to the current Code context. Native occurrences become children of the existing Python emission and share its Session.

Callbacks borrow metadata synchronously; persistent evidence uses copied values rather than object addresses. The extension plan and binary identity are part of the fixed context. Ground callbacks inside this cross-language path are not yet supported and fail explicitly; Python Ground and standalone C++ Ground are separate supported paths.

## Ground and communication

Register Ground handlers with the workload before creating a Session. Python [`GroundCall`](maya/python/flexeva_maya_paper/frontend.py) associates a numerical consumption point with real inputs, a handler, and result selection. Standalone C++ `--ground qualified_function` emits scalar Ground requests; register handlers through `Options.grounds` using the schemas in [`code/maya.hpp`](code/include/flexeva/code/maya.hpp).

Ground supplies facts required by numerical control flow or shapes. [`ScalarControlGround` and `MoeRouteGround`](maya/python/flexeva_maya_paper/ground.py) consume explicit real CPU inputs, including immutable `FrozenCpuTensor` values. FakeCUDA tensors do not contain valid training results and cannot supply these numerical facts. Requests match declared inputs, rule identity, and lane evidence; reusable facts may be retained from the selected parent version. A Ground failure aborts the candidate, including when user code catches the originating exception.

[`FixedDistributed`](maya/python/flexeva_maya_paper/distributed.py) scopes standard Python `torch.distributed` calls to registered groups. [`LogicalCommunicator`](maya/include/flexeva/maya/logical_communication.hpp) provides the corresponding C++ interface. Supported operations include all-reduce, broadcast, all-gather, reduce-scatter, grouped all-to-all, P2P, barrier, and unconditional wait. These operations model communication geometry without exchanging tensor values. Rendezvous, distributed launchers, DDP/FSDP, and polling-dependent control are outside this contract.

## State, reuse, and evaluator integration

A mutable region is the source range allowed to change. An execution window is its actual execution under a fixed context. A logical partition is a reuse unit whose events share an active-lane set and are contiguous within each participating lane. These are distinct from physical storage blocks.

`Session::init()` creates the first snapshot; `refresh(parent, input)` evaluates against an explicitly selected parent. `commit(snapshot)` promotes a completed candidate without rerunning the evaluator. Changes to fixed runtime context, registration rules, or cost semantics require compatible new evidence. RAS transactions preserve published snapshots on failure; they do not undo external side effects of workload code.

Host execution may run again to reconstruct model, gradient, and optimizer state. Reuse avoids generating and processing compatible trace outputs; it does not require a full program snapshot at every Boundary. Window cleanup runs in an auxiliary scope: Python scopes runtime lifetime, while C++ scopes selected-entry automatic destruction after evaluating a return expression. Explicit frees inside the modeled body retain their ordinary capture scope.

AL stores factorized `producer → Boundary → output` routes. It locates dependent outputs; it does not prove that an output is unchanged. Output Receipts own normalized backing and metadata. Logical trace partitions are materialized only when a downstream consumer needs them; a physical block rollover does not force a logical split.

| Core API | Contract |
| --- | --- |
| `Context::stage_output()` / `prepared_output()` | Stage an immutable body and Receipt during Capture/Partition for the corresponding Apply. |
| `Context::parent_output()` | Read the selected parent's output for the current layer's outgoing transform. |
| `Context::prepare()` / `adopt_prepared()` | Bind output preparation to the current operation, transform, Boundary, and evidence. |
| `Context::adopt_previous_output()` | Retain the current Boundary's output root from the selected parent. |
| `BoundaryView::output_range()` | Borrow a read-only range without creating a logical partition. |
| `Context::materialize_range()` | Create a validated logical range during Partition; publish it through `PartitionDelta`. |

[`TraceView`](maya/include/flexeva/maya/trace_view.hpp) exposes normalized outputs without implicitly enumerating their bodies. `Session::inspect()` explicitly scans the logical trace to compute diagnostic digests and counts; it is not part of normal output adoption or Retain.

Maya owns costs and replay. Costs are annotations on immutable normalized output; logical rank and resource projections remain relevant to cost identity. The scheduler retains a graph and timing annotations in RAS, updates affected nodes when topology is stable, and propagates until completion times stop changing. Structural or resource changes rebuild the graph. The full replay path remains available for comparison.

## Validation and simulation timing

Compare Refresh with complete execution under the same source, inputs, normalization contract, cost model, and optimized build. Check normalized output, dependencies, and feedback as well as actual rank invocations and generated work. `Mode::FullRank` and `Mode::FullCompact` independently initialize each candidate; the frontend must also be configured to execute every rank for an all-rank reference. `Mode::Refresh` uses the selected parent and permits representative execution when proven.

Use simulation execution as the main latency interval once source artifacts and runtime libraries are ready. Report Init, Refresh after a real change, and no-op Refresh separately. Record source analysis, compilation, imports, and DSO loading as preparation costs. `metrics.total_seconds` covers `Session::run()` after input validation through feedback decoding; it excludes external compilation, Python source analysis, and Python binding input parsing.

`Options.diagnostics` and Python `new_session(diagnostics=...)` default to `false`. Enable diagnostics to audit body reads and detailed maintenance, and disable it for latency measurements. A zero audit counter with diagnostics disabled means unmeasured, not zero work. Core's fine timers are also gated by diagnostics.

`capture_seconds` includes setup, host execution, and finalization. `normalization_seconds` and `retained_state_seconds` are nested within host execution; do not add them to capture again. Core's callback timer similarly includes evaluator phases. Work counters distinguish generated records, retained calls, output adoption, cost calls, replay reads, and rebuilt graphs.

Reuse does not make every refresh proportional to the changed body: Code metadata processing, descriptor enumeration, resource reconstruction, and structural replay fallback retain their own costs. Native V2 normalization also covers only the fields exposed by its capture ABI; it does not provide complete host timestamps or all CUDA launch-context semantics. Interpret output equality and simulation predictions within that contract.
