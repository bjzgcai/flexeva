# FlexEva

**Incremental performance evaluation for agentic ML workload optimization.**

FlexEva evaluates successive changes to distributed training programs while reusing compatible evaluator state. Its C++ implementation combines Resilient Anchor State (RAS), a backend-independent evaluation Core, Python and C++ code frontends, and the Maya simulation backend.

The frontends analyze a selected mutable source region and capture its execution. Maya intercepts CUDA, cuBLAS, and NCCL calls to generate a normalized trace without running GPU kernels. Dependencies and the association layer (AL) connect code changes to affected trace outputs, allowing later candidates to retain compatible outputs and reuse feedback computation. Equivalent logical ranks can share a representative execution selected before workload initialization.

This is the implementation repository for [*Lightweight Evaluation for Agentic ML Workload Optimization with Resilient Anchor State*](https://doi.org/10.1145/3842654.3848545). Paper experiments and reproduction instructions live in the separate [artifact evaluation repository](https://github.com/SpaceVector/flexeva-ae).

## Components

| Component | Purpose |
| --- | --- |
| [RAS](cpp/ras/) | In-memory versioned storage, immutable buffers, partitions, and persistent indexes. |
| [Core](cpp/core/) | Evaluator registration, dependency propagation, output reuse, AL routing, and Ground handlers. |
| [Code frontends](cpp/code/) | Python source analysis and Clang-based C++ instrumentation, execution evidence, and representative-rank planning. |
| [Maya](cpp/maya/) | Code-to-trace capture, normalization, cost annotations, scheduling, and performance feedback. |
| [FakeCUDA](backends/maya/cpp/fake_cuda/) | In-process CUDA, cuBLAS, and NCCL interception libraries used by Maya. |

RAS and Core can be embedded independently. The Code layer is also backend-independent; Maya owns CUDA semantics and replay. The [implementation guide](cpp/README.md) describes the interfaces, build options, and workload integration.

## Build RAS and Core

The native implementation targets Linux and requires CMake 3.20 or newer and a C++20 compiler. RAS and Core do not require Python, PyTorch, LLVM, or CUDA.

```bash
git clone https://github.com/bjzgcai/flexeva.git
cd flexeva
cmake -S cpp -B build/core -DCMAKE_BUILD_TYPE=Release \
  -DFLEXEVA_BUILD_CODE=OFF -DFLEXEVA_BUILD_MAYA=OFF \
  -DFLEXEVA_RAS_BUILD_TESTS=ON
cmake --build build/core --parallel
ctest --test-dir build/core/ras --output-on-failure
```

The CTest entry checks the RAS partition/refresh contract. Applications can link the `FlexEva::ras` and `FlexEva::core` CMake targets; [Core's runtime interface](cpp/core/include/flexeva/core/runtime.hpp) defines evaluator registration and execution.

## Build Maya and the frontends

Maya additionally requires OpenSSL development files. The Python binding needs an interpreter with matching development headers; PyTorch workloads need a CUDA-enabled PyTorch installation. The C++ source frontend needs LLVM/Clang 14 development packages, and LibTorch workloads need matching Torch headers and libraries.

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release \
  -DFLEXEVA_CODE_BUILD_CPP=ON \
  -DFLEXEVA_MAYA_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build/cpp --parallel
```

Set `FLEXEVA_CODE_BUILD_CPP=OFF` when only the Python frontend is needed. FakeCUDA libraries are built under `build/cpp/liboutput`; workload execution must redirect the CUDA libraries actually loaded by PyTorch or LibTorch to these interception libraries. Building the libraries or setting `PYTHONPATH` alone does not configure this execution environment. See the [Maya setup and API guide](cpp/README.md#maya-and-frontend-setup).

## Workload contract

Workloads specify a mutable source region, an execution window, and fixed inputs and runtime context. The supported frontend paths cover restricted Python/PyTorch and C++/LibTorch host programs, including ordinary modules, forward/backward execution, optimizer state, and Python calls into instrumented C++ extensions. CUDA kernel internals are outside the frontend's analysis scope.

Numerical decisions that a dry run cannot resolve require workload-registered **Ground** handlers with real inputs. Missing numerical facts fail explicitly. Python and standalone C++ workloads support Ground; Ground callbacks inside a Python-to-C++ extension are not yet supported. Fixed communication groups and explicit collective/P2P operations are modeled; distributed launchers, DDP/FSDP, arbitrary external side effects, and general alias analysis are outside the current contract.

Maya uses a fixed process context with the Torch CUDA caching allocator disabled. Host code may execute again to reconstruct program state, while compatible normalized trace outputs and feedback state are reused. Simulation timings should distinguish the initial evaluation, a refresh after a real change, and a no-op refresh; compilation and library loading are separate preparation costs.

## Citation

If you use FlexEva in your research, please cite our paper:

```bibtex
@inproceedings{flexeva2027eurosys,
  title     = {Lightweight Evaluation for Agentic {ML} Workload Optimization with Resilient Anchor State},
  author    = {Muxi Yan and Yinjie Wu and Xiaoting Wang and Bo Tang},
  year      = {2027},
  doi       = {10.1145/3842654.3848545},
  url       = {https://doi.org/10.1145/3842654.3848545},
  publisher = {ACM},
  isbn      = {979-8-4007-2971-3},
  booktitle = {22nd European Conference on Computer Systems (EuroSys '27), April 19--23, 2027, Rabat, Morocco}
}
```

## License

FlexEva is licensed under the [Apache License 2.0](LICENSE), matching the artifact evaluation repository. Third-party dependencies retain their respective licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
