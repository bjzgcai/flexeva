# License notices

The repository's top-level [LICENSE](LICENSE) contains the Apache License 2.0.
The imported components listed below retain their existing notices and license
terms; the top-level license does not replace those terms.

## FlexEva implementation

The implementation imported from [SpaceVector/flexeva](https://github.com/SpaceVector/flexeva)
in the following directories is covered by the MIT notice in
[LICENSES/FlexEva-MIT.txt](LICENSES/FlexEva-MIT.txt):

- `cpp/`
- `backends/maya/cpp/fake_cuda/`, except the vendored JSON header below
- `src/flexeva/`
- `workload-runtime/`
- `tools/run_fake_cuda.py`, adapted from the implementation's launch helper

Copyright (c) 2026 FlexEva authors. The workload runtime also retains its own
`workload-runtime/LICENSE`.

## JSON for Modern C++

`backends/maya/cpp/fake_cuda/include/device/json.hpp` vendors
[nlohmann/json 3.12.0](https://github.com/nlohmann/json/tree/v3.12.0).
Its MIT license is reproduced in
[LICENSES/nlohmann-json-MIT.txt](LICENSES/nlohmann-json-MIT.txt), from the
[upstream license](https://github.com/nlohmann/json/blob/v3.12.0/LICENSE.MIT).
The header retains its original copyright, attribution, and license declarations,
including those for Niels Lohmann, Evan Nemerson, Bjoern Hoehrmann, Florian
Loitsch, and the Abseil Authors.

The header identifies the included Google Abseil integer-sequence utilities as
Apache-2.0 code and preserves their upstream reference and attribution.
The complete Apache-2.0 terms are available in [LICENSE](LICENSE).
