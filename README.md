# InfiniTrain Backends

**English** | [简体中文](README.zh-CN.md)

[![Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/issues
)
[![PR](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/pulls
)
[![License](https://img.shields.io/github/license/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/blob/master/LICENSE
)

InfiniTrain Backends provides out-of-tree accelerator backends for
[InfiniTrain](https://github.com/InfiniTensor/InfiniTrain). It keeps
vendor-specific SDK integration, runtime support, collective communication,
and kernels outside the framework core while implementing InfiniTrain's
`PrivateUse1` backend interfaces.

Each provider is isolated under `backends/<provider>` and normally consumes the
InfiniTrain submodule pinned to `v0.7.0`. One build tree selects one provider, and a
process may register at most one provider for `DeviceType::kPrivateUse1`.

## Supported Backends

| Backend | Runtime | Collectives | Model examples |
| ------- | ------- | ----------- | -------------- |
| [MACA](backends/maca/README.md) | MACA | MCCL (optional) | GPT-2, LLaMA 3, Mixtral |
| [MLU](backends/mlu/README.md) (work in progress) | CNRT | — | GPT-2 |

The model sources normally come from the pinned InfiniTrain submodule. Each
backend supplies the runtime, kernels, collective implementation, and build
integration needed to run its supported examples on the corresponding
accelerator. See the linked backend guides for setup and usage.

## Common Requirements

- **Linux**
- **CMake**: 3.28 or newer
- **Git** with submodule support
- A **C++20** compiler and vendor SDK compatible with the selected backend

Additional dependencies and SDK setup are documented in each backend's guide.

## Build Options

These options apply across backends. Provider-specific options are documented
in the corresponding backend guide.

| Option | Default | Description |
| ------ | ------- | ----------- |
| `INFINITRAIN_BACKEND` | Required | Provider selected from `backends/<provider>` |
| `INFINITRAIN_SOURCE_DIR` | `third_party/InfiniTrain` | InfiniTrain source tree, normally the pinned submodule |
| `INFINITRAIN_BACKENDS_BUILD_EXAMPLES` | `ON` for a top-level build | Build provider-enabled InfiniTrain examples |
| `BUILD_TEST` | `OFF` | Build InfiniTrain's unit tests and variants for the selected provider |

Use a separate build directory for each provider so compiler and SDK cache
entries do not leak between them. Top-level builds enable backend examples by
default.

For development against another InfiniTrain checkout, add
`-DINFINITRAIN_SOURCE_DIR=/path/to/InfiniTrain` to the configure command. The
selected checkout must implement the PrivateUse1 extension API expected by
the backend.

PrivateUse1 providers require `USE_CUDA=OFF`; configuration fails rather than
silently overriding an explicit `USE_CUDA=ON`. Unit tests for the selected
provider use the PrivateUse1 device type and `PRIVATEUSE1/*` GTest prefix,
independent of the provider's name.

## Architecture

```text
backends/<provider>/
  README.md              provider setup and usage guide
  cmake/                 compiler and vendor SDK setup before project()
  include/               public provider API
  src/backend.cc         provider metadata registration entry point
  src/common/            provider-internal shared helpers
  src/runtime/           device, stream, event, and allocator integration
  src/kernels/           provider kernel implementations and registration
  src/ccl/               optional collective communication integration
  examples/              CMake adapters for upstream InfiniTrain examples
  scripts/               provider model-run configuration and wrappers

third_party/InfiniTrain/ framework source, normally a pinned git submodule
```

The root build selects and includes
`backends/<provider>/cmake/pre_project.cmake` before its first `project()` call.
This lets each provider select its compiler and prepare SDK-specific dependency
probes without adding vendor branches to the root `CMakeLists.txt`.

When adding this repository as a third-party library with `add_subdirectory()`,
select the compiler required by the provider in the parent project before
enabling the C++ language. After setting up the vendor SDK environment, use
the following order in the parent's `CMakeLists.txt`, replacing the compiler
path and provider name:

```cmake
cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_COMPILER "/path/to/vendor/compiler" CACHE FILEPATH "")
set(INFINITRAIN_BACKEND "<provider>" CACHE STRING "")

project(MyTrainingApp LANGUAGES CXX)
add_subdirectory(third_party/InfiniTrain-Backends)
```

CMake cannot replace a compiler after a language has been enabled.

## Development checks

See [formatting and static checks](scripts/README.md) for clang-format,
clang-tidy and IWYU setup, scope and manual usage (in Chinese).

## Cambricon MLU provider (work in progress)

`backends/mlu` registers a `PrivateUse1` backend named `mlu` for Cambricon MLU
devices. The optimized BF16 single-card path completed ten GPT-2 training steps on MLU590-M9.
See the validation documentation below for the measured loss and timings, historical FP32
ten-step results and the separate FP32 single-step regression.

```bash
cmake -S . -B build-mlu -G Ninja \
  -DINFINITRAIN_BACKEND=mlu -DUSE_CUDA=OFF -DBUILD_TEST=OFF
cmake --build build-mlu --target mlu_gpt2 mlu_op_test -j
```

See [the MLU provider guide](backends/mlu/README.md) for the SDK requirements and
[its validation status](backends/mlu/docs/validation.md) for what has actually
been run.

## License

InfiniTrain Backends is released under the [MIT License](LICENSE).
