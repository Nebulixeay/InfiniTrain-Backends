# MLU backend (Cambricon)

`backends/mlu` registers a `PrivateUse1` backend named `mlu` for Cambricon MLU devices. It is a
single-card integration with measured BF16 ten-step training and an FP32 single-step regression.
See [`docs/validation.md`](docs/validation.md) for the design, commands and original evidence.

## Architecture in one paragraph

InfiniTrain exposes third-party devices through the `PrivateUse1` mechanism: no new
`DeviceType` enum and no InfiniTrain source file changes. This provider registers the backend
from `src/backend.cc` (`infini_train::backends::mlu::RegisterBackend`):

- `src/runtime/` - device guard, stream (CNRT queue), event (notifier), dispatch helpers;
- `src/kernels/*.mlu` - pure BANG C device code, compiled by `cncc`;
- `src/kernels/*.cc` - host-side integration (dispatch + `REGISTER_KERNEL`), compiled by the host compiler;
- `src/kernels/mlu_kernel_api.h` - the launcher interface between the two;
- `include/infini_train_mlu/backend.h` - the only public header: the registration entry point
  used by examples and tools;
- `third_party/bang_host/` - two MIT-licensed Cambricon headers (`bang_fp16.h`, `bang_bf16.h`)
  vendored for host compilation; see the README there for why and for the single local
  modification.

## Requirements

| Item | Version used during development |
|---|---|
| Device | MLU590 (tested on MLU590-M9) |
| Driver | v6.5.52 |
| CNRT | 7.7.3 |
| cncc | 5.7.6 (clang 11.1 based) |
| Host compiler | GCC 13 (the framework needs C++20; GCC 11 lacks `<format>`) |
| CMake / Ninja | CMake >= 3.28, Ninja |
| InfiniTrain | pinned submodule `e19a56d` (upstream LayerNorm autocast-policy fix) |

`CNNL` is optional at configure time: if `cnnl.h` / `libcnnl` are not found the build prints a
warning and disables BLAS handle support. **`INFINITRAIN_MLU_WITH_CNNL=ON` currently only
controls whether the CNNL handle is compiled in; no kernel calls CNNL yet** (GEMM still uses the
host fallback, see [`docs/validation.md`](docs/validation.md)).

## Build

```bash
git submodule update --init --recursive

# CMAKE_POLICY_VERSION_MINIMUM=3.5 is required with CMake 4.x (third_party/gflags in the
# framework declares an older minimum).
cmake -S . -B build-mlu -G Ninja \
  -DINFINITRAIN_BACKEND=mlu -DUSE_CUDA=OFF -DBUILD_TEST=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-mlu --target mlu_gpt2 mlu_op_test -j
```

Targets: `infini_train_mlu` (runtime + host kernel integration), `infini_train_mlu_kernels`
(BANG C objects), `mlu_gpt2` (upstream GPT-2 example with this backend registered, written as
`gpt2` in the build root) and `mlu_op_test` (operator regression tool, written as `op_test`).
`-DINFINITRAIN_MLU_BUILD_TEST_TOOLS=OFF` skips the regression tool;
`-DINFINITRAIN_BACKENDS_BUILD_EXAMPLES=OFF` skips the example.

`BUILD_TEST=ON` is rejected on purpose: this provider does not wire the shared InfiniTrain test
suites yet, and CPU test results must not stand in for MLU validation. Use the MLU regression
tools instead.

## Run

```bash
GLOG_logtostderr=1 OMP_NUM_THREADS=192 ./build-mlu/gpt2 \
  --device mlu \
  --input_bin <DATA>/tiny_shakespeare_train.bin \
  --llmc_filepath <WEIGHTS>/gpt2_124M.bin \
  --dtype bfloat16 --batch_size 4 --sequence_length 64 \
  --total_batch_size 256 --num_iteration 10 --overfit_single_batch true \
  --val_loss_every 0 --sample_every 0
```

`GLOG_logtostderr=1` is required to see the step logs: glog otherwise writes them to
`/tmp/gpt2.*.INFO.*`.

This configuration completed ten steps on 2026-11-17. The measured FP32 regression changes
only `--dtype float32 --num_iteration 1`; historical FP32 ten-step runs are indexed separately.

## Regression tests

```bash
OMP_NUM_THREADS=4 ./build-mlu/op_test
OMP_NUM_THREADS=4 ./build-mlu/op_test --backward-group all
OMP_NUM_THREADS=4 ./build-mlu/op_test --linear-backward-gate
OMP_NUM_THREADS=4 ./build-mlu/op_test --softmax-backward-gate
./build-mlu/op_test --comparator-selftest  # host-only comparator checks
```

`--inject-finite` and `--inject-nan` are comparator negative tests: each must return
a nonzero status. `--bf16-gate` checks Cast, Linear and mixed-type addition, then
intentionally submits mismatched gradient accumulation; its expected outcome is an
explicit dtype rejection after the preceding checks pass. It is not a normal success test.
`--invalid-input-case slice-zero-step` and `--invalid-input-case layernorm-mixed-dtype`
likewise must abort with their respective validation errors before accessing an MLU queue.

See [validation](docs/validation.md) for measured results and current limitations.
