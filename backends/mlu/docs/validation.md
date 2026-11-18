# MLU single-card validation

## Implementation under review

The provider uses PrivateUse1 and keeps BF16 forward results under autocast.
InfiniTrain is pinned to upstream `e19a56dcb56bec11f1af63c089a1e8ae03ff24c7`, which
includes the LayerNorm autocast-policy fix. No framework source patch is required.

- Linear/Matmul backward converts incoming gradients to the saved operand's dtype.
  Following MACA's existing compatibility policy, BF16 compute returns FP32 gradients;
  Linear bias backward follows the same output policy. This does not recast forward results.
- Softmax backward follows CUDA type promotion: convert both inputs to their common
  dtype, then allocate and dispatch using that dtype. This fixes an FP32 gradient
  being read through a BF16 pointer and the resulting second-step NaN.
- Binary elementwise operations promote mixed input dtypes. Gradient accumulation
  rejects incompatible dtypes before enqueueing to prevent misinterpreted memory writes.

## Recorded results

These are single-card execution and numerical regression results, not formal accuracy
acceptance against a fully identified reference training run. The historical FP32
ten-step runs are distinct from the one-step FP32 regression of the BF16 fix.

| Date | Run | Result | Original evidence |
|---|---|---|---|
| 2026-11-05 | Historical FP32, 10 steps | 10/10, exit 0; the later optimized run has the same loss sequence | [FP32](https://github.com/Nebulixeay/InfiniTrain-Backends/tree/4988ae053992b46f8d128c45c7f0299293b7147b/backends/mlu/docs/experiments/single-card/2026-11-05-fp32-10-steps), [optimized FP32](https://github.com/Nebulixeay/InfiniTrain-Backends/tree/4988ae053992b46f8d128c45c7f0299293b7147b/backends/mlu/docs/experiments/single-card/2026-11-05-fp32-optimized-10-steps) |
| 2026-11-17 | Linear/Matmul backward | 12 combinations, 44 comparisons passed; max absolute error 4.768e-7 | [Operator results](https://github.com/Nebulixeay/InfiniTrain-Backends/tree/4988ae053992b46f8d128c45c7f0299293b7147b/backends/mlu/docs/experiments/single-card/2026-11-17-linear-backward-gate) |
| 2026-11-17 | Softmax backward | 9 dtype/shape cases passed; max absolute error 7.451e-8 | [Operator results](https://github.com/Nebulixeay/InfiniTrain-Backends/tree/4988ae053992b46f8d128c45c7f0299293b7147b/backends/mlu/docs/experiments/single-card/2026-11-17-softmax-backward-gate) |
| 2026-11-17 | BF16, 10 steps | 10/10, finite loss, complete forward/backward, exit 0; 9248.819 s | [BF16](https://github.com/Nebulixeay/InfiniTrain-Backends/tree/4988ae053992b46f8d128c45c7f0299293b7147b/backends/mlu/docs/experiments/single-card/2026-11-17-bf16-10-steps) |
| 2026-11-17 | FP32 regression, 1 step | Loss 5.356084, exit 0; 711.285 s | [FP32 regression](https://github.com/Nebulixeay/InfiniTrain-Backends/tree/4988ae053992b46f8d128c45c7f0299293b7147b/backends/mlu/docs/experiments/single-card/2026-11-17-fp32-1-step-regression) |

| Step | Historical FP32 (2026-11-05) | BF16 fix (2026-11-17) |
|---|---|---|
| 1 | 5.356084 | 5.314695 |
| 2 | 5.060423 | 5.026468 |
| 3 | 4.860268 | 4.856012 |
| 4 | 4.961235 | 4.939342 |
| 5 | 4.888132 | 4.898680 |
| 6 | 5.099115 | 5.073312 |
| 7 | 4.860406 | 4.836881 |
| 8 | 4.944980 | 4.883745 |
| 9 | 5.121128 | 5.095389 |
| 10 | 5.270833 | 5.239595 |

The two columns record separate experiments; they are not an accuracy acceptance comparison.
The old blanket BF16-to-FP32 forward-recast experiment is superseded and is not evidence
for this implementation. The links above are pinned to the archived publication commit,
preserving complete logs, commands and source/binary identities without carrying all
development logs in the current source tree.

## Review cleanup validation (2026-11-18)

The probe-free implementation completed one BF16 step with loss **5.314695**, full
forward/backward and exit code 0 in 926.600 s. This matches the November 17 first step;
no additional ten-step run was performed. Subsequent cleanup added pre-launch rejection
of zero Slice steps and mixed LayerNorm input/parameter dtypes, prefixed runtime error
macros to avoid SDK name collisions, and organized direct includes.

The final regression checks passed: default operators, backward operators, mixed-dtype
Linear/Matmul and Softmax, comparator self-tests, and expected rejection of injected
finite errors, NaNs and incompatible accumulation. Both new invalid-input cases abort
with their specific error messages before accessing an MLU queue. The final change after
these checks only reordered includes and was rebuilt and rechecked for header dependencies.

- GCC 13 / cncc build: passed, including all nine device translation units.
- Six provider headers compiled independently: passed.
- clang-tidy 18.1.8 with the repository plugin: 15 host translation units, no diagnostics.
- IWYU 0.22 / LLVM 18: the same 15 host translation units, no remaining suggestions.
- clang-format 16.0.6 and the applicable Black check: passed.

The LLVM checks use a separate real MLU CMake configuration with Clang 18, GCC 13's
standard library and LLVM OpenMP; the provider also compiled in that configuration.
The device sources are covered by cncc, not the host-only static analysis.

## Reproduce

Build from the repository root using the [backend instructions](../README.md).
The recorded environment used MLU590-M9, driver 6.5.52, CNRT 7.7.3, cncc 5.7.6 and GCC 13.

```bash
OMP_NUM_THREADS=4 ./build-mlu/op_test --linear-backward-gate --device 0
OMP_NUM_THREADS=4 ./build-mlu/op_test --softmax-backward-gate --device 0

GLOG_logtostderr=1 OMP_NUM_THREADS=192 ./build-mlu/gpt2 \
  --device mlu \
  --input_bin <DATA>/tiny_shakespeare_train.bin \
  --llmc_filepath <WEIGHTS>/gpt2_124M.bin \
  --dtype bfloat16 --batch_size 4 --sequence_length 64 \
  --total_batch_size 256 --num_iteration 10 --overfit_single_batch true \
  --val_loss_every 0 --sample_every 0
```

For the recorded FP32 regression, change `--dtype float32 --num_iteration 1`.
The original November 17 runs enabled the now-removed Linear timing probe; the command
above omits that probe and retains the training configuration. This is not a claim of
a second ten-step run after cleanup.

Input SHA-256:

- `tiny_shakespeare_train.bin`: `8a70606be574040c26d225694f5f9759973b419852d22f7fe5c118e1b359dcc8`
- `gpt2_124M.bin`: `3da8b207584030bcdcd207cf7a99952e3421dce92da218b351071857511bf162`

The November 17 implementation was tested as base `2150ed7` plus the recorded patch,
then committed as `3a788f3`. Its GPT-2 binary SHA-256 was
`cfacde3069c51c0bcf53377fddf81692f6264e1e1a40c70eb29f6417c9c0e1be`.
Subsequent publication changes only adjusted formatting and comments. Historical FP32
source and patch identities are in its linked runtime manifest. Weights and binaries are
not included in this repository.

## Current limits

- Linear/Matmul uses FP32 host GEMM with device-to-host and host-to-device transfers.
  Optional CNNL support currently provides a handle, not a device GEMM implementation.
- Embedding backward rounds after each token addition for FP16/BF16 buffers; it does
  not yet accumulate an entire row in FP32 before one final conversion.
- Device coverage is single-card. Distributed training and formal training-accuracy
  acceptance remain outside these results.
