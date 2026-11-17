# MLU single-card validation

The current implementation preserves BF16 forward outputs under autocast. Mixed-dtype operations perform the
required conversion at the operator boundary, rather than recasting all Linear/Matmul forward outputs to FP32.
The pinned InfiniTrain revision is `e19a56dcb56bec11f1af63c089a1e8ae03ff24c7`, including the upstream LayerNorm
autocast-policy fix. No change from the old forward-output-recast PR is required.

## Current results (2026-11-17)

| Validation | Observed result | Evidence |
|---|---|---|
| Linear/Matmul backward | 12 combinations, 44 comparisons passed; max absolute error 4.768e-7 | [Operator results](experiments/single-card/2026-11-17-linear-backward-gate/README.md) |
| Softmax backward | 9 dtype/shape cases passed; max absolute error 7.451e-8 | [Before/after reproduction](experiments/single-card/2026-11-17-softmax-backward-gate/README.md) |
| BF16 two-step regression | 2/2, losses 5.314695 and 5.026468, rc0 | [Two steps](experiments/single-card/2026-11-17-bf16-2-step-regression/README.md) |
| BF16 single-card ten steps | 10/10, finite loss at every step, complete forward/loss/backward, rc0; 9248.819 s | [Full logs and commands](experiments/single-card/2026-11-17-bf16-10-steps/README.md) |
| FP32 single-step regression | 1/1, loss 5.356084, exactly matches the historical MLU first step, rc0; 711.285 s | [Full log and command](experiments/single-card/2026-11-17-fp32-1-step-regression/README.md) |

These are single-card execution/regression results. Reference comparisons retain their diagnostic classification in
the experiment metadata. The FP32 result on this revision is one step; historical FP32 ten-step runs are indexed separately.

## Design changes being reviewed

- `LinearBackwardInput/Weight` and `MatmulBackwardInput/Other` convert incoming gradients to the saved operand's
  compute dtype. Following the existing MACA compatibility policy, BF16 compute produces FP32 output gradients.
  `LinearBackwardBias` similarly produces FP32 output when its incoming gradient is BF16. This is a backward
  compatibility policy, not a blanket promotion of forward results.
- `SoftmaxBackward` follows CUDA's `PromoteDataTypes`: both input tensors are converted to the common dtype, and
  allocation and dispatch use that same dtype. Previously an FP32 gradient could be read through a BF16 pointer.
- The pre-existing MLU binary elementwise implementation already promotes mixed input types. The framework's
  corrected LayerNorm autocast policy is used directly.
- Gradient-accumulation dtype checks remain enabled. The prior backward gradient-merge failure and the subsequent
  second-step NaN are retained in the [experiment index](experiments/single-card/README.md).

## Reproduce

Use the build instructions in the [backend README](../README.md), including the pinned submodule and its dependencies.
The tested environment used MLU590-M9, driver6.5.52, CNRT7.7.3, cncc5.7.6 and GCC13.
Detailed build logs are available [here](experiments/single-card/2026-11-17-build/README.md).

```bash
OMP_NUM_THREADS=4 ./build-mlu/op_test --linear-backward-gate --device 0
OMP_NUM_THREADS=4 ./build-mlu/op_test --softmax-backward-gate --device 0

GLOG_logtostderr=1 OMP_NUM_THREADS=192 MLU_LINEAR_PROFILE=1 ./build-mlu/gpt2 \
  --device mlu \
  --input_bin <DATA>/tiny_shakespeare_train.bin \
  --llmc_filepath <WEIGHTS>/gpt2_124M.bin \
  --dtype bfloat16 --batch_size 4 --sequence_length 64 \
  --total_batch_size 256 --num_iteration 10 --overfit_single_batch true \
  --val_loss_every 0 --sample_every 0
```

For the measured FP32 regression, change only `--dtype float32 --num_iteration 1`.
Keep `MLU_DEBUG_SYNC` and `MLU_DEBUG_CHECK` unset for these runs.
`--softmax-backward-probe` is the safe-direction reproducer: it fails on the pre-fix implementation and passes after
the fix. The separate mixed-dtype accumulation refusal test intentionally aborts; it is not a normal success test.

## Source and evidence identity

Current kernel content was tested as base `2150ed7` plus the recorded patch and is now committed as
`3a788f3` (full identity in each current experiment's manifest). Publication-only formatting and comment changes
are mapped in [publication-source-mapping.json](publication-source-mapping.json); they were not a second training run.

The current GPT-2 binary SHA-256 is `cfacde3069c51c0bcf53377fddf81692f6264e1e1a40c70eb29f6417c9c0e1be`.
Each experiment includes original logs, command/environment metadata, exit code, input/binary identity and file hashes.
Neither weights nor binaries are included in this repository.

Historical FP32 ten-step experiments, the superseded BF16 output-recast experiment, failure reproductions and the
current results are all available in the [single-card experiment index](experiments/single-card/README.md).
The legacy BF16 result is not evidence for the current implementation.
