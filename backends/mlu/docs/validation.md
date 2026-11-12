# MLU backend - validation status (stage snapshot)

This file records what has actually been run and what has not. It is deliberately conservative:
anything not listed as verified below should be treated as unverified.

## Recorded 2026-11-11; runs executed 2026-11-05 / 2026-11-10 (single-card training evidence)

The complete raw logs, argv, hashes, exit codes and step tables are committed under
[`experiments/single-card`](experiments/single-card/README.md). These are
single-card path results, not formal cross-framework accuracy passes.

| Item | Result | Evidence |
|---|---|---|
| GPT-2 FP32 single-card 10 steps | `MEASURED_10_STEPS`; rc=0, unique `step 1/10` through `step 10/10`, complete forward/loss/backward path, loss changes across the fixed batch, device returned idle; 7,417 s | [`023_fp32_10_steps`](experiments/single-card/023_fp32_10_steps/README.md) |
| GPT-2 BF16 single-card 10 steps | `BF16_10_STEPS_COMPLETED_DIAGNOSTIC`; rc=0, unique `step 1/10` through `step 10/10`, complete forward/loss/backward path, loss changes across the fixed batch, step1 loss `5.290029`, `phase_done=490`, device returned idle; 8,456.023 s | [`036_bf16_10_steps`](experiments/single-card/036_bf16_10_steps/README.md) |
| Formal loss acceptance | **Blocked independently**: A100 reference `weights_sha256` and `data_sha256` remain unknown, and the FP32 cross-framework comparison is already `DIAGNOSTIC_FAIL` at step 2 (`abs_diff=1.620e-4`, above the original threshold). Results stay diagnostic-only even if the missing A100 hashes are later supplied. | `backends/mlu/tools/compare_loss_strict.py` registry gate |

The FP32 run used runtime HEAD `72f81e2` plus its recorded dirty worktree, and binary
`93ece3b6…`; the BF16 run used parent `ea5d11c8`, submodule `c90338ce` plus dirty
`function.cc`, and binary `94a886b7…`. The two runs are historical evidence from their
recorded source identities and must not be described as the same binary. The hashes in each
`run.argv.txt` identify local MLU assets, not the A100 reference assets.

## Verified on 2026-11-03 (this provider layout)

Built from a fresh CMake cache against the pinned InfiniTrain content (`c90338c`, v0.7.0) with
GCC 13.4.0, cncc 5.7.6, CMake 4.4.5 and Ninja 1.13.2; targets `mlu_gpt2` and `mlu_op_test`.

| Item | Evidence |
|---|---|
| Fresh configuration and build | `configure` and `build` of `mlu_gpt2` + `mlu_op_test` both succeeded; all **9** BANG C sources were compiled by `cncc` |
| Operator comparison vs CPU (`op_test`) | 35 small-shape FP32 cases: Linear (both weight layouts, bias on/off, non-square 3x5->7), LayerNorm, Softmax, Add broadcast, MulScalar, Transpose, CrossEntropy, Pow domain and Pow(x,3) fwd/bwd. All pass with `abs(got-ref) <= 1e-5 + 1e-4*abs(ref)` (Pow uses 1e-6/1e-5) |
| Negative comparison paths | `--inject-finite` and `--inject-nan` return exit code 1 with `RESULT: FAIL`; the controlled fatal case aborts with the expected marker and no post-fatal work |
| Device math functions (`tests/math_selftest`) | exp / log / tanh / sqrt / rsqrt / pow / identity: 0/1024 mismatches under the tool's combined tolerance (`abs <= 1e-6 + 1e-5*abs(ref)`); the largest absolute error is 4.883e-4 on `exp` at x=8.5, where the reference is 4914.77 (relative 1.19e-7) |
| Example executable | `gpt2 --help` loads and prints usage. **No model training step was run** |
| Absence of stale framework paths | the generated compile database contains no path to the previously used `5a7be78` source tree |

Note: Linear and Matmul currently execute through a **host fallback** - the operator tests above
prove that fallback path, not a device GEMM.

## Verified earlier (historical, pre-migration layout)

These were observed while the backend still lived under `mlu/` next to a separate framework tree.
They are recorded for traceability and were **not** re-executed for this layout change.

| Item | Historical evidence |
|---|---|
| Backend registration | the `--device mlu` validator accepted the backend and kernels dispatched on `kPrivateUse1` |
| Build dependency tracking | touching the launcher header rebuilt all `.mlu` objects; a following build did nothing |

## Not verified

| Item | Status |
|---|---|
| Standard recursive-submodule download | the validation build used an archive of the pinned framework tree plus shallow retrieval of the four dependencies at their pinned commits. File content matched the pinned revisions, but the plain `git submodule update --init --recursive` procedure was **not** exercised |
| Cross-provider regression | no MACA or CUDA build was run. The untouched official files are not evidence of cross-platform regression coverage |
| Device GEMM (CNNL) | not implemented; `src/kernels/linear.cc` does D2H -> host GEMM -> H2D and is marked `NOTE(MLU-FALLBACK)` |
| Full GPT-2 FP32 single-card 10 steps | completed for the recorded GPT-2 124M configuration; broader shapes and formal cross-framework accuracy remain unverified |
| Backward coverage at model scale | completed for the recorded GPT-2 single-card FP32/BF16 configurations; broader shapes and dedicated operator-level coverage remain unverified |
| BF16 beyond the recorded GPT-2 single-card path | broader shapes/models and formal accuracy remain unverified; `src/kernels/embedding.mlu` still rounds per token (`TODO(BF16-MUST-FIX)`); the recorded 10-step path is documented above |
| Multi-card (CNCL) | not implemented (communication kernels absent) |
| Shared InfiniTrain test suites | not wired; `BUILD_TEST=ON` is rejected on purpose and the MLU regression tools are used instead |
| Performance | kernels run single task / serial (`cnrtFuncTypeBlock`); no multi-core or NRAM vectorisation yet |

## Known platform notes

- `__bang_atomic_add` triggered a queue error in our usage, so `EmbeddingBackward` accumulates
  with a row-partitioned scan instead (deterministic, no atomics).
- Device `powf` works once kernels run single-task/serial with `cnrtFuncTypeBlock`; with the
  earlier multi-task `cnrtFuncTypeUnion1` launch, loop writes to global memory did not persist.
- `MluGuardImpl::Memcpy` must copy through the current stream (`cnrtMemcpyAsync` + queue sync);
  plain `cnrtMemcpy` reads stale data while kernels are in flight.

## Next steps

1. provide the missing A100 reference asset hashes and resolve the recorded FP32 diagnostic
   difference before attempting a formal accuracy claim;
2. implement device GEMM via CNNL with the handle bound to the current stream, then re-run
   the operator tests;
3. locate the unresolved large-shape queue failure. GPT-2 1 -> 2 -> 10 steps is complete for
   the recorded single-card configuration, but broader shapes and the formal A100 comparison
   remain open;
4. if broader BF16 coverage is required, add focused shape/operator gates rather than repeating
   the completed GPT-2 ten-step run; keep the embedding per-token rounding TODO visible;
5. multi-card: CNCL registration and a 2-card DP milestone;
6. wire the shared InfiniTrain test suites and run a MACA/CUDA regression before claiming
   cross-provider compatibility.

## Strict loss acceptance tool (2026-11-03, this provider layout)

`backends/mlu/tools/compare_loss_strict.py` compares a run's per-step loss against the frozen
reference and decides pass/fail/reject on host only. It was migrated from the earlier prototype
tree (`InfiniTrain-Backends/tools/`, kept read-only) and fixed there; the prototype is not part of
this provider.

Verified on 2026-11-03 with the bundled Python (3.14.2) on Windows, no device and no training:

| Item | Result |
|---|---|
| Selftest `backends/mlu/tools/tests/test_compare_loss_strict.py` | `PASS (66/66 cases)`, rc=0, ~9 s. The count grows with added regressions: 54 cases in the first round, 62 after the schema/rank fixes, 66 after the config-type and reference-provenance fixes |
| Formatting | `scripts/format.py --path backends/mlu/tools --check --py <black 26.1.0>` rc=0 (isolated venv, not a repo dependency) |
| Real registry (formal mode) | rc=3 `formal mode rejects unknown_fields=['weights_sha256','data_sha256']` — the raw log hash, the single-case hash, the case derivation and the 10 registered steps are all validated before that gate, and the run then stops because the A100 asset hashes are still unknown |
| Reference provenance | the single-case file must be byte-identical to the segment cut from the raw log at the unique `Running: gpt2_1 (fp32)` title; a duplicate title, a tampered case file or a missing raw log are rejected |

Notes on the decision contract:

- `criterion_source=user_pdf_and_a100_log` (FP32 `max_abs<=1e-5`, BF16 `max_rel<=1e-2`) and
  `comparison_policy_source=program_convention` (error denominator, zero reference, `<=` boundary)
  are reported separately in the terminal output and in the JSON.
- Formal mode never falls back to registry numbers: the registry `steps` only cross-check the
  parsed log values and must match exactly.
- Schema and identity formats are validated before any comparison: `source_commit` must be a full
  40-hex token on both the actual metadata and the registered reference case, `binary_sha256` and
  the asset hashes a full 64-hex token (trailing characters are rejected), the actual metadata /
  registry / config / `unknown_fields` must have the documented container types, and a `--rank`
  run refuses to mix tagged and untagged loss records. Config fields that are numeric by meaning
  (`batch_size`, `sequence_length`, `total_batch_size`, `learning_rate`) are type-checked on both
  sides before any comparison, so a bool/string/`null` there is a rejection rather than a silent
  coercion. Structural or type errors are reported as `REJECTED` with rc=3, never as a traceback.
- Diagnostic mode prints `DIAGNOSTIC_MATCH`/`DIAGNOSTIC_FAIL` only, with `acceptance_ready=false`.
- A BF16 row with a zero reference and a non-zero actual yields `rel_diff = inf`, so the JSON
  contains the non-standard token `Infinity`. It never raises, but strict JSON parsers reject it;
  the output contract is left unchanged on purpose.
- The A100 log and its single-case derivation are redistributed under the upstream MIT license
  (`references/LICENSE.InfiniTrain-Test`, `InfiniTensor/InfiniTrain-Test@5a0f27a5`).
- This tool only proves the acceptance logic. It is **not** evidence that GPT-2 FP32 ten steps or
  any accuracy target has passed; the assets (`weights_sha256`, `data_sha256`) are still unknown.

## Key backward cases in `op_test` (source prepared; build/device pending)

Added on 2026-11-03 under 007-P1 v2 / 007-P2 v1. **Nothing here has been compiled or run** on this
Windows workspace; the MLU build and device execution need a separate, explicitly authorized
server batch. No PASS result for these new cases exists yet.

- New entry point: `op_test --backward-group a|b|all`. Default behaviour is unchanged: without that
  flag the existing suite still runs exactly as before. An unknown group value, or combining the
  flag with `--comparator-selftest` / `--debug-status-test` / `--probe-case`, exits with rc=2 before
  the backend is registered.
- Group A: `EmbeddingBackward` (repeated tokens, unused rows must stay exactly zero),
  `CrossEntropyBackward` (`[8,32]` and `[1,32]`, scalar upstream gradient `g=0.5`),
  `AccumulateGrad` (in place, `rate=+0.5` for gradient accumulation and `rate=-0.5` for the SGD
  update role; kernel-level semantics, not the training accumulation rate).
- Group B: `SoftmaxBackward` (`dim=1`, finite / `-inf` mask / `float32_lowest` mask, every row keeps
  at least one unmasked logit; masked columns must have probability and gradient **exactly 0** and
  finite; both the CPU and the MLU forward outputs are guards-checked and finiteness-checked before
  they are fed to a backward call, and a non-finite forward output records a failure and skips that
  mask instead of polluting the gradient), `LayerNormBackward` (3D input `[1,2,4]`, `mean`/`rstd`
  must be FP32 `[1,2]`, three gradients checked separately against both the CPU kernel and the
  upstream expected values), `MatmulBackwardInput/Other` (2D and batched, dA and dB separately).
- Comparison: existing `Compare(name, ref, got, atol, 0.0, allow_special_ref=false)` — absolute
  threshold only, `rtol=0`, no special-value allowance for the new cases. Every CPU and MLU return
  value (including intermediate `mean`/`rstd`, the softmax forward output used as a backward input,
  and the in-place target after an `AccumulateGrad` call on either device) is checked for
  null/shape/dtype before it is read or fed to the next kernel; size and analytic-reference
  finiteness (including the double→float cast) are checked before `Compare`, and any failure sets
  the failure flag without touching the data, so the process exits non-zero.
- Data movement: the blocking `ToHost` helper is used for every device-to-host read, inside an
  explicit `DeviceGuard` for the MLU device. `ToHost` returns a `std::vector<float>`; the void
  return that cannot report its own exit code is the inner `MluGuardImpl::Memcpy`
  (`cnrtMemcpyAsync` on the current queue followed by `cnrtQueueSync`, failures raised by
  `CNRT_CHECK`). Each MLU case logs its own `[bwd] case=... device=... stream=...
  sync=blocking_Memcpy_current_queue host_read_after_tohost_returned` line only **after** the
  relevant read-backs have returned (LayerNorm logs after all three gradients were copied out).
  `Tensor::To(kCPU)` is deliberately **not** used: its D2H path only enqueues the copy.
- Negative paths: `--inject-finite` / `--inject-nan` apply to a copy of the compared result vector
  and must still end in a non-zero exit; they never modify the CPU or analytic reference.
- Not formatted: `clang-format-16` is not available in this workspace, so the new C++ has **not**
  been format-checked yet; this must be done before the server build.

## GPT-2 single-card follow-up (historical notes, superseded by the committed evidence above)

- FP32 completed all 10 training steps on one MLU. With the gradient accumulation
  optimization, the median gap between logged steps fell from about 47.8 s to
  23.9 s, and all 10 reported losses matched the previous MLU run exactly.
  The comparison against the A100 reference remains diagnostic and exceeds the
  FP32 acceptance threshold.
- BF16 autocast initially stopped at a gradient merge in Function::BackwardPartial because an
  FP32 gradient was being added to a BF16 buffer. The subsequent 034/035 structural cleanup
  and 036 run completed the recorded BF16 single-card ten-step path; the earlier note remains
  as history and is not the current status.
- These are staged-run observations from the corresponding changes. The branch
  keeps the existing InfiniTrain submodule revision and the original accuracy
  thresholds.
