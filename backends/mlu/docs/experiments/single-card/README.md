# Single-card training experiments

Current design and commands: [validation](../../validation.md). Current results preserve BF16 forward outputs; the legacy output-recast experiment is explicitly separate.

| Experiment | Evidence |
|---|---|
| Historical FP32 ten steps | [2026-11-05-fp32-10-steps](2026-11-05-fp32-10-steps/README.md) |
| Historical BF16 output-recast experiment | [legacy-bf16-output-recast-10-steps](legacy-bf16-output-recast-10-steps/README.md) |
| Historical FP32 ten steps after accumulation optimization | [2026-11-05-fp32-optimized-10-steps](2026-11-05-fp32-optimized-10-steps/README.md) |
| Linear/Matmul backward dtype regression | [2026-11-17-linear-backward-gate](2026-11-17-linear-backward-gate/README.md) |
| BF16 gradient-merge failure before backward alignment | [2026-11-13-bf16-gradient-merge-failure](2026-11-13-bf16-gradient-merge-failure/README.md) |
| BF16 failure before the Softmax correction | [2026-11-17-bf16-pre-softmax-failure](2026-11-17-bf16-pre-softmax-failure/README.md) |
| Softmax backward reproduction and regression | [2026-11-17-softmax-backward-gate](2026-11-17-softmax-backward-gate/README.md) |
| BF16 two-step regression after Softmax correction | [2026-11-17-bf16-2-step-regression](2026-11-17-bf16-2-step-regression/README.md) |
| Current BF16 ten-step experiment | [2026-11-17-bf16-10-steps](2026-11-17-bf16-10-steps/README.md) |
| Current FP32 single-step regression | [2026-11-17-fp32-1-step-regression](2026-11-17-fp32-1-step-regression/README.md) |

Each directory contains original logs, commands/identity metadata and a file-hash manifest. Historic runtime revisions may include dirty patches; their original identities are preserved rather than relabelled as the current commit. The tested current kernel content corresponds to `3a788f3cfa6d729424d894490c8e189269e231b3`; subsequent publication cleanup is documented in [publication source mapping](../../publication-source-mapping.json).

[Build logs and environment](2026-11-17-build/README.md).
