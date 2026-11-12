# 单卡训练实验结果索引

本目录把任务二已完成的单卡实验结果放进产品仓库，确保后续 PR 的 Files changed 中可直接看到 loss 和运行证据。

| 实验 | 结论 | 证据目录 |
|---|---|---|
| FP32 单卡十步 | `MEASURED_10_STEPS`；rc=0、10/10；formal 精度仍 blocked | [`023_fp32_10_steps`](023_fp32_10_steps/README.md) |
| BF16 单卡十步 | `BF16_10_STEPS_COMPLETED_DIAGNOSTIC`；rc=0、10/10；step1 loss 一致 | [`036_bf16_10_steps`](036_bf16_10_steps/README.md) |

## 共同配置

GPT-2 124M、device0、batch_size=4、sequence_length=64、total_batch_size=256、overfit_single_batch=true、learning_rate=1e-4。具体 argv、环境、源码/二进制/输入 hash 以各实验的 `run.argv.txt` 和 `manifest.json` 为准。

## 结果口径

这些记录证明单卡训练路径和十步运行行为，包含完整的 forward、loss、backward 路径；固定单 batch 下 loss 随 step 变化，说明其后发生了参数更新，但原始日志没有单独记录 optimizer 行。不能单独宣称跨框架 formal 精度通过。A100 参考的 `weights_sha256`、`data_sha256` 尚未闭合；FP32 现有跨框架比较的最差 step 2 `abs_diff=1.620e-4` 已超过原始阈值，即使补齐参考资产 hash 也仍是 `DIAGNOSTIC_FAIL`。各 `run.argv.txt` 中的 `weights_sha256`/`data_sha256` 是本机 MLU 权重和数据资产，不是 A100 参考资产。多卡 DP/CNCL 不在这些单卡结果范围内。

两轮目录还保留了运行起始状态、设备中段快照和判定工具输出，便于同事从 PR 追溯“跑了什么、运行到哪一步、设备是否正常收尾”。

原始运行产物按字节保留。个别文件包含抓取时的 SSH 登录横幅、空首行、本机路径、容器主机名或内部事件标识；这些内容不含凭据，保留它们是为了让证据可以回溯到原始现场。
