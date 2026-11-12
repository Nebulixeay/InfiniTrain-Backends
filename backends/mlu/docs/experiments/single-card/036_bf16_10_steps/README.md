# BF16 单卡十步正式验证

- 结论：`BF16_10_STEPS_COMPLETED_DIAGNOSTIC`
- dtype：`bfloat16`
- 运行时源码：父仓 `ea5d11c8c2fe178010d511de64db4b837715b07e`，子模块 `c90338ceb2dc528816bf78c422f1f5483f2be1ae`，工作树含 dirty `function.cc`（sha256=`13830660ca994d995b2f363b5dca18aa2aa90740415ee2ba0d45ecc53ebfba9e`）。`8ca1ce54590f8e4b2c2597ec119d71fba5621d65` 是内容等价提交，不是该二进制直接构建的运行时父仓状态。
- binary sha256：`94a886b7c62f273da2d73a02779e64abbc6fa4ed6fe0e6eca330d6090b1205df`
- 原始训练日志 sha256：`44223de2c468069d664ba65ae201f4f135b647e5f1fa66366577c13bb3911662`
- 原始日志行数：`2514`；退出码：`0`
- 运行耗时：`8456.023 s`
- 证据范围：只记录已实际运行的 GPT-2 124M、device0、batch=4、sequence=64、total_batch=256、overfit_single_batch 配置。
- 训练阶段：10 个 step 均经过完整 forward、loss、backward 路径；固定单 batch 下 loss 逐 step 变化，说明其后发生了参数更新，但原始日志没有单独记录 optimizer 行。

## Loss 表

| step | loss | lr | step_ms | peak MB |
|---:|---:|---:|---:|---:|
| 1 | 5.290029 | 1.00e-04 | 820001.47 | 1517 |
| 2 | 5.029475 | 1.00e-04 | 845049.88 | 1517 |
| 3 | 4.862110 | 1.00e-04 | 845174.29 | 1517 |
| 4 | 4.941002 | 1.00e-04 | 846161.59 | 1517 |
| 5 | 4.915048 | 1.00e-04 | 845742.74 | 1517 |
| 6 | 5.080010 | 1.00e-04 | 845669.58 | 1517 |
| 7 | 4.849368 | 1.00e-04 | 846174.44 | 1517 |
| 8 | 4.897724 | 1.00e-04 | 844615.63 | 1517 |
| 9 | 5.097935 | 1.00e-04 | 845645.55 | 1517 |
| 10 | 5.235152 | 1.00e-04 | 846141.79 | 1517 |

## 阶段/边界

phase_enter=490；phase_done=490；MLU_LINEAR_PROFILE=2440；forward 共 4346.2 s、loss forward 共 292.3 s、backward 共 3576.4 s；9 个 gap 的 median=23.939 s（min=23.934，max=23.939）。实际 9 个 gap 为 `23.937 / 23.934 / 23.939 / 23.939 / 23.939 / 23.939 / 23.935 / 23.939 / 23.939 s`。

## 文件

- `manifest.json`：身份、hash、逐步结果和结论。
- `run.argv.txt`：实际 argv/env/timeout/起止时间。
- `train.txt`：完整原始训练日志。
- `train.rc` / `train.secs*`：退出码和耗时。

## 解释边界

rc=0、10/10、step1=5.290029、loss 全 finite、无 FATAL/CNRT/NaN/Inf/core/cndump、设备回空闲；不是 formal accuracy PASS。

该二进制来自带未提交改动的工作树；`content_equivalent_commit` 只用于标识等价内容，不能暗示从该提交直接重建运行时二进制。

## 追加证据

- `start_state_036.txt` / `pre_run_check_036.txt`：运行起始源码、dirty 状态、二进制身份和启动前检查。
- `cnmon_before_bf16_036.txt` / `cnmon_mid_bf16_036.txt` / `cnmon_after_bf16_036.txt`：设备前、第 5 步后、结束快照。
- `verdict_036.txt`：十步完成、错误扫描、阶段统计和设备收尾摘要。
