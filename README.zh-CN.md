# InfiniTrain Backends

[English](README.md) | **简体中文**

[![Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/issues
)
[![PR](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/pulls
)
[![License](https://img.shields.io/github/license/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/blob/master/LICENSE
)

InfiniTrain Backends 为 [InfiniTrain](https://github.com/InfiniTensor/InfiniTrain)
提供独立于框架核心维护的加速器后端。各后端实现 InfiniTrain 的 `PrivateUse1`
接口，将厂商 SDK 集成、运行时支持、集合通信和算子实现放在框架核心之外。

每个后端位于独立的 `backends/<provider>` 目录中，通常使用固定为 `v0.7.0` 的
InfiniTrain 子模块。每个构建目录只选择一个后端，每个进程最多注册一个
`DeviceType::kPrivateUse1` 后端。

## 支持的后端

| 后端 | 运行时 | 集合通信 | 模型示例 |
| ---- | ------ | -------- | -------- |
| [MACA](backends/maca/README.zh-CN.md) | MACA | MCCL（可选） | GPT-2、LLaMA 3、Mixtral |
| [MLU](backends/mlu/README.md)（开发中） | CNRT | — | GPT-2 |

模型源码通常来自锁定版本的 InfiniTrain 子模块。各后端提供运行时、算子、集合通信
实现和构建集成，以便在对应加速器上运行其支持的模型示例。环境配置和使用方式见上表
链接的各后端指南。

## 通用依赖

- **Linux**
- **CMake**：3.28 或更新版本
- 支持子模块的 **Git**
- 与所选后端兼容的 **C++20** 编译器和厂商 SDK

额外依赖及 SDK 配置方式见对应后端指南。

## 构建选项

以下选项适用于各后端。后端专用选项见对应后端指南。

| 选项 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `INFINITRAIN_BACKEND` | 必填 | 选择 `backends/<provider>` 下的后端 |
| `INFINITRAIN_SOURCE_DIR` | `third_party/InfiniTrain` | InfiniTrain 源码目录，通常为锁定版本的子模块 |
| `INFINITRAIN_BACKENDS_BUILD_EXAMPLES` | 独立构建本仓库时为 `ON` | 构建启用所选后端的 InfiniTrain 示例 |
| `BUILD_TEST` | `OFF` | 构建 InfiniTrain 单元测试及所选后端的测试版本 |

请为不同后端使用独立的构建目录，避免混用编译器和 SDK 缓存。独立构建本仓库时，
默认启用后端示例。

如需使用其他 InfiniTrain 工作副本进行开发，可在 CMake 配置命令中添加
`-DINFINITRAIN_SOURCE_DIR=/path/to/InfiniTrain`。所选源码版本必须实现该后端所需的
PrivateUse1 扩展接口。

PrivateUse1 后端要求 `USE_CUDA=OFF`。如果显式设置 `USE_CUDA=ON`，配置会报错，
而不会静默覆盖该选项。所选后端的单元测试使用 PrivateUse1 设备类型和
`PRIVATEUSE1/*` GTest 前缀，与后端名称无关。

## 架构

```text
backends/<provider>/
  README.md              后端环境配置与使用指南
  cmake/                 在 project() 前配置编译器和厂商 SDK
  include/               后端公共接口
  src/backend.cc         后端元数据注册入口
  src/common/            后端内部共享辅助代码
  src/runtime/           设备、流、事件和内存分配集成
  src/kernels/           后端算子实现及注册
  src/ccl/               可选的集合通信集成
  examples/              上游 InfiniTrain 示例的 CMake 适配
  scripts/               后端模型运行配置及封装脚本

third_party/InfiniTrain/ 框架源码，通常为锁定版本的 Git 子模块
```

根目录构建脚本在首次调用 `project()` 之前，根据所选后端加载
`backends/<provider>/cmake/pre_project.cmake`。各后端可在此选择编译器并准备
SDK 相关的依赖探测配置，无需在根目录 `CMakeLists.txt` 中增加厂商分支。

通过 `add_subdirectory()` 将本仓库作为第三方库添加到父项目时，应在启用 C++
语言前选择后端所需的编译器。配置好厂商 SDK 环境后，可按以下顺序编写父项目的
`CMakeLists.txt`，并替换编译器路径和后端名称：

```cmake
cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_COMPILER "/path/to/vendor/compiler" CACHE FILEPATH "")
set(INFINITRAIN_BACKEND "<provider>" CACHE STRING "")

project(MyTrainingApp LANGUAGES CXX)
add_subdirectory(third_party/InfiniTrain-Backends)
```

CMake 无法在启用某种语言后更换其编译器。

## 开发检查

clang-format、clang-tidy 和 IWYU 的范围、依赖及手动使用方式见
[代码格式和静态检查](scripts/README.md)。

## 寒武纪 MLU 后端（开发中）

`backends/mlu` 通过 `PrivateUse1` 注册名为 `mlu` 的后端。11月17日的 BF16 单卡实现
已完成十步训练，其 FP32 单步回归与历史 MLU 首步一致。后续 Cast 优化通过了完整 BF16
单步对照，尚未重新验证十步。算子回归及单独标注版本的训练结果见下方验证文档。

```bash
cmake -S . -B build-mlu -G Ninja \
  -DINFINITRAIN_BACKEND=mlu -DUSE_CUDA=OFF -DBUILD_TEST=OFF
cmake --build build-mlu --target mlu_gpt2 mlu_op_test -j
```

SDK 依赖见 [MLU 后端指南](backends/mlu/README.md)，实际验证状态见
[验证状态](backends/mlu/docs/validation.md)。

## 许可证

InfiniTrain Backends 使用 [MIT 许可证](LICENSE)。
