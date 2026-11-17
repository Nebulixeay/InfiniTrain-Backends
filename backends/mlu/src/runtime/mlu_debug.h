#pragma once

// 调试探针：MLU_DEBUG_CHECK=1 时，在关键 kernel 启动后把输出张量拷回 host，
// 打印 NaN/Inf 计数与数值范围，用来定位"第一个产生 NaN 的 kernel"。
// 只在排障时开启，正常训练零开销（一次 getenv）。
//
// 约定：
//   - 按张量真实 dtype 解码，不用 sizeof(T) 猜 FP16/BF16，也不把整型当浮点解码；
//   - 统计值只统计有限元素，min/max 从首个有限值初始化；无有限值时明确打印；
//   - 采样有上限，日志必须能区分"全量"与"前缀采样"；
//   - 启用探针时，真实 SDK 同步/拷贝失败立即终止进程；数值上出现 NaN/Inf 不终止
//     （因果 mask 的 -inf 是预期语义）。

#include <cstdint>

#include <cnrt.h>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/tensor.h"

namespace infini_train::core::mlu {

// 开关解析：unset / "0" / "false" 关闭；"1" / "true" 打开；其他值警告后关闭。
bool DebugFlagEnabled(const char *env_name);

// 可单测的同步结果处理：成功返回 true；失败在探针开启时 LOG(FATAL) 终止进程。
bool CheckSyncResult(cnrtRet_t ret, const char *kernel_name);

void DebugSyncQueue(cnrtQueue_t queue, const char *kernel_name);

// dtype 用 infini_train::DataType 的整数值传入（避免在纯 C 侧引入框架头）。
void DebugCheckTensor(const void *dev_ptr, int64_t num_elements, int dtype, const char *name);

template <typename T> void DebugCheck(const std::shared_ptr<infini_train::Tensor> &t, const char *name) {
    if (!t) {
        return;
    }
    DebugCheckTensor(t->DataPtr(), static_cast<int64_t>(t->NumElements()), static_cast<int>(t->Dtype()), name);
}

} // namespace infini_train::core::mlu
