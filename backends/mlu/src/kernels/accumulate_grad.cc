// 对应上游 src/kernels/cuda/accumulate_grad.cu 的 host 集成部分；设备 kernel 在 accumulate_grad.mlu。

#include <cmath>
#include <cstdint>
#include <memory>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "kernels/mlu_kernel_api.h"
#include "runtime/mlu_dispatch.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    // 029：grad 与 param 必须同 dtype。设备 kernel 以 gradient->Dtype() 选定 T，并把 tensor 指针按同一 T
    // 解释；一旦不一致（典型来源是 autocast 产生的 bf16 grad + fp32 param），就会按错误的元素宽度写入，
    // 产生静默错值（028 实测 changed=32/64、loss 仍有限）。这里在任何取 queue / 入队之前显式中止，
    // 与 CUDA/CPU 侧"grad 与 param 同 dtype"的既有不变量对齐。FP32 训练两侧同 dtype，恒不触发。
    CHECK(gradient->Dtype() == tensor->Dtype())
        << "MLU AccumulateGrad dtype mismatch: grad=" << kDataTypeToDesc.at(gradient->Dtype())
        << " param=" << kDataTypeToDesc.at(tensor->Dtype()) << " (refusing to reinterpret memory)";
    auto device = tensor->GetDevice();
    const auto queue = dynamic_cast<infini_train::core::mlu::MluStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->mlu_queue();

    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        gradient->Dtype(),
        [=]<typename T>() {
            AccumulateGradLaunch(gradient->DataPtr(), rate, tensor->DataPtr(), MluTypeTag<T>(),
                                 static_cast<int64_t>(gradient->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "AccumulateGrad");
        },
        "MLU AccumulateGrad");
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // 029：Adam 的 temp 状态 m/v 与 param 同样按 grad 的 T 写入，四者必须同 dtype，否则同样会静默错值。
    CHECK(grad->Dtype() == param->Dtype() && grad->Dtype() == m->Dtype() && grad->Dtype() == v->Dtype())
        << "MLU AdamAccumulateGrad dtype mismatch: grad=" << kDataTypeToDesc.at(grad->Dtype())
        << " param=" << kDataTypeToDesc.at(param->Dtype()) << " m=" << kDataTypeToDesc.at(m->Dtype())
        << " v=" << kDataTypeToDesc.at(v->Dtype()) << " (refusing to reinterpret memory)";
    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    auto device = grad->GetDevice();
    const auto queue = dynamic_cast<infini_train::core::mlu::MluStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->mlu_queue();

    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            AdamAccumulateGradLaunch(grad->DataPtr(), param->DataPtr(), m->DataPtr(), v->DataPtr(), learning_rate,
                                     beta1, beta2, eps, bias_correction_m, bias_correction_v, MluTypeTag<T>(),
                                     static_cast<int64_t>(grad->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "AdamAccumulateGrad");
        },
        "MLU AdamAccumulateGrad");
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_MLU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_MLU_ACCUMULATE_GRAD_KERNEL
