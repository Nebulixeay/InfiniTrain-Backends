#include <cmath>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

#include "glog/logging.h"
#include <cnrt.h>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "kernels/mlu_kernel_api.h"
#include "runtime/mlu_dispatch.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    // Require identical gradient/parameter dtypes before enqueueing: the kernel accesses both through the gradient
    // type.
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
        },
        "MLU AccumulateGrad");
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // Adam state and parameter buffers must match the gradient dtype used by the kernel.
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
