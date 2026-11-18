#include <cstdint>
#include <memory>
#include <ostream>
#include <tuple>
#include <vector>

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
namespace {
cnrtQueue_t CurrentQueue(const Device &device) {
    return dynamic_cast<infini_train::core::mlu::MluStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->mlu_queue();
}
} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                 const std::shared_ptr<Tensor> &bias, const float eps) {
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_LE(input->Dims()[2], weight->Dims()[0]);
    CHECK_LE(input->Dims()[2], bias->Dims()[0]);

    const int64_t batch_size = input->Dims()[0];
    const int64_t max_seqlen = input->Dims()[1];
    const int64_t embed_dim = input->Dims()[2];

    auto dtype = input->Dtype();
    CHECK(dtype == weight->Dtype() && dtype == bias->Dtype()) << "MLU LayerNormForward dtype mismatch";
    auto output = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());
    auto mean = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32,
                                         input->GetDevice());
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen}, DataType::kFLOAT32,
                                         input->GetDevice());

    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            LayerNormForwardLaunch(input->DataPtr(), weight->DataPtr(), bias->DataPtr(),
                                   static_cast<float *>(mean->DataPtr()), static_cast<float *>(rstd->DataPtr()),
                                   output->DataPtr(), eps, batch_size * max_seqlen, embed_dim, MluTypeTag<T>(), queue);
        },
        "MLU LayerNormForward");
    return {output, mean, rstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LayerNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &mean,
                  const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_LE(input->Dims()[2], weight->Dims()[0]);
    CHECK_LE(input->Dims()[2], bias->Dims()[0]);

    const int64_t batch_size = input->Dims()[0];
    const int64_t max_seqlen = input->Dims()[1];
    const int64_t embed_dim = input->Dims()[2];
    const int64_t num_tokens = batch_size * max_seqlen;

    auto dtype = input->Dtype();
    CHECK(dtype == weight->Dtype() && dtype == bias->Dtype() && dtype == grad_output->Dtype()
          && mean->Dtype() == DataType::kFLOAT32 && rstd->Dtype() == DataType::kFLOAT32);

    auto grad_input = std::make_shared<Tensor>(input->Dims(), dtype, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), dtype, grad_output->GetDevice());
    auto grad_bias = std::make_shared<Tensor>(bias->Dims(), dtype, grad_output->GetDevice());

    const auto queue = CurrentQueue(grad_output->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            LayerNormBackwardLaunch(
                input->DataPtr(), grad_output->DataPtr(), static_cast<const float *>(mean->DataPtr()),
                static_cast<const float *>(rstd->DataPtr()), weight->DataPtr(), grad_input->DataPtr(),
                grad_weight->DataPtr(), grad_bias->DataPtr(), num_tokens, embed_dim, MluTypeTag<T>(), queue);
        },
        "MLU LayerNormBackward");
    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_LAYERNORM_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_LAYERNORM_KERNEL(LayerNormForward)
REGISTER_MLU_LAYERNORM_KERNEL(LayerNormBackward)

#undef REGISTER_MLU_LAYERNORM_KERNEL
