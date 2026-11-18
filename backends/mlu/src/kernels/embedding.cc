#include <cstddef>
#include <cstdint>
#include <memory>
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

std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight) {
    CHECK(input->Dtype() == DataType::kINT64);
    CHECK_EQ(weight->Dims().size(), 2);

    const int64_t batch_size = input->Dims().size() == 2 ? input->Dims()[0] : 1;
    const int64_t max_seqlen = input->Dims().size() == 2 ? input->Dims()[1] : input->Dims()[0];
    const int64_t vocab_size = weight->Dims()[0];
    const int64_t embed_dim = weight->Dims()[1];
    auto output_dims = input->Dims();
    output_dims.push_back(embed_dim);

    auto output = std::make_shared<Tensor>(output_dims, weight->Dtype(), input->GetDevice());
    const auto queue = CurrentQueue(input->GetDevice());

    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        weight->Dtype(),
        [=]<typename T>() {
            EmbeddingForwardLaunch(input->DataPtr(), output->DataPtr(), weight->DataPtr(), batch_size * max_seqlen,
                                   embed_dim, vocab_size, MluTypeTag<T>(), queue);
        },
        "MLU EmbeddingForward");
    return output;
}

std::shared_ptr<Tensor> EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &weight_dims,
                                          const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->Dtype() == DataType::kINT64);
    CHECK_EQ(weight_dims.size(), 2);

    const int64_t vocab_size = weight_dims[0];
    const int64_t embedding_dim = weight_dims[1];
    CHECK_EQ(input->Dims().size() + 1, grad_output->Dims().size());
    for (size_t idx = 0; idx < input->Dims().size(); ++idx) { CHECK_EQ(input->Dims()[idx], grad_output->Dims()[idx]); }
    CHECK_EQ(*grad_output->Dims().rbegin(), embedding_dim);

    auto grad_weight = std::make_shared<Tensor>(weight_dims, grad_output->Dtype(), grad_output->GetDevice());
    // Zero the gradient buffer before accumulation.
    grad_weight->Fill(0.0);
    const auto queue = CurrentQueue(input->GetDevice());

    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            EmbeddingBackwardLaunch(input->DataPtr(), grad_output->DataPtr(), grad_weight->DataPtr(),
                                    input->NumElements(), embedding_dim, vocab_size, MluTypeTag<T>(), queue);
        },
        "MLU EmbeddingBackward");
    return grad_weight;
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_EMBEDDING_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_EMBEDDING_KERNEL(EmbeddingForward)
REGISTER_MLU_EMBEDDING_KERNEL(EmbeddingBackward)

#undef REGISTER_MLU_EMBEDDING_KERNEL
