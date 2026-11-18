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

void ComputeSoftmaxSizes(const std::vector<int64_t> &dims, int64_t dim, int64_t *outer_size, int64_t *axis_size,
                         int64_t *inner_size) {
    *outer_size = 1;
    *axis_size = dims[dim];
    *inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) { *outer_size *= dims[i]; }
    for (size_t i = dim + 1; i < dims.size(); ++i) { *inner_size *= dims[i]; }
}
} // namespace

std::shared_ptr<Tensor> SoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim) {
    const auto &input_dims = input->Dims();
    dim = dim < 0 ? dim + input_dims.size() : dim;
    CHECK(dim >= 0 && dim < static_cast<int64_t>(input_dims.size()));

    auto output = std::make_shared<Tensor>(input_dims, input->Dtype(), input->GetDevice());

    int64_t outer_size, axis_size, inner_size;
    ComputeSoftmaxSizes(input_dims, dim, &outer_size, &axis_size, &inner_size);
    if (axis_size == 0 || outer_size == 0) {
        return output;
    }

    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            SoftmaxForwardLaunch(output->DataPtr(), input->DataPtr(), outer_size, axis_size, inner_size,
                                 MluTypeTag<T>(), queue);
        },
        "MLU SoftmaxForward");
    return output;
}

std::shared_ptr<Tensor> SoftmaxBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &output, int64_t dim) {
    // Match CUDA: both pointers and the result must use the promoted dtype.
    const auto promoted_type = PromoteDataTypes(grad_output->Dtype(), output->Dtype());
    auto grad_output_promoted = grad_output->Dtype() == promoted_type
                                  ? grad_output
                                  : std::make_shared<Tensor>(grad_output->To(promoted_type));
    auto output_promoted
        = output->Dtype() == promoted_type ? output : std::make_shared<Tensor>(output->To(promoted_type));
    const auto &output_dims = output->Dims();
    dim = dim < 0 ? dim + output_dims.size() : dim;
    CHECK(dim >= 0 && dim < static_cast<int64_t>(output_dims.size()));

    auto grad_input = std::make_shared<Tensor>(output_dims, promoted_type, output->GetDevice());

    int64_t outer_size, axis_size, inner_size;
    ComputeSoftmaxSizes(output_dims, dim, &outer_size, &axis_size, &inner_size);
    if (axis_size == 0 || outer_size == 0) {
        return grad_input;
    }

    const auto queue = CurrentQueue(output->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        promoted_type,
        [=]<typename T>() {
            SoftmaxBackwardLaunch(grad_input->DataPtr(), grad_output_promoted->DataPtr(), output_promoted->DataPtr(),
                                  outer_size, axis_size, inner_size, MluTypeTag<T>(), queue);
        },
        "MLU SoftmaxBackward");
    return grad_input;
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_SOFTMAX_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_SOFTMAX_KERNEL(SoftmaxForward)
REGISTER_MLU_SOFTMAX_KERNEL(SoftmaxBackward)

#undef REGISTER_MLU_SOFTMAX_KERNEL
