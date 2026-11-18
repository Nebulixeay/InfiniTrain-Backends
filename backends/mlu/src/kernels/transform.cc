#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include <cnrt.h>

#include "infini_train/include/common/common.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/runtime_common.h"
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

bool IsLeadMaskShape(const std::vector<int64_t> &in, const std::vector<int64_t> &mk) {
    if (mk.empty() || in.empty() || mk.size() > in.size()) {
        return false;
    }
    for (size_t d = 0; d < mk.size(); ++d) {
        if (!(mk[d] == in[d] || mk[d] == 1)) {
            return false;
        }
    }
    return true;
}

bool IsTailMaskShape(const std::vector<int64_t> &in, const std::vector<int64_t> &mk) {
    if (mk.size() > in.size()) {
        return false;
    }
    const size_t k = mk.size();
    for (size_t i = 0; i < k; ++i) {
        if (!(mk[i] == in[in.size() - k + i] || mk[i] == 1)) {
            return false;
        }
    }
    return true;
}

// Prefer a trailing mask when both layouts match, consistent with CUDA.
bool DecideMaskLead(const std::vector<int64_t> &in, const std::vector<int64_t> &mk) {
    const bool lead = IsLeadMaskShape(in, mk);
    const bool tail = IsTailMaskShape(in, mk);
    CHECK(lead || tail) << "Mask must align/broadcast to either leading or trailing axes.";
    return !tail;
}

MluIndexMeta MakeSliceMeta(const std::vector<int64_t> &in_dims, const std::vector<int64_t> &new_dims,
                           const std::vector<int64_t> &starts, const std::vector<int64_t> &steps) {
    CHECK_LE(in_dims.size(), MLU_MAX_META_DIMS);
    MluIndexMeta meta{};
    meta.ndim = static_cast<int32_t>(in_dims.size());
    const auto in_strides = ComputeStrides(in_dims);
    const auto out_strides = ComputeStrides(new_dims);
    for (size_t i = 0; i < in_dims.size(); ++i) {
        meta.dims[i] = new_dims[i];
        meta.in_strides[i] = in_strides[i];
        meta.out_strides[i] = out_strides[i];
        meta.starts[i] = starts[i];
        meta.steps[i] = steps[i];
    }
    return meta;
}

std::vector<int64_t> SliceNewDims(const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                  const std::vector<int64_t> &steps) {
    std::vector<int64_t> new_dims;
    for (size_t i = 0; i < starts.size(); ++i) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LT(0, steps[i]) << "MLU Slice requires a positive step";
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }
    return new_dims;
}

} // namespace

// ----------------------------- Tril / Triu -----------------------------

std::shared_ptr<Tensor> TrilForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            TrilTriuLaunch(input->DataPtr(), output->DataPtr(), input->Dims()[0], input->Dims()[1], diagonal, false,
                           MluTypeTag<T>(), queue);
        },
        "MLU TrilForward");
    return output;
}

std::shared_ptr<Tensor> TrilBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    return TrilForward(grad_output, diagonal);
}

std::shared_ptr<Tensor> TriuForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            TrilTriuLaunch(input->DataPtr(), output->DataPtr(), input->Dims()[0], input->Dims()[1], diagonal, true,
                           MluTypeTag<T>(), queue);
        },
        "MLU TriuForward");
    return output;
}

std::shared_ptr<Tensor> TriuBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    return TriuForward(grad_output, diagonal);
}

// ----------------------------- Transpose -----------------------------

std::shared_ptr<Tensor> TransposeForward(const std::shared_ptr<Tensor> &input, int64_t dim0, int64_t dim1) {
    const auto in_dims = input->Dims();
    CHECK_LE(in_dims.size(), MLU_MAX_META_DIMS);
    const int64_t ndim = in_dims.size();
    dim0 = dim0 < 0 ? dim0 + ndim : dim0;
    dim1 = dim1 < 0 ? dim1 + ndim : dim1;
    CHECK(dim0 >= 0 && dim0 < ndim && dim1 >= 0 && dim1 < ndim);

    auto out_dims = in_dims;
    std::swap(out_dims[dim0], out_dims[dim1]);
    auto output = std::make_shared<Tensor>(out_dims, input->Dtype(), input->GetDevice());

    MluIndexMeta meta{};
    meta.ndim = static_cast<int32_t>(ndim);
    const auto in_strides = ComputeStrides(in_dims);
    const auto out_strides = ComputeStrides(out_dims);
    for (int64_t i = 0; i < ndim; ++i) {
        meta.in_strides[i] = in_strides[i];
        meta.out_strides[i] = out_strides[i];
    }

    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            TransposeLaunch(input->DataPtr(), output->DataPtr(), meta, dim0, dim1,
                            static_cast<int64_t>(input->NumElements()), MluTypeTag<T>(), queue);
        },
        "MLU TransposeForward");
    return output;
}

// Backward transposes the same pair of dimensions.
std::shared_ptr<Tensor> TransposeBackward(const std::shared_ptr<Tensor> &grad_output, int64_t dim0, int64_t dim1) {
    return TransposeForward(grad_output, dim0, dim1);
}

// ----------------------------- Mask（MaskedFill） -----------------------------

std::shared_ptr<Tensor> MaskForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &mask,
                                    float value) {
    const auto dtype = input->Dtype();
    auto mask_casted = mask->Dtype() == dtype ? mask : std::make_shared<Tensor>(mask->To(dtype));
    CHECK_EQ(static_cast<int>(dtype), static_cast<int>(mask_casted->Dtype()))
        << "For now, input/mask dtypes must match.";

    const bool lead = DecideMaskLead(input->Dims(), mask_casted->Dims());
    auto output = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());

    const int64_t mask_extent = mask_casted->NumElements();
    const int64_t num_elements = input->NumElements();
    const int64_t inner = num_elements / mask_extent;
    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        dtype,
        [=]<typename T>() {
            MaskForwardLaunch(input->DataPtr(), mask_casted->DataPtr(), output->DataPtr(), value, lead, mask_extent,
                              inner, num_elements, MluTypeTag<T>(), queue);
        },
        "MLU MaskForward");
    return output;
}

std::shared_ptr<Tensor> MaskBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask) {
    const auto dtype = grad_output->Dtype();
    auto mask_casted = mask->Dtype() == dtype ? mask : std::make_shared<Tensor>(mask->To(dtype));

    const bool lead = DecideMaskLead(grad_output->Dims(), mask_casted->Dims());
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), dtype, grad_output->GetDevice());

    const int64_t mask_extent = mask_casted->NumElements();
    const int64_t num_elements = grad_output->NumElements();
    const int64_t inner = num_elements / mask_extent;
    const auto queue = CurrentQueue(grad_output->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        dtype,
        [=]<typename T>() {
            MaskBackwardLaunch(grad_output->DataPtr(), mask_casted->DataPtr(), grad_input->DataPtr(), lead, mask_extent,
                               inner, num_elements, MluTypeTag<T>(), queue);
        },
        "MLU MaskBackward");
    return grad_input;
}

// ----------------------------- Slice -----------------------------

std::shared_ptr<Tensor> SliceForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                                     const std::vector<int64_t> &ends, const std::vector<int64_t> &steps) {
    CHECK_EQ(starts.size(), ends.size());
    CHECK_EQ(starts.size(), steps.size());
    const auto &dims = input->Dims();
    CHECK_EQ(starts.size(), dims.size());

    const auto new_dims = SliceNewDims(starts, ends, steps);
    auto output = std::make_shared<Tensor>(new_dims, input->Dtype(), input->GetDevice());
    const auto meta = MakeSliceMeta(dims, new_dims, starts, steps);

    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            SliceForwardLaunch(input->DataPtr(), output->DataPtr(), meta, static_cast<int64_t>(output->NumElements()),
                               MluTypeTag<T>(), queue);
        },
        "MLU SliceForward");
    return output;
}

std::shared_ptr<Tensor> SliceBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                      const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                      const std::vector<int64_t> &steps) {
    CHECK_EQ(starts.size(), ends.size());
    CHECK_EQ(starts.size(), steps.size());
    const auto &dims = input->Dims();
    CHECK_EQ(starts.size(), dims.size());

    const auto new_dims = SliceNewDims(starts, ends, steps);
    auto grad_input = std::make_shared<Tensor>(dims, grad_output->Dtype(), grad_output->GetDevice());
    // Keep the region outside the slice zero.
    grad_input->Fill(0.0);
    const auto meta = MakeSliceMeta(dims, new_dims, starts, steps);

    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            SliceBackwardLaunch(grad_output->DataPtr(), grad_input->DataPtr(), meta,
                                static_cast<int64_t>(grad_output->NumElements()), MluTypeTag<T>(), queue);
        },
        "MLU SliceBackward");
    return grad_input;
}

} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_TRANSFORM_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_TRANSFORM_KERNEL(TrilForward)
REGISTER_MLU_TRANSFORM_KERNEL(TrilBackward)
REGISTER_MLU_TRANSFORM_KERNEL(TriuForward)
REGISTER_MLU_TRANSFORM_KERNEL(TriuBackward)
REGISTER_MLU_TRANSFORM_KERNEL(TransposeForward)
REGISTER_MLU_TRANSFORM_KERNEL(TransposeBackward)
REGISTER_MLU_TRANSFORM_KERNEL(MaskForward)
REGISTER_MLU_TRANSFORM_KERNEL(MaskBackward)
REGISTER_MLU_TRANSFORM_KERNEL(SliceForward)
REGISTER_MLU_TRANSFORM_KERNEL(SliceBackward)

#undef REGISTER_MLU_TRANSFORM_KERNEL
