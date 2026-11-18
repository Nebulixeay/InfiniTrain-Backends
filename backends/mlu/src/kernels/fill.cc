#include <cstdint>
#include <cstring>
#include <memory>
// Used by std::is_same_v in MluScalarCast.
#include <type_traits> // IWYU pragma: keep

#include <bang_bf16.h>
#include <cnrt.h>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/scalar.h"
#include "infini_train/include/tensor.h"

#include "kernels/mlu_kernel_api.h"
#include "runtime/mlu_dispatch.h"
#include "runtime/mlu_runtime_common.h"

struct __half;

namespace infini_train::kernels::mlu {

namespace {

// Convert scalar double to float before half/BF16 to avoid ambiguous BANG constructors.
template <typename T> T MluScalarCast(const Scalar &scalar) {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, bfloat16_t>) {
        return T(scalar.to<float>());
    } else {
        return scalar.to<T>();
    }
}

} // namespace

void Fill(std::shared_ptr<Tensor> tensor, Scalar scalar) {
    auto device = tensor->GetDevice();
    const auto queue = dynamic_cast<infini_train::core::mlu::MluStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->mlu_queue();

    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            const T value = MluScalarCast<T>(scalar);
            static_assert(sizeof(T) <= sizeof(uint64_t));
            uint64_t value_bits = 0;
            std::memcpy(&value_bits, &value, sizeof(T));
            // Use synchronous cnrtMemset for zero fill to avoid slow scalar writes over large gradient buffers.
            if (value_bits == 0) {
                CNRT_CHECK(cnrtMemset(tensor->DataPtr(), 0, static_cast<size_t>(tensor->NumElements()) * sizeof(T)));
                return;
            }
            FillLaunch(tensor->DataPtr(), value_bits, MluTypeTag<T>(), static_cast<int64_t>(tensor->NumElements()),
                       queue);
        },
        "MLU Fill");
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_FILL_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_FILL_KERNEL(Fill)

#undef REGISTER_MLU_FILL_KERNEL
