#pragma once

// Host-only dtype dispatch using the vendored BANG headers.
// Device sources use mlu_kernel_api.h and must not include framework C++20 headers.

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include <bang_bf16.h>
#include <bang_fp16.h>

#include "infini_train/include/core/backend_type_map.h"
#include "infini_train/include/dtype_dispatch.h"

#include "kernels/mlu_kernel_api.h"

// Map FP16 to __half and BF16 to bfloat16_t; the host headers do not define half.
namespace infini_train::core {
template <> struct BackendTypeMap<Device::DeviceType::kPrivateUse1, DataType::kFLOAT16> {
    using type = __half;
};

template <> struct BackendTypeMap<Device::DeviceType::kPrivateUse1, DataType::kBFLOAT16> {
    using type = bfloat16_t;
};
} // namespace infini_train::core

// Register standard dtypes; the low-precision specializations are defined above.
INFINI_REGISTER_STANDARD_BACKEND_TYPES(infini_train::Device::DeviceType::kPrivateUse1)

// Exclude FP64: the device toolchain does not support double arithmetic.
// Unsupported dtype requests fail in dispatch.
#define INFINI_MLU_FLOATING_TYPES DataType::kFLOAT32, DataType::kFLOAT16, DataType::kBFLOAT16
#define INFINI_MLU_ALL_NUMERIC_TYPES INFINI_MLU_FLOATING_TYPES, INFINI_ALL_INTEGRAL_TYPES
#define INFINI_MLU_ALL_NUMERIC_AND_LOGICAL_TYPES INFINI_MLU_ALL_NUMERIC_TYPES, INFINI_LOGICAL_TYPES

namespace infini_train::core::mlu {

template <DataType DType> struct MluTypeMap : BackendTypeMap<Device::DeviceType::kPrivateUse1, DType> {};

// -----------------------------------------------------------------------------
// MLU dispatch helpers
// -----------------------------------------------------------------------------

template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchMluFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    return infini_train::DispatchByTypeMap<MluTypeMap, AllowedDTypes...>(
        dtype, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchMluFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                     Args &&...args) {
    return infini_train::DispatchByTypeMap<MluTypeMap, AllowedTypeLists...>(
        dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

} // namespace infini_train::core::mlu

// Map host scalar types to launcher dtype tags.
namespace infini_train::kernels::mlu {

template <typename T> constexpr MluScalarType MluTypeTag();
template <> constexpr MluScalarType MluTypeTag<bool>() { return MluScalarType::kBool; }
template <> constexpr MluScalarType MluTypeTag<uint8_t>() { return MluScalarType::kU8; }
template <> constexpr MluScalarType MluTypeTag<int8_t>() { return MluScalarType::kI8; }
template <> constexpr MluScalarType MluTypeTag<uint16_t>() { return MluScalarType::kU16; }
template <> constexpr MluScalarType MluTypeTag<int16_t>() { return MluScalarType::kI16; }
template <> constexpr MluScalarType MluTypeTag<uint32_t>() { return MluScalarType::kU32; }
template <> constexpr MluScalarType MluTypeTag<int32_t>() { return MluScalarType::kI32; }
template <> constexpr MluScalarType MluTypeTag<uint64_t>() { return MluScalarType::kU64; }
template <> constexpr MluScalarType MluTypeTag<int64_t>() { return MluScalarType::kI64; }
template <> constexpr MluScalarType MluTypeTag<__half>() { return MluScalarType::kF16; }
template <> constexpr MluScalarType MluTypeTag<bfloat16_t>() { return MluScalarType::kBF16; }
template <> constexpr MluScalarType MluTypeTag<float>() { return MluScalarType::kF32; }
template <> constexpr MluScalarType MluTypeTag<double>() { return MluScalarType::kF64; }

} // namespace infini_train::kernels::mlu
