// Linear / Matmul 的 MLU host 侧集成。
//
// NOTE(MLU-FALLBACK) 设备 GEMM 尚未接入：本文件目前用 D2H -> host 分块 GEMM -> H2D
// 作为临时实现，只为先把 FP32 数值链路跑通；正式验收要求替换为设备端 GEMM
// （例如 CNNL，且 handle 必须绑定到当前 stream）。
// 精度：host 侧为 float 顺序累加（k 递增），与分块累加的差异仅来自舍入。

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "runtime/mlu_debug.h"
#include "runtime/mlu_dispatch.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {
namespace {

// C(m,n) = op(A)(m,k) * op(B)(k,n)  + beta * C(m,n)
// 行主序；A 的 lda 为行跨度，B 的 ldb 为行跨度。alpha 固定为 1（本项目用不到）。
void HostGemm(bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k, const float *a, int64_t lda, const float *b,
              int64_t ldb, float beta, float *c, int64_t ldc) {
    // i-k-j 序：内层沿 n 连续访问 B、C，缓存友好；k 递增累加保证顺序确定性
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        float *c_row = c + i * ldc;
        for (int64_t j = 0; j < n; ++j) { c_row[j] = beta == 0.0f ? 0.0f : beta * c_row[j]; }
        for (int64_t p = 0; p < k; ++p) {
            const float av = trans_a ? a[p * lda + i] : a[i * lda + p];
            if (av == 0.0f) {
                continue;
            }
            const float *b_row = trans_b ? (b + p) : (b + p * ldb);
            const int64_t b_step = trans_b ? ldb : 1;
            for (int64_t j = 0; j < n; ++j) { c_row[j] += av * b_row[j * b_step]; }
        }
    }
}

// 设备张量 <-> host float 缓冲（16 位浮点经 float 中转）
std::vector<float> ToHostFloat(const std::shared_ptr<Tensor> &t, Device::DeviceType type) {
    const size_t num = t->NumElements();
    std::vector<float> host(num);
    auto *impl = infini_train::core::GetDeviceGuardImpl(type);
    switch (t->Dtype()) {
    case DataType::kFLOAT32:
        impl->Memcpy(host.data(), t->DataPtr(), num * sizeof(float), infini_train::core::MemcpyKind::kD2H);
        break;
    case DataType::kFLOAT16:
    case DataType::kBFLOAT16: {
        std::vector<uint16_t> raw(num);
        impl->Memcpy(raw.data(), t->DataPtr(), num * sizeof(uint16_t), infini_train::core::MemcpyKind::kD2H);
        for (size_t i = 0; i < num; ++i) {
            if (t->Dtype() == DataType::kFLOAT16) {
                host[i] = infini_train::detail::Fp16BitsToFloat(raw[i]);
            } else {
                host[i] = infini_train::detail::Bf16BitsToFloat(raw[i]);
            }
        }
        break;
    }
    default:
        LOG(FATAL) << "MLU GEMM fallback: unsupported dtype " << static_cast<int>(t->Dtype());
    }
    return host;
}

void FromHostFloat(const std::vector<float> &host, const std::shared_ptr<Tensor> &t, Device::DeviceType type) {
    const size_t num = t->NumElements();
    auto *impl = infini_train::core::GetDeviceGuardImpl(type);
    switch (t->Dtype()) {
    case DataType::kFLOAT32:
        impl->Memcpy(t->DataPtr(), host.data(), num * sizeof(float), infini_train::core::MemcpyKind::kH2D);
        break;
    case DataType::kFLOAT16:
    case DataType::kBFLOAT16: {
        std::vector<uint16_t> raw(num);
        for (size_t i = 0; i < num; ++i) {
            raw[i] = t->Dtype() == DataType::kFLOAT16 ? infini_train::detail::FloatToFp16Bits(host[i])
                                                      : infini_train::detail::FloatToBf16Bits(host[i]);
        }
        impl->Memcpy(t->DataPtr(), raw.data(), num * sizeof(uint16_t), infini_train::core::MemcpyKind::kH2D);
        break;
    }
    default:
        LOG(FATAL) << "MLU GEMM fallback: unsupported dtype " << static_cast<int>(t->Dtype());
    }
}

int64_t NumElementsOf(const std::vector<int64_t> &dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
}

// NOTE(014-P1 v1) 分段计时临时探针：仅 MLU_LINEAR_PROFILE=1 时生效，默认零日志、零行为变化。
// 开关语义沿用 mlu_debug.h 的约定：unset / "0" / "false" 关闭，其它值开启。
bool LinearProfileEnabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("MLU_LINEAR_PROFILE");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

double MsBetween(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int64_t NextLinearProfileCallId() {
    static int64_t next_call_id = 0;
    return next_call_id++;
}

std::string LinearProfileTag(int64_t call_id, int64_t rows, int64_t in_features, int64_t out_features, bool has_bias) {
    std::ostringstream oss;
    oss << "[MLU_LINEAR_PROFILE] call=" << call_id << " rows=" << rows << " in_features=" << in_features
        << " out_features=" << out_features << " has_bias=" << (has_bias ? 1 : 0);
    return oss.str();
}

} // namespace

// -----------------------------------------------------------------------------
// Linear
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int64_t out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    const int64_t rows = NumElementsOf(input_dims) / in_features;
    const bool profile = LinearProfileEnabled();
    const int64_t profile_call = profile ? NextLinearProfileCallId() : -1;
    std::string profile_tag;
    std::chrono::steady_clock::time_point t_enter;
    double tohost_input_weight_ms = 0.0;
    double gemm_ms = 0.0;
    double bias_tohost_ms = 0.0;
    double bias_add_ms = 0.0;
    double h2d_ms = 0.0;
    if (profile) {
        profile_tag = LinearProfileTag(profile_call, rows, in_features, out_features, bias != nullptr);
        t_enter = std::chrono::steady_clock::now();
        LOG(INFO) << profile_tag << " phase=enter";
    }

    auto in_host = ToHostFloat(input, input->GetDevice().type());
    auto w_host = ToHostFloat(weight, weight->GetDevice().type());
    std::vector<float> out_host(static_cast<size_t>(rows * out_features));
    if (profile) {
        const auto t_after = std::chrono::steady_clock::now();
        tohost_input_weight_ms = MsBetween(t_enter, t_after);
        LOG(INFO) << profile_tag << " phase=after_tohost_input_weight"
                  << " tohost_input_weight_ms=" << tohost_input_weight_ms;
    }

    // output = input(rows,in) * weight^T(in,out)（transpose=true 时 weight 为 (out,in)）
    HostGemm(false, transpose, rows, out_features, in_features, in_host.data(), in_features, w_host.data(),
             transpose ? in_features : out_features, 0.0f, out_host.data(), out_features);
    if (profile) {
        const auto t_after = std::chrono::steady_clock::now();
        gemm_ms = MsBetween(t_enter, t_after) - tohost_input_weight_ms;
        LOG(INFO) << profile_tag << " phase=after_gemm"
                  << " gemm_ms=" << gemm_ms;
    }

    if (bias) {
        const auto t_bias_enter = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto b_host = ToHostFloat(bias, bias->GetDevice().type());
        if (profile) {
            const auto t_after = std::chrono::steady_clock::now();
            bias_tohost_ms = MsBetween(t_bias_enter, t_after);
            LOG(INFO) << profile_tag << " phase=after_bias_tohost"
                      << " bias_tohost_ms=" << bias_tohost_ms;
        }
        const auto t_bias_add = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t j = 0; j < out_features; ++j) { out_host[r * out_features + j] += b_host[j]; }
        }
        if (profile) {
            const auto t_after = std::chrono::steady_clock::now();
            bias_add_ms = MsBetween(t_bias_add, t_after);
            LOG(INFO) << profile_tag << " phase=after_bias_add"
                      << " bias_add_ms=" << bias_add_ms;
        }
    }

    const auto t_h2d_enter = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    FromHostFloat(out_host, output, output->GetDevice().type());
    if (profile) {
        const auto t_done = std::chrono::steady_clock::now();
        h2d_ms = MsBetween(t_h2d_enter, t_done);
        LOG(INFO) << profile_tag << " phase=done"
                  << " tohost_ms=" << (tohost_input_weight_ms + bias_tohost_ms)
                  << " gemm_bias_ms=" << (gemm_ms + bias_add_ms) << " h2d_ms=" << h2d_ms
                  << " total_ms=" << MsBetween(t_enter, t_done);
    }
    if (output->Dtype() == DataType::kFLOAT32) {
        ::infini_train::core::mlu::DebugCheckTensor(output->DataPtr(), output->NumElements(),
                                                    static_cast<int>(output->Dtype()), "LinearForward");
    }
    return output;
}

std::shared_ptr<Tensor> LinearBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output, bool transpose,
                                            int64_t in_features, int64_t out_features,
                                            const std::vector<int64_t> &input_dims) {
    auto grad_input = std::make_shared<Tensor>(input_dims, grad_output->Dtype(), grad_output->GetDevice());
    const int64_t rows = NumElementsOf(input_dims) / in_features;

    auto g_host = ToHostFloat(grad_output, grad_output->GetDevice().type());
    auto w_host = ToHostFloat(weight, weight->GetDevice().type());
    std::vector<float> gi_host(static_cast<size_t>(rows * in_features));

    if (transpose) {
        // weight 为 [out, in]：grad_input(rows,in) = grad_output(rows,out) * weight(out,in)
        HostGemm(false, false, rows, in_features, out_features, g_host.data(), out_features, w_host.data(), in_features,
                 0.0f, gi_host.data(), in_features);
    } else {
        // weight 为 [in, out]：grad_input(rows,in) = grad_output(rows,out) * weight(in,out)^T
        HostGemm(false, true, rows, in_features, out_features, g_host.data(), out_features, w_host.data(), out_features,
                 0.0f, gi_host.data(), in_features);
    }

    FromHostFloat(gi_host, grad_input, grad_input->GetDevice().type());
    return grad_input;
}

std::shared_ptr<Tensor> LinearBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output, bool transpose,
                                             int64_t in_features, int64_t out_features) {
    const std::vector<int64_t> weight_dims
        = transpose ? std::vector<int64_t>{out_features, in_features} : std::vector<int64_t>{in_features, out_features};
    auto grad_weight = std::make_shared<Tensor>(weight_dims, grad_output->Dtype(), grad_output->GetDevice());
    const int64_t rows = input->NumElements() / in_features;

    auto in_host = ToHostFloat(input, input->GetDevice().type());
    auto g_host = ToHostFloat(grad_output, grad_output->GetDevice().type());
    std::vector<float> gw_host(static_cast<size_t>(out_features * in_features));

    if (transpose) {
        // grad_weight(out,in) = grad_output^T(out,rows) * input(rows,in)
        HostGemm(true, false, out_features, in_features, rows, g_host.data(), out_features, in_host.data(), in_features,
                 0.0f, gw_host.data(), in_features);
    } else {
        // grad_weight(in,out) = input^T(in,rows) * grad_output(rows,out)
        HostGemm(true, false, in_features, out_features, rows, in_host.data(), in_features, g_host.data(), out_features,
                 0.0f, gw_host.data(), out_features);
    }

    FromHostFloat(gw_host, grad_weight, grad_weight->GetDevice().type());
    return grad_weight;
}

std::shared_ptr<Tensor> LinearBackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_features) {
    auto grad_bias
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, grad_output->Dtype(), grad_output->GetDevice());
    const int64_t rows = grad_output->NumElements() / out_features;

    auto g_host = ToHostFloat(grad_output, grad_output->GetDevice().type());
    std::vector<float> gb_host(static_cast<size_t>(out_features), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < out_features; ++j) { gb_host[j] += g_host[r * out_features + j]; }
    }

    FromHostFloat(gb_host, grad_bias, grad_bias->GetDevice().type());
    return grad_bias;
}

// -----------------------------------------------------------------------------
// Matmul（批量，最后两维为矩阵，batch 维必须一致）
// -----------------------------------------------------------------------------

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);
    const int64_t n = other_dims[other_dims.size() - 1];
    const int64_t bs = NumElementsOf(std::vector<int64_t>(input_dims.begin(), input_dims.end() - 2));

    auto output_dims = input_dims;
    output_dims.back() = n;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    auto a_host = ToHostFloat(input, input->GetDevice().type());
    auto b_host = ToHostFloat(other, other->GetDevice().type());
    std::vector<float> out_host(static_cast<size_t>(bs * m * n));

    for (int64_t b = 0; b < bs; ++b) {
        HostGemm(false, false, m, n, k, a_host.data() + b * m * k, k, b_host.data() + b * k * n, n, 0.0f,
                 out_host.data() + b * m * n, n);
    }

    FromHostFloat(out_host, output, output->GetDevice().type());
    return output;
}

std::shared_ptr<Tensor> MatmulBackwardInput(const std::shared_ptr<Tensor> &other,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims) {
    const auto &other_dims = other->Dims();
    const auto &grad_dims = grad_output->Dims();
    const int64_t m = grad_dims[grad_dims.size() - 2];
    const int64_t n = grad_dims[grad_dims.size() - 1];
    const int64_t k = other_dims[other_dims.size() - 2];
    const int64_t bs = NumElementsOf(std::vector<int64_t>(grad_dims.begin(), grad_dims.end() - 2));

    auto grad_input = std::make_shared<Tensor>(input_dims, grad_output->Dtype(), grad_output->GetDevice());
    auto g_host = ToHostFloat(grad_output, grad_output->GetDevice().type());
    auto b_host = ToHostFloat(other, other->GetDevice().type());
    std::vector<float> gi_host(static_cast<size_t>(bs * m * k));

    // grad_input(bs,m,k) = grad(bs,m,n) * other(bs,k,n)^T
    for (int64_t b = 0; b < bs; ++b) {
        HostGemm(false, true, m, k, n, g_host.data() + b * m * n, n, b_host.data() + b * k * n, n, 0.0f,
                 gi_host.data() + b * m * k, k);
    }

    FromHostFloat(gi_host, grad_input, grad_input->GetDevice().type());
    return grad_input;
}

std::shared_ptr<Tensor> MatmulBackwardOther(const std::shared_ptr<Tensor> &input1,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &other_dims) {
    const auto &input_dims = input1->Dims();
    const auto &grad_dims = grad_output->Dims();
    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = grad_dims[grad_dims.size() - 1];
    const int64_t bs = NumElementsOf(std::vector<int64_t>(input_dims.begin(), input_dims.end() - 2));

    auto grad_other = std::make_shared<Tensor>(other_dims, grad_output->Dtype(), grad_output->GetDevice());
    auto a_host = ToHostFloat(input1, input1->GetDevice().type());
    auto g_host = ToHostFloat(grad_output, grad_output->GetDevice().type());
    std::vector<float> go_host(static_cast<size_t>(bs * k * n));

    // grad_other(bs,k,n) = input(bs,m,k)^T * grad(bs,m,n)
    for (int64_t b = 0; b < bs; ++b) {
        HostGemm(true, false, k, n, m, a_host.data() + b * m * k, k, g_host.data() + b * m * n, n, 0.0f,
                 go_host.data() + b * k * n, n);
    }

    FromHostFloat(go_host, grad_other, grad_other->GetDevice().type());
    return grad_other;
}

} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_LINEAR_KERNEL(LinearForward)
REGISTER_MLU_LINEAR_KERNEL(LinearBackwardInput)
REGISTER_MLU_LINEAR_KERNEL(LinearBackwardWeight)
REGISTER_MLU_LINEAR_KERNEL(LinearBackwardBias)
REGISTER_MLU_LINEAR_KERNEL(MatmulForward)
REGISTER_MLU_LINEAR_KERNEL(MatmulBackwardInput)
REGISTER_MLU_LINEAR_KERNEL(MatmulBackwardOther)

#undef REGISTER_MLU_LINEAR_KERNEL
