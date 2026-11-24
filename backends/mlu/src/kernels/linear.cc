#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#ifdef USE_CNNL
#include <cnnl.h>
#include <cnrt.h>

#include "common/common_mlu.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "runtime/mlu_runtime_common.h"
#endif

namespace infini_train::kernels::mlu {
namespace {

#ifdef USE_CNNL
constexpr int kProjectionRows = 256;
constexpr int kEmbeddingWidth = 768;
constexpr int kVocabularySize = 50257;

cnnlTensorDescriptor_t MakeBf16Matrix(int rows, int columns) {
    cnnlTensorDescriptor_t descriptor;
    int dims[] = {rows, columns};
    MLU_CNNL_CHECK(cnnlCreateTensorDescriptor(&descriptor));
    MLU_CNNL_CHECK(cnnlSetTensorDescriptor(descriptor, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BFLOAT16, 2, dims));
    return descriptor;
}

// Use the validated vocabulary projection only when CNNL offers an algorithm without extra workspace.
bool TryVocabularyProjection(const Tensor &input, const Tensor &weight, Tensor &output) {
    const auto device = input.GetDevice();
    auto *stream = dynamic_cast<core::mlu::MluStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device));
    CHECK(stream);
    const auto queue = stream->mlu_queue();
    cnnlHandle_t handle;
    MLU_CNNL_CHECK(cnnlCreate(&handle));
    MLU_CNNL_CHECK(cnnlSetQueue(handle, queue));
    auto a = MakeBf16Matrix(kProjectionRows, kEmbeddingWidth);
    auto b = MakeBf16Matrix(kVocabularySize, kEmbeddingWidth);
    auto d = MakeBf16Matrix(kProjectionRows, kVocabularySize);
    cnnlMatMulDescriptor_t operation;
    cnnlMatMulAlgo_t algorithm;
    cnnlMatMulHeuristicResult_t result;
    MLU_CNNL_CHECK(cnnlCreateMatMulDescriptor(&operation));
    MLU_CNNL_CHECK(cnnlCreateMatMulAlgo(&algorithm));
    MLU_CNNL_CHECK(cnnlCreateMatMulHeuristicResult(&result));
    int32_t yes = 1, no = 0;
    auto compute = CNNL_DTYPE_FLOAT;
    MLU_CNNL_CHECK(cnnlSetMatMulDescAttr(operation, CNNL_MATMUL_DESC_COMPUTE_TYPE, &compute, sizeof(compute)));
    MLU_CNNL_CHECK(cnnlSetMatMulDescAttr(operation, CNNL_MATMUL_DESC_TRANSB, &yes, sizeof(yes)));
    MLU_CNNL_CHECK(cnnlSetMatMulDescAttr(operation, CNNL_MATMUL_USE_BETA, &no, sizeof(no)));
    MLU_CNNL_CHECK(cnnlSetMatMulDescAttr(operation, CNNL_MATMUL_ALLOW_TF32, &no, sizeof(no)));
    int count = 0;
    size_t workspace = 0;
    auto query = cnnlGetMatMulAlgoHeuristic(handle, operation, a, b, d, d, nullptr, 1, &result, &count);
    bool supported = query != CNNL_STATUS_NOT_SUPPORTED && query != CNNL_STATUS_ARCH_MISMATCH;
    if (supported) {
        MLU_CNNL_CHECK(query);
        if (count == 1) {
            MLU_CNNL_CHECK(cnnlGetMatMulHeuristicResult(result, algorithm, &workspace));
        }
        supported = count == 1 && workspace == 0;
    }
    if (supported) {
        // CNNL requires a valid C operand even when beta is zero. Keep it alive until the queue completes.
        Tensor scratch(output.Dims(), output.Dtype(), output.GetDevice());
        MLU_CNRT_CHECK(cnrtMemset(scratch.DataPtr(), 0, output.NumElements() * sizeof(uint16_t)));
        float alpha = 1, beta = 0;
        MLU_CNNL_CHECK(cnnlMatMul_v2(handle, operation, algorithm, &alpha, a, input.DataPtr(), b, weight.DataPtr(),
                                     &beta, d, scratch.DataPtr(), nullptr, workspace, d, output.DataPtr()));
        MLU_CNRT_CHECK(cnrtQueueSync(queue));
    }
    MLU_CNNL_CHECK(cnnlDestroyMatMulHeuristicResult(result));
    MLU_CNNL_CHECK(cnnlDestroyMatMulAlgo(algorithm));
    MLU_CNNL_CHECK(cnnlDestroyMatMulDescriptor(operation));
    MLU_CNNL_CHECK(cnnlDestroyTensorDescriptor(a));
    MLU_CNNL_CHECK(cnnlDestroyTensorDescriptor(b));
    MLU_CNNL_CHECK(cnnlDestroyTensorDescriptor(d));
    MLU_CNNL_CHECK(cnnlDestroy(handle));
    return supported;
}
#endif

// Row-major storage; lda/ldb/ldc are row strides and alpha is one.
void HostGemm(bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k, const float *a, int64_t lda, const float *b,
              int64_t ldb, float beta, float *c, int64_t ldc) {
    // Traverse i-k-j for contiguous inner-loop access and deterministic accumulation order.
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

// Convert device tensors to host float buffers, including 16-bit floating-point storage.
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
#ifdef USE_CNNL
    if (transpose && !bias && rows == kProjectionRows && in_features == kEmbeddingWidth
        && out_features == kVocabularySize && input->Dtype() == DataType::kBFLOAT16
        && weight->Dtype() == DataType::kBFLOAT16 && input->IsContiguous() && weight->IsContiguous()
        && input->GetDevice() == weight->GetDevice() && TryVocabularyProjection(*input, *weight, *output)) {
        return output;
    }
#endif
    auto in_host = ToHostFloat(input, input->GetDevice().type());
    auto w_host = ToHostFloat(weight, weight->GetDevice().type());
    std::vector<float> out_host(static_cast<size_t>(rows * out_features));

    // output = input * weight^T when transpose is true and weight is [out, in].
    HostGemm(false, transpose, rows, out_features, in_features, in_host.data(), in_features, w_host.data(),
             transpose ? in_features : out_features, 0.0f, out_host.data(), out_features);

    if (bias) {
        auto b_host = ToHostFloat(bias, bias->GetDevice().type());

        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t j = 0; j < out_features; ++j) { out_host[r * out_features + j] += b_host[j]; }
        }
    }

    FromHostFloat(out_host, output, output->GetDevice().type());

    return output;
}

std::shared_ptr<Tensor> LinearBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output, bool transpose,
                                            int64_t in_features, int64_t out_features,
                                            const std::vector<int64_t> &input_dims) {
    const auto compute_dtype = weight->Dtype();
    auto grad_compute = grad_output->Dtype() == compute_dtype
                          ? grad_output
                          : std::make_shared<Tensor>(grad_output->To(compute_dtype));
    // Match MACA's temporary backward output policy until autograd/autocast owns this conversion.
    const auto output_dtype = compute_dtype == DataType::kBFLOAT16 ? DataType::kFLOAT32 : compute_dtype;
    auto grad_input = std::make_shared<Tensor>(input_dims, output_dtype, grad_output->GetDevice());
    const int64_t rows = NumElementsOf(input_dims) / in_features;

    auto g_host = ToHostFloat(grad_compute, grad_compute->GetDevice().type());
    auto w_host = ToHostFloat(weight, weight->GetDevice().type());
    std::vector<float> gi_host(static_cast<size_t>(rows * in_features));

    if (transpose) {
        // grad_input = grad_output * weight for weight shaped [out, in].
        HostGemm(false, false, rows, in_features, out_features, g_host.data(), out_features, w_host.data(), in_features,
                 0.0f, gi_host.data(), in_features);
    } else {
        // grad_input = grad_output * weight^T for weight shaped [in, out].
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
    const auto compute_dtype = input->Dtype();
    auto grad_compute = grad_output->Dtype() == compute_dtype
                          ? grad_output
                          : std::make_shared<Tensor>(grad_output->To(compute_dtype));
    const auto output_dtype = compute_dtype == DataType::kBFLOAT16 ? DataType::kFLOAT32 : compute_dtype;
    auto grad_weight = std::make_shared<Tensor>(weight_dims, output_dtype, grad_output->GetDevice());
    const int64_t rows = input->NumElements() / in_features;

    auto in_host = ToHostFloat(input, input->GetDevice().type());
    auto g_host = ToHostFloat(grad_compute, grad_compute->GetDevice().type());
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
    const auto compute_dtype = grad_output->Dtype();
    const auto output_dtype = compute_dtype == DataType::kBFLOAT16 ? DataType::kFLOAT32 : compute_dtype;
    auto grad_bias
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, output_dtype, grad_output->GetDevice());
    const int64_t rows = grad_output->NumElements() / out_features;

    auto g_host = ToHostFloat(grad_output, grad_output->GetDevice().type());
    std::vector<float> gb_host(static_cast<size_t>(out_features), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < out_features; ++j) { gb_host[j] += g_host[r * out_features + j]; }
    }

    FromHostFloat(gb_host, grad_bias, grad_bias->GetDevice().type());
    return grad_bias;
}

// Batched Matmul: the last two dimensions are matrices and batch dimensions must match.

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

    const auto compute_dtype = other->Dtype();
    auto grad_compute = grad_output->Dtype() == compute_dtype
                          ? grad_output
                          : std::make_shared<Tensor>(grad_output->To(compute_dtype));
    const auto output_dtype = compute_dtype == DataType::kBFLOAT16 ? DataType::kFLOAT32 : compute_dtype;
    auto grad_input = std::make_shared<Tensor>(input_dims, output_dtype, grad_output->GetDevice());
    auto g_host = ToHostFloat(grad_compute, grad_compute->GetDevice().type());
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

    const auto compute_dtype = input1->Dtype();
    auto grad_compute = grad_output->Dtype() == compute_dtype
                          ? grad_output
                          : std::make_shared<Tensor>(grad_output->To(compute_dtype));
    const auto output_dtype = compute_dtype == DataType::kBFLOAT16 ? DataType::kFLOAT32 : compute_dtype;
    auto grad_other = std::make_shared<Tensor>(other_dims, output_dtype, grad_output->GetDevice());
    auto a_host = ToHostFloat(input1, input1->GetDevice().type());
    auto g_host = ToHostFloat(grad_compute, grad_compute->GetDevice().type());
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
