#pragma once

// Launcher ABI shared by host C++20 and device C++17 translation units.
// Keep framework headers out of this interface: cncc cannot compile them.

#include <stdint.h>

#include <cnrt.h>

namespace infini_train::kernels::mlu {

// Map host scalar types to launcher dtype tags.
enum class MluScalarType : int {
    kBool = 0,
    kU8,
    kI8,
    kU16,
    kI16,
    kU32,
    kI32,
    kU64,
    kI64,
    kF16,
    kBF16,
    kF32,
    kF64,
};

// Interpret the low bytes of value_bits according to the scalar type.
void FillLaunch(void *data, uint64_t value_bits, MluScalarType type, int64_t num_elements, cnrtQueue_t queue);

void CastLaunch(void *dst, const void *src, MluScalarType dst_type, MluScalarType src_type, int64_t num_elements,
                cnrtQueue_t queue);

// Accumulate tensor += grad * rate.
void AccumulateGradLaunch(const void *grad, float rate, void *tensor, MluScalarType type, int64_t num_elements,
                          cnrtQueue_t queue);

// Adam update with bias-correction factors computed by the host.
void AdamAccumulateGradLaunch(const void *grad, void *param, void *m, void *v, float learning_rate, float beta1,
                              float beta2, float eps, float bias_correction_m, float bias_correction_v,
                              MluScalarType type, int64_t num_elements, cnrtQueue_t queue);

// -----------------------------------------------------------------------------
// elementwise
// -----------------------------------------------------------------------------

// POD broadcast metadata passed by value. Shapes are right-aligned; strides assume contiguous storage.
#define MLU_MAX_BROADCAST_DIMS 8
struct MluBroadcastMeta {
    int32_t ndim;
    int64_t a_strides[MLU_MAX_BROADCAST_DIMS];
    int64_t b_strides[MLU_MAX_BROADCAST_DIMS];
    int64_t out_strides[MLU_MAX_BROADCAST_DIMS];
    int64_t a_shape[MLU_MAX_BROADCAST_DIMS];
    int64_t b_shape[MLU_MAX_BROADCAST_DIMS];
};

enum class MluUnaryOp : int {
    kNeg = 0,
    kReciprocal,
    kTanh,
    kRsqrt,
    kExp,
    kLog,
    kSin,
    kCos,
    kSigmoid,
};

enum class MluBinaryOp : int {
    kAdd = 0,
    kSub,
    kMul,
    kDiv,
};

// Apply out[i] = op(in[i]); low-precision arithmetic uses float intermediates.
void UnaryForwardLaunch(void *out, const void *in, MluUnaryOp op, MluScalarType type, int64_t num_elements,
                        cnrtQueue_t queue);

// aux is the forward output for tanh/exp/sigmoid and the forward input for other unary operations.
void UnaryBackwardLaunch(void *grad_in, const void *grad_out, const void *aux, MluUnaryOp op, MluScalarType type,
                         int64_t num_elements, cnrtQueue_t queue);

// scalar_is_base selects exponent^in instead of in^exponent.
void PowForwardLaunch(void *out, const void *in, float exponent, bool scalar_is_base, MluScalarType type,
                      int64_t num_elements, cnrtQueue_t queue);
void PowBackwardLaunch(void *grad_in, const void *grad_out, const void *in, float exponent, bool scalar_is_base,
                       MluScalarType type, int64_t num_elements, cnrtQueue_t queue);

// Apply the binary operation after mapping output indices to broadcast input offsets.
void BinaryForwardLaunch(void *out, const void *a, const void *b, MluBinaryOp op, MluScalarType type,
                         MluBroadcastMeta meta, int64_t num_elements, cnrtQueue_t queue);

// Use direct writes for equal shapes; broadcast gradients use deterministic float gather reductions.
void BinaryBackwardLaunch(void *grad_a, void *grad_b, const void *grad_out, const void *a, const void *b,
                          MluBinaryOp op, MluScalarType type, MluBroadcastMeta meta, int64_t numel_a, int64_t numel_b,
                          int64_t numel_out, cnrtQueue_t queue);

// Scalar Add/Mul; the framework reduces scalar Sub/Div to these operations.
void ScalarForwardLaunch(void *out, const void *in, float scalar, MluBinaryOp op, MluScalarType type,
                         int64_t num_elements, cnrtQueue_t queue);
void ScalarBackwardLaunch(void *grad_in, const void *grad_out, float scalar, MluBinaryOp op, MluScalarType type,
                          int64_t num_elements, cnrtQueue_t queue);

// Equality produces bool output and has no backward kernel.
void EqualsForwardLaunch(void *out, const void *a, const void *b, MluScalarType type, MluBroadcastMeta meta,
                         int64_t num_elements, cnrtQueue_t queue);
void EqualsScalarForwardLaunch(void *out, const void *in, double scalar, MluScalarType type, int64_t num_elements,
                               cnrtQueue_t queue);

// -----------------------------------------------------------------------------
// embedding
// -----------------------------------------------------------------------------

// Token IDs are int64; out-of-range IDs are skipped, matching CUDA.
void EmbeddingForwardLaunch(const void *input, void *output, const void *weight, int64_t num_tokens, int64_t embed_dim,
                            int64_t vocab_size, MluScalarType type, cnrtQueue_t queue);
// Zero grad_weight before launch; each vocabulary row has a single writer.
void EmbeddingBackwardLaunch(const void *input, const void *grad_output, void *grad_weight, int64_t num_tokens,
                             int64_t embed_dim, int64_t vocab_size, MluScalarType type, cnrtQueue_t queue);

// Row-wise normalization and loss reductions use float intermediates.

// input is [num_tokens, embed_dim]; mean_out and rstd_out are non-null FP32 buffers.
void LayerNormForwardLaunch(const void *input, const void *weight, const void *bias, float *mean_out, float *rstd_out,
                            void *output, float eps, int64_t num_tokens, int64_t embed_dim, MluScalarType type,
                            cnrtQueue_t queue);
void LayerNormBackwardLaunch(const void *input, const void *grad_output, const float *mean, const float *rstd,
                             const void *weight, void *grad_input, void *grad_weight, void *grad_bias,
                             int64_t num_tokens, int64_t embed_dim, MluScalarType type, cnrtQueue_t queue);

// The host decomposes dimensions into outer, axis and inner extents.
void SoftmaxForwardLaunch(void *output, const void *input, int64_t outer_size, int64_t axis_size, int64_t inner_size,
                          MluScalarType type, cnrtQueue_t queue);
void SoftmaxBackwardLaunch(void *grad_input, const void *grad_output, const void *output, int64_t outer_size,
                           int64_t axis_size, int64_t inner_size, MluScalarType type, cnrtQueue_t queue);

// Targets are uint8 or int64. Return one loss per sample; the host averages in FP64, matching CUDA.
void CrossEntropyForwardLaunch(const void *input, const void *target, void *loss, int64_t bs, int64_t num_classes,
                               MluScalarType target_type, MluScalarType input_type, cnrtQueue_t queue);
void CrossEntropyBackwardLaunch(const void *input, void *grad_input, const void *target, const void *grad_output,
                                int64_t bs, int64_t num_classes, MluScalarType target_type, MluScalarType input_type,
                                cnrtQueue_t queue);

// -----------------------------------------------------------------------------
// transform (transpose / slice / tril / triu / mask)
// -----------------------------------------------------------------------------

// POD indexing metadata passed by value to device kernels.
#define MLU_MAX_META_DIMS 8
struct MluIndexMeta {
    int32_t ndim;
    int64_t dims[MLU_MAX_META_DIMS]; // Slice output dimensions.
    int64_t in_strides[MLU_MAX_META_DIMS];
    int64_t out_strides[MLU_MAX_META_DIMS];
    int64_t starts[MLU_MAX_META_DIMS]; // Slice only.
    int64_t steps[MLU_MAX_META_DIMS];  // Slice only.
};

// Forward and backward both retain selected elements and zero the rest; upper selects Triu.
void TrilTriuLaunch(const void *input, void *output, int64_t rows, int64_t cols, int64_t diagonal, bool upper,
                    MluScalarType type, cnrtQueue_t queue);

// Use input/output strides; backward transposes the same dimension pair.
void TransposeLaunch(const void *input, void *output, MluIndexMeta meta, int64_t dim0, int64_t dim1,
                     int64_t num_elements, MluScalarType type, cnrtQueue_t queue);

// Populate output dims, both strides, starts and steps; zero grad_input before backward launch.
void SliceForwardLaunch(const void *input, void *output, MluIndexMeta meta, int64_t num_elements, MluScalarType type,
                        cnrtQueue_t queue);
void SliceBackwardLaunch(const void *grad_output, void *grad_input, MluIndexMeta meta, int64_t num_elements,
                         MluScalarType type, cnrtQueue_t queue);

// lead selects leading-dimension masks; otherwise use trailing dimensions.
// The host casts the mask to the input dtype; entries equal to one select replacement.
void MaskForwardLaunch(const void *input, const void *mask, void *output, float value, bool lead, int64_t mask_extent,
                       int64_t inner, int64_t num_elements, MluScalarType type, cnrtQueue_t queue);
void MaskBackwardLaunch(const void *grad_output, const void *mask, void *grad_input, bool lead, int64_t mask_extent,
                        int64_t inner, int64_t num_elements, MluScalarType type, cnrtQueue_t queue);

} // namespace infini_train::kernels::mlu
