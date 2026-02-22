// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kernels/kernels.h"
#include "binding_utils.h"

#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

#include <nanobind/stl/optional.h>

namespace nb = nanobind;

using CudaArray = nb::ndarray<nb::c_contig, nb::device::cuda>;
using CPUArray = nb::ndarray<nb::c_contig, nb::device::cpu>;


Tensor to_tensor(const CudaArray& array) {
    const ETensorDType dtype = from_dlpack_dtype(array.dtype());
    const std::vector<long> shape(array.shape_ptr(), array.shape_ptr() + array.ndim());
    return Tensor::from_pointer(static_cast<std::byte*>(array.data()), array.device_id(), dtype, shape);
}

Tensor to_tensor(const CPUArray& array) {
    const ETensorDType dtype = from_dlpack_dtype(array.dtype());
    const std::vector<long> shape(array.shape_ptr(), array.shape_ptr() + array.ndim());
    return Tensor::from_pointer(static_cast<std::byte*>(array.data()), -1, dtype, shape);
}

Tensor to_tensor(const std::optional<CudaArray>& array) {
    if (array.has_value()) {
        return to_tensor(array.value());
    } else {
        return Tensor{};
    }
}

/// This function is to be called with the list of dimensions of different tensors that should be
/// equal (e.g., the first dimension of input and output should be the same batch size B)
static long get_dimension_checked(const std::initializer_list<std::size_t> values, const char* dim_name) {
    const long value = *values.begin();
    for (auto it = values.begin(); it != values.end(); ++it) {
        if (*it != value)
            throw std::invalid_argument("All dimensions must be equal for dimension " + std::string(dim_name));
    }
    return value;
}

static cudaStream_t as_stream(std::uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

static float* get_abs_max_ptr(const std::optional<CudaArray>& abs_max_tensor) {
    if (!abs_max_tensor.has_value()) return nullptr;
    if (from_dlpack_dtype(abs_max_tensor->dtype()) != ETensorDType::FP32) {
        throw std::invalid_argument("abs_max_tensor must be a FP32 tensor");
    }
    return static_cast<float*>(abs_max_tensor->data());
}

static float* device_scalar_float(CudaArray& array) {
    if (from_dlpack_dtype(array.dtype()) != ETensorDType::FP32) {
        throw std::invalid_argument("array must be a FP32 tensor");
    }
    return static_cast<float*>(array.data());
}

static const float* device_scalar_float(const CudaArray& array) {
    if (from_dlpack_dtype(array.dtype()) != ETensorDType::FP32) {
        throw std::invalid_argument("array must be a FP32 tensor");
    }
    return static_cast<const float*>(array.data());
}

static cudaDeviceProp get_device_prop(int device) {
    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, device));
    return dp;
}

// ---------------------------------------------------------------------------------------------------------------------
// below are the wrapper functions. each wrapper should extract the input shapes, using `NB_CHECK_NDIMS` and
// `get_dimension_checked` to validate that dimensions that should be equal have the same value.
// Each _output_ tensor should be created as a new Tensor object (these are passed as non-const reference),
// whereas input tensors (const ref) can be converted on the fly using to_tensor.

void bind_encoder_forward(const CudaArray& out, const CudaArray& inp, const CudaArray& wte, const std::optional<CudaArray>& wpe, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(out, 3);
    NB_CHECK_NDIMS(inp, 3);
    NB_CHECK_NDIMS(wte, 2);

    // TODO wpe dimension check
    const long B = get_dimension_checked({out.shape(0), inp.shape(0)}, "B");
    const long T = get_dimension_checked({out.shape(1), inp.shape(1)}, "T");
    const long V = wte.shape(0);
    const long C = get_dimension_checked({out.shape(2), wte.shape(1)}, "C");
    Tensor out_t = to_tensor(out);
    encoder_forward(out_t, to_tensor(inp), to_tensor(wte), to_tensor(wpe), B, T, C, V, as_stream(stream));
}

// ---- RMSNorm ----
void bind_rmsnorm_forward(const CudaArray& out, const CudaArray& rms, const CudaArray& inp, const CudaArray& weight,
                          const std::optional<CudaArray>& abs_max, float epsilon, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(out, 3);
    NB_CHECK_NDIMS(inp, 3);
    NB_CHECK_NDIMS(rms, 2);
    NB_CHECK_NDIMS(weight, 1);
    const long B = get_dimension_checked({inp.shape(0), out.shape(0), rms.shape(0)}, "B");
    const long T = get_dimension_checked({inp.shape(1), out.shape(1), rms.shape(1)}, "T");
    const long C = get_dimension_checked({inp.shape(2), out.shape(2), weight.shape(0)}, "C");
    Tensor out_t = to_tensor(out);
    Tensor rms_t = to_tensor(rms);
    rmsnorm_forward(out_t, rms_t, to_tensor(inp), to_tensor(weight), get_abs_max_ptr(abs_max),
        epsilon, B, T, C, as_stream(stream));
}

void bind_rmsnorm_backward(const CudaArray& dinp, const CudaArray& dweight, const CudaArray& scratch,
                           const CudaArray& dresidual, const CudaArray& dout, const CudaArray& inp, const CudaArray& weight,
                           const CudaArray& rstd, const std::optional<CudaArray>& abs_max,
                           const std::uintptr_t stream) {
    NB_CHECK_NDIMS(dinp, 3);
    NB_CHECK_NDIMS(dweight, 1);
    NB_CHECK_NDIMS(scratch, 1);
    NB_CHECK_NDIMS(dresidual, 3);
    NB_CHECK_NDIMS(dout, 3);
    NB_CHECK_NDIMS(inp, 3);
    NB_CHECK_NDIMS(weight, 1);
    NB_CHECK_NDIMS(rstd, 2);

    auto dp = get_device_prop(inp.device_id());
    const long B = get_dimension_checked({inp.shape(0), dinp.shape(0), dout.shape(0), rstd.shape(0), dresidual.shape(0)}, "B");
    const long T = get_dimension_checked({inp.shape(1), dinp.shape(1), dout.shape(1), rstd.shape(1), dresidual.shape(0)}, "T");
    const long C = get_dimension_checked({inp.shape(2), dinp.shape(2), dout.shape(2), dresidual.shape(2), weight.shape(0)}, "C");

    Tensor dinp_t = to_tensor(dinp);
    Tensor dweight_t = to_tensor(dweight);
    Tensor scratch_t = to_tensor(scratch);
    rmsnorm_backward(dinp_t, dweight_t, scratch_t,
        to_tensor(dresidual), to_tensor(dout), to_tensor(inp), to_tensor(weight), to_tensor(rstd),
        get_abs_max_ptr(abs_max), B, T, C, dp, as_stream(stream));
}

// ---- Fused residual + rmsnorm ----
void bind_fused_residual_rmsnorm_forward(const CudaArray& residual, const CudaArray& normed, const CudaArray& rrms,
                                         const CudaArray& inp1, const CudaArray& inp2, const CudaArray& weight,
                                         const std::optional<CudaArray>& absmax, float epsilon, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(residual, 3);
    NB_CHECK_NDIMS(normed, 3);
    NB_CHECK_NDIMS(rrms, 2);
    NB_CHECK_NDIMS(inp1, 3);
    NB_CHECK_NDIMS(inp2, 3);
    NB_CHECK_NDIMS(weight, 1);
    const long B = get_dimension_checked({inp1.shape(0), inp2.shape(0), residual.shape(0), normed.shape(0), rrms.shape(0)}, "B");
    const long T = get_dimension_checked({inp1.shape(1), inp2.shape(1), residual.shape(1), normed.shape(1), rrms.shape(1)}, "T");
    const long C = get_dimension_checked({inp1.shape(2), inp2.shape(2), residual.shape(2), normed.shape(2), weight.shape(0)}, "C");
    const long N = B * T;
    Tensor residual_t = to_tensor(residual);
    Tensor normed_t = to_tensor(normed);
    Tensor rrms_t = to_tensor(rrms);
    fused_residual_rmsnorm_forward(residual_t, normed_t, rrms_t, to_tensor(inp1), to_tensor(inp2), to_tensor(weight),
        get_abs_max_ptr(absmax), epsilon, N, C, as_stream(stream));
}

// ---- Rope ----
void bind_rope_forward(const CudaArray& out, const CudaArray& in, const CudaArray& freqs_cis,
                       const std::optional<CudaArray>& abs_max, int Nq, int Nkv, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(in, 3);
    NB_CHECK_NDIMS(out, 3);
    NB_CHECK_NDIMS(freqs_cis, 2);
    const long B = get_dimension_checked({in.shape(0), out.shape(0)}, "B");
    const long T = get_dimension_checked({in.shape(1), out.shape(1)}, "T");
    const long C = get_dimension_checked({in.shape(2), out.shape(2), freqs_cis.shape(1)}, "C");

    const long head_dim = div_exact(C, (long)Nq + 2*Nkv);

    Tensor out_t = to_tensor(out);
    rope_forward(out_t, to_tensor(in), to_tensor(freqs_cis),
        get_abs_max_ptr(abs_max), B, T, Nq, Nkv, head_dim, as_stream(stream));
}

void bind_rope_backward(const CudaArray& dinp, const CudaArray& dout, const CudaArray& freqs_cis,
                        const std::optional<CudaArray>& abs_max, int Nq, int Nkv, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(dinp, 3);
    NB_CHECK_NDIMS(dout, 3);
    NB_CHECK_NDIMS(freqs_cis, 2);
    const long B = get_dimension_checked({dinp.shape(0), dout.shape(0)}, "B");
    const long T = get_dimension_checked({dinp.shape(1), dout.shape(1)}, "T");
    const long C = get_dimension_checked({dinp.shape(2), dout.shape(2), freqs_cis.shape(1)}, "C");

    const long head_dim = div_exact(C, (long)Nq + 2*Nkv);

    Tensor dinp_t = to_tensor(dinp);
    rope_backward(dinp_t, to_tensor(dout), to_tensor(freqs_cis),
        get_abs_max_ptr(abs_max), B, T, Nq, Nkv, head_dim, as_stream(stream));
}

// ---- SwiGLU ----
void bind_swiglu_forward(const CudaArray& out, const CudaArray& inp, const std::optional<CudaArray>& abs_max,
                         const std::uintptr_t stream) {
    NB_CHECK_NDIMS(inp, 3);
    NB_CHECK_NDIMS(out, 3);
    const long B = get_dimension_checked({inp.shape(0), out.shape(0)}, "B");
    const long T = get_dimension_checked({inp.shape(1), out.shape(1)}, "T");
    const long C = get_dimension_checked({div_exact(inp.shape(2),  2ul), out.shape(2)}, "C");
    Tensor out_t = to_tensor(out);
    swiglu_forward(out_t, to_tensor(inp), get_abs_max_ptr(abs_max), B, T, C, as_stream(stream));
}

void bind_swiglu_forward_quant(const CudaArray& out, CudaArray scale, const CudaArray& inp,
                               const std::optional<CudaArray>& abs_max, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(inp, 3);
    NB_CHECK_NDIMS(out, 3);

    const long B = get_dimension_checked({inp.shape(0), out.shape(0)}, "B");
    const long T = get_dimension_checked({inp.shape(1), out.shape(1)}, "T");
    const long C = get_dimension_checked({div_exact(inp.shape(2), 2ul), out.shape(2)}, "C");
    Tensor out_t = to_tensor(out);
    swiglu_forward_quant(out_t, device_scalar_float(scale), to_tensor(inp), get_abs_max_ptr(abs_max), B, T, C, as_stream(stream));
}

void bind_swiglu_backward(const CudaArray& dinp, const CudaArray& dout, const CudaArray& inp,
                          const std::optional<CudaArray>& absmax, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(inp, 3);
    NB_CHECK_NDIMS(dout, 3);
    NB_CHECK_NDIMS(dinp, 3);
    const long B = get_dimension_checked({inp.shape(0), dout.shape(0), dinp.shape(0)}, "B");
    const long T = get_dimension_checked({inp.shape(1), dout.shape(1), dinp.shape(1)},  "T");
    const long C = get_dimension_checked({inp.shape(2), dinp.shape(2), 2*dout.shape(2)}, "C");
    Tensor dinp_t = to_tensor(dinp);
    swiglu_backward(dinp_t, to_tensor(dout), to_tensor(inp), get_abs_max_ptr(absmax), B, T, C, as_stream(stream));
}

// ---- Attention (cuDNN) ----
void bind_attention_forward(const CudaArray& out, const CudaArray& stats, const CudaArray& inp,
                            const CudaArray& workspace, std::uintptr_t cudnn_handle,
                            int Hq, int Hkv, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(inp, 3);
    NB_CHECK_NDIMS(out, 3);
    NB_CHECK_NDIMS(stats, 2);

    const long B = get_dimension_checked({inp.shape(0), out.shape(0), stats.shape(0)}, "B");
    const long T = get_dimension_checked({inp.shape(1), out.shape(1), stats.shape(1)}, "T");
    const long C = get_dimension_checked({inp.shape(2)}, "C_in");
    const long head_dim = div_exact(C, (long)Hq + 2*Hkv);
    (void)get_dimension_checked({out.shape(2), (unsigned long)head_dim * Hq}, "C_out");

    Tensor out_t = to_tensor(out);
    Tensor stats_t = to_tensor(stats);
    Tensor ws_t = to_tensor(workspace);
    attention_forward_cudnn(out_t, stats_t,  to_tensor(inp), ws_t,
                            reinterpret_cast<cudnnHandle_t>(cudnn_handle), B, T, Hq, Hkv, head_dim, as_stream(stream));
}

void bind_attention_backward(const CudaArray& dqkv, const CudaArray& stats, const CudaArray& out, const CudaArray& dout,
                             const CudaArray& qkv, const CudaArray& workspace, std::uintptr_t cudnn_handle,
                             int Hq, int Hkv, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(dout, 3);
    NB_CHECK_NDIMS(out, 3);
    NB_CHECK_NDIMS(dqkv, 3);
    NB_CHECK_NDIMS(qkv, 3);
    NB_CHECK_NDIMS(stats, 2);
    const long B = get_dimension_checked({qkv.shape(0), dqkv.shape(0), out.shape(0), dout.shape(0), stats.shape(0)}, "B");
    const long T = get_dimension_checked({qkv.shape(1), dqkv.shape(1), out.shape(1), dout.shape(1), stats.shape(1)}, "T");
    const long C = get_dimension_checked({qkv.shape(2), dqkv.shape(2)}, "C_in");
    const long head_dim = div_exact(C, (long)Hq + 2*Hkv);
    (void)get_dimension_checked({dout.shape(2), (unsigned long)head_dim * Hq}, "C_out");

    Tensor dqkv_t = to_tensor(dqkv);
    Tensor ws_t = to_tensor(workspace);
    attention_backward_cudnn(dqkv_t, to_tensor(stats), to_tensor(out), to_tensor(dout), to_tensor(qkv), ws_t,
                             reinterpret_cast<cudnnHandle_t>(cudnn_handle), B, T, Hq, Hkv, head_dim, as_stream(stream));
}

// ---- Classifier / losses ----
void bind_fused_classifier(const CudaArray& logits, const CudaArray& losses, const CudaArray& lse,
                           float dloss, const CudaArray& targets, float z_reg, bool write_dlogits, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(logits, 3);
    NB_CHECK_NDIMS(losses, 2);
    NB_CHECK_NDIMS(lse, 2);
    NB_CHECK_NDIMS(targets, 2);

    const long B = get_dimension_checked({logits.shape(0), losses.shape(0), lse.shape(0), targets.shape(0)}, "B");
    const long T = get_dimension_checked({logits.shape(1), losses.shape(1), lse.shape(1), targets.shape(1)}, "T");
    const long V = get_dimension_checked({logits.shape(2)}, "V");

    Tensor logits_t = to_tensor(logits);
    Tensor losses_t = to_tensor(losses);
    Tensor lse_t = to_tensor(lse);
    fused_classifier(logits_t, losses_t, lse_t, dloss, to_tensor(targets), z_reg, B*T, V, V, write_dlogits, as_stream(stream));
}

void bind_grouped_loss_sum(const CudaArray& out, const CudaArray& per_token_loss, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(out, 1);
    NB_CHECK_NDIMS(per_token_loss, 2);
    const long B = get_dimension_checked({per_token_loss.shape(0)}, "B");
    const long T = get_dimension_checked({per_token_loss.shape(1)}, "T");
    Tensor out_t = to_tensor(out);
    grouped_loss_sum(out_t, to_tensor(per_token_loss), B, T, as_stream(stream));
}

// ---- Matmul ----
void bind_matmul(const CudaArray& c, const CudaArray& a, const CudaArray& b, const std::optional<CudaArray>& bias,
                 const std::optional<CudaArray>& scale_a, const std::optional<CudaArray>& scale_b,
                 std::uintptr_t cublaslt_handle, const CudaArray& workspace,
                 int M, int N, int K, EMMTranspose mode, bool accumulate, const std::uintptr_t stream) {
    // TODO shape validation
    const float* scale_a_ptr = scale_a.has_value() ? static_cast<const float*>(scale_a->data()) : nullptr;
    const float* scale_b_ptr = scale_b.has_value() ? static_cast<const float*>(scale_b->data()) : nullptr;
    Tensor c_t = to_tensor(c);
    Tensor ws_t = to_tensor(workspace);
    matmul(c_t, to_tensor(a), to_tensor(b), to_tensor(bias), scale_a_ptr, scale_b_ptr,
           reinterpret_cast<cublasLtHandle_t>(cublaslt_handle), ws_t, M, N, K, mode, accumulate, as_stream(stream));
}

// ---- Bias add/backward ----
void bind_backward_bias(const CudaArray& dbias, const CudaArray& dout, const std::optional<CudaArray>& scale_a, const std::optional<CudaArray>& scale_b,
                        const CudaArray& dbias_buffer, const std::uintptr_t stream) {
    NB_CHECK_NDIMS(dbias, 1);
    NB_CHECK_NDIMS(dout, 3);
    int device = dout.device_id();
    auto dp = get_device_prop(device);
    long B = get_dimension_checked({dout.shape(0)}, "B");
    long T = get_dimension_checked({dout.shape(1)}, "T");
    long OC = get_dimension_checked({dbias.shape(0), dout.shape(0)}, "OC");
    const float* scale_a_ptr = scale_a.has_value() ? static_cast<const float*>(scale_a->data()) : nullptr;
    const float* scale_b_ptr = scale_b.has_value() ? static_cast<const float*>(scale_b->data()) : nullptr;
    Tensor dbias_t = to_tensor(dbias);
    Tensor buf_t = to_tensor(dbias_buffer);
    backward_bias(dbias_t, to_tensor(dout), scale_a_ptr, scale_b_ptr, buf_t, B, T, OC, dp, as_stream(stream));
}

// ---- Quantization helpers ----
void bind_abs_max(CudaArray scale, const CudaArray& in, const std::uintptr_t stream) {
    int device = in.device_id();
    auto dp = get_device_prop(device);
    long N = in.size();
    Tensor in_t = to_tensor(in);
    abs_max(device_scalar_float(scale), in_t, N, dp, as_stream(stream));
}

void bind_quantize_with_abs_max(const CudaArray& out, CudaArray scale, const CudaArray& in, const CudaArray& abs_max,
                                const std::uintptr_t stream) {
    int device = in.device_id();
    auto dp = get_device_prop(device);
    long N =  get_dimension_checked({in.size(), out.size()}, "nelem");
    Tensor out_t = to_tensor(out);
    Tensor in_t = to_tensor(in);
    quantize_with_abs_max(out_t, device_scalar_float(scale), in_t, device_scalar_float(abs_max.data()), N, dp, as_stream(stream));
}

void bind_quantize_and_transpose_with_abs_max(const CudaArray& out, CudaArray scale, const CudaArray& in, const CudaArray& abs_max,
                                              int rows, int cols, const std::uintptr_t stream) {
    int device = in.device_id();
    auto dp = get_device_prop(device);
    Tensor out_t = to_tensor(out);
    Tensor in_t = to_tensor(in);
    quantize_and_transpose_with_abs_max(out_t, device_scalar_float(scale), in_t, device_scalar_float(abs_max), rows, cols, dp, as_stream(stream));
}

// ---- Transpose ----
void bind_transpose(const CudaArray& dst, const CudaArray& src, int rows, int cols, const std::uintptr_t stream) {
    Tensor dst_t = to_tensor(dst);
    Tensor src_t = to_tensor(src);
    transpose(dst_t, src_t, rows, cols, as_stream(stream));
}

// ---- Vector ops ----
void bind_vector_add_sr(const CudaArray& dest, const CudaArray& left, const CudaArray& right, float scale, unsigned seed, const std::uintptr_t stream) {
    const long nelem = get_dimension_checked({dest.size(), left.size(), right.size()}, "nelem");
    Tensor dest_t = to_tensor(dest);
    vector_add_sr(dest_t, to_tensor(left), to_tensor(right), scale, nelem, seed, as_stream(stream));
}

void bind_vector_reduce_sr(const CudaArray& dest, const CudaArray& src, float scale, int n_shards, int skip, bool accumulate, unsigned seed, const std::uintptr_t stream) {
    const long nelem = get_dimension_checked({dest.size() * n_shards, src.size()}, "nelem");
    Tensor dest_t = to_tensor(dest);
    vector_reduce_sr(dest_t, to_tensor(src), scale, n_shards, skip, nelem, accumulate, seed, as_stream(stream));
}

// ---- Global norm ----
void bind_global_norm_squared(const CudaArray& out, const CudaArray& values, const std::uintptr_t stream) {
    auto dp = get_device_prop(values.device_id());
    int blocks = get_max_num_block_sums(dp);
    if (out.size() < blocks) {
        throw std::runtime_error("Global norm output buffer too small");
    }
    Tensor out_t = to_tensor(out);
    global_norm_squared(out_t, to_tensor(values), values.size(), dp, as_stream(stream));
}

// ---- AdamW ----
void bind_adamw_update(const CudaArray& params, const CudaArray& grads, const CudaArray& m, const CudaArray& v,
                       size_t num_parameters, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                       const std::optional<CudaArray>& grad_scale, const std::optional<CudaArray>& m_scales,
                       const std::optional<CudaArray>& abs_max,
                       unsigned int seed, const std::uintptr_t stream) {
    const float* grad_scale_ptr = grad_scale.has_value() ? static_cast<const float*>(grad_scale->data()) : nullptr;
    // TODO validate shape compatibility
    Tensor params_t = to_tensor(params);
    Tensor grads_t = to_tensor(grads);
    Tensor m_t = to_tensor(m);
    Tensor v_t = to_tensor(v);
    Tensor m_scales_t = to_tensor(m_scales);
    adamw_update(params_t, grads_t, m_t, v_t, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay,
                 grad_scale_ptr, m_scales_t, get_abs_max_ptr(abs_max), seed, as_stream(stream));
}

// ---- Fillers ----
void bind_fill_normal(const CudaArray& dst, float mean, float std, unsigned long long seed, unsigned long long subsequence, const std::uintptr_t stream) {
    Tensor dst_t = to_tensor(dst);
    fill_normal(dst_t, dst.size(), mean, std, seed, subsequence, as_stream(stream));
}

void bind_fill_constant(const CudaArray& dst, float value, const std::uintptr_t stream) {
    Tensor dst_t = to_tensor(dst);
    fill_constant(dst_t, value, dst.size(), as_stream(stream));
}

// ---- Encoder backward ----
void bind_encoder_backward(const CudaArray& dwte, const CudaArray& scratch, const CPUArray& workload_indices, const CPUArray& bucket_info,
                           const CudaArray& dout, const CudaArray& inp, const CPUArray& inputs_cpu,
                           unsigned int seed, const std::uintptr_t stream, std::uintptr_t sync_event, std::uintptr_t copy_stream) {
    const long B = get_dimension_checked({inp.shape(0), dout.shape(0), inputs_cpu.shape(0)}, "B");
    const long T = get_dimension_checked({inp.shape(1), dout.shape(1), inputs_cpu.shape(1)}, "T");
    const long C = get_dimension_checked({dout.shape(2), dwte.shape(1)}, "C");
    Tensor dwte_t = to_tensor(dwte);
    Tensor scratch_t = to_tensor(scratch);
    Tensor wl_t = to_tensor(workload_indices);
    Tensor bucket_t = to_tensor(bucket_info);
    encoder_backward(dwte_t, scratch_t, wl_t, bucket_t, to_tensor(dout), to_tensor(inp), to_tensor(inputs_cpu),
                     B, T, C, seed, as_stream(stream), reinterpret_cast<cudaEvent_t>(sync_event), as_stream(copy_stream));
}


void register_kernels(nanobind::module_& m) {
    m.def("encoder_forward", &bind_encoder_forward, nb::arg("out"), nb::arg("inp"), nb::arg("wte"), nb::arg("wpe") = std::nullopt, nb::arg("stream") = 0);

    // RMSNorm
    m.def("rmsnorm_forward", &bind_rmsnorm_forward, nb::arg("out"), nb::arg("rms"), nb::arg("inp"), nb::arg("weight"), nb::arg("absmax") = std::nullopt, nb::arg("epsilon"), nb::arg("stream") = 0);
    m.def("rmsnorm_backward", &bind_rmsnorm_backward, nb::arg("dinp"), nb::arg("dweight"), nb::arg("scratch"), nb::arg("dresidual"), nb::arg("dout"), nb::arg("inp"), nb::arg("weight"), nb::arg("rstd"), nb::arg("absmax") = std::nullopt, nb::arg("stream") = 0);

    // Fused residual + rmsnorm
    m.def("fused_residual_rmsnorm_forward", &bind_fused_residual_rmsnorm_forward, nb::arg("residual"), nb::arg("normed"), nb::arg("rrms"), nb::arg("inp1"), nb::arg("inp2"), nb::arg("weight"), nb::arg("absmax") = std::nullopt, nb::arg("epsilon"), nb::arg("stream") = 0);

    // Rope
    m.def("rope_forward", &bind_rope_forward, nb::arg("out"), nb::arg("in"), nb::arg("freqs_cis"), nb::arg("absmax") = std::nullopt, nb::arg("Nq"), nb::arg("Nkv"), nb::arg("stream") = 0);
    m.def("rope_backward", &bind_rope_backward, nb::arg("dinp"), nb::arg("dout"), nb::arg("freqs_cis"), nb::arg("absmax") = std::nullopt, nb::arg("Nq"), nb::arg("Nkv"),  nb::arg("stream") = 0);

    // SwiGLU
    m.def("swiglu_forward", &bind_swiglu_forward, nb::arg("out"), nb::arg("inp"), nb::arg("absmax") = std::nullopt, nb::arg("stream") = 0);
    m.def("swiglu_forward_quant", &bind_swiglu_forward_quant, nb::arg("out"), nb::arg("scale"), nb::arg("inp"), nb::arg("absmax") = std::nullopt, nb::arg("stream") = 0);
    m.def("swiglu_backward", &bind_swiglu_backward, nb::arg("dinp"), nb::arg("dout"), nb::arg("inp"), nb::arg("absmax") = std::nullopt, nb::arg("stream") = 0);

    // Attention (cuDNN)
    m.def("attention_forward", &bind_attention_forward, nb::arg("out"), nb::arg("stats"), nb::arg("inp"), nb::arg("workspace"), nb::arg("cudnn_handle"), nb::arg("Hq"), nb::arg("Hkv"), nb::arg("stream") = 0);
    m.def("attention_backward", &bind_attention_backward, nb::arg("dqkv"), nb::arg("stats"), nb::arg("out"), nb::arg("dout"), nb::arg("qkv"), nb::arg("workspace"), nb::arg("cudnn_handle"), nb::arg("Hq"), nb::arg("Hkv"), nb::arg("stream") = 0);

    // Classifier
    m.def("fused_classifier", &bind_fused_classifier, nb::arg("logits"), nb::arg("losses"), nb::arg("lse"), nb::arg("dloss"), nb::arg("targets"), nb::arg("z_reg"), nb::arg("write_dlogits"), nb::arg("stream") = 0);
    m.def("grouped_loss_sum", &bind_grouped_loss_sum, nb::arg("out"), nb::arg("per_token_loss"), nb::arg("stream") = 0);

    // Matmul
    m.def("matmul", &bind_matmul, nb::arg("c"), nb::arg("a"), nb::arg("b"), nb::arg("bias") = std::nullopt, nb::arg("scale_a") = std::nullopt, nb::arg("scale_b") = std::nullopt, nb::arg("cublaslt_handle"), nb::arg("workspace"), nb::arg("M"), nb::arg("N"), nb::arg("K"), nb::arg("mode"), nb::arg("accumulate") = false, nb::arg("stream") = 0);

    // Bias backward
    m.def("backward_bias", &bind_backward_bias, nb::arg("dbias"), nb::arg("dout"), nb::arg("scale_a") = std::nullopt, nb::arg("scale_b") = std::nullopt, nb::arg("dbias_buffer"), nb::arg("stream") = 0);

    // Quantization utils
    m.def("abs_max", &bind_abs_max, nb::arg("scale"), nb::arg("in"), nb::arg("stream") = 0);
    m.def("quantize_with_abs_max", &bind_quantize_with_abs_max, nb::arg("out"), nb::arg("scale"), nb::arg("in"), nb::arg("abs_max"), nb::arg("stream") = 0);
    m.def("quantize_and_transpose_with_abs_max", &bind_quantize_and_transpose_with_abs_max, nb::arg("out"), nb::arg("scale"), nb::arg("in"), nb::arg("abs_max"), nb::arg("rows"), nb::arg("cols"), nb::arg("stream") = 0);

    // Transpose
    m.def("transpose", &bind_transpose, nb::arg("dst"), nb::arg("src"), nb::arg("rows"), nb::arg("cols"), nb::arg("stream") = 0);

    // Vector ops
    m.def("vector_add_sr", &bind_vector_add_sr, nb::arg("dest"), nb::arg("left"), nb::arg("right"), nb::arg("scale"), nb::arg("seed"), nb::arg("stream") = 0);
    m.def("vector_reduce_sr", &bind_vector_reduce_sr, nb::arg("dest"), nb::arg("src"), nb::arg("scale"), nb::arg("n_shards"), nb::arg("skip") = -1, nb::arg("accumulate") = false, nb::arg("seed") = 0u, nb::arg("stream") = 0);

    // Fillers
    m.def("fill_normal", &bind_fill_normal, nb::arg("dst"), nb::arg("mean"), nb::arg("std"), nb::arg("seed"), nb::arg("subsequence") = 0ull, nb::arg("stream") = 0);
    m.def("fill_constant", &bind_fill_constant, nb::arg("dst"), nb::arg("value"), nb::arg("stream") = 0);

    // Encoder backward
    m.def("encoder_backward", &bind_encoder_backward,
          nb::arg("dwte"), nb::arg("scratch"), nb::arg("workload_indices"), nb::arg("bucket_info"),
          nb::arg("dout"), nb::arg("inp"), nb::arg("inputs_cpu"), nb::arg("seed"), nb::arg("stream") = 0, nb::arg("sync_event") = 0, nb::arg("copy_stream") = 0);

    // Global norm
    m.def("global_norm_squared", &bind_global_norm_squared, nb::arg("out"), nb::arg("values"), nb::arg("stream") = 0);

    // AdamW
    m.def("adamw_update", &bind_adamw_update, nb::arg("params"), nb::arg("grads"), nb::arg("m"), nb::arg("v"), nb::arg("num_parameters"), nb::arg("learning_rate"), nb::arg("beta1"), nb::arg("beta2"), nb::arg("t"), nb::arg("eps"), nb::arg("weight_decay"), nb::arg("grad_scale") = std::nullopt, nb::arg("m_scales") = std::nullopt, nb::arg("abs_max") = std::nullopt, nb::arg("seed") = 0u, nb::arg("stream") = 0);
}
