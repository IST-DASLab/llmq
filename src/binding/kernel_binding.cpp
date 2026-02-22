// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kernels/kernels.h"
#include "binding_utils.h"

#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

namespace nb = nanobind;

using CudaArray = nb::ndarray<nb::c_contig, nb::device::cuda>;


Tensor to_tensor(const CudaArray& array) {
    const ETensorDType dtype = from_dlpack_dtype(array.dtype());
    const std::vector<long> shape(array.shape_ptr(), array.shape_ptr() + array.ndim());
    return Tensor::from_pointer(static_cast<std::byte*>(array.data()), array.device_id(), dtype, shape);
}
Tensor to_tensor(const std::optional<CudaArray>& array) {
    if (array.has_value()) {
        return to_tensor(array.value());
    } else {
        return Tensor{};
    }
}

long get_dimension(const std::initializer_list<std::size_t> values) {
    const long value = *values.begin();
    for (auto it = values.begin(); it != values.end(); ++it) {
        if (*it != value)
            throw std::invalid_argument("All dimensions must be equal");
    }
    return value;
}

static inline long numel(const CudaArray& a) {
    long n = 1;
    for (std::size_t i = 0; i < a.ndim(); ++i) n *= static_cast<long>(a.shape(i));
    return n;
}

void bind_encoder_forward(const CudaArray& out, const CudaArray& inp, const CudaArray& wte, const std::optional<CudaArray>& wpe, const std::uintptr_t stream) {
    // TODO wpe dimension check
    long B = get_dimension({out.shape(0), inp.shape(0)});
    long T = get_dimension({out.shape(1), inp.shape(1)});
    long V = wte.shape(0);
    long C = get_dimension({out.shape(2), wte.shape(1)});
    Tensor out_t = to_tensor(out);
    Tensor inp_t = to_tensor(inp);
    Tensor wte_t = to_tensor(wte);
    Tensor wpe_t = to_tensor(wpe);
    encoder_forward(out_t, inp_t, wte_t, wpe_t, B, T, C, V, reinterpret_cast<cudaStream_t>(stream));
}

static inline cudaStream_t as_stream(std::uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

static inline cudaDeviceProp get_device_prop(int device) {
    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, device));
    return dp;
}

// ---- RMSNorm ----
void bind_rmsnorm_forward(const CudaArray& out, const CudaArray& rms, const CudaArray& inp, const CudaArray& weight,
                          const std::optional<CudaArray>& absmax, float epsilon, const std::uintptr_t stream) {
    long B = inp.shape(0);
    long T = inp.shape(1);
    long C = inp.shape(2);
    float* abs_max_ptr = absmax.has_value() ? static_cast<float*>(absmax->data()) : nullptr;
    Tensor out_t = to_tensor(out);
    Tensor rms_t = to_tensor(rms);
    Tensor inp_t = to_tensor(inp);
    Tensor w_t = to_tensor(weight);
    rmsnorm_forward(out_t, rms_t, inp_t, w_t, abs_max_ptr, epsilon, B, T, C, as_stream(stream));
}

void bind_rmsnorm_backward(const CudaArray& dinp, const CudaArray& dweight, const CudaArray& scratch,
                           const CudaArray& dresidual, const CudaArray& dout, const CudaArray& inp, const CudaArray& weight,
                           const CudaArray& rstd, const std::optional<CudaArray>& absmax,
                           const std::uintptr_t stream) {
    int device = inp.device_id();
    auto dp = get_device_prop(device);
    long B = inp.shape(0);
    long T = inp.shape(1);
    long C = inp.shape(2);
    float* abs_max_ptr = absmax.has_value() ? static_cast<float*>(absmax->data()) : nullptr;
    Tensor dinp_t = to_tensor(dinp);
    Tensor dweight_t = to_tensor(dweight);
    Tensor scratch_t = to_tensor(scratch);
    Tensor dres_t = to_tensor(dresidual);
    Tensor dout_t = to_tensor(dout);
    Tensor inp_t = to_tensor(inp);
    Tensor w_t = to_tensor(weight);
    Tensor rstd_t = to_tensor(rstd);
    rmsnorm_backward(dinp_t, dweight_t, scratch_t, dres_t, dout_t, inp_t, w_t, rstd_t, abs_max_ptr,
                     B, T, C, dp, as_stream(stream));
}

// ---- Fused residual + rmsnorm ----
void bind_fused_residual_rmsnorm_forward(const CudaArray& residual, const CudaArray& normed, const CudaArray& rrms,
                                         const CudaArray& inp1, const CudaArray& inp2, const CudaArray& weight,
                                         const std::optional<CudaArray>& absmax, float epsilon, const std::uintptr_t stream) {
    long N = inp1.shape(0) * inp1.shape(1);
    long C = inp1.shape(2);
    float* abs_max_ptr = absmax.has_value() ? static_cast<float*>(absmax->data()) : nullptr;
    Tensor residual_t = to_tensor(residual);
    Tensor normed_t = to_tensor(normed);
    Tensor rrms_t = to_tensor(rrms);
    Tensor inp1_t = to_tensor(inp1);
    Tensor inp2_t = to_tensor(inp2);
    Tensor w_t = to_tensor(weight);
    fused_residual_rmsnorm_forward(residual_t, normed_t, rrms_t, inp1_t, inp2_t, w_t, abs_max_ptr,
                                   epsilon, N, C, as_stream(stream));
}

// ---- Rope ----
void bind_rope_forward(const CudaArray& out, const CudaArray& in, const CudaArray& freqs_cis,
                       const std::optional<CudaArray>& absmax, int Nq, int Nkv, int head_dim, const std::uintptr_t stream) {
    long B = in.shape(0);
    long T = in.shape(1);
    float* abs_max_ptr = absmax.has_value() ? static_cast<float*>(absmax->data()) : nullptr;
    Tensor out_t = to_tensor(out);
    Tensor in_t = to_tensor(in);
    Tensor f_t = to_tensor(freqs_cis);
    rope_forward(out_t, in_t, f_t, abs_max_ptr, B, T, Nq, Nkv, head_dim, as_stream(stream));
}

void bind_rope_backward(const CudaArray& dinp, const CudaArray& dout, const CudaArray& freqs_cis,
                        const std::optional<CudaArray>& absmax, int Nq, int Nkv, int head_dim, const std::uintptr_t stream) {
    long B = dout.shape(0);
    long T = dout.shape(1);
    float* abs_max_ptr = absmax.has_value() ? static_cast<float*>(absmax->data()) : nullptr;
    Tensor dinp_t = to_tensor(dinp);
    Tensor dout_t = to_tensor(dout);
    Tensor f_t = to_tensor(freqs_cis);
    rope_backward(dinp_t, dout_t, f_t, abs_max_ptr, B, T, Nq, Nkv, head_dim, as_stream(stream));
}

// ---- SwiGLU ----
void bind_swiglu_forward(const CudaArray& out, const CudaArray& inp, const std::optional<CudaArray>& absmax,
                         const std::uintptr_t stream) {
    long B = inp.shape(0);
    long T = inp.shape(1);
    long C = inp.shape(2);
    float* abs_max_ptr = absmax.has_value() ? static_cast<float*>(absmax->data()) : nullptr;
    Tensor out_t = to_tensor(out);
    Tensor inp_t = to_tensor(inp);
    swiglu_forward(out_t, inp_t, abs_max_ptr, B, T, C, as_stream(stream));
}

void bind_swiglu_forward_quant(const CudaArray& out, const CudaArray& scale, const CudaArray& inp,
                               const std::optional<CudaArray>& absmax, const std::uintptr_t stream) {
    long B = inp.shape(0);
    long T = inp.shape(1);
    long C = inp.shape(2);
    const float* abs_max_ptr = absmax.has_value() ? static_cast<const float*>(absmax->data()) : nullptr;
    Tensor out_t = to_tensor(out);
    Tensor inp_t = to_tensor(inp);
    swiglu_forward_quant(out_t, static_cast<float*>(scale.data()), inp_t, abs_max_ptr, B, T, C, as_stream(stream));
}

void bind_swiglu_backward(const CudaArray& dinp, const CudaArray& dout, const CudaArray& inp,
                          const std::optional<CudaArray>& absmax, const std::uintptr_t stream) {
    long B = inp.shape(0);
    long T = inp.shape(1);
    long C = inp.shape(2);
    float* abs_max_ptr = absmax.has_value() ? static_cast<float*>(absmax->data()) : nullptr;
    Tensor dinp_t = to_tensor(dinp);
    Tensor dout_t = to_tensor(dout);
    Tensor inp_t = to_tensor(inp);
    swiglu_backward(dinp_t, dout_t, inp_t, abs_max_ptr, B, T, C, as_stream(stream));
}

// ---- Attention (cuDNN) ----
void bind_attention_forward(const CudaArray& out, const CudaArray& stats, const CudaArray& inp,
                            const CudaArray& workspace, std::uintptr_t cudnn_handle,
                            int Hq, int Hkv, int HS, const std::uintptr_t stream) {
    long B = inp.shape(0);
    long T = inp.shape(1);
    Tensor out_t = to_tensor(out);
    Tensor stats_t = to_tensor(stats);
    Tensor inp_t = to_tensor(inp);
    Tensor ws_t = to_tensor(workspace);
    attention_forward_cudnn(out_t, stats_t, inp_t, ws_t,
                            reinterpret_cast<cudnnHandle_t>(cudnn_handle), B, T, Hq, Hkv, HS, as_stream(stream));
}

void bind_attention_backward(const CudaArray& dqkv, const CudaArray& stats, const CudaArray& out, const CudaArray& dout,
                             const CudaArray& qkv, const CudaArray& workspace, std::uintptr_t cudnn_handle,
                             int Hq, int Hkv, int HS, const std::uintptr_t stream) {
    long B = qkv.shape(0);
    long T = qkv.shape(1);
    Tensor dqkv_t = to_tensor(dqkv);
    Tensor stats_t = to_tensor(stats);
    Tensor out_t = to_tensor(out);
    Tensor dout_t = to_tensor(dout);
    Tensor qkv_t = to_tensor(qkv);
    Tensor ws_t = to_tensor(workspace);
    attention_backward_cudnn(dqkv_t, stats_t, out_t, dout_t, qkv_t, ws_t,
                             reinterpret_cast<cudnnHandle_t>(cudnn_handle), B, T, Hq, Hkv, HS, as_stream(stream));
}

// ---- Classifier / losses ----
void bind_fused_classifier(const CudaArray& logits, const CudaArray& losses, const CudaArray& lse,
                           float dloss, const CudaArray& targets, float z_reg, bool write_dlogits, const std::uintptr_t stream) {
    long BT = numel(targets);
    long V = logits.shape(1);
    long P = logits.shape(2);
    Tensor logits_t = to_tensor(logits);
    Tensor losses_t = to_tensor(losses);
    Tensor lse_t = to_tensor(lse);
    Tensor targets_t = to_tensor(targets);
    fused_classifier(logits_t, losses_t, lse_t, dloss, targets_t, z_reg, BT, V, P, write_dlogits, as_stream(stream));
}

void bind_grouped_loss_sum(const CudaArray& out, const CudaArray& per_token_loss, const std::uintptr_t stream) {
    long B = per_token_loss.shape(0);
    long T = per_token_loss.shape(1);
    Tensor out_t = to_tensor(out);
    Tensor ptl_t = to_tensor(per_token_loss);
    grouped_loss_sum(out_t, ptl_t, B, T, as_stream(stream));
}

// ---- Matmul ----
void bind_matmul(const CudaArray& c, const CudaArray& a, const CudaArray& b, const std::optional<CudaArray>& bias,
                 const std::optional<CudaArray>& scale_a, const std::optional<CudaArray>& scale_b,
                 std::uintptr_t cublaslt_handle, const CudaArray& workspace,
                 int M, int N, int K, EMMTranspose mode, bool accumulate, const std::uintptr_t stream) {
    const float* scale_a_ptr = scale_a.has_value() ? static_cast<const float*>(scale_a->data()) : nullptr;
    const float* scale_b_ptr = scale_b.has_value() ? static_cast<const float*>(scale_b->data()) : nullptr;
    Tensor c_t = to_tensor(c);
    Tensor a_t = to_tensor(a);
    Tensor b_t = to_tensor(b);
    Tensor bias_t = to_tensor(bias);
    Tensor ws_t = to_tensor(workspace);
    matmul(c_t, a_t, b_t, bias_t, scale_a_ptr, scale_b_ptr,
           reinterpret_cast<cublasLtHandle_t>(cublaslt_handle), ws_t, M, N, K, mode, accumulate, as_stream(stream));
}

// ---- Bias add/backward ----
void bind_backward_bias(const CudaArray& dbias, const CudaArray& dout, const std::optional<CudaArray>& scale_a, const std::optional<CudaArray>& scale_b,
                        const CudaArray& dbias_buffer, int OC, const std::uintptr_t stream) {
    int device = dout.device_id();
    auto dp = get_device_prop(device);
    long B = dout.shape(0);
    long T = dout.shape(1);
    const float* scale_a_ptr = scale_a.has_value() ? static_cast<const float*>(scale_a->data()) : nullptr;
    const float* scale_b_ptr = scale_b.has_value() ? static_cast<const float*>(scale_b->data()) : nullptr;
    Tensor dbias_t = to_tensor(dbias);
    Tensor dout_t = to_tensor(dout);
    Tensor buf_t = to_tensor(dbias_buffer);
    backward_bias(dbias_t, dout_t, scale_a_ptr, scale_b_ptr, buf_t, B, T, OC, dp, as_stream(stream));
}

// ---- Quantization helpers ----
void bind_abs_max(const CudaArray& scale, const CudaArray& in, const std::uintptr_t stream) {
    int device = in.device_id();
    auto dp = get_device_prop(device);
    long N = numel(in);
    Tensor in_t = to_tensor(in);
    abs_max(static_cast<float*>(scale.data()), in_t, N, dp, as_stream(stream));
}

void bind_quantize_with_abs_max(const CudaArray& out, const CudaArray& scale, const CudaArray& in, const CudaArray& abs_max,
                                const std::uintptr_t stream) {
    int device = in.device_id();
    auto dp = get_device_prop(device);
    long N = numel(in);
    Tensor out_t = to_tensor(out);
    Tensor in_t = to_tensor(in);
    quantize_with_abs_max(out_t, static_cast<float*>(scale.data()), in_t, static_cast<const float*>(abs_max.data()), N, dp, as_stream(stream));
}

void bind_quantize_and_transpose_with_abs_max(const CudaArray& out, const CudaArray& scale, const CudaArray& in, const CudaArray& abs_max,
                                              int rows, int cols, const std::uintptr_t stream) {
    int device = in.device_id();
    auto dp = get_device_prop(device);
    Tensor out_t = to_tensor(out);
    Tensor in_t = to_tensor(in);
    quantize_and_transpose_with_abs_max(out_t, static_cast<float*>(scale.data()), in_t, static_cast<const float*>(abs_max.data()), rows, cols, dp, as_stream(stream));
}

// ---- Transpose ----
void bind_transpose(const CudaArray& dst, const CudaArray& src, int rows, int cols, const std::uintptr_t stream) {
    Tensor dst_t = to_tensor(dst);
    Tensor src_t = to_tensor(src);
    transpose(dst_t, src_t, rows, cols, as_stream(stream));
}

// ---- Vector ops ----
void bind_vector_add_sr(const CudaArray& dest, const CudaArray& left, const CudaArray& right, float scale, unsigned seed, const std::uintptr_t stream) {
    long nelem = numel(dest);
    Tensor dest_t = to_tensor(dest);
    Tensor left_t = to_tensor(left);
    Tensor right_t = to_tensor(right);
    vector_add_sr(dest_t, left_t, right_t, scale, nelem, seed, as_stream(stream));
}

void bind_vector_reduce_sr(const CudaArray& dest, const CudaArray& src, float scale, int n_shards, int skip, bool accumulate, unsigned seed, const std::uintptr_t stream) {
    long nelem = numel(dest);
    Tensor dest_t = to_tensor(dest);
    Tensor src_t = to_tensor(src);
    vector_reduce_sr(dest_t, src_t, scale, n_shards, skip, nelem, accumulate, seed, as_stream(stream));
}

// ---- Global norm ----
void bind_global_norm_squared(const CudaArray& out, const CudaArray& values, const std::uintptr_t stream) {
    int device = values.device_id();
    auto dp = get_device_prop(device);
    size_t count = static_cast<size_t>(numel(values));
    Tensor out_t = to_tensor(out);
    Tensor vals_t = to_tensor(values);
    global_norm_squared(out_t, vals_t, count, dp, as_stream(stream));
}

// ---- AdamW ----
void bind_adamw_update(const CudaArray& params, const CudaArray& grads, const CudaArray& m, const CudaArray& v,
                       size_t num_parameters, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                       const std::optional<CudaArray>& grad_scale, const std::optional<CudaArray>& m_scales,
                       const std::optional<CudaArray>& abs_max,
                       unsigned int seed, const std::uintptr_t stream) {
    const float* grad_scale_ptr = grad_scale.has_value() ? static_cast<const float*>(grad_scale->data()) : nullptr;
    float* abs_max_ptr = abs_max.has_value() ? static_cast<float*>(abs_max->data()) : nullptr;
    Tensor params_t = to_tensor(params);
    Tensor grads_t = to_tensor(grads);
    Tensor m_t = to_tensor(m);
    Tensor v_t = to_tensor(v);
    Tensor m_scales_t = to_tensor(m_scales);
    adamw_update(params_t, grads_t, m_t, v_t, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay,
                 grad_scale_ptr, m_scales_t, abs_max_ptr, seed, as_stream(stream));
}

// ---- Fillers ----
void bind_fill_normal(const CudaArray& dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, const std::uintptr_t stream) {
    Tensor dst_t = to_tensor(dst);
    fill_normal(dst_t, count, mean, std, seed, subsequence, as_stream(stream));
}

void bind_fill_constant(const CudaArray& dst, float value, std::size_t count, const std::uintptr_t stream) {
    Tensor dst_t = to_tensor(dst);
    fill_constant(dst_t, value, count, as_stream(stream));
}

// ---- Encoder backward ----
void bind_encoder_backward(const CudaArray& dwte, const CudaArray& scratch, const CudaArray& workload_indices, const CudaArray& bucket_info,
                           const CudaArray& dout, const CudaArray& inp, const CudaArray& inputs_cpu,
                           unsigned int seed, const std::uintptr_t stream, std::uintptr_t sync_event, std::uintptr_t copy_stream) {
    long B = inp.shape(0);
    long T = inp.shape(1);
    long C = dout.shape(2);
    Tensor dwte_t = to_tensor(dwte);
    Tensor scratch_t = to_tensor(scratch);
    Tensor wl_t = to_tensor(workload_indices);
    Tensor bucket_t = to_tensor(bucket_info);
    Tensor dout_t = to_tensor(dout);
    Tensor inp_t = to_tensor(inp);
    Tensor cpu_t = to_tensor(inputs_cpu);
    encoder_backward(dwte_t, scratch_t, wl_t, bucket_t, dout_t, inp_t, cpu_t,
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
    m.def("rope_forward", &bind_rope_forward, nb::arg("out"), nb::arg("in"), nb::arg("freqs_cis"), nb::arg("absmax") = std::nullopt, nb::arg("Nq"), nb::arg("Nkv"), nb::arg("head_dim"), nb::arg("stream") = 0);
    m.def("rope_backward", &bind_rope_backward, nb::arg("dinp"), nb::arg("dout"), nb::arg("freqs_cis"), nb::arg("absmax") = std::nullopt, nb::arg("Nq"), nb::arg("Nkv"), nb::arg("head_dim"), nb::arg("stream") = 0);

    // SwiGLU
    m.def("swiglu_forward", &bind_swiglu_forward, nb::arg("out"), nb::arg("inp"), nb::arg("absmax") = std::nullopt, nb::arg("stream") = 0);
    m.def("swiglu_forward_quant", &bind_swiglu_forward_quant, nb::arg("out"), nb::arg("scale"), nb::arg("inp"), nb::arg("absmax") = std::nullopt, nb::arg("stream") = 0);
    m.def("swiglu_backward", &bind_swiglu_backward, nb::arg("dinp"), nb::arg("dout"), nb::arg("inp"), nb::arg("absmax") = std::nullopt, nb::arg("stream") = 0);

    // Attention (cuDNN)
    m.def("attention_forward", &bind_attention_forward, nb::arg("out"), nb::arg("stats"), nb::arg("inp"), nb::arg("workspace"), nb::arg("cudnn_handle"), nb::arg("Hq"), nb::arg("Hkv"), nb::arg("HS"), nb::arg("stream") = 0);
    m.def("attention_backward", &bind_attention_backward, nb::arg("dqkv"), nb::arg("stats"), nb::arg("out"), nb::arg("dout"), nb::arg("qkv"), nb::arg("workspace"), nb::arg("cudnn_handle"), nb::arg("Hq"), nb::arg("Hkv"), nb::arg("HS"), nb::arg("stream") = 0);

    // Classifier
    m.def("fused_classifier", &bind_fused_classifier, nb::arg("logits"), nb::arg("losses"), nb::arg("lse"), nb::arg("dloss"), nb::arg("targets"), nb::arg("z_reg"), nb::arg("write_dlogits"), nb::arg("stream") = 0);
    m.def("grouped_loss_sum", &bind_grouped_loss_sum, nb::arg("out"), nb::arg("per_token_loss"), nb::arg("stream") = 0);

    // Matmul
    m.def("matmul", &bind_matmul, nb::arg("c"), nb::arg("a"), nb::arg("b"), nb::arg("bias") = std::nullopt, nb::arg("scale_a") = std::nullopt, nb::arg("scale_b") = std::nullopt, nb::arg("cublaslt_handle"), nb::arg("workspace"), nb::arg("M"), nb::arg("N"), nb::arg("K"), nb::arg("mode"), nb::arg("accumulate") = false, nb::arg("stream") = 0);

    // Bias backward
    m.def("backward_bias", &bind_backward_bias, nb::arg("dbias"), nb::arg("dout"), nb::arg("scale_a") = std::nullopt, nb::arg("scale_b") = std::nullopt, nb::arg("dbias_buffer"), nb::arg("OC"), nb::arg("stream") = 0);

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
    m.def("fill_normal", &bind_fill_normal, nb::arg("dst"), nb::arg("count"), nb::arg("mean"), nb::arg("std"), nb::arg("seed"), nb::arg("subsequence") = 0ull, nb::arg("stream") = 0);
    m.def("fill_constant", &bind_fill_constant, nb::arg("dst"), nb::arg("value"), nb::arg("count"), nb::arg("stream") = 0);

    // Encoder backward
    m.def("encoder_backward", &bind_encoder_backward,
          nb::arg("dwte"), nb::arg("scratch"), nb::arg("workload_indices"), nb::arg("bucket_info"),
          nb::arg("dout"), nb::arg("inp"), nb::arg("inputs_cpu"), nb::arg("seed"), nb::arg("stream") = 0, nb::arg("sync_event") = 0, nb::arg("copy_stream") = 0);

    // Global norm
    m.def("global_norm_squared", &bind_global_norm_squared, nb::arg("out"), nb::arg("values"), nb::arg("stream") = 0);

    // AdamW
    m.def("adamw_update", &bind_adamw_update, nb::arg("params"), nb::arg("grads"), nb::arg("m"), nb::arg("v"), nb::arg("num_parameters"), nb::arg("learning_rate"), nb::arg("beta1"), nb::arg("beta2"), nb::arg("t"), nb::arg("eps"), nb::arg("weight_decay"), nb::arg("grad_scale") = std::nullopt, nb::arg("m_scales") = std::nullopt, nb::arg("abs_max") = std::nullopt, nb::arg("seed") = 0u, nb::arg("stream") = 0);
}
