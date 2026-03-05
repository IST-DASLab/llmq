// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include "utilities/tensor.h"

template<typename TensorLike>
static std::string tensor_dtype_str(const TensorLike& t) {
    if (t.empty()) {
        return "<null>";
    }
    return dtype_to_str(t.DType);
}

template<typename... T>
static void throw_unsupported_dtype(const char* func, T&&... args) {
    std::array<std::string, sizeof...(T)> dtypes = {tensor_dtype_str(args)...};
    throw std::invalid_argument(fmt::format("{}: unsupported dtypes. Got {}", func, dtypes));
}

#define UNSUPPORTED_DTYPE(...) throw_unsupported_dtype(__FUNCTION__, __VA_ARGS__)

void rmsnorm_forward(Tensor& out, Tensor& rms, const Tensor& inp, const Tensor& weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::BF16) {
        rmsnorm_forward(out.get<nv_bfloat16>(), rms.get<float>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), abs_max_ptr, epsilon, B, T, C, stream);
    } else if (out.DType == ETensorDType::FP32) {
        rmsnorm_forward(out.get<float>(), rms.get<float>(), inp.get<float>(), weight.get<float>(), abs_max_ptr, epsilon, B, T, C, stream);
    } else {
        UNSUPPORTED_DTYPE(out, rms, inp, weight);
    }
}

void rmsnorm_backward(Tensor& dinp, Tensor& dweight, Tensor& scratch, const Tensor& dresidual, const Tensor& dout, const Tensor& inp, const Tensor& weight, const Tensor& rstd, float* abs_max_ptr,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::BF16) {
        rmsnorm_backward(dinp.get<nv_bfloat16>(), dweight.get<nv_bfloat16>(), scratch.Data, dresidual.get<nv_bfloat16>(),
            dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), rstd.get<float>(), abs_max_ptr, B, T, C, dp, stream);
    } else if (dinp.DType == ETensorDType::FP32) {
        rmsnorm_backward(dinp.get<float>(), dweight.get<float>(), scratch.Data, dresidual.get<float>(),
            dout.get<float>(), inp.get<float>(), weight.get<float>(), rstd.get<float>(), abs_max_ptr, B, T, C, dp, stream);
    } else {
        UNSUPPORTED_DTYPE(dinp, dweight, scratch, dresidual, dout, inp, weight, rstd);
    }
}

void fused_residual_rmsnorm_forward(Tensor& residual, Tensor& normed, Tensor& rrms, const Tensor& inp1, const Tensor& inp2, const Tensor& weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream)
{
    if(residual.DType == ETensorDType::BF16) {
        fused_residual_rmsnorm_forward(residual.get<nv_bfloat16>(), normed.get<nv_bfloat16>(), rrms.get<float>(),
            inp1.get<nv_bfloat16>(), inp2.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), abs_max_ptr, epsilon, N, C, stream);
    } else if (residual.DType == ETensorDType::FP32) {
        fused_residual_rmsnorm_forward(residual.get<float>(), normed.get<float>(), rrms.get<float>(),
            inp1.get<float>(), inp2.get<float>(), weight.get<float>(), abs_max_ptr, epsilon, N, C, stream);
    } else {
        UNSUPPORTED_DTYPE(residual, normed, rrms, inp1, inp2, weight);
    }
}

void swiglu_forward(Tensor& out, const Tensor& inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP32) {
        swiglu_forward(out.get<float>(), inp.get<float>(), abs_max_ptr, B, T, C, stream);
    } else if (out.DType == ETensorDType::BF16) {
        swiglu_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), abs_max_ptr, B, T, C, stream);
    } else {
        UNSUPPORTED_DTYPE(out, inp);
    }
}

void swiglu_forward_quant(Tensor& out, float* scale_ptr, const Tensor& inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP8_E4M3 && inp.DType == ETensorDType::BF16) {
        swiglu_forward_quant(out.get<__nv_fp8_e4m3>(), scale_ptr, inp.get<nv_bfloat16>(), abs_max_ptr, B, T, C, stream);
    } else {
        UNSUPPORTED_DTYPE(out, inp);
    }
}

void swiglu_backward(Tensor& dinp, const Tensor& dout, const Tensor& inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::FP32) {
        swiglu_backward(dinp.get<float>(), dout.get<float>(), inp.get<float>(), abs_max, B, T, C, stream);
    } else if (dinp.DType == ETensorDType::BF16) {
        swiglu_backward(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), abs_max, B, T, C, stream);
    } else {
        UNSUPPORTED_DTYPE(dinp, dout, inp);
    }
}

void rope_forward(Tensor& out, const Tensor& in, const Tensor& freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream)  {
    if(out.DType == ETensorDType::FP32) {
        rope_forward(out.get<float>(), in.get<float>(), freqs_cis.get<float>(), abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
    } else if(out.DType == ETensorDType::BF16) {
        rope_forward(out.get<nv_bfloat16>(), in.get<nv_bfloat16>(), freqs_cis.get<half>(), abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
    } else {
        UNSUPPORTED_DTYPE(out, in, freqs_cis);
    }
}

void rope_backward(Tensor& dinp, const Tensor& dout, const Tensor& freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::FP32) {
        rope_backward(dinp.get<float>(), dout.get<float>(), freqs_cis.get<float>(), abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
    } else if(dinp.DType == ETensorDType::BF16) {
        rope_backward(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), freqs_cis.get<half>(), abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
    } else {
        UNSUPPORTED_DTYPE(dinp, dout, freqs_cis);
    }
}

void qk_norm_forward(Tensor& out, Tensor& r_std, const Tensor& inp, const Tensor& q_wgt, const Tensor& k_wgt,
                     float epsilon, int BT, int Nq, int Nkv, int HeadDim, cudaStream_t stream) {
    if (out.DType == ETensorDType::BF16) {
        qk_norm_forward(out.get<nv_bfloat16>(), r_std.get<float>(), inp.get<nv_bfloat16>(),
            q_wgt.get<nv_bfloat16>(), k_wgt.get<nv_bfloat16>(), epsilon, BT, Nq, Nkv, HeadDim, stream);
    } else if (out.DType == ETensorDType::FP32) {
        qk_norm_forward(out.get<float>(), r_std.get<float>(), inp.get<float>(),
            q_wgt.get<float>(), k_wgt.get<float>(), epsilon, BT, Nq, Nkv, HeadDim, stream);
    } else {
        UNSUPPORTED_DTYPE(out, r_std, inp, q_wgt, k_wgt);
    }
}

void fused_classifier(Tensor& logits, Tensor& losses, Tensor& lse,
                      float dloss, const Tensor& targets, float z_reg,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    if(logits.DType == ETensorDType::FP32) {
        fused_classifier(logits.get<float>(), losses.get<float>(), lse.get<float>(), dloss, targets.get<int>(), z_reg, BT, V, P, write_dlogits, stream);
    } else if(logits.DType == ETensorDType::BF16) {
        fused_classifier(logits.get<nv_bfloat16>(), losses.get<float>(), lse.get<float>(), dloss, targets.get<int>(), z_reg, BT, V, P, write_dlogits, stream);
    } else {
        UNSUPPORTED_DTYPE(logits, losses, lse);
    }
}

void grouped_loss_sum(Tensor& out, const Tensor& per_token_loss, int B, int T, cudaStream_t stream) {
    grouped_loss_sum(out.get<float>(), per_token_loss.get<float>(), B, T, stream);
}

void encoder_forward(Tensor& out, const Tensor& inp, const Tensor& wte, const Tensor& wpe, int B, int T, int C, int V, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP32) {
        encoder_forward(out.get<float>(), inp.get<std::int32_t>(), wte.get<float>(), wpe.get_optional<float>(), B, T, C, V, stream);
    } else if(out.DType == ETensorDType::BF16) {
        encoder_forward(out.get<nv_bfloat16>(), inp.get<std::int32_t>(), wte.get<nv_bfloat16>(),  wpe.get_optional<nv_bfloat16>(), B, T, C, V, stream);
    } else {
        UNSUPPORTED_DTYPE(out, inp, wte, wpe);
    }
}

void encoder_backward(Tensor& dwte, Tensor& scratch,
                      Tensor& workload_indices, Tensor& bucket_info,
                      const Tensor& dout, const Tensor& inp, const Tensor& inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream) {
    assert(workload_indices.Device == -1);
    assert(bucket_info.Device == -1);
    if(dwte.DType == ETensorDType::FP32) {
        encoder_backward(dwte.get<float>(), scratch.get<int>(), workload_indices.get<int>(),
            (int4*)bucket_info.get<int>(), dout.get<float>(), inp.get<std::int32_t>(), inputs_cpu.get<std::int32_t>(),
            B, T, C, seed, stream, sync_event, copy_stream);
    } else if(dwte.DType == ETensorDType::BF16) {
        encoder_backward(dwte.get<nv_bfloat16>(), scratch.get<int>(), workload_indices.get<int>(),
            (int4*)bucket_info.get<int>(), dout.get<nv_bfloat16>(), inp.get<std::int32_t>(), inputs_cpu.get<std::int32_t>(),
            B, T, C, seed, stream, sync_event, copy_stream);
    } else {
        UNSUPPORTED_DTYPE(dwte, scratch, workload_indices, bucket_info, dout, inp, inputs_cpu);
    }
}

void global_norm_squared(Tensor& out, const Tensor& values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(values.DType == ETensorDType::FP32) {
        global_norm_squared(out.get<float>(), values.get<float>(), count, dp, stream);
    } else if(values.DType == ETensorDType::BF16) {
        global_norm_squared(out.get<float>(), values.get<nv_bfloat16>(), count, dp, stream);
    } else {
        UNSUPPORTED_DTYPE(out, values);
    }
}


void adamw_update(Tensor& params_memory, const Tensor& grads_memory, Tensor& m_memory, Tensor& v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, Tensor& m_scales, float* abs_max, unsigned int seed, cudaStream_t stream) {
    if (params_memory.nelem() != grads_memory.nelem() || params_memory.nelem() != m_memory.nelem() || params_memory.nelem() != v_memory.nelem()) {
        throw std::invalid_argument("adamw_update: shape mismatch");
    }
    if(params_memory.DType == ETensorDType::FP32) {
        adamw_update(params_memory.get<float>(), grads_memory.get<float>(), m_memory.get<float>(), v_memory.get<float>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
    } else if(params_memory.DType == ETensorDType::BF16 && m_memory.DType == ETensorDType::FP32 && v_memory.DType == ETensorDType::FP32) {
        adamw_update(params_memory.get<nv_bfloat16>(), grads_memory.get<nv_bfloat16>(), m_memory.get<float>(), v_memory.get<float>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
    } else if(params_memory.DType == ETensorDType::BF16 && m_memory.DType == ETensorDType::BF16 && v_memory.DType == ETensorDType::FP32) {
        adamw_update(params_memory.get<nv_bfloat16>(), grads_memory.get<nv_bfloat16>(), m_memory.get<nv_bfloat16>(), v_memory.get<float>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
    } else if(params_memory.DType == ETensorDType::BF16 && m_memory.DType == ETensorDType::BF16 && v_memory.DType == ETensorDType::BF16) {
        adamw_update(params_memory.get<nv_bfloat16>(), grads_memory.get<nv_bfloat16>(), m_memory.get<nv_bfloat16>(), v_memory.get<nv_bfloat16>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
    }  else if(params_memory.DType == ETensorDType::BF16 && m_memory.DType == ETensorDType::FP8_E4M3 && v_memory.DType == ETensorDType::BF16) {
        adamw_update(params_memory.get<nv_bfloat16>(), grads_memory.get<nv_bfloat16>(), m_memory.get<__nv_fp8_e4m3>(), v_memory.get<nv_bfloat16>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, m_scales.get<float>(), abs_max, seed, stream);
    } else {
        UNSUPPORTED_DTYPE(params_memory, grads_memory, m_memory, v_memory);
    }
}

void transpose(Tensor& dst, const Tensor& src, int rows, int cols, cudaStream_t stream) {
    if(dst.DType == ETensorDType::FP32) {
        transpose(dst.get<float>(), src.get<float>(), rows, cols, stream);
    } else if(dst.DType == ETensorDType::BF16) {
        transpose(dst.get<nv_bfloat16>(), src.get<nv_bfloat16>(), rows, cols, stream);
    } else if(dst.DType == ETensorDType::FP8_E4M3) {
        transpose(dst.get<__nv_fp8_e4m3>(), src.get<__nv_fp8_e4m3>(), rows, cols, stream);
    }  else if(dst.DType == ETensorDType::FP8_E5M2) {
        transpose(dst.get<__nv_fp8_e5m2>(), src.get<__nv_fp8_e5m2>(), rows, cols, stream);
    } else {
        UNSUPPORTED_DTYPE(dst, src);
    }
}

void vector_add_sr(Tensor& dest, const Tensor& left, const Tensor& right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    if(dest.DType == ETensorDType::FP32) {
        vector_add_sr(dest.get<float>(), left.get<float>(), right.get<float>(), scale, nelem, seed, stream);
    } else if(dest.DType == ETensorDType::BF16) {
        vector_add_sr(dest.get<nv_bfloat16>(), left.get<nv_bfloat16>(), right.get<nv_bfloat16>(), scale, nelem, seed, stream);
    } else {
        UNSUPPORTED_DTYPE(dest, left, right);
    }
}

void vector_reduce_sr(Tensor& dest, const Tensor& src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream) {
    if(dest.DType == ETensorDType::FP32) {
        vector_reduce_sr(dest.get<float>(), src.get<float>(), scale, n_shards, skip, nelem, accumulate, seed, stream);
    } else if(dest.DType == ETensorDType::BF16) {
        vector_reduce_sr(dest.get<nv_bfloat16>(), src.get<nv_bfloat16>(), scale, n_shards, skip, nelem, accumulate, seed, stream);
    } else {
        UNSUPPORTED_DTYPE(dest, src);
    }
}

void abs_max(float* scale, const Tensor& in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (in.DType == ETensorDType::FP32) {
        abs_max(scale, in.get<float>(), N, dp, stream);
    } else if (in.DType == ETensorDType::BF16) {
        abs_max(scale, in.get<nv_bfloat16>(), N, dp, stream);
    } else {
        UNSUPPORTED_DTYPE(in);
    }
}

void quantize_with_abs_max(Tensor& out, float* scale_ptr, const Tensor& in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (in.DType == ETensorDType::FP32) {
        if (out.DType == ETensorDType::BF16) {
            quantize_with_abs_max(out.get<nv_bfloat16>(), scale_ptr, in.get<float>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::FP8_E4M3) {
            quantize_with_abs_max(out.get<__nv_fp8_e4m3>(), scale_ptr, in.get<float>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::FP8_E5M2) {
            quantize_with_abs_max(out.get<__nv_fp8_e5m2>(), scale_ptr, in.get<float>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::INT8) {
            quantize_with_abs_max(out.get<int8_t>(), scale_ptr, in.get<float>(), abs_max, N, dp, stream);
        } else {
            UNSUPPORTED_DTYPE(out, in);
        }
    } else if (in.DType == ETensorDType::BF16) {
        if (out.DType == ETensorDType::FP8_E4M3) {
            quantize_with_abs_max(out.get<__nv_fp8_e4m3>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::FP8_E5M2) {
            quantize_with_abs_max(out.get<__nv_fp8_e5m2>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::INT8) {
            quantize_with_abs_max(out.get<int8_t>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, N, dp, stream);
        } else {
            UNSUPPORTED_DTYPE(out, in);
        }
    } else {
        UNSUPPORTED_DTYPE(out, in);
    }
}

void quantize_and_transpose_with_abs_max(Tensor& out, float* scale_ptr, const Tensor& in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(out.DType == ETensorDType::BF16 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<nv_bfloat16>(), scale_ptr, in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::INT8 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<std::int8_t>(), scale_ptr, in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::INT8 && in.DType == ETensorDType::BF16) {
        quantize_and_transpose_with_abs_max(out.get<std::int8_t>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::FP8_E4M3 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<__nv_fp8_e4m3>(), scale_ptr, in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::FP8_E4M3 && in.DType == ETensorDType::BF16) {
        quantize_and_transpose_with_abs_max(out.get<__nv_fp8_e4m3>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, rows, cols, dp, stream);
    } else {
        UNSUPPORTED_DTYPE(out, in);
    }
}

void fill_normal(Tensor& dest, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream) {
    if (dest.DType == ETensorDType::FP32) {
        fill_normal(dest.get<float>(), count, mean, std, seed, subsequence, stream);
    } else if (dest.DType == ETensorDType::BF16) {
        fill_normal(dest.get<nv_bfloat16>(), count, mean, std, seed, subsequence, stream);
    } else {
        UNSUPPORTED_DTYPE(dest);
    }
}

void fill_constant(Tensor& dest, float value, std::size_t count, cudaStream_t stream) {
    if (dest.DType == ETensorDType::FP32) {
        fill_constant(dest.get<float>(), value, count, stream);
    } else if (dest.DType == ETensorDType::BF16) {
        fill_constant(dest.get<nv_bfloat16>(), static_cast<nv_bfloat16>(value), count, stream);
    } else {
        UNSUPPORTED_DTYPE(dest);
    }
}

void matmul(Tensor& c, const Tensor& a, const Tensor& b, const Tensor& bias,
            const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, Tensor& workspace,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream, EMatmulBackend backend) {
    std::byte* ws = workspace.get<std::byte>();
    std::size_t ws_size = workspace.bytes();
    if(c.DType == ETensorDType::FP32 && a.DType == ETensorDType::FP32) {
        const float* bias_ptr = bias.get_optional<float>();
        matmul(c.get<float>(), a.get<float>(), b.get<float>(), bias_ptr,
            scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream, backend);
    } else if(c.DType == ETensorDType::FP32 && a.DType == ETensorDType::BF16) {
        const float* bias_ptr = bias.get_optional<float>();
        matmul(c.get<float>(), a.get<nv_bfloat16>(), b.get<nv_bfloat16>(), bias_ptr,
            scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream, backend);
    } else if(c.DType == ETensorDType::FP32 && a.DType == ETensorDType::FP8_E4M3) {
        if(!bias.empty()) {
            if(bias.DType == ETensorDType::BF16) {
                matmul(c.get<float>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e4m3>(), bias.get<nv_bfloat16>(),
                    scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream, backend);
            } else {
                matmul(c.get<float>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e4m3>(), bias.get<float>(),
                    scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream, backend);
            }
        } else {
            matmul(c.get<float>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e4m3>(), (nv_bfloat16*)nullptr,
                scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream, backend);
        }
    } else if(c.DType == ETensorDType::BF16 && a.DType == ETensorDType::FP8_E4M3 && b.DType == ETensorDType::FP8_E4M3) {
        const nv_bfloat16* bias_ptr = bias.get_optional<nv_bfloat16>();
        matmul(c.get<nv_bfloat16>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e4m3>(), bias_ptr,
            scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream, backend);
    } else if(c.DType == ETensorDType::BF16 && a.DType == ETensorDType::FP8_E4M3 && b.DType == ETensorDType::FP8_E5M2) {
        const nv_bfloat16* bias_ptr = bias.get_optional<nv_bfloat16>();
        matmul(c.get<nv_bfloat16>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e5m2>(), bias_ptr,
            scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream, backend);
    } else if(c.DType == ETensorDType::BF16) {
        const nv_bfloat16* bias_ptr = bias.get_optional<nv_bfloat16>();
        matmul(c.get<nv_bfloat16>(), a.get<nv_bfloat16>(), b.get<nv_bfloat16>(), bias_ptr,
            scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream, backend);
    } else {
        UNSUPPORTED_DTYPE(c, a, b, bias, workspace);
    }
}

void backward_bias(Tensor& dbias, const Tensor& dout, const float* scale_a, const float* scale_b, Tensor& dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    std::size_t expected_buffer_size =  get_bias_backward_scratch_size(dbias.DType, OC, dp);
    if (dbias_buffer.bytes() < expected_buffer_size) {
        throw std::invalid_argument(fmt::format("scratch buffer too small: expected at least {} bytes, got {}", expected_buffer_size, dbias_buffer.bytes()));
    }
    if(dbias.DType == ETensorDType::FP32 && dout.DType == ETensorDType::FP32) {
        backward_bias(dbias.get<float>(), dout.get<float>(), scale_a, scale_b, dbias_buffer.get<float>(), B, T, OC, dp, stream);
    } else if(dbias.DType == ETensorDType::BF16 && dout.DType == ETensorDType::BF16) {
        backward_bias(dbias.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), scale_a, scale_b, dbias_buffer.get<float>(), B, T, OC, dp, stream);
    } else if(dbias.DType == ETensorDType::BF16 && dout.DType == ETensorDType::FP8_E4M3) {
        backward_bias(dbias.get<nv_bfloat16>(), dout.get<__nv_fp8_e4m3>(), scale_a, scale_b, dbias_buffer.get<float>(), B, T, OC, dp, stream);
    }  else if(dbias.DType == ETensorDType::BF16 && dout.DType == ETensorDType::FP8_E5M2) {
        backward_bias(dbias.get<nv_bfloat16>(), dout.get<__nv_fp8_e5m2>(), scale_a, scale_b, dbias_buffer.get<float>(), B, T, OC, dp, stream);
    } else {
        UNSUPPORTED_DTYPE(dbias, dout, dbias_buffer);
    }
}
