// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//

#include "kernels.h"

#include "utilities/tensor.h"

void rmsnorm_forward(Tensor& out, Tensor& rms, const Tensor& inp, const Tensor& weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::BF16) {
        rmsnorm_forward(out.get<nv_bfloat16>(), rms.get<float>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), abs_max_ptr, epsilon, B, T, C, stream);
    } else if (out.DType == ETensorDType::FP32) {
        rmsnorm_forward(out.get<float>(), rms.get<float>(), inp.get<float>(), weight.get<float>(), abs_max_ptr, epsilon, B, T, C, stream);
    } else {
        throw std::logic_error("rmsnorm_forward: unsupported dtype");
    }
}

void rmsnorm_backward(Tensor& dinp, Tensor& dweight, Tensor& scratch, const Tensor& dresidual, const Tensor& dout, const Tensor& inp, const Tensor& weight, const Tensor& rstd,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::BF16) {
        rmsnorm_backward(dinp.get<nv_bfloat16>(), dweight.get<nv_bfloat16>(), scratch.Data, dresidual.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), rstd.get<float>(), B, T, C, dp, stream);
    } else if (dinp.DType == ETensorDType::FP32) {
        rmsnorm_backward(dinp.get<float>(), dweight.get<float>(), scratch.Data, dresidual.get<float>(), dout.get<float>(), inp.get<float>(), weight.get<float>(), rstd.get<float>(), B, T, C, dp, stream);
    } else {
        throw std::logic_error("rmsnorm_backward: unsupported dtype");
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
        throw std::logic_error("fused_residual_rmsnorm_forward: unsupported dtype");
    }
}

void swiglu_forward(Tensor& out, const Tensor& inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP32) {
        swiglu_forward(out.get<float>(), inp.get<float>(), abs_max_ptr, B, T, C, stream);
    } else if (out.DType == ETensorDType::BF16) {
        swiglu_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), abs_max_ptr, B, T, C, stream);
    } else {
        throw std::logic_error("swiglu_forward: unsupported dtype");
    }
}

void swiglu_forward_quant(Tensor& out, const Tensor& inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP8_E4M3 && inp.DType == ETensorDType::BF16) {
        swiglu_forward_quant(out.get<__nv_fp8_e4m3>(), inp.get<nv_bfloat16>(), abs_max_ptr, B, T, C, stream);
    } else {
        throw std::logic_error("swiglu_forward_quant: unsupported dtype");
    }
}

void swiglu_backward(Tensor& dinp, const Tensor& dout, const Tensor& inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::FP32) {
        swiglu_backward(dinp.get<float>(), dout.get<float>(), inp.get<float>(), abs_max, B, T, C, stream);
    } else if (dinp.DType == ETensorDType::BF16) {
        swiglu_backward(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), abs_max, B, T, C, stream);
    } else {
        throw std::logic_error("swiglu_backward: unsupported dtype");
    }
}

void rope_forward(Tensor& out, const Tensor& in, const Tensor& freqs_cis, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream)  {
    if(out.DType == ETensorDType::FP32) {
        rope_forward(out.get<float>(), in.get<float>(), freqs_cis.get<float>(), B, T, Nq, Nkv, head_dim, stream);
    } else if(out.DType == ETensorDType::BF16) {
        rope_forward(out.get<nv_bfloat16>(), in.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(), B, T, Nq, Nkv, head_dim, stream);
    } else {
        throw std::logic_error("rope_forward: unsupported dtype");
    }
}

void rope_backward(Tensor& dinp, const Tensor& dout, const Tensor& freqs_cis, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::FP32) {
        rope_backward(dinp.get<float>(), dout.get<float>(), freqs_cis.get<float>(), B, T, Nq, Nkv, head_dim, stream);
    } else if(dinp.DType == ETensorDType::BF16) {
        rope_backward(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(), B, T, Nq, Nkv, head_dim, stream);
    } else {
        throw std::logic_error("rope_backward: unsupported dtype");
    }
}

void fused_classifier(Tensor& logits, Tensor& losses,
                      float dloss, const Tensor& targets,
                      int B, int T, int V, int P, bool write_dlogits, cudaStream_t stream) {
    if(logits.DType == ETensorDType::FP32) {
        fused_classifier(logits.get<float>(), losses.get<float>(), dloss, targets.get<int>(), B, T, V, P, write_dlogits, stream);
    } else if(logits.DType == ETensorDType::BF16) {
        fused_classifier(logits.get<nv_bfloat16>(), losses.get<float>(), dloss, targets.get<int>(), B, T, V, P, write_dlogits, stream);
    } else {
        throw std::runtime_error("fused_classifier: unsupported dtype");
    }
}

void encoder_forward(Tensor& out, const Tensor& inp, const Tensor& wte, std::optional<Tensor> wpe, int B, int T, int C, int V, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP32) {
        encoder_forward(out.get<float>(), inp.get<std::int32_t>(), wte.get<float>(), wpe.has_value() ? wpe->get<float>() : nullptr, B, T, C, V, stream);
    } else if(out.DType == ETensorDType::BF16) {
        encoder_forward(out.get<nv_bfloat16>(), inp.get<std::int32_t>(), wte.get<nv_bfloat16>(),  wpe.has_value() ? wpe->get<nv_bfloat16>() : nullptr, B, T, C, V, stream);
    } else {
        throw std::runtime_error("encoder_forward: unsupported dtype");
    }
}

void encoder_backward(Tensor& dwte, Tensor& scratch,
                      Tensor& workload_indices, Tensor& bucket_info,
                      const Tensor& dout, const Tensor& inp, const Tensor& inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    assert(workload_indices.Device == -1);
    assert(bucket_info.Device == -1);
    if(dwte.DType == ETensorDType::FP32) {
        encoder_backward(dwte.get<float>(), scratch.get<int>(), workload_indices.get<int>(), (int4*)bucket_info.get<int>(), dout.get<float>(), inp.get<std::int32_t>(), inputs_cpu.get<std::int32_t>(), B, T, C, seed, stream);
    } else if(dwte.DType == ETensorDType::BF16) {
        encoder_backward(dwte.get<nv_bfloat16>(), scratch.get<int>(), workload_indices.get<int>(), (int4*)bucket_info.get<int>(), dout.get<nv_bfloat16>(), inp.get<std::int32_t>(), inputs_cpu.get<std::int32_t>(), B, T, C, seed, stream);
    } else {
        throw std::logic_error("encoder_backward: unsupported dtype");
    }
}

void global_norm_squared(Tensor& out, const Tensor& values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(values.DType == ETensorDType::FP32) {
        global_norm_squared(out.get<float>(), values.get<float>(), count, dp, stream);
    } else if(values.DType == ETensorDType::BF16) {
        global_norm_squared(out.get<float>(), values.get<nv_bfloat16>(), count, dp, stream);
    } else {
        throw std::logic_error("global_norm_squared: unsupported dtype");
    }
}


void adamw_update(Tensor& params_memory, const Tensor& grads_memory, Tensor& m_memory, Tensor& v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    if (params_memory.nelem() != grads_memory.nelem() || params_memory.nelem() != m_memory.nelem() || params_memory.nelem() != v_memory.nelem()) {
        throw std::runtime_error("adamw_update: shape mismatch");
    }
    if(params_memory.DType == ETensorDType::FP32) {
        adamw_update(params_memory.get<float>(), grads_memory.get<float>(), m_memory.get<float>(), v_memory.get<float>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
    } else if(params_memory.DType == ETensorDType::BF16 && m_memory.DType == ETensorDType::FP32 && v_memory.DType == ETensorDType::FP32) {
        adamw_update(params_memory.get<nv_bfloat16>(), grads_memory.get<nv_bfloat16>(), m_memory.get<float>(), v_memory.get<float>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
    } else if(params_memory.DType == ETensorDType::BF16 && m_memory.DType == ETensorDType::BF16 && v_memory.DType == ETensorDType::FP32) {
        adamw_update(params_memory.get<nv_bfloat16>(), grads_memory.get<nv_bfloat16>(), m_memory.get<nv_bfloat16>(), v_memory.get<float>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
    } else if(params_memory.DType == ETensorDType::BF16 && m_memory.DType == ETensorDType::BF16 && v_memory.DType == ETensorDType::BF16) {
        adamw_update(params_memory.get<nv_bfloat16>(), grads_memory.get<nv_bfloat16>(), m_memory.get<nv_bfloat16>(), v_memory.get<nv_bfloat16>(), num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
    } else {
        throw std::logic_error("adamw_update: unsupported dtype");
    }
}

void transpose(Tensor& dst, const Tensor& src, int rows, int cols, cudaStream_t stream) {
    if(dst.DType == ETensorDType::FP32) {
        transpose(dst.get<float>(), src.get<float>(), rows, cols, stream);
    } else if(dst.DType == ETensorDType::BF16) {
        transpose(dst.get<nv_bfloat16>(), src.get<nv_bfloat16>(), rows, cols, stream);
    } else if(dst.DType == ETensorDType::FP8_E4M3) {
        transpose(dst.get<__nv_fp8_e4m3>(), src.get<__nv_fp8_e4m3>(), rows, cols, stream);
    } else {
        throw std::logic_error("transpose: unsupported dtype");
    }
}

void vector_add_sr(Tensor& dest, const Tensor& left, const Tensor& right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    if(dest.DType == ETensorDType::FP32) {
        vector_add_sr(dest.get<float>(), left.get<float>(), right.get<float>(), scale, nelem, seed, stream);
    } else if(dest.DType == ETensorDType::BF16) {
        vector_add_sr(dest.get<nv_bfloat16>(), left.get<nv_bfloat16>(), right.get<nv_bfloat16>(), scale, nelem, seed, stream);
    } else {
        throw std::logic_error("vector_add_sr: unsupported dtype");
    }
}

void abs_max(float* scale, const Tensor& in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (in.DType == ETensorDType::FP32) {
        abs_max(scale, in.get<float>(), N, dp, stream);
    } else if (in.DType == ETensorDType::BF16) {
        abs_max(scale, in.get<nv_bfloat16>(), N, dp, stream);
    } else {
        throw std::logic_error("absmax_scale: unsupported dtype");
    }
}

void quantize_with_abs_max(Tensor& out, const Tensor& in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (in.DType == ETensorDType::FP32) {
        if (out.DType == ETensorDType::BF16) {
            quantize_with_abs_max(out.get<nv_bfloat16>(), in.get<float>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::FP8_E4M3) {
            quantize_with_abs_max(out.get<__nv_fp8_e4m3>(), in.get<float>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::INT8) {
            quantize_with_abs_max(out.get<int8_t>(), in.get<float>(), abs_max, N, dp, stream);
        } else {
            throw std::logic_error("quantize_with_abs_max: unsupported dtype");
        }
    } else if (in.DType == ETensorDType::BF16) {
        if (out.DType == ETensorDType::FP8_E4M3) {
            quantize_with_abs_max(out.get<__nv_fp8_e4m3>(), in.get<nv_bfloat16>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::INT8) {
            quantize_with_abs_max(out.get<int8_t>(), in.get<nv_bfloat16>(), abs_max, N, dp, stream);
        } else {
            throw std::logic_error("quantize_with_abs_max: unsupported dtype");
        }
    } else {
        throw std::logic_error("quantize_with_abs_max: unsupported dtype");
    }
}

void quantize_and_transpose_with_abs_max(Tensor& out, const Tensor& in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(out.DType == ETensorDType::BF16 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<nv_bfloat16>(), in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::INT8 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<std::int8_t>(), in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::INT8 && in.DType == ETensorDType::BF16) {
        quantize_and_transpose_with_abs_max(out.get<std::int8_t>(), in.get<nv_bfloat16>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::FP8_E4M3 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<__nv_fp8_e4m3>(), in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::FP8_E4M3 && in.DType == ETensorDType::BF16) {
        quantize_and_transpose_with_abs_max(out.get<__nv_fp8_e4m3>(), in.get<nv_bfloat16>(), abs_max, rows, cols, dp, stream);
    } else {
        throw std::logic_error("Invalid DType combination");
    }
}

void fill_normal(Tensor& dest, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream) {
    if (dest.DType == ETensorDType::FP32) {
        fill_normal(dest.get<float>(), count, mean, std, seed, subsequence, stream);
    } else if (dest.DType == ETensorDType::BF16) {
        fill_normal(dest.get<nv_bfloat16>(), count, mean, std, seed, subsequence, stream);
    } else {
        throw std::logic_error("fill_normal: unsupported dtype");
    }
}

void fill_constant(Tensor& dest, float value, std::size_t count, cudaStream_t stream) {
    if (dest.DType == ETensorDType::FP32) {
        fill_constant(dest.get<float>(), value, count, stream);
    } else if (dest.DType == ETensorDType::BF16) {
        fill_constant(dest.get<nv_bfloat16>(), value, count, stream);
    } else {
        throw std::logic_error("fill_constant: unsupported dtype");
    }
}
