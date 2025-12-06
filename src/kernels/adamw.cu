// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <algorithm>

#include "squirrel_noise.cuh"
#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
// TODO remove this in favour of std::lerp?
__device__ float fast_lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

template <int VecElems, typename FloatIn>
__device__ GenericVector<float, VecElems> load_vector(const FloatIn* memory, const float* scales, long idx) {
    float factor = 1.f;
    auto in = GenericVector<FloatIn, VecElems>::load(memory + idx);
    GenericVector<float, VecElems> out;
    if(scales != nullptr) {
        factor = scales[idx / 128];
    }
    for (int i = 0; i < VecElems; i++) {
        out[i] = (float)in[i] * factor;
    }

    return out;
}

template <typename FloatOut, std::size_t VecElems>
__device__ void store_vector(FloatOut* memory, const GenericVector<float, VecElems>& in, float* scales, long idx, unsigned active_mask) {
    float factor = 1.f;
    unsigned int rng = get_noise_2d(idx, blockIdx.y, 51245);
    if(scales != nullptr) {
        float abs_max = 0.0f;
        for (int i = 0; i < VecElems; i++) {
            abs_max = std::max(abs_max, fabs((float)in[i]));
        }
        constexpr int Threads = 128 / VecElems;
        static_assert(Threads <= 32, "#threads > warp size");
        for(int i = 1; i < Threads; i *= 2) {
            abs_max = std::max(abs_max, __shfl_xor_sync(active_mask, abs_max, i, Threads));
        }
        if(abs_max > 1e-10f) {
            factor = 448.f / abs_max;
            scales[idx/128] = 1.f / factor;
        } else {
            factor = 1.f;
            scales[idx/128] = 1.f;
        }
    }
    GenericVector<FloatOut, VecElems> out;
    for (int i = 0; i < VecElems; i++) {
        stochastic_rounding(in[i] * factor, &out[i], rng, false);
        rng = __funnelshift_l(rng, rng, 7); // rotate by 7 bits
    }
    out.store(memory + idx);
}

template <typename floatX, typename floatM, typename floatV>
__device__ auto adamw_update(floatX* params_memory, const floatX* grads_memory, floatM* m_memory, floatV* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, float* m_scales, unsigned int seed, long idx) {
    constexpr int VecElems = std::min({16 / sizeof(floatX), 16 / sizeof(floatM), 16 / sizeof(floatV)});
    using vec_f_t = GenericVector<float, VecElems>;
    using vec_x_t = GenericVector<floatX, VecElems>;
    using vec_m_t = GenericVector<floatM, VecElems>;
    using vec_v_t = GenericVector<floatV, VecElems>;

    const unsigned active_mask = __ballot_sync(0xffffffff, idx < num_parameters);
    if (idx >= num_parameters) { return vec_x_t::zeros(); }  // guard

    vec_f_t m = load_vector<VecElems>(m_memory, m_scales, idx);
    vec_v_t v = vec_v_t::load(v_memory + idx);
    vec_x_t g = vec_x_t::load(grads_memory + idx);
    vec_x_t p = vec_x_t::load(params_memory + idx);

    vec_x_t p_new;
    vec_v_t v_new;

    for (int i = 0; i < VecElems; i++) {
        float grad = grad_scale * (float)g[i];
        float v_i = (float)v[i];
        m[i] = fast_lerp(grad, m[i], beta1);
        v_i = fast_lerp(grad * grad, v_i, beta2);

        // random number generation (reuse same rng shifted, since 32 bits is overkill for FP32->BF16)
        // note this all gets optimised away by the compiler if everything is FP32
        unsigned int random = get_noise_2d(idx + i, blockIdx.y, seed);
        unsigned int random_v = __funnelshift_l(random, random, 20); // rotate by 20 bits

        stochastic_rounding(v_i, &v_new[i], random_v, false);

        float m_hat = m[i] / beta1_correction;
        float v_hat = v_i / beta2_correction;
        float old_param = (float)p[i];
        float param = old_param - (learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * old_param));
        stochastic_rounding(param, &p_new[i], random, false);
    }

    p_new.store(params_memory + idx);
    store_vector(m_memory, m, m_scales, idx, active_mask);
    v_new.store(v_memory + idx);

    return p_new;
}

template <typename floatX, typename floatM, typename floatV>
__global__ void adamw_kernel(floatX* params_memory, const floatX* grads_memory, floatM* m_memory, floatV* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             const float* grad_scale, float* m_scales, float* abs_max_ptr, unsigned int seed) {
    constexpr int VecElems = std::min({16 / sizeof(floatX), 16 / sizeof(floatM), 16 / sizeof(floatV)});
    using vec_x_t = GenericVector<floatX, VecElems>;
    __shared__ float block_abs_max;
    if(threadIdx.x == 0) {
        block_abs_max = 0.f;
    }

    float thread_abs_max = 0.0f;
    vec_x_t p_new = adamw_update(params_memory,
                 grads_memory,
                 m_memory,
                 v_memory,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, *grad_scale,
                 m_scales,
                 seed,
                 VecElems * (blockIdx.x * blockDim.x + threadIdx.x));
    for (int i = 0; i < VecElems; i++) {
        thread_abs_max = std::max(thread_abs_max, (float)p_new[i]);
    }

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);
}

template <typename floatX, typename floatM, typename floatV>
void adamw_update_imp(floatX* params_memory, const floatX* grads_memory, floatM* m_memory, floatV* v_memory, size_t num_parameters,
                      float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                      const float* grad_scale, float* m_scales, float* abs_max, unsigned int seed, cudaStream_t stream) {
    constexpr int VecElems = std::min({16 / sizeof(floatX), 16 / sizeof(floatM), 16 / sizeof(floatV)});
    // AdamW update
    int block_size = 512;
    int num_blocks = div_ceil(num_parameters, (size_t)(block_size * VecElems));
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel<<<num_blocks, block_size, 0, stream>>>(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                                                        learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                        grad_scale, m_scales, abs_max, seed);
    CUDA_CHECK(cudaGetLastError());
}

void adamw_update(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, nullptr, abs_max, seed, stream);
}

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, nullptr, abs_max, seed, stream);
}

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, nv_bfloat16* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, nullptr, abs_max, seed, stream);
}

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, nv_bfloat16* m_memory, nv_bfloat16* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, nullptr, abs_max, seed, stream);
}

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, __nv_fp8_e4m3* m_memory, nv_bfloat16* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* m_scales, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, m_scales, abs_max, seed, stream);
}
