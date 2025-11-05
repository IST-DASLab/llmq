// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cuda/atomic>
#include <algorithm>

#include "squirrel_noise.cuh"
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

template <typename floatX, typename floatM, typename floatV>
__device__ auto adamw_update(floatX* params_memory, const floatX* grads_memory, floatM* m_memory, floatV* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed, long idx) {
    constexpr int VecElems = std::max({16 / sizeof(floatX), 16 / sizeof(floatM), 16 / sizeof(floatV)});
    using vec_x_t = GenericVector<floatX, VecElems>;
    using vec_m_t = GenericVector<floatM, VecElems>;
    using vec_v_t = GenericVector<floatV, VecElems>;

    if (idx >= num_parameters) { return vec_x_t::zeros(); }  // guard

    vec_m_t m = vec_m_t::load(m_memory + idx);
    vec_v_t v = vec_v_t::load(v_memory + idx);
    vec_x_t g = vec_x_t::load(grads_memory + idx);
    vec_x_t p = vec_x_t::load(params_memory + idx);

    vec_x_t p_new;
    vec_m_t m_new;
    vec_v_t v_new;

    for (int i = 0; i < VecElems; i++) {
        float grad = grad_scale * (float)g[i];
        float m_i = (float)m[i];
        float v_i = (float)v[i];
        m_i = fast_lerp(grad, m_i, beta1);
        v_i = fast_lerp(grad * grad, v_i, beta2);

        // random number generation (reuse same rng shifted, since 32 bits is overkill for FP32->BF16)
        // note this all gets optimised away by the compiler if everything is FP32
        unsigned int random = get_noise_2d(idx + i, blockIdx.y, seed);
        unsigned int random_m = __funnelshift_l(random, random, 10); // rotate by 10 bits
        unsigned int random_v = __funnelshift_l(random, random, 20); // rotate by 20 bits

        stochastic_rounding(m_i, &m_new[i], random_m, false);
        stochastic_rounding(v_i, &v_new[i], random_v, false);

        float m_hat = m_i / beta1_correction;
        float v_hat = v_i / beta2_correction;
        float old_param = (float)p[i];
        float param = old_param - (learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * old_param));
        stochastic_rounding(param, &p_new[i], random, false);
    }

    p_new.store(params_memory + idx);
    m_new.store(m_memory + idx);
    v_new.store(v_memory + idx);

    return p_new;
}

template <typename floatX, typename floatM, typename floatV>
__global__ void adamw_kernel(floatX* params_memory, const floatX* grads_memory, floatM* m_memory, floatV* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             const float* grad_scale, float* abs_max, unsigned int seed) {

    constexpr int VecElems = std::max({16 / sizeof(floatX), 16 / sizeof(floatM), 16 / sizeof(floatV)});
    using vec_x_t = GenericVector<floatX, VecElems>;
    __shared__ float local_max;
    if(threadIdx.x == 0) {
        local_max = 1e-10f;
    }

    float abs_max_local = 0.0f;
    vec_x_t p_new = adamw_update(params_memory,
                 grads_memory,
                 m_memory,
                 v_memory,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, *grad_scale,
                 seed,
                 VecElems * (blockIdx.x * blockDim.x + threadIdx.x));
    for (int i = 0; i < VecElems; i++) {
        abs_max_local = std::max(abs_max_local, (float)p_new[i]);
    }

    __syncthreads();
    atomicMax_block(reinterpret_cast<unsigned int*>(&local_max), *reinterpret_cast<unsigned int*>(&abs_max_local));
    __syncthreads();
    if(threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(abs_max), *reinterpret_cast<unsigned int*>(&local_max));
    }
}

template <typename floatX, typename floatM, typename floatV>
void adamw_update_imp(floatX* params_memory, const floatX* grads_memory, floatM* m_memory, floatV* v_memory, size_t num_parameters,
                      float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                      const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    constexpr int VecElems = std::max({16 / sizeof(floatX), 16 / sizeof(floatM), 16 / sizeof(floatV)});
    // AdamW update
    int block_size = 512;
    int num_blocks = div_ceil(num_parameters, (size_t)(block_size * VecElems));
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel<<<num_blocks, block_size, 0, stream>>>(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                                                        learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                        grad_scale, abs_max, seed);
    CUDA_CHECK(cudaGetLastError());
}

void adamw_update(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
}

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
}

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, nv_bfloat16* m_memory, float* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
}

void adamw_update(nv_bfloat16* params_memory, const nv_bfloat16* grads_memory, nv_bfloat16* m_memory, nv_bfloat16* v_memory, size_t num_parameters,
                  float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  const float* grad_scale, float* abs_max, unsigned int seed, cudaStream_t stream) {
    adamw_update_imp(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, abs_max, seed, stream);
}
