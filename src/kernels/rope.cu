// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cuda_bf16.h>

#include <cassert>

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

template<typename floatX>
void precompute_freqs_cis_imp(floatX *freqs_cis, int dim, int end, float theta) {
    // helper function that (on the CPU!) precomputes the freqs_cis for the RoPE rotation
    // same as precompute_freqs_cis_real in rope.py
    for (int i = 0; i < dim / 2; i++) {

        // calculate the frequency for the (i, i+1)th dimension
        float inv_freq = 1.0f / powf(theta, (float)(2 * i) / dim);

        // iterate over all time steps, calculate the angle, and store the cos/sin
        for (int t = 0; t < end; t++) {
            float angle = (float)t * inv_freq;
            freqs_cis[t * dim + 2 * i] = (floatX)cosf(angle);     // real part
            freqs_cis[t * dim + 2 * i + 1] = (floatX)sinf(angle); // imaginary part
        }
    }
}

void precompute_freqs_cis(float *freqs_cis, int dim, int end, float theta) {
    return precompute_freqs_cis_imp(freqs_cis, dim, end, theta);
}

void precompute_freqs_cis(nv_bfloat16 *freqs_cis, int dim, int end, float theta) {
    return precompute_freqs_cis_imp(freqs_cis, dim, end, theta);
}

template<bool Backward, typename floatX>
__global__ void rope_kernel(floatX *out, const floatX *inp, const floatX *freqs_cis, float* abs_max_ptr,
                            int B, int T, int Nq, int Nkv, int head_dim, std::bool_constant<Backward> bw = {}) {
    using x64 = GenericVector<floatX, 8/sizeof(floatX)>;
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    __shared__ float block_abs_max;
    if (abs_max_ptr) {
        if(threadIdx.x == 0)
            block_abs_max = 1e-10f;
        __syncthreads();
    }
    float thread_abs_max = 1e-10f;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x64::size;
    int head_dim_half = head_dim / 2;
    int N = Nq + 2*Nkv;
    if (idx >= B * T * N * head_dim_half) return;
    // decode the qkv index early so we can early exit if it's a value index
    int h = (idx / head_dim_half) % N;
    int qkv = 2;
    if(h < Nq) {
        qkv = 0;        // query head
    } else if (h < Nq + Nkv) {
        qkv = 1;        // key head
        h -= Nq;
    }

    if (qkv == 2) {
        if(abs_max_ptr) {
            x128 val = x128::load_cs(inp + 2 * idx);
            for(int k = 0; k < x128::size; k++) {
                thread_abs_max = fmaxf(thread_abs_max, fabsf(val[k]));
            }
            if (out != inp) {
                val.store(out + 2 * idx);
            }
            // this thread is not participating in the block-level reduction, so we need to update
            // globally before exiting
            atomicMax(reinterpret_cast<unsigned int*>(abs_max_ptr), __float_as_uint(thread_abs_max));
            return;
        } else {
            // if not in place, need to copy the value heads
            if (out != inp) {
                x128::load_cs(inp + 2 * idx).store(out + 2 * idx);
            }
            return;
        }
    }
    // decode the individual indices and get the input index
    int b = idx / (T * N * head_dim_half);
    int t = (idx / (N * head_dim_half)) % T;
    int d = idx % head_dim_half;
    int idx_bt = b * (T * N * head_dim) + t * (N * head_dim);
    int idx_bth = idx_bt + qkv * (Nq * head_dim) + h * head_dim;
    int idxi = idx_bth + d; // index in the input

    x128 freqs_vec = x128::load_ldg(freqs_cis + t * head_dim + 2 * d);
    x64 v_real = x64::load(inp + idxi);
    x64 v_imag = x64::load(inp + idxi + head_dim_half);
    x64 o_real;
    x64 o_imag;
    for(int k = 0; k < x64::size; k++) {
        float cos = (float)freqs_vec[2*k];
        float sin = (float)freqs_vec[2*k+1];
        if constexpr (Backward) {
            sin = -sin;
        }
        float real = (float)v_real[k];
        float imag = (float)v_imag[k];
        o_real[k] = real * cos - imag * sin;
        o_imag[k] = real * sin + imag * cos;
        if (abs_max_ptr) {
            thread_abs_max = fmaxf(thread_abs_max, fabsf(o_real[k]));
            thread_abs_max = fmaxf(thread_abs_max, fabsf(o_imag[k]));
        }
    }
    o_real.store(out + idxi);
    o_imag.store(out + idxi + head_dim_half);

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);
}

template<bool Backward, class floatX>
void rope_imp(floatX* out, const floatX* in, const floatX *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream, std::bool_constant<Backward> bw = {}) {
    // the input and output to this kernel are (B, T, Nq + Nk + Nv, HD)
    if (abs_max_ptr)
        CUDA_CHECK(cudaMemsetAsync(abs_max_ptr, 0, sizeof(float), stream));

    const int block_size = 128;
    using x64 = GenericVector<floatX, 8/sizeof(floatX)>;
    assert(head_dim % (2*x64::size) == 0);
    int total_threads = (B * T * (Nq + 2*Nkv) * head_dim / 2) / x64::size;
    int num_blocks = div_ceil(total_threads, block_size);
    rope_kernel<<<num_blocks, block_size, 0, stream>>>(out, in, freqs_cis, abs_max_ptr, B, T, Nq, Nkv, head_dim, bw);
    CUDA_CHECK(cudaGetLastError());
}

void rope_forward(float* out, const float* in, const float *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    rope_imp(out, in, freqs_cis, abs_max_ptr, B, T, Nq, Nkv, head_dim, stream, std::bool_constant<false>());
}

void rope_forward(nv_bfloat16* out, const nv_bfloat16* in, const nv_bfloat16 *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream)  {
    rope_imp(out, in, freqs_cis,  abs_max_ptr, B, T, Nq, Nkv, head_dim, stream, std::bool_constant<false>());
}

void rope_backward(float* dinp, const float* dout, const float *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    rope_imp(dinp, dout, freqs_cis, abs_max_ptr, B, T, Nq, Nkv, head_dim, stream, std::bool_constant<true>());
}

void rope_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16 *freqs_cis, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream)  {
    rope_imp(dinp, dout, freqs_cis, abs_max_ptr, B, T, Nq, Nkv, head_dim, stream, std::bool_constant<true>());
}
