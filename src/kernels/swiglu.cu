// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//
// SPDX-License-Identifier: MIT
// Based on llm.c https://github.com/karpathy/llm.c

#include <cassert>

#include <cuda_bf16.h>

#include "kernels.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template<typename floatX>
__global__ void swiglu_forward_kernel1(floatX* out, const floatX* inp, float* abs_max_ptr, int B, int T, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    floatX* out_ptr = out + idx;
    // b,t,c in the output
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;

    const floatX* up_ptr = inp + (b * T * C * 2 + t * C * 2 + c);
    const floatX* gate_ptr = up_ptr + C;

    __shared__ float block_max;
    // only handle abs-max if requested; these are guaranteed to be warp-convergent branches,
    // so they don't cost us in this memory-bound kernel.
    if (abs_max_ptr) {
        if(threadIdx.x == 0) {
            block_max = 1e-10f;
        }
        __syncthreads();
    }
    float thread_max = 0.f;

    x128 packed_out;
    x128 up_inp = x128::load_cs(up_ptr);
    x128 gate_inp = x128::load_cs(gate_ptr);
    for(int k = 0; k < up_inp.size; ++k) {
        float x1 = (float)up_inp[k];
        float x2 = (float)gate_inp[k];
        packed_out[k] = (floatX)((x1 * x2) / (1.0f + expf(-x2)));
        if (abs_max_ptr) {
            thread_max = std::max(thread_max, fabsf(packed_out[k]));
        }
    }
    packed_out.store(out_ptr);

    // if we requested an absmax, do the block-wise and global reduction
    if (abs_max_ptr) {
        atomicMax_block(reinterpret_cast<unsigned int*>(&block_max), __float_as_uint(thread_max));
        __syncthreads();
        if(threadIdx.x == 0) {
            atomicMax(reinterpret_cast<unsigned int*>(abs_max_ptr), __float_as_uint(block_max));
        }
    }
}

template<typename floatX>
__global__ void swiglu_forward_quant_kernel(__nv_fp8_e4m3* out, const floatX* inp, const float* abs_max_ptr, int B, int T, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f8v_t = GenericVector<__nv_fp8_e4m3, 16 / sizeof(floatX)>;

    float scale = 448.f / *abs_max_ptr;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    __nv_fp8_e4m3* out_ptr = out + idx;
    // b,t,c in the output
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;

    const floatX* up_ptr = inp + (b * T * C * 2 + t * C * 2 + c);
    const floatX* gate_ptr = up_ptr + C;

    f8v_t packed_out;
    x128 up_inp = x128::load_cs(up_ptr);
    x128 gate_inp = x128::load_cs(gate_ptr);
    for(int k = 0; k < up_inp.size; ++k) {
        float x1 = (float)up_inp[k];
        float x2 = (float)gate_inp[k];
        float result = (x1 * x2) / (1.0f + expf(-x2));
        floatX qr = (floatX)result;
        __nv_fp8_e4m3 quant;
        quant.__x = __nv_cvt_float_to_fp8(scale * (float)qr, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        packed_out[k] = quant;
    }
    packed_out.store(out_ptr);
}

template<typename floatX>
__global__ void swiglu_backward_kernel1(floatX* dinp, const floatX* dout, const floatX* inp, float* abs_max_ptr, int B, int T, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    const floatX* dout_ptr = dout + idx;
    // b,t,c in the output
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;
    // coords in input
    int C2 = C * 2;
    const floatX* inp1_ptr = inp + (b * T * C2 + t * C2 + c);
    const floatX* inp2_ptr = inp1_ptr + C;
    floatX* dinp1_ptr = dinp + (b * T * C2 + t * C2 + c);
    floatX* dinp2_ptr = dinp1_ptr + C;
    // backward
    x128 dinp1;
    x128 dinp2;
    x128 packed_dout = x128::load_cs(dout_ptr);
    x128 packed_inp1 = x128::load_cs(inp1_ptr); // fc1
    x128 packed_inp2 = x128::load_cs(inp2_ptr); // fc2

    __shared__ float block_max;
    // only handle abs-max if requested; these are guaranteed to be warp-convergent branches,
    // so they don't cost us in this memory-bound kernel.
    if (abs_max_ptr) {
        if(threadIdx.x == 0) {
            block_max = 1e-10f;
        }
        __syncthreads();
    }

    float thread_max = 0.f;

    for(int k = 0; k < packed_inp1.size; ++k) {
        float x1 = (float)packed_inp1[k];
        float x2 = (float)packed_inp2[k];
        float dout = (float)packed_dout[k];

        float sx2 = 1.0f / (1.0f + expf(-x2)); // sigmoid of x2
        float dx1 = dout * x2 * sx2;
        float dx2 = dout * x1 * sx2 * (1.0f + x2 * (1.0f - sx2));

        dinp1[k] = (floatX)dx1;
        dinp2[k] = (floatX)dx2;

        if (abs_max_ptr) {
            thread_max = std::max(thread_max, fabsf(dinp1[k]));
            thread_max = std::max(thread_max, fabsf(dinp2[k]));
        }
    }
    dinp1.store(dinp1_ptr);
    dinp2.store(dinp2_ptr);

    // if we requested an absmax, do the block-wise and global reduction
    if (abs_max_ptr) {
        atomicMax_block(reinterpret_cast<unsigned int*>(&block_max), __float_as_uint(thread_max));
        __syncthreads();
        if(threadIdx.x == 0) {
            atomicMax(reinterpret_cast<unsigned int*>(abs_max_ptr), __float_as_uint(block_max));
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

template<typename floatX>
void swiglu_forward_impl(floatX* out, const floatX* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    // input is (B, T, 2C), output is (B, T, C)
    // we have that inp[b, t, :] = [fc1, fc2] (i.e. they are concatenated in each C-fiber)
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    if (abs_max_ptr)
        CUDA_CHECK(cudaMemsetAsync(abs_max_ptr, 0, sizeof(float), stream));

    const int block_size = 128;
    assert(C % x128::size == 0);
    assert((B*T*C) % (block_size * x128::size) == 0);
    const int grid_size = div_ceil(B*T*C, (int)(block_size * x128::size));
    swiglu_forward_kernel1<<<grid_size, block_size, 0, stream>>>(out, inp, abs_max_ptr, B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_forward(nv_bfloat16* out, const nv_bfloat16* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    swiglu_forward_impl(out, inp, abs_max_ptr, B, T, C, stream);
}

void swiglu_forward(float* out, const float* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    swiglu_forward_impl(out, inp, abs_max_ptr, B, T, C, stream);
}

void swiglu_forward_quant(__nv_fp8_e4m3* out, const nv_bfloat16* inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    using x128 = GenericVector<nv_bfloat16, 16/sizeof(nv_bfloat16)>;
    const int block_size = 128;
    assert(C % x128::size == 0);
    assert((B*T*C) % (block_size * x128::size) == 0);
    const int grid_size = div_ceil(B*T*C, (int)(block_size * x128::size));
    swiglu_forward_quant_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, abs_max_ptr, B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

template<typename floatX>
void swiglu_backward_impl(floatX* dinp, const floatX* dout, const floatX* inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    // input is (B, T, 2C), output is (B, T, C)
    // we have that inp[b, t, :] = [fc1, fc2] (i.e. they are concatenated in each C-fiber)

    NVTX_RANGE_FN();

    if (abs_max)
        CUDA_CHECK(cudaMemsetAsync(abs_max, 0, sizeof(float), stream));

    const int block_size = 256;
    assert((B*T*C) % (block_size * x128::size) == 0);
    const int grid_size = div_ceil((size_t)B*T*C, block_size * x128::size);
    swiglu_backward_kernel1<<<grid_size, block_size, 0, stream>>>(dinp, dout, inp, abs_max, B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16* inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    swiglu_backward_impl(dinp, dout, inp, abs_max, B, T, C, stream);
}

void swiglu_backward(float* dinp, const float* dout, const float* inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    swiglu_backward_impl(dinp, dout, inp, abs_max, B, T, C, stream);
}
