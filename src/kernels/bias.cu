// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include "utilities/vec.cuh"
#include "utilities/utils.h"
#include "utilities/dtype.h"

#include <type_traits>
#include <cassert>


template<class floatO, class floatB>
__global__ void add_bias_kernel(floatO* out, const floatB* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

template<class floatO, class floatB>
void add_bias_impl(floatO* out, const floatB* bias, int B, int T, int OC, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = div_ceil(OC * B * T, block_size);
    add_bias_kernel<<<grid_size, block_size, 0, stream>>>(out, bias, B, T, OC);
    CUDA_CHECK(cudaGetLastError());
}

void add_bias(float* out, const float* bias, int B, int T, int OC, cudaStream_t stream) {
    add_bias_impl(out, bias, B, T, OC, stream);
}

void add_bias(nv_bfloat16* out, const nv_bfloat16* bias, int B, int T, int OC, cudaStream_t stream) {
    add_bias_impl(out, bias, B, T, OC, stream);
}

template<typename floatX, typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel(OutFloat* dbias, const floatX* dout, const float* scale_a, const float* scale_b, int B, int T, int OC,
                                            std::bool_constant<UseAuxBuffer>) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;
    constexpr const int bdx = 4;
    constexpr const int bdy = 32 / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    float scale = 1.f;
    if(scale_a != nullptr && scale_b != nullptr) {
        scale = *scale_a * *scale_b;
    }

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = x128::load(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[x128::size][32][bdy];

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a * scale + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a * scale;
            }
        }
    }
}

template<class floatX>
__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = f128::load(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}

int get_bias_backward_scratch_size(ETensorDType dtype, int OC, const cudaDeviceProp& dp) {
    const int block_size = dp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
    const int OC_per_warp = 8 * ( 16 / get_dtype_size(dtype) ); // 64 at BF16
    const int grid_size_x = div_ceil(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
    const int grid_size_y = max(1, block_size * dp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!
    return grid_size_y * OC * sizeof(float);
}

template<class floatX, class FloatY>
void backward_bias_imp(floatX* dbias, const FloatY* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
    // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;

    const int block_size = dp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

    dim3 block_dim = {4, 8, (unsigned)block_size/32};
    const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
    const int grid_size_x = div_ceil(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
    const int grid_size_y = max(1, block_size * dp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

    if( (scale_a == nullptr) != (scale_b == nullptr) ) {
        throw std::logic_error("backward_bias: scale_a and scale_b must be both nullptr or both non-nullptr");
    }

    // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
    // and write results directly to the output.
    if(grid_size_y == 1) {
        matmul_backward_bias_kernel<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, scale_a, scale_b, B, T, OC, std::bool_constant<false>());
        CUDA_CHECK(cudaGetLastError());
    } else {
        // kernel 9 overwrites temp buffer, so no need to memset
        matmul_backward_bias_kernel<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, scale_a, scale_b, B, T, OC, std::bool_constant<true>());
        CUDA_CHECK(cudaGetLastError());
        reduce_add_sum_kernel<<<div_ceil((size_t)OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
        CUDA_CHECK(cudaGetLastError());
    }
}

void backward_bias(float* dbias, const float* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream)  {
    backward_bias_imp(dbias, dout, scale_a, scale_b, dbias_buffer, B, T, OC, dp, stream);
}

void backward_bias(nv_bfloat16* dbias, const nv_bfloat16* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream)  {
    backward_bias_imp(dbias, dout, scale_a, scale_b, dbias_buffer, B, T, OC, dp, stream);
}

void backward_bias(nv_bfloat16* dbias, const __nv_fp8_e4m3* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream)  {
    backward_bias_imp(dbias, dout, scale_a, scale_b, dbias_buffer, B, T, OC, dp, stream);
}

void backward_bias(nv_bfloat16* dbias, const __nv_fp8_e5m2* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream)  {
    backward_bias_imp(dbias, dout, scale_a, scale_b, dbias_buffer, B, T, OC, dp, stream);
}
