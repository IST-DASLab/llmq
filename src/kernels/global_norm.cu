// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c
#include <cassert>
#include <cmath>
#include <cstddef>

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>


#include "utilities/utils.h"
#include "kernel_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template<class T>
__device__ float global_norm_squared_for_range(const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    accumulator = reduce_group_add(warp, accumulator);
    __shared__ float shared_accumulator[32];
    if(warp.thread_rank() == 0) {
        shared_accumulator[warp.meta_group_rank()] = accumulator;
    }
    __syncthreads();
    // block-level reduce
    float total = warp.thread_rank() < warp.meta_group_size() ? shared_accumulator[warp.thread_rank()] : 0.f;
    total = reduce_group_add(warp, total);
    return total;
}

template<class T>
__global__ void global_norm_squared_kernel(float* out, const T* data, size_t count) {
    float block_sum = global_norm_squared_for_range(data, count);
    // each block accumulates its partial sum to out[blockIdx]
    // we want to avoid using atomic addition here, so we combine this kernel with another kernel call
    // that sums up the partial block sums
    if(threadIdx.x == 0) {
        out[blockIdx.x] = out[blockIdx.x] + block_sum;
    }
}

template<class floatX>
__global__ void deterministic_sum_kernel(float* out, const floatX* data, std::size_t count) {
    assert(gridDim.x == 1);     // only a single block!
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)data[index];
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    float warp_sum = reduce_group_add(warp, thread_sum);
    __shared__ float shared_accumulator[32];
    if(warp.thread_rank() == 0) {
        shared_accumulator[warp.meta_group_rank()] = warp_sum;
    }
    __syncthreads();
    // block-level reduce
    if(warp.meta_group_rank() == 0) {
        float total = warp.thread_rank() < warp.meta_group_size() ? shared_accumulator[warp.thread_rank()] : 0.f;
        total = reduce_group_add(warp, total);
        if (threadIdx.x == 0) {
            *out = total;
        }
    }
}

__global__ void global_norm_sqrt_kernel(float* out, float* out_cpu, float grad_clip) {
    float n_squared = out[0];
    float norm = std::sqrt(n_squared);
    if(n_squared > grad_clip) {
        out[1] = grad_clip / n_squared;     // out[1] contains the grad scaling factor
    }  else {
        out[1] = 1.f;
    }
    *out_cpu = norm;
}


// ----------------------------------------------------------------------------
// kernel launcher

// Helper function determines the maximum number of block sums
int get_max_num_block_sums(const cudaDeviceProp& dp) {
    // NOTE: this needs to be kept in sync with `global_norm_squared` below.
    const int block_size = 512;
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = dp.maxThreadsPerMultiProcessor * dp.multiProcessorCount / block_size;

    return grid_size;
}

template<typename T>
void global_norm_squared_imp(float* out, const T* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    // out points to an array of get_max_num_block_sums elements
    const int block_size = 512;
    const int max_grid_size = get_max_num_block_sums(dp);

    // for tiny tensors, using a device-wide grid is a waste of resources.
    const int max_useful_blocks = div_ceil(count, (size_t)block_size);
    const int grid_size = std::min(max_grid_size, max_useful_blocks);
    assert(grid_size > 0);      // gives a better error than letting the call below fail

    global_norm_squared_kernel<<<grid_size, block_size, 0, stream>>>(out, values, count);
    CUDA_CHECK(cudaGetLastError());
}

void global_norm_squared(float* out, const float* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    global_norm_squared_imp(out, values, count, dp, stream);
}

void global_norm_squared(float* out, const nv_bfloat16* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    global_norm_squared_imp(out, values, count, dp, stream);
}

void global_norm_sqrt(float* out, float* out_cpu, float grad_clip, const cudaDeviceProp& dp, cudaStream_t stream) {
    global_norm_sqrt_kernel<<<1,  1, 0, stream>>>(out, out_cpu, grad_clip);
}

void deterministic_sum(float* out, const float* values, std::size_t count, cudaStream_t stream) {
    deterministic_sum_kernel<<<1, 512, 0, stream>>>(out, values, count);
    CUDA_CHECK(cudaGetLastError());
}

void deterministic_sum(float* out, const nv_bfloat16* values, std::size_t count, cudaStream_t stream) {
    deterministic_sum_kernel<<<1, 512, 0, stream>>>(out, values, count);
    CUDA_CHECK(cudaGetLastError());
}
