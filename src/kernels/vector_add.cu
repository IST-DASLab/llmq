// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "squirrel_noise.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

// note: this calculates `a(x+y)` , _not_ `ax + y` as  in saxpy
template<typename T>
__global__ void vector_add_sr_kernel(T* dest, const T* left, const T* right, float scale, long nelem, unsigned seed) {
    using vec_t = GenericVector<T, 16/sizeof(T)>;
    long idx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_t::size;
    if(idx < nelem) {
        vec_t a = vec_t::load_cs(left + idx);
        vec_t b = vec_t::load_cs(right + idx);
        vec_t c = vec_t::zeros();
        for(int j = 0; j < vec_t::size; ++j) {
            float sum = scale * ((float)a[j] + (float)b[j]);
            stochastic_rounding(sum, &c[j], seed + idx);
        }
        c.store(dest + idx);
    }
}

template<typename T>
__global__ void vector_reduce_sr_kernel(T* dest, const T* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed) {
    using vecx_t = GenericVector<T, 16/sizeof(T)>;
    using vecf_t = GenericVector<float, 16/sizeof(T)>;
    long idx = (blockIdx.x * blockDim.x + threadIdx.x) * vecx_t::size;

    if(idx < nelem) {
        vecf_t accumulator = vecf_t::zeros();
        if(accumulate) {
            vecx_t v = vecx_t::load_cs(dest + idx);
            for(int j = 0; j < vecx_t::size; ++j) {
                accumulator[j] += (float) v[j];
            }
        }
        for (int k = 0; k < n_shards; ++k) {
            if(k == skip) continue;
            vecx_t v = vecx_t::load_cs(src + idx + k * nelem);
            for(int j = 0; j < vecx_t::size; ++j) {
                accumulator[j] += (float) v[j];
            }
        }

        vecx_t result;
        for(int j = 0; j < vecx_t::size; ++j) {
            float sum = scale * accumulator[j];
            stochastic_rounding(sum, &result[j], seed + idx);
        }
        result.store(dest + idx);
    }
}

template<typename T>
void vector_add_sr_imp(T* dest, const T* left, const T* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    long block_size = 512;
    long grid_size = div_ceil(nelem, block_size);
    vector_add_sr_kernel<T><<<grid_size, block_size, 0, stream>>>(dest, left, right, scale, nelem, seed);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void vector_reduce_sr_imp(T* dest, const T* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream) {
    long block_size = 512;
    long grid_size = div_ceil(nelem, block_size);
    vector_reduce_sr_kernel<T><<<grid_size, block_size, 0, stream>>>(dest, src, scale, n_shards, skip, nelem, accumulate, seed);
    CUDA_CHECK(cudaGetLastError());
}

void vector_add_sr(float* dest, const float* left, const float* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    vector_add_sr_imp(dest, left, right, scale, nelem, seed, stream);
}

void vector_add_sr(nv_bfloat16* dest, const nv_bfloat16* left, const nv_bfloat16* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    vector_add_sr_imp(dest, left, right, scale, nelem, seed, stream);
}

void vector_reduce_sr(float* dest, const float* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream) {
    vector_reduce_sr_imp(dest, src, scale, n_shards, skip, nelem, accumulate, seed, stream);
}

void vector_reduce_sr(nv_bfloat16* dest, const nv_bfloat16* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream) {
    vector_reduce_sr_imp(dest, src, scale, n_shards, skip, nelem, accumulate, seed, stream);
}
