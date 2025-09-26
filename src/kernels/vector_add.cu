// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
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
void vector_add_sr_imp(T* dest, const T* left, const T* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    long block_size = 512;
    long grid_size = div_ceil(nelem, block_size);
    vector_add_sr_kernel<T><<<grid_size, block_size, 0, stream>>>(dest, left, right, scale, nelem, seed);
    CUDA_CHECK(cudaGetLastError());
}

void vector_add_sr(float* dest, const float* left, const float* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    vector_add_sr_imp(dest, left, right, scale, nelem, seed, stream);
}

void vector_add_sr(nv_bfloat16* dest, const nv_bfloat16* left, const nv_bfloat16* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    vector_add_sr_imp(dest, left, right, scale, nelem, seed, stream);
}
