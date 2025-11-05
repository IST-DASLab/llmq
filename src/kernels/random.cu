// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>

#include <curand_kernel.h>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

template<typename floatX>
__global__ void rng_normal_kernel(floatX* dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence) {
    curandStatePhilox4_32_10_t state;
    long id = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
    if (id >= count) return;

    curand_init(seed, subsequence, id, &state);
    float4 normal = curand_normal4(&state);
    GenericVector<floatX, 4> cvt;
    cvt[0] = static_cast<floatX>(normal.x * std + mean);
    cvt[1] = static_cast<floatX>(normal.y * std + mean);
    cvt[2] = static_cast<floatX>(normal.z * std + mean);
    cvt[3] = static_cast<floatX>(normal.w * std + mean);
    cvt.store(dst + id);
}

template<typename floatX>
void rng_normal_imp(floatX* dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream) {
    assert(count % 4 == 0);
    rng_normal_kernel<<<div_ceil(count, static_cast<std::size_t>(4*256)), 256, 0, stream>>> (dst, count, mean, std, seed, subsequence);
    CUDA_CHECK(cudaGetLastError());
}

void fill_normal(float* dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream) {
    rng_normal_imp<float>(dst, count, mean, std, seed, subsequence, stream);
}

void fill_normal(nv_bfloat16* dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream) {
    rng_normal_imp<nv_bfloat16>(dst, count, mean, std, seed, subsequence, stream);
}
