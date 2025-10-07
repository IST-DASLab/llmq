// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//

#include <cassert>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

template<typename floatX>
__global__ void fill_kernel(floatX* dst, floatX value, std::size_t count) {
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count) return;
    // TODO vectorize
    dst[id] = value;
}

template<typename floatX>
void fill_imp(floatX* dst, floatX value, std::size_t count, cudaStream_t stream) {
    fill_kernel<<<div_ceil(count, static_cast<std::size_t>(256)), 256, 0, stream>>> (dst, value, count);
    CUDA_CHECK(cudaGetLastError());
}


void fill_constant(float* dst, float value, std::size_t count, cudaStream_t stream) {
    fill_imp(dst, value, count, stream);
}


void fill_constant(nv_bfloat16* dst, nv_bfloat16 value, std::size_t count, cudaStream_t stream) {
    fill_imp(dst, value, count, stream);
}
