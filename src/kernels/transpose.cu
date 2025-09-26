// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//
// SPDX-License-Identifier: MIT

#include "transpose_template.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

template<int VEC_SIZE, typename T>
__global__ void transpose_kernel(T* dest, const T* src, int rows, int cols) {
    apply_and_transpose_helper<VEC_SIZE>([](auto&& a){ return a; }, dest, src, rows, cols);
}

template<typename T>
void transpose_imp(T* dst, const T* src, int rows, int cols, cudaStream_t stream) {
     if(rows % 16 == 0 && cols % 16 == 0 && sizeof(T) == 1) {
        dim3 block_size = {8, 8};
        dim3 grid_size = {(unsigned)div_ceil(rows, 16*(int)block_size.x), (unsigned)div_ceil(cols, 16*(int)block_size.y)};
         transpose_kernel<16><<<grid_size, block_size, 0, stream>>>(dst, src, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    } else if (rows % 8 == 0 && cols % 8 == 0 && sizeof(T) <= 2) {
        dim3 block_size = {8, 8};
        dim3 grid_size = {(unsigned)div_ceil(rows, 8*(int)block_size.x), (unsigned)div_ceil(cols, 8*(int)block_size.y)};
         transpose_kernel<8><<<grid_size, block_size, 0, stream>>>(dst, src, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    } else {
        dim3 block_size = {8, 8};
        dim3 grid_size = {(unsigned)div_ceil(rows, (int)block_size.x), (unsigned)div_ceil(cols, (int)block_size.y)};
        transpose_kernel<1><<<grid_size, block_size, 0, stream>>>(dst, src, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    }
}

void transpose(float* dst, const float* src, int rows, int cols, cudaStream_t stream) {
    transpose_imp(dst, src, rows, cols, stream);
}

void transpose(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src, int rows, int cols, cudaStream_t stream) {
    transpose_imp(dst, src, rows, cols, stream);
}
void transpose(nv_bfloat16* dst, const nv_bfloat16* src, int rows, int cols, cudaStream_t stream) {
    transpose_imp(dst, src, rows, cols, stream);
}
