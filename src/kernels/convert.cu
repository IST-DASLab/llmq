// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

template<typename Src, typename Dst>
__global__ void convert_dtype_kernel(Dst* target, const Src* source, std::size_t size) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= size) {
        return;
    }
    target[tid] = static_cast<Dst>(source[tid]);
    // TODO vectorize
}

template<typename Src, typename Dst>
void convert_dtype_launcher(Dst* target, const Src* source, std::size_t size) {
    unsigned long n_blocks = div_ceil(size, 128ul);
    convert_dtype_kernel<Src, Dst><<<n_blocks, 128>>>(target, source, size);
}

void convert_dtype(float* target, const nv_bfloat16* source, std::size_t size) {
    convert_dtype_launcher(target, source, size);
}

void convert_dtype(nv_bfloat16* target, const float* source, std::size_t size) {
    convert_dtype_launcher(target, source, size);
}

void convert_dtype(nv_bfloat16* target, const half* source, std::size_t size) {
    convert_dtype_launcher(target, source, size);
}
