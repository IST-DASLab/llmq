// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_KERNELS_TRANSPOSE_TEMPLATE_CUH
#define LLMQ_SRC_KERNELS_TRANSPOSE_TEMPLATE_CUH

#include "utilities/vec.cuh"

template<int VEC_SIZE, typename F, typename TD, class TS>
__device__ void apply_and_transpose_helper(F&& op, TD* dest, const TS* src, int rows, int cols) {
    long r = VEC_SIZE*(blockIdx.x * blockDim.x + threadIdx.x);
    long c = VEC_SIZE*(blockIdx.y * blockDim.y + threadIdx.y);
    if(c >= cols || r >= rows) {
        return;
    }

    using src_vec_t = GenericVector<TS, VEC_SIZE>;
    src_vec_t cache[VEC_SIZE];
    for(int i = 0; i < VEC_SIZE; i++) {
        cache[i] = src_vec_t::load(src + c + (i + r) * cols);
    }
    using dst_vec_t = GenericVector<TD, VEC_SIZE>;
    dst_vec_t save[VEC_SIZE];
    for(int i = 0; i < VEC_SIZE; i++) {
        for(int j = 0; j < VEC_SIZE; j++) {
            save[i][j] = TD{op(cache[j][i])};
        }
    }

    for(int j = 0; j < VEC_SIZE; j++) {
        save[j].store(dest + (c+j) * rows + r);
    }
}

#endif //LLMQ_SRC_KERNELS_TRANSPOSE_TEMPLATE_CUH
