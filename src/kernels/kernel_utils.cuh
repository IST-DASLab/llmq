// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH
#define LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH

#include <cassert>

static __forceinline__ __device__ void handle_absmax_reduction(float* __restrict__ abs_max_ptr, float* __restrict__ block_max, float thread_max) {
    if (abs_max_ptr) {
        // this code is only guaranteed to be correct if it is warp convergent
        // (in theory, ensuring thread 0 hasn't exited would be enough...)
        assert(__activemask() == 0xffffffff);
        auto warp_max = __reduce_max_sync(0xffffffff, __float_as_uint(thread_max));
        if(threadIdx.x % 32 == 0) {
            atomicMax_block(reinterpret_cast<unsigned*>(block_max), warp_max);
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            atomicMax(reinterpret_cast<unsigned int*>(abs_max_ptr), __float_as_uint(*block_max));
        }
    }
}

#endif //LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH
