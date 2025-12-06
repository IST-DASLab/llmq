// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH
#define LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH

#include <cassert>
#include <cooperative_groups.h>

#ifndef __HIP__
#include <cooperative_groups/reduce.h>
#endif

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

template<typename Group, typename Element>
static __forceinline__ __device__ Element reduce_group_add(Group& group, Element value) {
    return cooperative_groups::reduce(group, value, cooperative_groups::plus<Element>());
}

template<typename Group, typename Element>
static __forceinline__ __device__ Element reduce_group_max(Group& group, Element value) {
    return cooperative_groups::reduce(group, value, cooperative_groups::greater<Element>());
}

static __forceinline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

#endif //LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH
