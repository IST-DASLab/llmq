// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH
#define LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH

#include <cassert>
#include <cooperative_groups.h>

#ifndef __HIP__
#include <cooperative_groups/reduce.h>
#endif

#include "utilities/vec.cuh"

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

static __forceinline__ __device__ nv_bfloat16 dispatch_max(nv_bfloat16 a, nv_bfloat16 b) {
    return __hmax(a, b);
}

static __forceinline__ __device__ float dispatch_max(float a, float b) {
    return fmaxf(a, b);
}

template<class Float>
__device__ Float warpReduceMax(Float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = dispatch_max(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

template<std::size_t Size>
static __forceinline__ __device__ float vecReduceMax(GenericVector<float, Size> val) {
    static_assert(Size % 2 == 0, "Size must be even for vecReduceMax");
    float max = fmaxf(val[0], val[1]);

    // two-level reduction for ILP: the inner fmaxf of iteration k+2 can
    // overlap with the outer fmaxf of iteration k
    #pragma unroll
    for (int k = 2; k < Size; k += 2) {
        max = fmaxf(max, fmaxf(val[k], val[k+1]));
    }
    return max;
}

template<std::size_t Size>
static __forceinline__ __device__ nv_bfloat16 vecReduceMax(GenericVector<nv_bfloat16, Size> val) {
    static_assert(Size % 2 == 0, "Size must be even for vecReduceMax");
    // use nv_bfloat162 reduction instructions to require only half as many instructions as a naive implementation.
    for (std::size_t end = Size / 2; end > 1; end /= 2) {
        for (int k = 0; k < end; k += 2) {
            nv_bfloat162 m = __hmax2(make_bfloat162(val[k], val[k+1]), make_bfloat162(val[k + end / 2], val[k + end / 2 + 1]));
            val[k] = m.x;
            val[k + 1] = m.y;
        }
    }
    return __hmax(val[0], val[1]);
}

static __device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

#endif //LLMQ_SRC_KERNELS_KERNEL_UTILS_CUH
