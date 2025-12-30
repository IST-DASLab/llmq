// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/barrier>
#include <cooperative_groups.h>

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

namespace cg = cooperative_groups;

namespace {
    // dummy struct to imbue dynamic smem with 128 byte alignment
    struct alignas(128) AlignedSmem  {
        std::byte content[128];
    };
}

static __forceinline__ __device__ void reduce_max_4(float* dst, const float* smem_src) {
    if (threadIdx.x < 4) {
        float val = smem_src[threadIdx.x];
        val = dispatch_max(val, __shfl_xor_sync(0x0000000Fu, val, 2));
        val = dispatch_max(val, __shfl_xor_sync(0x0000000Fu, val, 1));
        if (threadIdx.x == 0) {
            *dst = val;
        }
    }
}

static __forceinline__ __device__ void reduce_max_4(nv_bfloat16* dst, const nv_bfloat16* smem_src) {
    if(threadIdx.x == 0) {
        using vec_t = GenericVector<nv_bfloat16, 4>;
        vec_t val = vec_t::load(smem_src);
        nv_bfloat162 m = __hmax2(make_bfloat162(val[0], val[1]), make_bfloat162(val[2], val[3]));
        *dst = __hmax(m.x, m.y);
    }
}

template<class Float>
__device__ Float warpReduceMax8(Float val) {
    for (int offset = 4; offset > 0; offset /= 2) {
        val = dispatch_max(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

// ----------------------------------------------------------------------------
// CUDA kernels

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <class floatX>
//__global__ void __block_size__((1024, 1, 1), (8, 1, 1))
__global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(128, 2)
    fused_classifier_tma_kernel(floatX* logits, float* losses, float* lse_out,
                             const float dloss, const int* targets, float z_reg,
                             int V, int P) {
    #if __CUDA_ARCH__ >= 900
    constexpr int BLOCK_SIZE = 128;
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using barrier = cuda::barrier<cuda::thread_scope_block>;

    // 1. Initialize shared memory barrier with the number of threads participating in the barrier.
    __shared__ barrier bar;
    __shared__ floatX max_reduction_buffer[4];
    __shared__ float sum_reduction_buffer[4];
    extern __shared__ AlignedSmem smem[];
    cg::cluster_group cluster = cg::this_cluster();

    floatX* smem_logits = reinterpret_cast<floatX*>(reinterpret_cast<std::byte*>(smem) + 128);
    if (threadIdx.x == 0) {
        init(&bar, BLOCK_SIZE);
    }

    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = blockIdx.x / 8;
    int ix = targets[idx];
    if(ix == -100) {
        x128 zero = x128::zeros();
        for (int i = (threadIdx.x + cluster.block_rank() * BLOCK_SIZE) * x128::size; i < P; i += BLOCK_SIZE * 8 * x128::size) {
            zero.store(logits + idx * P + i);
        }
        return;     // mask
    }
    // adjust pointers to point to current token
    const int logits_per_block = P / 8;
    assert(0 <= ix && ix < V);
    __syncthreads();
    int block_start_ix = cluster.block_rank() * logits_per_block;
    int block_end_ix = block_start_ix + logits_per_block;
    logits += idx * P;

    if (threadIdx.x == 0) {
        const int bytes = (block_end_ix - block_start_ix) * sizeof(floatX);
        cuda::device::memcpy_async_tx(smem_logits, logits + block_start_ix, cuda::aligned_size_t<16>(bytes), bar);
        cuda::device::barrier_expect_tx(bar, bytes);
    }

    // 3a. All threads arrive on the barrier.
    barrier::arrival_token token = bar.arrive();

    floatX thread_max = -INFINITY;
    float thread_sum = 0.0f;
    int lane_id = threadIdx.x % 32;

    // 3b. Wait for the data to have arrived.
    bar.wait(std::move(token));

    // OK, all logits for _this_ block are ready here. Calculate logsumexp
    {
        int i = threadIdx.x * x128::size;
        for (; i + x128::size * BLOCK_SIZE < logits_per_block; i += 2 * x128::size * BLOCK_SIZE) {
            x128 v1 = x128::load(smem_logits + i);
            x128 v2 = x128::load(smem_logits + i + x128::size * BLOCK_SIZE);
            floatX old_maxval = thread_max;
            thread_max = dispatch_max(vecReduceMax(v1), vecReduceMax(v2));

            // write this as a two-level reduction. this does not require any additional instructions (we change FMUL to
            // FMA, but that costs the same), but could give slightly less rounding error accumulation.
            float vec_sum = 0.f;
            for(int k = 0; k < x128::size; ++k) {
                vec_sum += expf(static_cast<float>(v1[k] - thread_max));
                vec_sum += expf(static_cast<float>(v2[k] - thread_max));
            }
            thread_sum = thread_sum * expf(static_cast<float>(old_maxval - thread_max)) + vec_sum;
        }

        // epilogue
        if (i  < logits_per_block) {
            x128 v = x128::load(smem_logits + i);
            floatX old_maxval = thread_max;
            for(int k = 0; k < x128::size; ++k) {
                thread_max = dispatch_max(thread_max, v[k]);
            }
            // write this as a two-level reduction. this does not require any additional instructions (we change FMUL to
            // FMA, but that costs the same), but could give slightly less rounding error accumulation.
            float vec_sum = 0.f;
            for(int k = 0; k < x128::size; ++k) {
                vec_sum += expf(static_cast<float>(v[k] - thread_max));
            }
            thread_sum = thread_sum * expf(static_cast<float>(old_maxval - thread_max)) + vec_sum;
        }
    }

    floatX warp_max = warpReduceMax(thread_max);
    if(lane_id == 0) {
        max_reduction_buffer[threadIdx.x / 32] = warp_max;
    }
    __syncthreads();
    reduce_max_4(reinterpret_cast<floatX*>(smem), max_reduction_buffer);

    // cluster data exchange for max
    cluster.sync();
    const std::byte* remote_mem_ptr;
    if (threadIdx.x < 8) {
        remote_mem_ptr = cluster.map_shared_rank(reinterpret_cast<const std::byte*>(smem), threadIdx.x);
        max_reduction_buffer[threadIdx.x] = reinterpret_cast<const nv_bfloat16*>(remote_mem_ptr)[0];
    }
    __syncthreads();
    floatX cluster_max = warpReduceMax8(max_reduction_buffer[lane_id % 8]);

    thread_sum *= expf(static_cast<float>(thread_max - cluster_max));
    float warp_sum = warpReduceSum(thread_sum);
    if(lane_id == 0) {
        sum_reduction_buffer[threadIdx.x / 32] = warp_sum;
    }
    __syncthreads();
    float block_sum = warpReduceSum(lane_id < 4 ? sum_reduction_buffer[lane_id] : 0.f);
    if(lane_id == 0) {
        reinterpret_cast<float*>(smem)[1] = block_sum;
    }
    cluster.sync();
    if (threadIdx.x < 8) {
        sum_reduction_buffer[threadIdx.x] = reinterpret_cast<const float*>(remote_mem_ptr)[1];
    }
    cg::cluster_group::arrival_token dsmem_done_token = cluster.barrier_arrive();
    __syncthreads();
    float cluster_sum = warpReduceSum(lane_id < 8 ? sum_reduction_buffer[lane_id % 8] : 0.f);
    float scale = 1.f / cluster_sum;
    float lse = logf(cluster_sum) + (float)cluster_max;

    // calculate the probability needed for the loss and update (single-threaded)
    // warp 0 gets all the extra work during reductions, so we let this be handled by warp 1
    if(threadIdx.x == 32 && block_start_ix < ix && ix < block_end_ix) {
        // here, it is important that the subtraction is performed in float precision,
        // otherwise we can't match the test's required accuracy
        float prob = expf((float)(smem_logits[ix - block_start_ix]) - (float)cluster_max) * scale;
        losses[idx] -= logf(prob);
        lse_out[idx] = lse;
    }

    int ix_by_v = (ix / x128::size) * x128::size;
    float loss_scale = dloss * scale + z_reg * lse * scale;

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    for (int i = block_start_ix + threadIdx.x * x128::size; i < block_end_ix; i += x128::size * BLOCK_SIZE) {
        x128 v = x128::load(smem_logits + i - block_start_ix);
        if ( i != ix_by_v ) {
            for(int k = 0; k < x128::size; ++k) {
                float d_logit = expf((float)(v[k] - cluster_max)) * loss_scale;
                v[k] = (floatX)d_logit;
            }
        } else {
            for(int k = 0; k < x128::size; ++k) {
                int element = i + k;
                float loss_neg = expf((float)(v[k] - cluster_max)) * loss_scale;
                float indicator = (element == ix) ? 1.0f : 0.0f;
                v[k] = (floatX)(loss_neg - indicator * dloss);
            }
        }
        // reduce cache persistence for the overwritten logits
        // to maximise the probability that logits remain in cache between prepare_softmax and here
        v.store_cs(logits + i);
    }

    // ensure that no block exits until dsmem communication is done
    cluster.barrier_wait(std::move(dsmem_done_token));
    #endif
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type>
void fused_classifier_tma_imp(Type* logits, float* losses, float* lse,
                             const float dloss, const int* targets, float z_reg,
                             int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 128;
    const int grid_size = BT * 8;
    const int smem = 128 + P * sizeof(Type) / 8;
    CUDA_CHECK(cudaFuncSetAttribute(fused_classifier_tma_kernel<Type>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    fused_classifier_tma_kernel<<<grid_size, block_size, smem, stream>>>(
        logits, losses, lse, dloss, targets, z_reg, V, P);
    CUDA_CHECK(cudaGetLastError());
}

void fused_classifier_tma(float* logits, float* losses, float* lse,
                          float dloss, const int* targets, float z_reg,
                          int BT, int V, int P, cudaStream_t stream) {
    fused_classifier_tma_imp(logits, losses, lse, dloss, targets, z_reg, BT, V, P, stream);
}

void fused_classifier_tma(nv_bfloat16* logits, float* losses, float* lse,
                          float dloss, const int* targets, float z_reg,
                          int BT, int V, int P, cudaStream_t stream) {
    fused_classifier_tma_imp(logits, losses, lse, dloss, targets, z_reg, BT, V, P, stream);
}
