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

    struct alignas(8) SoftmaxStats {
        float Sum;
        float Max;
    };
}

template<int Size, class Float>
__device__ Float subWarpReduceMax(Float val) {
    for (int offset = Size/2; offset > 0; offset /= 2) {
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
__global__ void
#if __CUDA_ARCH__ >= 900
__cluster_dims__(8, 1, 1) __launch_bounds__(128, 2)
#endif
    fused_classifier_tma_kernel(floatX* logits, float* losses, float* lse_out,
                             const float dloss, const int* targets, float z_reg,
                             int V, int P) {
    #if __CUDA_ARCH__ >= 900
    constexpr int BLOCK_SIZE = 128;
    constexpr float kLog2e = 1.4426950408889634f;
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using barrier = cuda::barrier<cuda::thread_scope_block>;

    // 1. Initialize shared memory barrier with the number of threads participating in the barrier.
    __shared__ barrier bar;
    __shared__ SoftmaxStats reduction_buffer[8];
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
    const int logits_per_block = V / 8;
    assert(0 <= ix && ix < V);
    __syncthreads();
    int block_start_ix = cluster.block_rank() * logits_per_block;
    int block_middle_idx = block_start_ix + round_down(logits_per_block / 2, static_cast<int>(2 * x128::size * BLOCK_SIZE));
    int block_end_ix = block_start_ix + logits_per_block;
    logits += idx * P;

    if (threadIdx.x == 0) {
        const int bytes = (block_middle_idx - block_start_ix) * sizeof(floatX);
        cuda::device::memcpy_async_tx(smem_logits, logits + block_start_ix, cuda::aligned_size_t<16>(bytes), bar);
        cuda::device::barrier_expect_tx(bar, bytes);
    }

    // 3a. All threads arrive on the barrier.
    barrier::arrival_token token_a = bar.arrive();

    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    int lane_id = threadIdx.x % 32;

    // 3b. Wait for the data to have arrived.
    bar.wait(std::move(token_a));

    // start the rest of the transfer
    if (threadIdx.x == 0) {
        const int bytes = (block_end_ix - block_middle_idx) * sizeof(floatX);
        cuda::device::memcpy_async_tx(smem_logits + (block_middle_idx - block_start_ix), logits + block_middle_idx, cuda::aligned_size_t<16>(bytes), bar);
        cuda::device::barrier_expect_tx(bar, bytes);
    }
    barrier::arrival_token token_b = bar.arrive();

    // handle the first part of the block. By construction, this is a multiple of 2 x128 * block
    for (int i = threadIdx.x * x128::size; i < (block_middle_idx - block_start_ix); i += 2 * x128::size * BLOCK_SIZE) {
        x128 v1 = x128::load(smem_logits + i);
        x128 v2 = x128::load(smem_logits + i + x128::size * BLOCK_SIZE);
        float old_maxval = thread_max;
        thread_max = static_cast<float>(dispatch_max(vecReduceMax(v1), vecReduceMax(v2)));
        thread_max = fmaxf(thread_max, old_maxval);

        // write this as a two-level reduction. this does not require any additional instructions (we change FMUL to
        // FMA, but that costs the same), but could give slightly less rounding error accumulation.
        float vec_sum = 0.f;
        float log2_thread_max = thread_max * kLog2e;

        for(int k = 0; k < x128::size; ++k) {
            vec_sum += exp2f(static_cast<float>(v1[k]) * kLog2e - log2_thread_max);
            vec_sum += exp2f(static_cast<float>(v2[k]) * kLog2e - log2_thread_max);
        }
        thread_sum = thread_sum * exp2f(old_maxval * kLog2e - log2_thread_max) + vec_sum;
    }

    // now wait for the second half of the data
    bar.wait(std::move(token_b));

    // Now handle the rest. This might be not a multiple of block_size * 2 * x128, so we have an epilogue
    {
        int i = (block_middle_idx - block_start_ix) + threadIdx.x * x128::size;
        for (; i + x128::size * BLOCK_SIZE < logits_per_block; i += 2 * x128::size * BLOCK_SIZE) {
            x128 v1 = x128::load(smem_logits + i);
            x128 v2 = x128::load(smem_logits + i + x128::size * BLOCK_SIZE);
            float old_maxval = thread_max;
            thread_max = static_cast<float>(dispatch_max(vecReduceMax(v1), vecReduceMax(v2)));
            thread_max = fmaxf(thread_max, old_maxval);

            // write this as a two-level reduction. this does not require any additional instructions (we change FMUL to
            // FMA, but that costs the same), but could give slightly less rounding error accumulation.
            float vec_sum = 0.f;
            float log2_thread_max = thread_max * kLog2e;
            for(int k = 0; k < x128::size; ++k) {
                vec_sum += exp2f(static_cast<float>(v1[k]) * kLog2e - log2_thread_max);
                vec_sum += exp2f(static_cast<float>(v2[k]) * kLog2e - log2_thread_max);
            }
            thread_sum = thread_sum * exp2f(old_maxval * kLog2e - log2_thread_max) + vec_sum;
        }

        // epilogue
        if (i < logits_per_block) {
            x128 v = x128::load(smem_logits + i);
            float old_maxval = thread_max;
            for(int k = 0; k < x128::size; ++k) {
                thread_max = fmaxf(thread_max, static_cast<float>(v[k]));
            }
            // write this as a two-level reduction. this does not require any additional instructions (we change FMUL to
            // FMA, but that costs the same), but could give slightly less rounding error accumulation.
            float vec_sum = 0.f;
            float log2_thread_max = thread_max * kLog2e;
            for(int k = 0; k < x128::size; ++k) {
                vec_sum += exp2f(static_cast<float>(v[k]) * kLog2e - log2_thread_max);
            }
            thread_sum = thread_sum * exp2f(old_maxval * kLog2e - log2_thread_max) + vec_sum;
        }
    }

    float warp_max = warpReduceMax(thread_max);
    if(lane_id == 0) {
        reduction_buffer[threadIdx.x / 32].Max = warp_max;
    }
    __syncthreads();
    float block_max = subWarpReduceMax<4>(reduction_buffer[lane_id % 4].Max);
    thread_sum *= expf(thread_max - block_max);
    float warp_sum = warpReduceSum(thread_sum);
    if(lane_id == 0) {
        reduction_buffer[threadIdx.x / 32].Sum = warp_sum;
    }
    __syncthreads();
    float block_sum = warpReduceSum(lane_id < 4 ? reduction_buffer[lane_id].Sum : 0.f);

    if (threadIdx.x == 0) {
        SoftmaxStats stats{block_sum, block_max};
        reinterpret_cast<SoftmaxStats*>(smem)[0] = stats;
    }

    // cluster data exchange
    cluster.sync();
    if (threadIdx.x < 8) {
        const std::byte* remote_mem_ptr = cluster.map_shared_rank(reinterpret_cast<const std::byte*>(smem), threadIdx.x);
        reduction_buffer[threadIdx.x] = *reinterpret_cast<const SoftmaxStats*>(remote_mem_ptr);
    }
    __syncthreads();
    SoftmaxStats other_stats = reduction_buffer[lane_id % 8];
    float cluster_max = subWarpReduceMax<8>(other_stats.Max);
    other_stats.Sum *= expf(other_stats.Max - cluster_max);
    cg::cluster_group::arrival_token dsmem_done_token = cluster.barrier_arrive();

    float cluster_sum = warpReduceSum(lane_id < 8 ? other_stats.Sum : 0.f);
    float scale = 1.f / cluster_sum;
    float lse = logf(cluster_sum) + cluster_max;

    // is this block responsible for the ground-truth index
    bool block_has_ix = block_start_ix <= ix && ix < block_end_ix;

    // calculate the probability needed for the loss and update (single-threaded)
    // warp 0 gets all the extra work during reductions, so we let this be handled by warp 1
    if(threadIdx.x == 32 && block_has_ix) {
        losses[idx] -= (float)(smem_logits[ix - block_start_ix]) - lse;
        lse_out[idx] = lse;
    }

    // figure out which warp will encounter the ground-truth token
    float shift = (logf((dloss + z_reg * lse) * scale) - cluster_max) * kLog2e;

    // treat all classes as if they are negative. This allows us to avoid any conditionals inside this loop.
    for (int i = block_start_ix + threadIdx.x * x128::size; i < block_end_ix; i += x128::size * BLOCK_SIZE) {
        x128 v = x128::load(smem_logits + i - block_start_ix);
        for(int k = 0; k < x128::size; ++k) {
            float d_logit = exp2f((float)v[k] * kLog2e + shift);
            v[k] = (floatX)d_logit;
        }
        v.store_cg(logits + i);
    }

    // write the correct dlogit for the true class
    int thread_with_ix = -1;
    if (block_has_ix) {
        thread_with_ix = ((ix - block_start_ix) / x128::size) % BLOCK_SIZE;
    }
    if (threadIdx.x == thread_with_ix) {
        floatX logit = smem_logits[ix - block_start_ix];
        float loss_neg = exp2f((float)logit * kLog2e + shift);
        logits[ix] = (floatX)(loss_neg - dloss);
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
