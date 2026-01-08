// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cassert>

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

struct SoftmaxParams {
    float SumExp;
    float Offset;
};

template<class floatX>
__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*static_cast<int>(x128::size) > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_max;
            thread_max = fmaxf(thread_max, v);
            thread_sum *= expf((old_maxval - thread_max));
            thread_sum += expf(v - thread_max);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = x128::load(x + i * x128::size); // load and keep in cache until fused_classifier loop
        // two-pass calculation: First, determine the new max and adjust the existing
        // thread_sum, then add the new values.
        // having two loops almost halves the number of expf calls required.
        float old_maxval = thread_max;
        auto vec_max = vecReduceMax(packed_x);
        thread_max = fmaxf(thread_max, static_cast<float>(vec_max));
        constexpr float kLog2e = 1.4426950408889634f;
        float log2_thread_max = thread_max * kLog2e;
        thread_sum *= exp2f(old_maxval * kLog2e - log2_thread_max);
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            thread_sum += exp2f(v * kLog2e - log2_thread_max);
        }
    }

    __shared__ float smem_max[32];
    __shared__ float smem_sum[32];

    int lane_id = threadIdx.x % 32;
    float warp_max = warpReduceMax(thread_max);
    if(lane_id == 0) {
        smem_max[threadIdx.x / 32] = warp_max;
    }
    __syncthreads();
    float block_max = warpReduceMax(smem_max[lane_id]);

    thread_sum *= expf(thread_max - block_max);
    float warp_sum = warpReduceSum(thread_sum);
    if(lane_id == 0) {
        smem_sum[threadIdx.x / 32] = warp_sum;
    }
    __syncthreads();
    float block_sum = warpReduceSum(smem_sum[lane_id]);

    // return the softmax parameters
    return SoftmaxParams{block_sum, block_max};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <class floatX, bool WriteDLogits, bool ZLoss>
__global__ void __launch_bounds__(1024, 1)
    fused_classifier_kernel5(floatX* logits, float* losses, float* lse_out,
                             const float dloss, const int* targets, float z_reg,
                             int V, int P, std::bool_constant<WriteDLogits>, std::bool_constant<ZLoss>) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = gridDim.x - (blockIdx.x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];
    if(ix == -100) {
        if (WriteDLogits){
            x128 zero = x128::zeros();
            for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
                zero.store(logits + idx * P + i * x128::size);
            }
        }
        return;     // mask
    }
    assert(0 <= ix && ix < V);

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    float scale = 1.f / sp.SumExp;
    float lse = logf(sp.SumExp) + sp.Offset;
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * scale;
        losses[idx] -= logf(prob);
        lse_out[idx] = lse;
    }

    if constexpr (ZLoss) {
        z_reg = z_reg * lse;
    }

    // without this synchronization point we have a race condition:
    // the logits used above to compute the loss are concurrently (race) modified to carry backward pass grads.
    __syncthreads();


    int ix_by_v = ix / x128::size;
    float shift = logf((dloss + z_reg) * scale) - sp.Offset;

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    constexpr float kLog2e = 1.4426950408889634f;
    shift *= kLog2e;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = x128::load(logits_vec + i * x128::size); // rely on cs of store128cs
        if ( i != ix_by_v ) {
            for(int k = 0; k < x128::size; ++k) {
                packed_logits_vec[k] = static_cast<floatX>(exp2f((float)packed_logits_vec[k] * kLog2e + shift));
            }
        } else {
            for(int k = 0; k < x128::size; ++k) {
                int element = i*x128::size + k;
                float loss_neg = exp2f((float)packed_logits_vec[k] * kLog2e + shift);
                float indicator = (element == ix) ? 1.0f : 0.0f;
                packed_logits_vec[k] = (floatX)(loss_neg - indicator * dloss);
            }
        }
        if constexpr (WriteDLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise the probability that logits remain in cache between prepare_softmax and here
            packed_logits_vec.store_cs(logits + idx * P + i * x128::size);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = threadIdx.x + unaligned_start; i < V; i += blockDim.x) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if constexpr (ZLoss) {
            dlogit += z_reg * prob;
        }

        if (WriteDLogits){
            __stcs(logits + idx * P + i, (floatX)dlogit);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type>
void fused_classifier_imp(Type* logits, float* losses, float* lse,
                      const float dloss, const int* targets, const float z_loss,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    const int block_size = 1024;
    const int grid_size = BT;
    if(write_dlogits) {
        if (z_loss != 0.f) {
            fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, lse, dloss, targets,
                                                                           z_loss, V, P, std::bool_constant<true>(), std::bool_constant<true>());
        } else {
            fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, lse, dloss, targets,
                                                                           z_loss, V, P, std::bool_constant<true>(), std::bool_constant<false>());
        }
    } else {
        fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, lse, dloss, targets,
                                                                       z_loss, V, P, std::bool_constant<false>(), std::bool_constant<false>());
    }
    CUDA_CHECK(cudaGetLastError());
}

void fused_classifier(float* logits, float* losses, float* lse,
                      const float dloss, const int* targets, const float z_loss,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    fused_classifier_imp(logits, losses, lse, dloss, targets, z_loss, BT, V, P, write_dlogits, stream);
}
void fused_classifier(nv_bfloat16* logits, float* losses, float* lse,
                      const float dloss, const int* targets, const float z_loss,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    fused_classifier_imp(logits, losses, lse, dloss, targets, z_loss, BT, V, P, write_dlogits, stream);
}


// ------------------------------------------------------------------------------
// Logit partition function (Z) tracking

__global__ void reduce_lse_stats_kernel(float* __restrict__ stats, const float* __restrict__ lse, long N, bool first_step) {
    using vec_t = GenericVector<float, 4>;

    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];

    float thread_max = 0.f;
    float thread_sum = 0.f;
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(lse + i);
        for(int j = 0; j < vec_t::size; ++j) {
            thread_max = fmaxf(thread_max, values[j]);
            thread_sum += values[j];
        }
    }

    // warp reduction
    thread_max = warpReduceMax(thread_max);
    thread_sum = warpReduceSum(thread_sum);

    if (threadIdx.x % 32 == 0) {
        warp_max[threadIdx.x / 32] = thread_max;
        warp_sum[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();
    // one warp handles max, another handles sum.
    if (threadIdx.x < 32) {
        thread_max = warpReduceMax(warp_max[threadIdx.x]);
        if (threadIdx.x == 0) {
            if (first_step) {
                stats[0] = thread_max;
            } else {
                stats[0] = fmaxf(thread_max, stats[0]);
            }
        }
    } else if (threadIdx.x < 64) {
        thread_sum = warpReduceSum(warp_sum[threadIdx.x - 32]);
        if (threadIdx.x == 32) {
            if (first_step) {
                stats[1] = thread_sum;
            } else {
                stats[1] = thread_sum + stats[1];
            }
        }
    }
}

void reduce_lse_stats(float* result, const float* in, long N, bool first_step, cudaStream_t stream) {
    reduce_lse_stats_kernel<<<1, 1024, 0, stream>>>(result, in, N, first_step);
    CUDA_CHECK(cudaGetLastError());
}
