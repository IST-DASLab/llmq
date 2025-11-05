// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cassert>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// warp-level reduction for finding the maximum value
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}


// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = float (*) (float);
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ float shared_val[32];
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}


struct SoftmaxParams {
    float Scale;
    float Offset;
};

template<class floatX>
__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = x128::load(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <class floatX, bool WriteDLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, 1)
    fused_classifier_kernel5(floatX* logits, float* losses, floatX* probs,
                             const float dloss, const int* targets,
                             int B, int T, int V, int P, std::bool_constant<WriteDLogits>) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = gridDim.x - (blockIdx.x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];
    assert(0 <= ix && ix < V);

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] -= logf(prob);
    }

    // without this synchronization point we have a race condition:
    // the logits used above to compute the loss are concurrently (race) modified to carry backward pass grads.
    __syncthreads();

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = x128::load(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteDLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise the probability that logits remain in cache between prepare_softmax and here
            packed_logits_vec.store(logits + idx * P + i * x128::size); // TODO CS
        }
        if (WriteProbs) {
            packed_probs.store(probs + idx * P + i * x128::size);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteDLogits){
            __stcs(logits + idx * P + i, (floatX)dlogit);
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type>
void fused_classifier_imp(Type* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, bool write_dlogits, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    if(write_dlogits) {
        fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (Type*) NULL, dloss, targets, B,
                                                                       T, V, P, std::bool_constant<true>());
    } else {
        fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (Type*) NULL, dloss, targets, B,
                                                                       T, V, P, std::bool_constant<false>());
    }
    CUDA_CHECK(cudaGetLastError());
}

void fused_classifier(float* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, bool write_dlogits, cudaStream_t stream) {
    fused_classifier_imp(logits, losses, dloss, targets, B, T, V, P, write_dlogits, stream);
}
void fused_classifier(nv_bfloat16* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, bool write_dlogits, cudaStream_t stream) {
    fused_classifier_imp(logits, losses, dloss, targets, B, T, V, P, write_dlogits, stream);
}
