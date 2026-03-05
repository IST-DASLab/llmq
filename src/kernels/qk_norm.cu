// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

constexpr const int GroupSize = 8;

template<typename Float>
__global__ void qk_norm_simple_kernel(Float* out, float* r_std, const Float* inp, const Float* q_wgt, const Float* k_wgt, float epsilon, int BT, int Nq, int Nkv, int HeadDim) {
   using x128 = GenericVector<Float, 16/sizeof(Float)>;

    const int lane = threadIdx.x % GroupSize;
    const int group = threadIdx.x / GroupSize;

    // load weights into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* smem[];

    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_q_wgt = reinterpret_cast<x128*>(smem);
    x128* s_k_wgt = reinterpret_cast<x128*>(smem) + (HeadDim / x128::size);
    x128* s_in = reinterpret_cast<x128*>(s_k_wgt) + (HeadDim / x128::size) + group * (HeadDim / x128::size);

    for(int i =  threadIdx.x * x128::size; i < HeadDim; i += blockDim.x * x128::size) {
        s_q_wgt[i/x128::size] = x128::load(q_wgt + i);
        s_k_wgt[i/x128::size] = x128::load(k_wgt + i);
    }

    __syncthreads();
    int idx = blockIdx.x * (blockDim.x / GroupSize) + group;

    int h = idx % (Nq + 2 * Nkv);
    int bt = idx / (Nq + 2 * Nkv);

    if (bt >= BT) return;

    // adjust pointers to current token
    inp += idx * HeadDim;
    out += idx * HeadDim;

    const x128* wgt_src = nullptr;
    if(h < Nq) {
        wgt_src = s_q_wgt;
    } else if (h < Nq + Nkv) {
        wgt_src = s_k_wgt;
    } else if (inp == out) {
        return;
    } else {
        for(int c = lane * x128::size; c < HeadDim; c += GroupSize * x128::size) {
            x128 in_data = x128::load_cs(inp + c);
            in_data.store(out + c);
        }
        return;
    }

    float acc = 0.f;

    for(int c = lane * x128::size; c < HeadDim; c += GroupSize * x128::size) {
        const x128 in_data = x128::load_cs(inp + c);
        s_in[c / x128::size] = in_data;
        for(int k = 0; k < x128::size; ++k) {
            float data_k = (float)in_data[k];
            acc += data_k * data_k;
        }
    }

    unsigned int active_mask = __ballot_sync(0xffffffffu, true);
    for (int offset = GroupSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(active_mask, acc, offset);
    }
    acc /= HeadDim;
    float s = rsqrtf(acc + epsilon);

    for(int c = lane * x128::size; c < HeadDim; c += GroupSize * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w = wgt_src[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * (float)in_data[k]; // normalized output
            // Note: would make sense to do this in fp32, but transformers uses bf16 here,
            // so we try to match
            out_data[k] = (Float)n * (Float)w[k]; // scale
        }

        out_data.store(out + c);    // TODO cs
    }

    // store the rms, no need to cache it
    if(lane == 0 && r_std != nullptr) {
        __stcs(r_std + idx, s);
    }
}

template<typename Float>
void qk_norm_forward(Float* out, float* r_std,
                     const Float* inp,
                     const Float* q_wgt, const Float* k_wgt,
                     float epsilon,
                     int BT, int Nq, int Nkv, int HeadDim,
                     cudaStream_t stream) {
    constexpr int block_size = 512; // larger blocks mean fewer redundant weight loads
    static_assert(block_size % GroupSize == 0);

    const int groups_per_block = block_size / GroupSize;
    const int total_heads = BT * (Nq + 2 * Nkv);
    const int grid_size = div_ceil(total_heads, groups_per_block);

    // smem: q_wgt + k_wgt + one input buffer per group
    size_t smem = (2 + groups_per_block) * HeadDim * sizeof(Float);

    CUDA_CHECK(cudaFuncSetAttribute(
        qk_norm_simple_kernel<Float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));

    qk_norm_simple_kernel<Float><<<grid_size, block_size, smem, stream>>>(
        out, r_std, inp, q_wgt, k_wgt, epsilon, BT, Nq, Nkv, HeadDim);

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
void qk_norm_forward(float* out, float* r_std, const float* inp,
                     const float* q_wgt, const float* k_wgt,
                     float epsilon, int BT, int Nq, int Nkv, int HeadDim,
                     cudaStream_t stream) {
    qk_norm_forward<float>(out, r_std, inp, q_wgt, k_wgt, epsilon, BT, Nq, Nkv, HeadDim, stream);
}

void qk_norm_forward(nv_bfloat16* out, float* r_std, const nv_bfloat16* inp,
                     const nv_bfloat16* q_wgt, const nv_bfloat16* k_wgt,
                     float epsilon, int BT, int Nq, int Nkv, int HeadDim,
                     cudaStream_t stream) {
    qk_norm_forward<nv_bfloat16>(out, r_std, inp, q_wgt, k_wgt, epsilon, BT, Nq, Nkv, HeadDim, stream);
}
