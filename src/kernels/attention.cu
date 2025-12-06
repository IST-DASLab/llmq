// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

// This is a relatively simple baseline implementation of memory-efficient attention.
// Its main purpose is to allow running in *32-bit* precision, which is not supported
// by cuDNN.

#include <cmath>

#include <cooperative_groups.h>
#include "kernels/kernels.h"
#include "utilities/tensor.h"
#include "utilities/vec.cuh"
#include "kernel_utils.cuh"

namespace cg = cooperative_groups;

template<int E, class scalar_t>
__global__ void __launch_bounds__(512) attention_forward_gpu_kernel(
    scalar_t* out, float* stats, float scale,
    const scalar_t* qkv,
    int B, int T, int Hq, int Hkv) {
    constexpr const int SubWarpSize = 16;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);

    extern __shared__ float scratch[];

    int h = blockIdx.x;
    int b = blockIdx.y;
    int t = blockIdx.z;

    int hkv = h * Hkv / Hq;
    int TH = Hq + 2*Hkv;
    ptrdiff_t batch_offset = b * T * TH * E;
    qkv += batch_offset;
    const scalar_t* query = qkv + t * TH * E + h * E;
    const scalar_t* keys = qkv + (Hq + hkv) * E;
    const scalar_t* values = qkv + (Hq + Hkv + hkv) * E;

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using q_cache_t = GenericVector<float, E / SubWarpSize>;
    q_cache_t q_cache;

    // combine values
    using v_cache_t = GenericVector<float, E / SubWarpSize>;
    v_cache_t v_cache = v_cache_t::zeros();

    // determine maximum and online logsumexp
    float maximum = std::numeric_limits<float>::lowest();
    float lse = 0;

    for (int ee = 0; ee < E / (SubWarpSize * vec_t::size); ++ee) {
        int e = (ee * SubWarpSize + sub_warp.thread_rank()) * vec_t::size;
        vec_t qv = vec_t::load(query + e);
        for (int j = 0; j < vec_t::size; ++j) {
            q_cache[ee * vec_t::size + j] = (float)qv[j];
        }
    }

    for (int l = sub_warp.meta_group_rank(); l <= t; l += sub_warp.meta_group_size()) {
        ptrdiff_t kv_offset = l * TH * E;
        float qk = 0;
        for (int ee = 0; ee < E / (SubWarpSize * vec_t::size); ++ee) {
            int e = (ee * SubWarpSize + sub_warp.thread_rank()) * vec_t::size;
            vec_t kv = vec_t::load(keys + kv_offset + e);
            for (int j = 0; j < vec_t::size; ++j) {
                qk += q_cache[ee * vec_t::size + j] * (float)kv[j];
            }
        }
        qk = reduce_group_add(sub_warp, qk);
        if (qk > maximum) {
            float rescale = std::exp(scale * (maximum - qk));
            for (int j = 0; j < v_cache_t::size; ++j) {
                v_cache[j] *= rescale;
            }
            lse *= rescale;
            maximum = qk;
        }
        float att = std::exp(scale * (qk - maximum));
        lse += std::exp(scale * (qk - maximum));

        for (int ee = 0; ee < E / (SubWarpSize * vec_t::size); ++ee) {
            int e = (ee * SubWarpSize + sub_warp.thread_rank()) * vec_t::size;
            vec_t vv = vec_t::load(values + kv_offset + e);
            for (int j = 0; j < vec_t::size; ++j) {
                v_cache[ee * vec_t::size + j] += att * (float)vv[j];
            }
        }
    }

    // combine split-k results
    if (sub_warp.thread_rank() == 0) {
        scratch[sub_warp.meta_group_rank()] = maximum;
        scratch[sub_warp.meta_group_rank() + sub_warp.meta_group_size()] = lse;
    }

    __syncthreads();
    float r_max = maximum;
    float l_max = maximum;
    float r_lse = 0;
    if (warp.thread_rank() < sub_warp.meta_group_size()) {
        r_max = scratch[warp.thread_rank()];
        r_lse = scratch[warp.thread_rank() + sub_warp.meta_group_size()];
    }

    maximum = reduce_group_max(warp, r_max);
    r_lse *= std::exp(scale * (r_max - maximum));
    lse = reduce_group_add(warp, r_lse);
    float rescale = std::exp(scale * (l_max - maximum)) / lse;
    for (int j = 0; j < v_cache_t::size; ++j) {
        v_cache[j] *= rescale;
    }
    if(threadIdx.x == 0) {
        stats[b * Hq * T + h * T + t] = scale * maximum + std::log(lse);
    }
    __syncthreads();

    for (int ee = 0; ee < E / (SubWarpSize * vec_t::size); ++ee) {
        int e = (ee * SubWarpSize + sub_warp.thread_rank()) * vec_t::size;
        fvec_t store;
        for (int j = 0; j < vec_t::size; ++j) {
            store[j] = v_cache[ee * vec_t::size + j];
        }
        store.store(scratch + e + E * sub_warp.meta_group_rank());
    }

    if (warp.meta_group_rank() != 0) return;
    __syncthreads();
    // write result
    for (int e = vec_t::size * warp.thread_rank(); e < E; e += vec_t::size * warp.size()) {
        fvec_t res = fvec_t::zeros();
        for (int j = 0; j < sub_warp.meta_group_size(); ++j) {
            fvec_t sv = fvec_t::load(scratch + e + E * j);
            for (int jj = 0; jj < vec_t::size; ++jj) {
                res[jj] += sv[jj];
            }
        }
        vec_t cv;
        for (int j = 0; j < vec_t::size; ++j) {
            cv[j] = (scalar_t)res[j];
        }
        cv.store(out + ((b * T + t) * Hq + h) * E + e);
    }
}

template<int E, class scalar_t>
__global__ void __launch_bounds__(512) attention_backward_gpu_kernel(
        scalar_t* dqkv, const float* stats, float scale,
        const scalar_t* out, const scalar_t* dout, const scalar_t* qkv,
        int B, int T, int Hq, int Hkv) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = blockIdx.z;

    const int hkv = h * Hkv / Hq;
    const int TH = Hq + 2*Hkv;

    qkv += b * T * TH * E;
    dqkv += b * T * TH * E;
    out += b * T * Hq * E + h * E;
    dout += b * T * Hq * E + h * E;

    const scalar_t* query = qkv + h * E;
    const scalar_t* keys = qkv + (Hq + hkv) * E;
    const scalar_t* values = qkv + (Hq + Hkv + hkv) * E;

    scalar_t* dquery = dqkv + h * E;
    scalar_t* dkeys = dqkv + (Hq + hkv) * E;
    scalar_t* dvalues = dqkv + (Hq + Hkv + hkv) * E;

    float lse = stats[b * Hq * T + h * T + t];
    float D = 0.0;
    for(int i = 0; i < E; ++i) {
        D += dout[t * Hq * E + i] * out[t * Hq * E + i];
    }

    ptrdiff_t q_offset = t * TH * E;
    for (int l = threadIdx.x; l <= t; l += blockDim.x) {
        ptrdiff_t kv_offset = l * TH * E;
        float qk = 0;
        for (int i = 0; i < E; ++i) {
            qk += (float)query[q_offset + i] * (float)keys[kv_offset + i];
        }

        float att = std::exp(scale * qk - lse);
        float datt = 0.0;

        // Update V gradient and calculate attention gradient
        for (int i = 0; i < E; ++i) {
            float do_t = dout[t * Hq * E + i];
            atomicAdd(dvalues + kv_offset + i, att * do_t);
            datt += do_t * values[kv_offset + i];
        }

        float dqk = scale * att * (datt - D);

        // Update QK gradients
        for (int i = 0; i < E; ++i) {
            atomicAdd(dquery + q_offset + i, dqk * keys[kv_offset + i]);
            atomicAdd(dkeys + kv_offset + i, dqk * query[q_offset + i]);
        }
    }
}

template<class floatX>
cudaError_t attention_gpu_forward(floatX* out, float* stats, float scale,
                          const floatX* qkv,
                          int B, int T, int Hq, int Hkv, int Hs, cudaStream_t stream) {
    dim3 grid_dim{(unsigned)Hq, (unsigned)B, (unsigned)T};
    dim3 block_dim{512, 1, 1};
    size_t smem = Hs * sizeof(float) * block_dim.x / 16;

    if (Hs == 128) {
        attention_forward_gpu_kernel<128><<<grid_dim, block_dim, smem, stream>>>(
            out, stats, scale, qkv, B, T, Hq, Hkv);
    } else if (Hs == 64) {
        attention_forward_gpu_kernel<64><<<grid_dim, block_dim, smem, stream>>>(
            out, stats, scale, qkv,  B, T, Hq, Hkv);
    } else {
        printf("Unsupported head dimension");
    }
    return cudaGetLastError();
}

void attention_forward_cudnn(float* out,  // output: (B, T, Nq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const float* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    attention_gpu_forward(out, stats, 1.f / sqrtf(HS), inp, B, T, Hq, Hkv, HS, stream);
}

template<class floatX>
cudaError_t attention_gpu_backward(floatX* dqkv, const float* stats, float scale,
                                   const floatX* out, const floatX* dout, const floatX* qkv,
                                   int B, int T, int Hq, int Hkv, int Hs, cudaStream_t stream) {
    dim3 grid_dim{(unsigned)Hq, (unsigned)B, (unsigned)T};
    dim3 block_dim{512, 1, 1};
    size_t smem = Hs * sizeof(float) * block_dim.x / 16;
    cudaMemsetAsync(dqkv, 0, sizeof(float)*B*T*(Hq + 2*Hkv) * Hs, stream);
    if (Hs == 128) {
        attention_backward_gpu_kernel<128><<<grid_dim, block_dim, smem, stream>>>(
            dqkv, stats, scale, out, dout, qkv, B, T, Hq, Hkv);
    } else if (Hs == 64) {
        attention_backward_gpu_kernel<64><<<grid_dim, block_dim, smem, stream>>>(
            dqkv, stats, scale, out, dout, qkv, B, T, Hq, Hkv);
    } else {
        printf("Unsupported head dimension");
    }
    return cudaGetLastError();
}

void attention_backward_cudnn(float* dqkv, const float* stats,
                              const float* out, const float* dout, const float* qkv, std::byte* workspace,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    attention_gpu_backward(dqkv, stats, 1.f / sqrtf(HS), out, dout, qkv, B, T, Hq, Hkv, HS, stream);
}

void attention_forward_cudnn(Tensor& out,  // output: (B, T, Hq, HS)
                             Tensor& stats, // output for backward pass: (B, Hq, T)
                             const Tensor& inp,  // input: (B, T, Hq + Hk + Hv, HS) QKV
                             Tensor& workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    if(out.DType == ETensorDType::FP32) {
        attention_forward_cudnn(out.get<float>(), stats.get<float>(), inp.get<float>(), ws, handle, B, T, Hq, Hkv, HS, stream);
    } else if(out.DType == ETensorDType::BF16) {
        attention_forward_cudnn(out.get<nv_bfloat16>(), stats.get<float>(), inp.get<nv_bfloat16>(), ws, handle, B, T, Hq, Hkv, HS, stream);
    } else {
        throw std::logic_error("attention_forward: unsupported dtype");
    }
}

void attention_backward_cudnn(Tensor& dqkv, const Tensor& stats,
                              const Tensor& out, const Tensor& dout, const Tensor& qkv,
                              Tensor& workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    if(out.DType == ETensorDType::FP32) {
        attention_backward_cudnn(dqkv.get<float>(), stats.get<float>(), out.get<float>(), dout.get<float>(), qkv.get<float>(), ws, B, T, Hq, Hkv, HS, stream);
    } else if(out.DType == ETensorDType::BF16) {
        // TODO make argument order consistent
        attention_backward_cudnn(dqkv.get<nv_bfloat16>(), stats.get<float>(), dout.get<nv_bfloat16>(), qkv.get<nv_bfloat16>(), out.get<nv_bfloat16>(), ws, handle, B, T, Hq, Hkv, HS, stream);
    } else {
        throw std::logic_error("attention_backward: unsupported dtype");
    }
}
