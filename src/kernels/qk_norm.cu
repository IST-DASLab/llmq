// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

constexpr const int GroupSize = 8;

__device__ float reduce_group_sum(float acc) {
    unsigned int active_mask = __ballot_sync(0xffffffffu, true);
    for (int offset = GroupSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(active_mask, acc, offset);
    }
    return acc;
}

template<typename Float>
__global__ void qk_norm_forward_simple_kernel(Float* out, float* r_rms, const Float* inp, const Float* q_wgt, const Float* k_wgt, float epsilon, int BT, int Nq, int Nkv, int HeadDim) {
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

    acc = reduce_group_sum(acc) / HeadDim;
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

        out_data.store(out + c);
    }

    // store the rms, no need to cache it
    if(lane == 0 && r_rms != nullptr) {
        __stcs(r_rms + idx, s);
    }
}



template<class Float>
__global__ void __launch_bounds__(512, 2)
qk_norm_backward_kernel(Float* dinp, std::byte* scratch,
                        const Float* dout, const Float* inp, const Float* q_wgt, const Float* k_wgt,
                        const float* rstd, float* abs_max_ptr, float epsilon, int BT, int Nq, int Nkv, int HeadDim) {
    // formulas:
    //   rms = sqrt(sum x_j^2 / C + eps)
    //   y_i = w_i x_i / rms
    //
    //   o_i := dL/dy_i
    //   dw_i = sum (x_i o_i / rms)
    //   xow := sum x_i o_i w_i
    //   dy_j/drms = - w_j x_j / rms²
    //   drms/dx_j = x_j/(C rms)
    //   dx_i = dL/dy_j (dy_j/dx_i + dy_j/drms drms/dx_i)
    //        = o_i w_i / rms - sum_j o_j w_j x_j / rms² x_i/(C rms)
    //        = o_i w_i / rms - x_i (xow/C) / rms³
    using x128 = GenericVector<Float, 16/sizeof(Float)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;
    using fvec = GenericVector<float, x128::size>;

    constexpr int XF_STRIDE = div_exact(x128::size, f128::size);

    const int lane = threadIdx.x % GroupSize;
    const int group = threadIdx.x / GroupSize;
    const int num_groups = blockDim.x / GroupSize;

    // load weights into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* smem[];

    __shared__ float block_abs_max;
    float thread_abs_max = 0.f;

    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_q_wgt = reinterpret_cast<x128*>(smem);
    x128* s_k_wgt = s_q_wgt + (HeadDim / x128::size);
    float* s_dq_wgt_base = reinterpret_cast<float*>(s_k_wgt) + HeadDim;
    float* s_dk_wgt_base = s_dq_wgt_base + HeadDim * num_groups;
    float* s_dq_wgt = s_dq_wgt_base + HeadDim * group;
    float* s_dk_wgt = s_dk_wgt_base + HeadDim * group;

    for(int i = threadIdx.x * x128::size; i < HeadDim; i += blockDim.x * x128::size) {
        s_q_wgt[i/x128::size] = x128::load(q_wgt + i);
        s_k_wgt[i/x128::size] = x128::load(k_wgt + i);
    }
    for(int i = threadIdx.x * f128::size; i < HeadDim; i += blockDim.x * f128::size) {
        f128::zeros().store(s_q_wgt + i);
        f128::zeros().store(s_k_wgt + i);
    }

    if (abs_max_ptr && threadIdx.x == 0) {
        block_abs_max = 0.f;
    }

    __syncthreads();

    const int groups_in_grid = blockDim.x / GroupSize * gridDim.x;
    const int start_idx = blockIdx.x * (blockDim.x / GroupSize) + group;
    for (int idx = start_idx; idx < BT * (Nq + 2 * Nkv); idx += groups_in_grid) {
        int h = idx % (Nq + 2 * Nkv);
        int bt = idx / (Nq + 2 * Nkv);

        // adjusted pointers to current token
        const Float* inp_i = inp + idx * HeadDim;
        const Float* dout_i = dout + idx * HeadDim;
        Float* dinp_i = dinp + idx * HeadDim;
        const float rstd_i = rstd[idx];

        const x128* wgt_src = nullptr;
        float* dwgt_dst = nullptr;
        if(h < Nq) {
            wgt_src = s_q_wgt;
            dwgt_dst = s_dq_wgt;
        } else if (h < Nq + Nkv) {
            wgt_src = s_k_wgt;
            dwgt_dst = s_dk_wgt;
        } else if (dout_i == dinp_i) {
            return;
        } else {
            for(int c = lane * x128::size; c < HeadDim; c += GroupSize * x128::size) {
                x128 in_data = x128::load_cs(dout_i + c);
                in_data.store(dinp_i + c);
            }
            continue;
        }

        float sum_xow = 0.0f;
        for (int i = lane * x128::size; i < HeadDim; i += GroupSize * x128::size) {
            x128 o = x128::load(dout_i + i);
            x128 x = x128::load(inp_i  + i);
            x128 w = x128::load(wgt_src + i);
            for (int k = 0; k < x128::size; k++) {
                sum_xow += (float)w[k] * (float)o[k] * (float)x[k];
            }
        }

        sum_xow = reduce_group_sum(sum_xow);
        const float xow_norm = sum_xow / HeadDim * rstd_i;

        for (int i = lane * x128::size; i < HeadDim; i += GroupSize * x128::size) {
            x128 o = x128::load_cs(dout_i + i);
            x128 x = x128::load_cs(inp_i + i);
            x128 w = x128::load(wgt_src + i);
            x128 dx = x128::load(dinp_i + i);

            fvec dw;
            for(int j = 0; j < x128::size / f128::size; ++j) {
                f128 dw_128 = f128::load(dwgt_dst + i / XF_STRIDE + j * HeadDim / XF_STRIDE + group * HeadDim);
                for (int k = 0; k < f128::size; k++) {
                    dw[j * f128::size + k] = dw_128[k];
                }
            }

            for (int k = 0; k < x128::size; k++) {
                float xn = (float)x[k] * rstd_i;
                dw[k] += xn * (float)o[k];
                float dx_k = ((float)o[k] * (float)w[k] - xn * xow_norm) * rstd_i + (float)dx[k];
                thread_abs_max = fmaxf(thread_abs_max, fabsf(dx_k));
                dx[k] = static_cast<Float>(dx_k);
            }

            dx.store(dinp_i + i);

            // Cache per-warp partial dweight in shared memory, addressed by absolute feature index
            for(int j = 0; j < x128::size / f128::size; ++j) {
                f128 dw_128;
                for (int k = 0; k < f128::size; k++) {
                    dw_128[k] = dw[j * f128::size + k];
                }
                dw_128.store(dwgt_dst + i / XF_STRIDE + j * HeadDim / XF_STRIDE + group * HeadDim);
            }
        }
    }

    // ok, at this point, dx is done, and each warp has partial dw results. Now we need to reduce across warps.
    // now we reduce across warps
    __syncthreads();

    float* src_dw = nullptr;
    float* scratch_dweight = nullptr;
    int l_idx;
    if (threadIdx.x < 32) {
        src_dw = s_dq_wgt_base;
        scratch_dweight = reinterpret_cast<float*>(scratch) + 2*HeadDim*blockIdx.x;
        l_idx = threadIdx.x;
    } else if (threadIdx.x < 64) {
        src_dw = s_dk_wgt_base;
        scratch_dweight = reinterpret_cast<float*>(scratch) + 2*HeadDim*blockIdx.x + HeadDim;
        l_idx = threadIdx.x - 32;
    }

    if (src_dw != nullptr) {
        for (int i = l_idx * f128::size; i < HeadDim; i += 32 * f128::size) {
            f128 dw_sum = f128::zeros();
            for (int w = 0; w < num_groups; ++w) {
                // Sum contributions from all warps
                f128 dw_other = f128::load(src_dw + w * HeadDim + i);
                for (int k = 0; k < f128::size; k++) {
                    dw_sum[k] += dw_other[k];
                }
            }
            dw_sum.store(scratch_dweight + i);
        }
    }

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);
}

template<class Float>
__global__ void __launch_bounds__(64, 1)
qk_norm_backward_reduce_kernel(Float* dq_wgt, Float* dk_wgt, const std::byte* scratch, int blocks, int HeadDim) {
    using x128 = GenericVector<Float, 16/sizeof(Float)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;
    using fvec = GenericVector<float, x128::size>;
    constexpr int XF_STRIDE = div_exact(x128::size, f128::size);

    const float* scratch_dweight = reinterpret_cast<const float*>(scratch);

    Float* dst_dw = nullptr;
    int l_idx;
    if (threadIdx.x < 32) {
        dst_dw = dq_wgt;
        l_idx = threadIdx.x;
    } else if (threadIdx.x < 64) {
        dst_dw = dk_wgt;
        scratch_dweight = scratch_dweight + HeadDim;
        l_idx = threadIdx.x - 32;
    }

    for (int b = 0; b < blocks; ++b) {
        for (int i = l_idx * x128::size; i < HeadDim; i += blockDim.x * x128::size) {
            x128 old_dw = x128::load(dst_dw + i);
            fvec summed;
            for (int k = 0; k < x128::size; k++) {
                summed[k] = static_cast<float>(old_dw[k]);
            }

            for(int j = 0; j < x128::size / f128::size; ++j) {
                // note: interleaved q/k means we move by 2x head dim for each block
                f128 dw_f = f128::load(scratch_dweight + i / XF_STRIDE + j * HeadDim / XF_STRIDE + b * 2 * HeadDim);
                for (int k = 0; k < f128::size; k++) {
                    summed[k + j * f128::size] += dw_f[k];
                }
            }

            for (int k = 0; k < x128::size; k++) {
                old_dw[k] = static_cast<Float>(summed[k]);
            }

            old_dw.store(dst_dw + i);
        }
    }
}


template<typename Float>
void qk_norm_forward(Float* out, float* r_rms,
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
        qk_norm_forward_simple_kernel<Float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));

    qk_norm_forward_simple_kernel<Float><<<grid_size, block_size, smem, stream>>>(
        out, r_rms, inp, q_wgt, k_wgt, epsilon, BT, Nq, Nkv, HeadDim);

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
void qk_norm_forward(float* out, float* r_rms, const float* inp,
                     const float* q_wgt, const float* k_wgt,
                     float epsilon, int BT, int Nq, int Nkv, int HeadDim,
                     cudaStream_t stream) {
    qk_norm_forward<float>(out, r_rms, inp, q_wgt, k_wgt, epsilon, BT, Nq, Nkv, HeadDim, stream);
}

void qk_norm_forward(nv_bfloat16* out, float* r_rms, const nv_bfloat16* inp,
                     const nv_bfloat16* q_wgt, const nv_bfloat16* k_wgt,
                     float epsilon, int BT, int Nq, int Nkv, int HeadDim,
                     cudaStream_t stream) {
    qk_norm_forward<nv_bfloat16>(out, r_rms, inp, q_wgt, k_wgt, epsilon, BT, Nq, Nkv, HeadDim, stream);
}
