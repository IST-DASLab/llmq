// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/dtype.h"
#include "utilities/vec.cuh"

constexpr const int GroupSize = 8;

__device__ float reduce_group_sum(float acc, unsigned int active_mask) {
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

    acc = reduce_group_sum(acc, 0xffffffff) / HeadDim;
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
    // strategy: y-dimension of block indicates which head is operated on
    //           x-dimension goes over tokens in groups of eight threads
    using x128 = GenericVector<Float, 16/sizeof(Float)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;
    using fvec = GenericVector<float, x128::size>;

    const int lane = threadIdx.x % GroupSize;
    const int group = threadIdx.x / GroupSize;
    const int num_groups = blockDim.x / GroupSize;
    const int h = blockIdx.y;
    const int Nh = Nq + 2 * Nkv;

    // load weights into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* smem[];

    __shared__ float block_abs_max;
    float thread_abs_max = 0.f;

    Float* s_wgt = reinterpret_cast<Float*>(smem);
    float* s_d_wgt_base = reinterpret_cast<float*>(s_wgt + HeadDim);
    float* s_d_wgt = s_d_wgt_base + HeadDim * group;

    for(int i = threadIdx.x * x128::size; i < HeadDim; i += blockDim.x * x128::size) {
        if(h < Nq) {
            x128::load(q_wgt + i).store(s_wgt + i);
        } else if (h < Nq + Nkv){
            x128::load(k_wgt + i).store(s_wgt + i);
        }
        // v-heads need no loading
    }

    for(int i = threadIdx.x * f128::size; i < HeadDim * num_groups; i += blockDim.x * f128::size) {
        f128::zeros().store(s_d_wgt_base + i);
    }

    if (abs_max_ptr && threadIdx.x == 0) {
        block_abs_max = 0.f;
    }

    __syncthreads();

    const int groups_in_grid = num_groups * gridDim.x;
    const int start_idx = blockIdx.x * (blockDim.x / GroupSize) + group;

    for (int bt = start_idx; ; bt += groups_in_grid) {
        bool valid = bt < BT;
        unsigned int active_mask = __ballot_sync(0xffffffffu, valid);
        bool all_finished = !__any_sync(0xffffffffu, valid);
        if (all_finished)
            break;
        if (!valid)
            continue;

        // adjusted pointers to current token
        const Float* inp_i = inp + bt * Nh * HeadDim + h * HeadDim;
        const Float* dout_i = dout + bt * Nh * HeadDim + h * HeadDim;
        Float* dinp_i = dinp + bt * Nh * HeadDim + h * HeadDim;
        const float rstd_i = rstd[bt * Nh + h];

        // V heads
        if (h >= Nq + Nkv) {
            if (dout_i != dinp_i) {
                for(int c = lane * x128::size; c < HeadDim; c += GroupSize * x128::size) {
                    x128 in_data = x128::load_cs(dout_i + c);
                    if (abs_max_ptr) {
                        for (int k = 0; k < x128::size; k++) {
                            thread_abs_max = fmaxf(thread_abs_max, fabsf(in_data[k]));
                        }
                    }
                    in_data.store(dinp_i + c);
                }
            }
            continue;
        }

        // QK heads
        float sum_xow = 0.0f;
        for (int i = lane * x128::size; i < HeadDim; i += GroupSize * x128::size) {
            x128 o = x128::load(dout_i + i);
            x128 x = x128::load(inp_i  + i);
            x128 w = x128::load(s_wgt  + i);
            for (int k = 0; k < x128::size; k++) {
                sum_xow += (float)w[k] * (float)o[k] * (float)x[k];
            }
        }

        sum_xow = reduce_group_sum(sum_xow, active_mask);
        const float xow_norm = sum_xow / HeadDim * rstd_i;

        for (int i = lane * x128::size; i < HeadDim; i += GroupSize * x128::size) {
            x128 o = x128::load_cs(dout_i + i);
            x128 x = x128::load_cs(inp_i + i);
            x128 w = x128::load(s_wgt + i);
            x128 dx = x128::load(dinp_i + i);

            fvec dw = fvec::load(s_d_wgt + i);
            for (int k = 0; k < x128::size; k++) {
                float xn = (float)x[k] * rstd_i;
                dw[k] += xn * (float)o[k];
                float dx_k = ((float)o[k] * (float)w[k] - xn * xow_norm) * rstd_i + (float)dx[k];
                thread_abs_max = fmaxf(thread_abs_max, fabsf(dx_k));
                dx[k] = static_cast<Float>(dx_k);
            }

            dx.store(dinp_i + i);

            // Cache per-warp partial dweight in shared memory
            dw.store(s_d_wgt_base + group * HeadDim + i);
        }
    }

    __syncthreads();
    // reduce across the block
    if (threadIdx.x < 32) {
        float* scratch_dweight = reinterpret_cast<float*>(scratch) + HeadDim * (blockIdx.x + blockIdx.y * gridDim.x);
        for (int i = threadIdx.x * f128::size; i < HeadDim; i += 32 * f128::size) {
            f128 accumulated = f128::zeros();
            for (int g = 0; g < num_groups; ++g) {
                f128 dw_128 = f128::load(s_d_wgt_base + g * HeadDim + i);
                for (int k = 0; k < f128::size; k++) {
                    accumulated[k] += dw_128[k];
                }
            }
            accumulated.store(scratch_dweight + i);
        }
    }

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);
}

template<class Float>
__global__ void __launch_bounds__(32, 1)
qk_norm_backward_reduce_kernel(Float* dq_wgt, Float* dk_wgt, const std::byte* scratch,
                               int x_blocks, int Nq, int Nkv, int HeadDim) {
    using f128 = GenericVector<float, 16/sizeof(float)>;

    const float* scratch_base = reinterpret_cast<const float*>(scratch);

    // blockIdx.y == 0 -> Q weights, blockIdx.y == 1 -> K weights
    const bool is_q = (blockIdx.y == 0);
    Float* dst_dw = is_q ? dq_wgt : dk_wgt;
    const int head_start = is_q ? 0 : Nq;
    const int num_heads  = is_q ? Nq : Nkv;

    for (int i = threadIdx.x * f128::size; i < HeadDim; i += 32 * f128::size) {
        f128 summed;
        // load old_dw directly as f128 via float cast
        for (int k = 0; k < f128::size; k++) {
            summed[k] = static_cast<float>(dst_dw[i + k]);
        }

        for (int bx = 0; bx < x_blocks; ++bx) {
            for (int hi = 0; hi < num_heads; ++hi) {
                int h = head_start + hi;
                const float* scratch_dweight = scratch_base + HeadDim * (bx + h * x_blocks);
                f128 dw_f = f128::load(scratch_dweight + i);
                for (int k = 0; k < f128::size; k++) {
                    summed[k] += dw_f[k];
                }
            }
        }

        for (int k = 0; k < f128::size; k++) {
            dst_dw[i + k] = static_cast<Float>(summed[k]);
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

// Get the amount of smem per block for backward
template<typename Float>
 size_t qk_norm_backward_smem(int HeadDim) {
    constexpr int block_size = 512;
    constexpr int num_groups = block_size / GroupSize;
    return HeadDim * sizeof(Float) + num_groups * HeadDim * sizeof(float);
}

// Get the maximum number of concurrent blocks in x direction
template<typename Float>
static int qk_norm_backward_x_blocks(int Nq, int Nkv, int HeadDim,
                                     const cudaDeviceProp& dp) {
    const int Nh = Nq + 2 * Nkv;
    const size_t smem = qk_norm_backward_smem<Float>(HeadDim);
    int blocks_per_sm;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, qk_norm_backward_kernel<Float>, 512, smem));

    int total_blocks = blocks_per_sm * dp.multiProcessorCount;
    if (total_blocks < Nh) {
        // won't be able to run all blocks in parallel, some will be serialized.
        return 1;
    }
    return total_blocks / Nh;
}

std::size_t qk_norm_backward_scratch_size(int Nq, int Nkv, int HeadDim,
                                      ETensorDType dtype, const cudaDeviceProp& dp) {
    const int Nh = Nq + 2 * Nkv;
    int x_blocks;
    switch(dtype) {
        case ETensorDType::FP32:
            x_blocks = qk_norm_backward_x_blocks<float>(Nq, Nkv, HeadDim, dp);
            break;
        case ETensorDType::BF16:
            x_blocks = qk_norm_backward_x_blocks<nv_bfloat16>(Nq, Nkv, HeadDim, dp);
            break;
        default:
            throw std::invalid_argument("Unsupported dtype");
    }
    return (size_t)HeadDim * x_blocks * Nh * sizeof(float);
}


template<class Float>
void qk_norm_backward_imp(Float* dinp, Float* dq_wgt, Float* dk_wgt, std::byte* scratch,
                          const Float* dout, const Float* inp,
                          const Float* q_wgt, const Float* k_wgt,
                          const float* rstd, float* abs_max_ptr,
                          float epsilon, int BT, int Nq, int Nkv, int HeadDim,
                          const cudaDeviceProp& dp, cudaStream_t stream)
{
    constexpr int block_size = 512;
    const int Nh = Nq + 2 * Nkv;
    const size_t smem = qk_norm_backward_smem<Float>(HeadDim);
    const int x_blocks = qk_norm_backward_x_blocks<Float>(Nq, Nkv, HeadDim, dp);

    dim3 grid(x_blocks, Nh);
    qk_norm_backward_kernel<Float><<<grid, block_size, smem, stream>>>(
        dinp, scratch, dout, inp, q_wgt, k_wgt, rstd, abs_max_ptr,
        epsilon, BT, Nq, Nkv, HeadDim);
    CUDA_CHECK(cudaGetLastError());

    dim3 reduce_grid(1, 2);
    qk_norm_backward_reduce_kernel<Float><<<reduce_grid, 32, 0, stream>>>(
        dq_wgt, dk_wgt, scratch, x_blocks, Nq, Nkv, HeadDim);
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

void qk_norm_backward(float* dinp, float* dq_wgt, float* dk_wgt, std::byte* scratch,
                      const float* dout, const float* inp,
                      const float* q_wgt, const float* k_wgt,
                      const float* rstd, float* abs_max_ptr,
                      float epsilon, int BT, int Nq, int Nkv, int HeadDim,
                      const cudaDeviceProp& dp, cudaStream_t stream) {
    qk_norm_backward_imp<float>(dinp, dq_wgt, dk_wgt, scratch, dout, inp, q_wgt, k_wgt, rstd, abs_max_ptr, epsilon, BT, Nq, Nkv, HeadDim, dp, stream);
}

void qk_norm_backward(nv_bfloat16* dinp, nv_bfloat16* dq_wgt, nv_bfloat16* dk_wgt, std::byte* scratch,
                      const nv_bfloat16* dout, const nv_bfloat16* inp,
                      const nv_bfloat16* q_wgt, const nv_bfloat16* k_wgt,
                      const float* rstd, float* abs_max_ptr,
                      float epsilon, int BT, int Nq, int Nkv, int HeadDim,
                      const cudaDeviceProp& dp, cudaStream_t stream) {
    qk_norm_backward_imp<nv_bfloat16>(dinp, dq_wgt, dk_wgt, scratch, dout, inp, q_wgt, k_wgt, rstd, abs_max_ptr, epsilon, BT, Nq, Nkv, HeadDim, dp, stream);
}
