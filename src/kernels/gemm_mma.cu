#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include "tensor_core_utils.h"
#include "utilities/vec.cuh"
#include <cstdio>
#undef NDEBUG
#include <cassert>
#include <type_traits>

unsigned div_ceil(unsigned a, unsigned b) {
    return (a + b - 1) / b;
}

template<int V>
using int_c = std::integral_constant<int, V>;

template<typename T>
std::type_identity<T> type_v = {};

template<typename AType, typename BType, typename BiasType, typename AccType>
__global__ __launch_bounds__(32*2*2, 2) void gemm_mma_tn_kernel(nv_bfloat16* __restrict__ out,
                                                                const AType* __restrict__ a, const BType* __restrict__ b,
                                                                int m, int n, int k, const float* __restrict__ scale,
                                                                const BiasType* __restrict__ bias,
                                                                bool accumulate,
                                                                std::type_identity<AccType> acc_type) {
    static_assert(sizeof(AType) == sizeof(BType), "index calculations assume sz(AType) == sz(BType)");
    // Note: you cannot change these numbers without breaking the kernel.
    // they are here only for convenience, not to parametrize the algorithm.
    // also note that some of the smaller loops have been unrolled by hand
    // to have a nicer experience in ncu
    constexpr int BI = 2;
    constexpr int BJ = 2;
    constexpr int WI = 4;
    constexpr int WJ = 4;
    constexpr int DEPTH = 4;

    constexpr int TI = 16;
    constexpr int TJ = 16;
    constexpr int TK = 2;       // in units of uint4

    int bi;
    int bj;
    if( m > n ) {
        bi = blockIdx.y * BI * WI;
        bj = blockIdx.x * BJ * WJ;
    } else {
        bi = blockIdx.x * BI * WI;
        bj = blockIdx.y * BJ * WJ;
    }
    int i = bi + threadIdx.y * WI;
    int j = bj + threadIdx.z * WJ;

    int wid = threadIdx.y + 2 * threadIdx.z;
    constexpr int NW = BI * BJ;

    int stride = k / (sizeof(uint4)/sizeof(AType));

    m16_n16_k32_c_fragment<AccType> acc[WI][WJ];
    __shared__ uint4 input_tiles[2 * DEPTH * WI * BI * TI * TK];
    int2 offsets = ldmatrix_offsets();
    constexpr int PIPE_OFFSET = WI * BI * TI * TK;
    constexpr int ROW_OFFSET = TI * TK;

    // instead of each thread computing addresses in both A and B, specialize 2 warps
    // for A-loading and 2 warps for B-loading.

    // const uint4* a_ptr = reinterpret_cast<const uint4*>(a) + (bi + wid) * TI * stride;
    // const uint4* b_ptr = reinterpret_cast<const uint4*>(b) + (bj + wid) * TJ * stride;
    // uint4* as_store_ptr = input_tiles + wid * ROW_OFFSET;
    // uint4* bs_store_ptr = input_tiles + wid * ROW_OFFSET + DEPTH * PIPE_OFFSET;

    const uint4* g_ptr;
    uint4* s_ptr;
    if(wid < 2) {
        g_ptr = reinterpret_cast<const uint4*>(a) + (bi + wid) * TI * stride;
        s_ptr = input_tiles + wid * ROW_OFFSET;
    } else {
        g_ptr = reinterpret_cast<const uint4*>(b) + (bj + wid - 2) * TJ * stride;
        s_ptr = input_tiles + (wid - 2) * ROW_OFFSET + DEPTH * PIPE_OFFSET;
    }

    global_to_shared_16_32_swizzle(&s_ptr, &g_ptr, stride);

    const uint4* as_load_ptr = input_tiles + offsets.x + threadIdx.y * WI*ROW_OFFSET;
    const uint4* bs_load_ptr = input_tiles + offsets.y + threadIdx.z * WJ*ROW_OFFSET + DEPTH * PIPE_OFFSET;

    static_assert(WI * BI % NW == 0, "WI * BI must be divisible by the number of warps per block");
    static_assert(WJ * BJ % NW == 0, "WI * BI must be divisible by the number of warps per block");

    auto loop_fraction = [&](auto stage_c, auto load_next_c, int ks) {
        constexpr int load_stage = decltype(stage_c)::value;
        constexpr int store_stage = (load_stage + 3) % DEPTH;
        constexpr bool load_next = decltype(load_next_c)::value;

        m16_n16_k32_a_fragment<AType> a_frag[WI];
        ptx_ldmatrix(a_frag[0].v, as_load_ptr + 0 * ROW_OFFSET + PIPE_OFFSET * load_stage);
        ptx_ldmatrix(a_frag[1].v, as_load_ptr + 1 * ROW_OFFSET + PIPE_OFFSET * load_stage);
        ptx_ldmatrix(a_frag[2].v, as_load_ptr + 2 * ROW_OFFSET + PIPE_OFFSET * load_stage);
        ptx_ldmatrix(a_frag[3].v, as_load_ptr + 3 * ROW_OFFSET + PIPE_OFFSET * load_stage);

        if constexpr(load_next) {
            __pipeline_memcpy_async(s_ptr + store_stage * PIPE_OFFSET + 0 * ROW_OFFSET, g_ptr + 0 * TI * stride + ks, 16);
            __pipeline_memcpy_async(s_ptr + store_stage * PIPE_OFFSET + 2 * ROW_OFFSET, g_ptr + 2 * TI * stride + ks, 16);
            __pipeline_memcpy_async(s_ptr + store_stage * PIPE_OFFSET + 4 * ROW_OFFSET, g_ptr + 4 * TI * stride + ks, 16);
            __pipeline_memcpy_async(s_ptr + store_stage * PIPE_OFFSET + 6 * ROW_OFFSET, g_ptr + 6 * TI * stride + ks, 16);
        }
        __pipeline_commit();
        for(int jj = 0; jj < WJ; jj++) {
            m16_n16_k32_b_fragment<BType> b_frag;
            ptx_ldmatrix(b_frag.v, bs_load_ptr + jj * ROW_OFFSET + PIPE_OFFSET * load_stage);
            for(int ii = 0; ii < WI; ii++) {
                mma_m16_n16_k32_sync(acc[ii][jj], a_frag[ii], b_frag, acc[ii][jj]);
            }
        }
        __pipeline_wait_prior(2);
        __syncthreads();
    };

    auto ldg_sts = [&](auto stage_c, int ks) {
        constexpr int stage = decltype(stage_c)::value;
        __pipeline_memcpy_async(s_ptr + stage * PIPE_OFFSET + 0 * ROW_OFFSET, g_ptr + 0 * TI * stride + ks, 16);
        __pipeline_memcpy_async(s_ptr + stage * PIPE_OFFSET + 2 * ROW_OFFSET, g_ptr + 2 * TI * stride + ks, 16);
        __pipeline_memcpy_async(s_ptr + stage * PIPE_OFFSET + 4 * ROW_OFFSET, g_ptr + 4 * TI * stride + ks, 16);
        __pipeline_memcpy_async(s_ptr + stage * PIPE_OFFSET + 6 * ROW_OFFSET, g_ptr + 6 * TI * stride + ks, 16);
    };

    // start up the pipeline
    ldg_sts(int_c<0>{}, 0);
    __pipeline_commit();
    ldg_sts(int_c<1>{}, 1 * TK);
    __pipeline_commit();
    ldg_sts(int_c<2>{}, 2 * TK);
    __pipeline_commit();

    std::bool_constant<true> true_v;
    std::bool_constant<false> false_v;

    __pipeline_wait_prior(2);
    __syncthreads();

    int ks = 0;
    while (ks + 6 * TK < stride) {
        loop_fraction(int_c<0>{}, true_v, ks + 3 * TK);
        loop_fraction(int_c<1>{}, true_v, ks + 4 * TK);
        loop_fraction(int_c<2>{}, true_v, ks + 5 * TK);
        loop_fraction(int_c<3>{}, true_v, ks + 6 * TK);

        ks += 4 * TK;
    }

    // last iteration
    loop_fraction(int_c<0>{}, true_v, ks + 3 * TK);
    loop_fraction(int_c<1>{}, false_v, ks + 4 * TK);
    loop_fraction(int_c<2>{}, false_v, ks + 5 * TK);
    loop_fraction(int_c<3>{}, false_v, ks + 6 * TK);

    if(scale != nullptr && *scale != 1.f) {
        for (auto ii = 0; ii < WI; ++ii) {
            for (int jj = 0; jj < WJ; ++jj) {
                acc[ii][jj].v[0] *= *scale;
                acc[ii][jj].v[1] *= *scale;
                acc[ii][jj].v[2] *= *scale;
                acc[ii][jj].v[3] *= *scale;
                acc[ii][jj].v[4] *= *scale;
                acc[ii][jj].v[5] *= *scale;
                acc[ii][jj].v[6] *= *scale;
                acc[ii][jj].v[7] *= *scale;
            }
        }
    }

    // note: loop_fraction ends with __syncthreads, so no need to sync here
    nv_bfloat16* out_shared = reinterpret_cast<nv_bfloat16*>(input_tiles) + (threadIdx.y + 2 * threadIdx.z) * TJ * TI;

    for(int ii = 0; ii < WI; ii++) {
        for (int jj = 0; jj < WJ; jj++) {
            store_fragment_row_major_sync(acc[ii][jj], out_shared, TJ);
            __syncwarp();
            int c = threadIdx.x % 2;
            int r = threadIdx.x / 2;

            if(accumulate) {
                auto old = GenericVector<nv_bfloat16, 8>::load(out + ((i + ii) * TI + r) * n + (j + jj) * TJ + 8 * c);
                auto upd = GenericVector<nv_bfloat16, 8>::load(out_shared + (c + 2 * r) * 8);
                for(int l = 0; l < 8; ++l) {
                    old[l] += upd[l];
                }
                old.store(out + ((i + ii) * TI + r) * n + (j + jj) * TJ + 8 * c);
            } else if (bias != nullptr) {
                auto old = GenericVector<BiasType, 8>::load(bias + (j + jj) * TJ + 8 * c);
                auto upd = GenericVector<nv_bfloat16, 8>::load(out_shared + (c + 2 * r) * 8);
                for(int l = 0; l < 8; ++l) {
                    old[l] += (nv_bfloat16)upd[l];
                }
                old.store(out + ((i + ii) * TI + r) * n + (j + jj) * TJ + 8 * c);
            } else {
                uint4 load = reinterpret_cast<uint4*>(out_shared)[c + 2 * r];
                *reinterpret_cast<uint4*>(out + ((i + ii) * TI + r) * n + (j + jj) * TJ + 8 * c) = load;
            }
        }
    }
}

template<class AType, class BType, class BiasType, class AccType>
void gemm_mma_tn_launcher(nv_bfloat16* out, const AType* a, const BType* b, int m, int n, int k, const float* scale, const BiasType* bias,
                          bool accumulate, std::type_identity<AccType>, cudaStream_t stream) {
    // our kernel is row-major, so to match cublas, we need to transpose everything => swapped a<->b, m<->n
    dim3 grid{(unsigned)n / 128, (unsigned)m / 128, 1};
    if( n > m ) {
        grid = {(unsigned)m / 128, (unsigned)n / 128, 1};
    } else {
        grid = {(unsigned)n / 128, (unsigned)m / 128, 1};
    }
    dim3 block{32, 2, 2};
    gemm_mma_tn_kernel<<<grid, block, 0, stream>>>(out, b, a, n, m, k, scale, bias, accumulate, type_v<AccType>);
}

void gemm_mma_tn(nv_bfloat16* out, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, int m, int n, int k, const float* scale, const nv_bfloat16* bias, bool accumulate, cudaStream_t stream) {
    gemm_mma_tn_launcher(out, a, b, m, n, k, scale, bias, accumulate, type_v<float>, stream);
    assert(cudaGetLastError() == cudaSuccess);
}

void gemm_mma_tn(nv_bfloat16* out, const nv_bfloat16* a, const nv_bfloat16* b, int m, int n, int k, const float* scale, const nv_bfloat16* bias, bool accumulate, cudaStream_t stream) {
    gemm_mma_tn_launcher(out, a, b, m, n, k, scale, bias, accumulate, type_v<float>, stream);
    assert(cudaGetLastError() == cudaSuccess);
}
