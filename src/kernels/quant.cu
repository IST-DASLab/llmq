// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_template.cuh"
#include "kernel_utils.cuh"
#include "utilities/tensor.h"
#include "utilities/vec.cuh"

template<class floatX>
__global__ void reduce_abs_max_kernel(float* __restrict__ result, const floatX* __restrict__ in, long N) {
    using vec_t = GenericVector<floatX, 16 / sizeof(floatX)>;

    __shared__ float block_abs_max;
    if(threadIdx.x == 0) {
        block_abs_max = 0.f;
    }
    __syncthreads();
    float thread_abs_max = 0.f;
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        for(int j = 0; j < vec_t::size; ++j) {
            thread_abs_max = fmaxf(thread_abs_max, fabsf((float)values[j]));
        }
    }

    handle_absmax_reduction(result, &block_abs_max, thread_abs_max);
}

template<class floatX>
void reduce_abs_max_launcher(float* result, const floatX* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    int block_size = dp.maxThreadsPerMultiProcessor == 2048 ? 1024 : 768;
    int n_blocks = dp.maxThreadsPerMultiProcessor / block_size * dp.multiProcessorCount;
    CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(float), stream));
    reduce_abs_max_kernel<<<n_blocks, block_size, 0, stream>>>(result, in, N);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void quantize_with_abs_max_kernel(nv_bfloat16* __restrict__ out, float* __restrict__ scale_ptr,
                                             const float* __restrict__ in, const float* __restrict__ abs_max, long N)
{
    using vec_t = GenericVector<float, 16 / sizeof(float)>;
    using bfv_t = GenericVector<nv_bfloat16, 16 / sizeof(nv_bfloat16)>;
    if(threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f;
    }

    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        bfv_t quants;
        for(int j = 0; j < vec_t::size; ++j) {
            quants[j] = (nv_bfloat16)values[j];
        }
        quants.store(out + i);
    }
}

template<class floatX>
__global__ void quantize_with_abs_max_kernel(std::int8_t* out, float* scale_ptr, const floatX* in, const float* abs_max, long N) {
    using vec_t = GenericVector<floatX, 16 / sizeof(floatX)>;
    using i8v_t = GenericVector<std::int8_t, 16 / sizeof(floatX)>;
    float scale = (float)std::numeric_limits<std::int8_t>::max() / *abs_max;
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        i8v_t quants;
        for(int j = 0; j < vec_t::size; ++j) {
            out[i] = (std::int8_t) std::max((float) std::numeric_limits<std::int8_t>::min(),
                                            std::min((float) std::numeric_limits<std::int8_t>::max(), scale * (float)values[j]));
        }
        quants.store(out + i);
    }
}
template<typename FloatOut, typename FloatIn,
    typename _ = std::enable_if_t<std::is_same_v<FloatOut, __nv_fp8_e4m3> || std::is_same_v<FloatOut, __nv_fp8_e5m2>, void>>
__global__ void quantize_with_abs_max_kernel(FloatOut* __restrict__ out, float* __restrict__ scale_ptr,
                                             const FloatIn* __restrict__ in, const float* __restrict__ abs_max, long N) {
    using vec_t = GenericVector<FloatIn, 16 / sizeof(FloatIn)>;
    using f8v_t = GenericVector<FloatOut, 16 / sizeof(FloatIn)>;
    float scale = 448.f / fmaxf(*abs_max, 1e-10f);
    if(threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        f8v_t quants;
        for(int j = 0; j < vec_t::size; ++j) {
            FloatOut result;
            result.__x = __nv_cvt_float_to_fp8(scale * (float)values[j], __nv_saturation_t::__NV_SATFINITE, fp8_interpretation_v<FloatOut>);
            quants[j] = result;
        }
        quants.store(out + i);
    }
}

template<class floatY, class floatX>
void quantize_with_abs_max_launcher(floatY* out, float* scale_ptr, const floatX* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    int block_size = dp.maxThreadsPerMultiProcessor == 2048 ? 1024 : 768;
    int n_blocks = dp.maxThreadsPerMultiProcessor / block_size * dp.multiProcessorCount;
    quantize_with_abs_max_kernel<<<n_blocks, block_size, 0, stream>>>(out, scale_ptr, in, abs_max, N);
    CUDA_CHECK(cudaGetLastError());
}

void abs_max(float* scale, const float* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    reduce_abs_max_launcher(scale, in, N, dp, stream);
}

void abs_max(float* scale, const nv_bfloat16* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    reduce_abs_max_launcher(scale, in, N, dp, stream);
}

void quantize_with_abs_max(nv_bfloat16* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}
void quantize_with_abs_max(std::int8_t* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}
void quantize_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}
void quantize_with_abs_max(__nv_fp8_e5m2* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

void quantize_with_abs_max(std::int8_t* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}
void quantize_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}
void quantize_with_abs_max(__nv_fp8_e5m2* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

template<int BLK>
__global__ void quantize_and_transpose_with_abs_max_kernel(nv_bfloat16* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols) {
    apply_and_transpose_helper<BLK>([](auto&& a){ return (nv_bfloat16)a; }, out, in, rows, cols);
}

template<int BLK, class floatX>
__global__ void quantize_and_transpose_with_abs_max_kernel(std::int8_t* out, float* scale_ptr, const floatX* in, const float* abs_max, int rows, int cols) {
    float scale = static_cast<float>(std::numeric_limits<std::int8_t>::max()) / *abs_max;
    auto cvt = [scale](auto&& in_val) -> std::int8_t {
        auto out_val = std::max((float) std::numeric_limits<std::int8_t>::min(),
                                std::min((float) std::numeric_limits<std::int8_t>::max(), scale * (float)in_val));
        return out_val;
    };

    apply_and_transpose_helper<BLK>(cvt, out, in, rows, cols);
}

template<int BLK, class floatX>
__global__ void quantize_and_transpose_with_abs_max_kernel(__nv_fp8_e4m3* out, float* scale_ptr, const floatX* in, const float* abs_max, int rows, int cols) {
    float scale = 448.f / fmaxf(*abs_max, 1e-10f);
    if(threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }
    auto cvt = [scale](auto&& in_val) -> __nv_fp8_e4m3 {
        __nv_fp8_e4m3 out_val;
        out_val.__x = __nv_cvt_float_to_fp8(scale * (float)in_val, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        return out_val;
    };

    apply_and_transpose_helper<BLK>(cvt, out, in, rows, cols);
}

template<class floatIn, class floatOut>
void quantize_and_transpose_with_abs_max_imp(floatOut* out, float* scale_ptr, const floatIn* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    dim3 block_size = {8, 8};
    const int BLK = std::is_same_v<floatIn, float> ? 4 : 8;
    dim3 grid_size = {(unsigned)div_ceil(rows, BLK*(int)block_size.x), (unsigned)div_ceil(cols, BLK*(int)block_size.y)};
    quantize_and_transpose_with_abs_max_kernel<BLK><<<grid_size, block_size, 0, stream>>>(out, scale_ptr, in, abs_max, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_and_transpose_with_abs_max(nv_bfloat16* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

void quantize_and_transpose_with_abs_max(std::int8_t* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

void quantize_and_transpose_with_abs_max(std::int8_t* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}
