// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//

#include "transpose_template.cuh"
#include "utilities/tensor.h"
#include "utilities/vec.cuh"

extern thread_local float* device_zero;
extern thread_local float* device_one;

template<class floatX>
__global__ void absmax_scale_kernel(float* result, const floatX* in, long N) {
    using vec_t = GenericVector<floatX, 16 / sizeof(floatX)>;

    __shared__ float local_max;
    if(threadIdx.x == 0) {
        local_max = 1e-10f;
    }
    __syncthreads();
    float thread_max = 0.f;
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        for(int j = 0; j < vec_t::size; ++j) {
            thread_max = std::max(thread_max, fabsf((float)values[j]));
        }
    }
    atomicMax_block(reinterpret_cast<unsigned int*>(&local_max), *reinterpret_cast<unsigned int*>(&thread_max));
    __syncthreads();
    if(threadIdx.x == 0) {
        thread_max = local_max;
        atomicMax(reinterpret_cast<unsigned int*>(result), *reinterpret_cast<unsigned int*>(&thread_max));
    }
    __syncthreads();
}

template<class floatX>
void absmax_scale_launcher(float* result, const floatX* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    int block_size = dp.maxThreadsPerMultiProcessor == 2048 ? 1024 : 768;
    int n_blocks = dp.maxThreadsPerMultiProcessor / block_size * dp.multiProcessorCount;
    CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(float), stream));
    absmax_scale_kernel<<<n_blocks, block_size, 0, stream>>>(result, in, N);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void quantize_with_abs_max_kernel(nv_bfloat16* out, const float* in, const float* abs_max, long N) {
    using vec_t = GenericVector<float, 16 / sizeof(float)>;
    using bfv_t = GenericVector<nv_bfloat16, 16 / sizeof(nv_bfloat16)>;
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        bfv_t quants;
        for(int j = 0; j < vec_t::size; ++j) {
            out[i] = (nv_bfloat16)values[i];
        }
        quants.store(out + i);
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = (nv_bfloat16) in[i];
    }
}

template<class floatX>
__global__ void quantize_with_abs_max_kernel(std::int8_t* out, const floatX* in, const float* abs_max, long N) {
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
template<class floatX>
__global__ void quantize_with_abs_max_kernel(__nv_fp8_e4m3* out, const floatX* in, const float* abs_max, long N) {
    using vec_t = GenericVector<floatX, 16 / sizeof(floatX)>;
    using f8v_t = GenericVector<__nv_fp8_e4m3, 16 / sizeof(floatX)>;
    float scale = 448.f / *abs_max;
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        f8v_t quants;
        for(int j = 0; j < vec_t::size; ++j) {
            __nv_fp8_e4m3 result;
            result.__x = __nv_cvt_float_to_fp8(scale * (float)values[j], __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
            quants[j] = result;
        }
        quants.store(out + i);
    }
}

template<class floatY, class floatX>
void quantize_with_abs_max_launcher(floatY* out, const floatX* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    int block_size = dp.maxThreadsPerMultiProcessor == 2048 ? 1024 : 768;
    int n_blocks = dp.maxThreadsPerMultiProcessor / block_size * dp.multiProcessorCount;
    quantize_with_abs_max_kernel<<<n_blocks, block_size, 0, stream>>>(out, in, abs_max, N);
    CUDA_CHECK(cudaGetLastError());
}

void abs_max(float* scale, const float* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    absmax_scale_launcher(scale, in, N, dp, stream);
}

void abs_max(float* scale, const nv_bfloat16* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    absmax_scale_launcher(scale, in, N, dp, stream);
}

void quantize_with_abs_max(nv_bfloat16* out, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, in, abs_max, N, dp, stream);
}
void quantize_with_abs_max(std::int8_t* out, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, in, abs_max, N, dp, stream);
}
void quantize_with_abs_max(__nv_fp8_e4m3* out, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, in, abs_max, N, dp, stream);
}

void quantize_with_abs_max(std::int8_t* out, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, in, abs_max, N, dp, stream);
}
void quantize_with_abs_max(__nv_fp8_e4m3* out, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, in, abs_max, N, dp, stream);
}

__global__ void matmul_out_scale_kernel(float* out, const float* a, const float* b, float host_scale) {
    *out = (*a) * (*b) * host_scale;
}

void matmul_out_scale(float* out, const float* a, const float* b, float dtype_scale, cudaStream_t stream) {
    matmul_out_scale_kernel<<<1, 1, 0, stream>>>(out, a, b, dtype_scale);
    CUDA_CHECK(cudaGetLastError());
}

template<int BLK>
__global__ void quantize_and_transpose_with_abs_max_kernel(nv_bfloat16* out, const float* in, const float*, int rows, int cols) {
    apply_and_transpose_helper<BLK>([](auto&& a){ return (nv_bfloat16)a; }, out, in, rows, cols);
}

template<int BLK, class floatX>
__global__ void quantize_and_transpose_with_abs_max_kernel(std::int8_t* out, const floatX* in, const float* abs_max, int rows, int cols) {
    float scale = static_cast<float>(std::numeric_limits<std::int8_t>::max()) / *abs_max;
    auto cvt = [scale](auto&& in_val) -> std::int8_t {
        auto out_val = std::max((float) std::numeric_limits<std::int8_t>::min(),
                                std::min((float) std::numeric_limits<std::int8_t>::max(), scale * (float)in_val));
        return out_val;
    };

    apply_and_transpose_helper<BLK>(cvt, out, in, rows, cols);
}

template<int BLK, class floatX>
__global__ void quantize_and_transpose_with_abs_max_kernel(__nv_fp8_e4m3* out, const floatX* in, const float* abs_max, int rows, int cols) {
    float scale = 448.f / *abs_max;
    auto cvt = [scale](auto&& in_val) -> __nv_fp8_e4m3 {
        __nv_fp8_e4m3 out_val;
        out_val.__x = __nv_cvt_float_to_fp8(scale * (float)in_val, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        return out_val;
    };

    apply_and_transpose_helper<BLK>(cvt, out, in, rows, cols);
}

template<class floatIn, class floatOut>
void quantize_and_transpose_with_abs_max_imp(floatOut* out, const floatIn* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    dim3 block_size = {8, 8};
    const int BLK = std::is_same_v<floatIn, float> ? 4 : 8;
    dim3 grid_size = {(unsigned)div_ceil(rows, BLK*(int)block_size.x), (unsigned)div_ceil(cols, BLK*(int)block_size.y)};
    quantize_and_transpose_with_abs_max_kernel<BLK><<<grid_size, block_size, 0, stream>>>(out, in, abs_max, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_and_transpose_with_abs_max(nv_bfloat16* out, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, in, abs_max, rows, cols, dp, stream);
}

void quantize_and_transpose_with_abs_max(std::int8_t* out, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, in, abs_max, rows, cols, dp, stream);
}

void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, in, abs_max, rows, cols, dp, stream);
}

void quantize_and_transpose_with_abs_max(std::int8_t* out, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, in, abs_max, rows, cols, dp, stream);
}

void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, in, abs_max, rows, cols, dp, stream);
}
