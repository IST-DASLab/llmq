// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//
// SPDX-License-Identifier: MIT
// Based on llm.c https://github.com/karpathy/llm.c

#include <cstdio>
#include <format>
#include <optional>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "kernels.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;

thread_local float* device_zero;
thread_local float* device_one;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
inline void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::format("cuBLAS ERROR ({}) at {}:{}", (int)status, file, line));
    }
}
#define CUBLAS_CHECK(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// Setup

cublasLtHandle_t create_cublaslt_handle() {
    cublasLtHandle_t handle;
    CUBLAS_CHECK(cublasLtCreate(&handle));
    return handle;
}

void setup_cublas() {
    // set up cuBLAS and cuBLASLt
    CUDA_CHECK(cudaMalloc(&device_zero, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_one, sizeof(float)));
    const float zero = 0.f;
    CUDA_CHECK(cudaMemcpy(device_zero, &zero, sizeof(float), cudaMemcpyHostToDevice));
    const float one = 1.f;
    CUDA_CHECK(cudaMemcpy(device_one, &one, sizeof(float), cudaMemcpyHostToDevice));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    bool enable_tf32 = false;//PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8 && override_enable_tf32;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

// ----------------------------------------------------------------------------
// kernel launchers

// Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
template<class floatO, class floatX, class floatB>
void matmul_cublaslt(floatO* d, const floatX* a, const floatX* b, const floatB* bias,
                     std::byte* workspace, std::size_t workspace_size,
                     int m, int n, int k, cudaStream_t stream, cublasLtHandle_t handle,
                     const float* scale=nullptr, bool transA=true, bool transB=false,
                     bool accumulate=false, bool backward=false)
{
    NVTX_RANGE_FN();
    bool has_bias = (bias != nullptr);

    // check alignment (some modes work unaligned, but it is always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        throw std::runtime_error("All cuBLASLt pointers must be aligned!");
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    if(std::is_same_v<floatX, std::int8_t>) {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    } else {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));
    }

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<floatX>, k, m, k));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<floatX>, m, k, m));
    }
    if (transB) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<floatX>, n, k, n));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<floatX>, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&CLayout, to_cuda_lib_type_enum<floatO>, m, n, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&DLayout, to_cuda_lib_type_enum<floatO>, m, n, m));

    // create a preference handle with specified max workspace
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_size, sizeof(workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = to_cuda_lib_type_enum<floatB>; // force BF16 bias for FP8 mode
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    cublasDataType_t scale_type = CUDA_R_32F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));
    if(scale != nullptr) {
        std::int32_t device_pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
        CUBLAS_CHECK(
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &device_pointer_mode,
                                           sizeof(device_pointer_mode)));
    }

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        throw std::runtime_error(std::format("No cuBLASLt algorithm: m: {}, n: {}, k: {}, bias: {}", n, m, k, has_bias));
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    float one = 1.f;
    float zero = 0.f;

    const float* alpha;
    const float* beta;
    if(scale != nullptr) {
        alpha = scale;
        beta = accumulate ? device_one : device_zero;
    } else {
        // For some reason, using device_one/device_zero for bf16 x bf16 -> bf16 matmuls doesn't
        // work reliably
        // TODO figure out what's going on with cublas
        alpha = &one;
        beta = &zero;
    }

    // call the matmul
    CUBLAS_CHECK(cublasLtMatmul(handle, operationDesc,
                               alpha, a, ALayout, b, BLayout, beta, d, CLayout, d, DLayout,
                               &heuristic.algo, workspace, workspace_size, stream));

    // cleanups
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(ALayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(BLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(CLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(DLayout));
    CUDA_CHECK(cudaGetLastError());
}

template<class floatO, class floatB>
__global__ void add_bias_kernel(floatO* out, const floatB* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

template<class floatO, class floatB>
void add_bias_impl(floatO* out, const floatB* bias, int B, int T, int OC, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = div_ceil(OC * B * T, block_size);
    add_bias_kernel<<<grid_size, block_size, 0, stream>>>(out, bias, B, T, OC);
    CUDA_CHECK(cudaGetLastError());
}

void add_bias(float* out, const float* bias, int B, int T, int OC, cudaStream_t stream) {
    add_bias_impl(out, bias, B, T, OC, stream);
}

void add_bias(nv_bfloat16* out, const nv_bfloat16* bias, int B, int T, int OC, cudaStream_t stream) {
    add_bias_impl(out, bias, B, T, OC, stream);
}

// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias, const float* scale,
                    cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                    int B, int T, int C, int OC, cudaStream_t stream) {
    matmul_cublaslt(out, weight, inp, bias, workspace, workspace_size, OC, B*T, C, stream, handle, scale, true, false, false, false);
}

void matmul_forward(float* out, const nv_bfloat16* inp, const nv_bfloat16* weight, const float* bias, const float* scale,
                    cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                    int B, int T, int C, int OC, cudaStream_t stream) {
    matmul_cublaslt(out, weight, inp, bias, workspace, workspace_size, OC, B*T, C, stream, handle,  scale, true, false, false, false);
}

void matmul_forward(float* out, const __nv_fp8_e4m3* inp, const __nv_fp8_e4m3* weight, const float* bias, const float* scale,
                    cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                    int B, int T, int C, int OC, cudaStream_t stream) {
    matmul_cublaslt(out, weight, inp, bias, workspace, workspace_size, OC, B*T, C, stream, handle,  scale, true, false, false, false);
}

void matmul_forward(float* out, const __nv_fp8_e4m3* inp, const __nv_fp8_e4m3* weight, const nv_bfloat16* bias, const float* scale,
                    cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                    int B, int T, int C, int OC, cudaStream_t stream) {
    matmul_cublaslt(out, weight, inp, bias, workspace, workspace_size, OC, B*T, C, stream, handle,  scale, true, false, false, false);
}

void matmul_forward(float* out, const std::int8_t* inp, const std::int8_t* weight, const float* bias, const float* scale,
                    cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                    int B, int T, int C, int OC, cudaStream_t stream) {
    matmul_cublaslt(out, weight, inp, bias, workspace, workspace_size, OC, B*T, C, stream, handle,  scale, true, false, false, false);
    if(bias) {
        add_bias(out, bias, B, T, OC, stream);
    }
}

void matmul_forward(nv_bfloat16* out, const nv_bfloat16* inp, const nv_bfloat16* weight, const nv_bfloat16* bias, const float* scale,
                    cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                    int B, int T, int C, int OC, cudaStream_t stream) {
    matmul_cublaslt(out, weight, inp, bias, workspace, workspace_size, OC, B*T, C, stream, handle,  scale, true, false, false, false);
}

void matmul_forward(nv_bfloat16* out, const __nv_fp8_e4m3* inp, const __nv_fp8_e4m3* weight, const nv_bfloat16* bias, const float* scale,
                    cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                    int B, int T, int C, int OC, cudaStream_t stream) {
    matmul_cublaslt(out, weight, inp, bias, workspace, workspace_size, OC, B*T, C, stream, handle,  scale, true, false, false, false);
}

void matmul_forward(Tensor& out, const Tensor& inp, const Tensor& weight, std::optional<Tensor> bias, const float* scale,
                    cublasLtHandle_t handle, Tensor& workspace,
                    int B, int T, int C, int OC, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    std::size_t ws_size = workspace.bytes();
    if(out.DType == ETensorDType::FP32 && inp.DType == ETensorDType::FP32) {
        float* bias_ptr = bias.has_value() ? bias.value().get<float>() : nullptr;
        matmul_forward(out.get<float>(), inp.get<float>(), weight.get<float>(), bias_ptr, scale,handle, ws, ws_size, B, T, C, OC, stream);
    } else if(out.DType == ETensorDType::FP32 && inp.DType == ETensorDType::BF16) {
        float* bias_ptr = bias.has_value() ? bias.value().get<float>() : nullptr;
        matmul_forward(out.get<float>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), bias_ptr, scale,handle, ws, ws_size, B, T, C, OC, stream);
    } else if(out.DType == ETensorDType::FP32 && inp.DType == ETensorDType::INT8) {
        float* bias_ptr = bias.has_value() ? bias.value().get<float>() : nullptr;
        matmul_forward(out.get<float>(), inp.get<std::int8_t>(), weight.get<std::int8_t>(), bias_ptr, scale,handle, ws, ws_size, B, T, C, OC, stream);
    } else if(out.DType == ETensorDType::FP32 && inp.DType == ETensorDType::FP8_E4M3) {
        if(bias.has_value()) {
            // even with 32-bit C, bias needs to be bf16 ?!
            if(bias.value().DType == ETensorDType::BF16) {
                matmul_forward(out.get<float>(), inp.get<__nv_fp8_e4m3>(), weight.get<__nv_fp8_e4m3>(), bias->get<nv_bfloat16>(), scale,handle, ws, ws_size, B, T, C, OC, stream);
            } else {
                matmul_forward(out.get<float>(), inp.get<__nv_fp8_e4m3>(), weight.get<__nv_fp8_e4m3>(), bias->get<float>(), scale,handle, ws, ws_size, B, T, C, OC, stream);
            }
        } else {
            matmul_forward(out.get<float>(), inp.get<__nv_fp8_e4m3>(), weight.get<__nv_fp8_e4m3>(), (nv_bfloat16*)nullptr, scale,handle, ws, ws_size, B, T, C, OC, stream);
        }
    } else if(out.DType == ETensorDType::BF16 && inp.DType == ETensorDType::FP8_E4M3) {
        nv_bfloat16* bias_ptr = bias.has_value() ? bias.value().get<nv_bfloat16>() : nullptr;
        matmul_forward(out.get<nv_bfloat16>(), inp.get<__nv_fp8_e4m3>(), weight.get<__nv_fp8_e4m3>(), bias_ptr, scale,handle, ws, ws_size, B, T, C, OC, stream);
    } else if(out.DType == ETensorDType::BF16) {
        nv_bfloat16* bias_ptr = bias.has_value() ? bias.value().get<nv_bfloat16>() : nullptr;
        matmul_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), bias_ptr, scale,handle, ws, ws_size, B, T, C, OC, stream);
    } else {
        throw std::logic_error("matmul_forward: invalid DType combination");
    }
}


template<typename floatX, typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout, const float* abs_max, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;
    constexpr const int bdx = 4;
    constexpr const int bdy = 32 / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = x128::load(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[x128::size][32][bdy];

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if(abs_max != nullptr) {
                a *= *abs_max;
                if constexpr (std::is_same_v<floatX, __nv_fp8_e4m3>) {
                    a /= 448.f;
                }
            }
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

template<class floatX>
__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = f128::load(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}

int get_bias_backward_scratch_size(ETensorDType dtype, int OC, const cudaDeviceProp& dp) {
    const int block_size = dp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
    const int OC_per_warp = 8 * ( 16 / get_dtype_size(dtype) ); // 64 at BF16
    const int grid_size_x = div_ceil(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
    const int grid_size_y = max(1, block_size * dp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!
    return grid_size_y * OC * sizeof(float);
}


template<class floatX>
void matmul_backward_imp(floatX* dinp, floatX* dweight, floatX* dbias,
                         const floatX* dout, const floatX* inp, const floatX* weight,
                         float* dbias_buffer, const float* dinp_scale, const float* dweight_scale,
                         bool accumulate_gradient,
                         cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                         int B, int T, int C, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    NVTX_RANGE_FN();
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;

    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = dp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim = {4, 8, (unsigned)block_size/32};
        const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
        const int grid_size_x = div_ceil(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, block_size * dp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, nullptr, B, T, OC, std::bool_constant<false>());
            CUDA_CHECK(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, nullptr, B, T, OC, std::bool_constant<true>());
            CUDA_CHECK(cudaGetLastError());
            reduce_add_sum_kernel<<<div_ceil((size_t)OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            CUDA_CHECK(cudaGetLastError());
        }
        dbias = NULL; // prevent dbias calculation from also being fused in matmul_cublaslt below (if we enabled fusion)
    }

    // backward to input, uses = in the backward pass (set the gradient)
    matmul_cublaslt(dinp, weight, dout, (float*)nullptr, workspace, workspace_size, C, B*T, OC, stream, handle, device_one, false, false, false, true);

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    matmul_cublaslt(dweight, inp, dout, (float*)nullptr /*dbias*/, workspace, workspace_size, C, OC, B*T, stream, handle, device_one, false, true,
                    accumulate_gradient, true);
}

template<class floatX, class Float8>
void matmul_backward_fp8_imp(floatX* dinp, floatX* dweight, floatX* dbias,
                             const Float8* dout, const Float8* dout_t, const Float8* inp, const Float8* weight,
                             float* dbias_buffer, const float* dinp_scale, const float* dweight_scale, const float* dout_abs_max,
                             bool accumulate_gradient,
                             cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                             int B, int T, int C, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    NVTX_RANGE_FN();
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;

    // TODO
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = dp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim = {4, 8, (unsigned)block_size/32};
        const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
        const int grid_size_x = div_ceil(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, block_size * dp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, dout_abs_max, B, T, OC, std::bool_constant<false>());
            CUDA_CHECK(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, dout_abs_max, B, T, OC, std::bool_constant<true>());
            CUDA_CHECK(cudaGetLastError());
            reduce_add_sum_kernel<<<div_ceil((size_t)OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            CUDA_CHECK(cudaGetLastError());
        }
        dbias = NULL; // prevent dbias calculation from also being fused in matmul_cublaslt below (if we enabled fusion)
    }

    // backward to input, uses = in the backward pass (set the gradient)
    matmul_cublaslt(dinp, weight, dout, (float*)nullptr, workspace, workspace_size, C, B*T, OC, stream, handle, dinp_scale, true, false, false, true);

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    matmul_cublaslt(dweight, inp, dout_t, (float*)nullptr /*dbias*/, workspace, workspace_size, C, OC, B*T, stream, handle, dweight_scale, true, false,
                    accumulate_gradient, true);
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     float* dbias_buffer, const float* dinp_scale, const float* dweight_scale,
                     bool accumulate_gradient,
                     cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                     int B, int T, int C, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    matmul_backward_imp(dinp, dweight, dbias, dout, inp, weight, dbias_buffer, dinp_scale, dweight_scale,
                        accumulate_gradient, handle, workspace, workspace_size, B, T, C, OC, dp, stream);
}

void matmul_backward(nv_bfloat16* dinp, nv_bfloat16* dweight, nv_bfloat16* dbias,
                     const nv_bfloat16* dout, const nv_bfloat16* inp, const nv_bfloat16* weight,
                     float* dbias_buffer, const float* dinp_scale, const float* dweight_scale,
                     bool accumulate_gradient,
                     cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                     int B, int T, int C, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    matmul_backward_imp(dinp, dweight, dbias, dout, inp, weight, dbias_buffer, dinp_scale, dweight_scale,
                        accumulate_gradient, handle, workspace, workspace_size, B, T, C, OC, dp, stream);
}

void matmul_backward(Tensor dinp, Tensor dweight, std::optional<Tensor> dbias,
                     const Tensor& dout, const Tensor& inp, const Tensor& weight,
                     std::optional<Tensor> dbias_buffer, const float* dinp_scale, const float* dweight_scale,
                     bool accumulate_gradient,
                     cublasLtHandle_t handle, Tensor& workspace,
                     int B, int T, int C, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    std::size_t ws_size = workspace.bytes();
    if(dinp.DType == ETensorDType::FP32 && inp.DType == ETensorDType::FP32) {
        float* d_bias_ptr = dbias.has_value() ? dbias.value().get<float>() : nullptr;
        float* bias_buffer_ptr = dbias_buffer.has_value() ? dbias_buffer.value().get<float>() : nullptr;
        matmul_backward(dinp.get<float>(), dweight.get<float>(), d_bias_ptr, dout.get<float>(), inp.get<float>(), weight.get<float>(),
                        bias_buffer_ptr, dinp_scale, dweight_scale, accumulate_gradient, handle, ws, ws_size, B, T, C, OC, dp, stream);
    } else if(dinp.DType == ETensorDType::BF16 && inp.DType == ETensorDType::BF16) {
        nv_bfloat16* d_bias_ptr = dbias.has_value() ? dbias.value().get<nv_bfloat16>() : nullptr;
        float* bias_buffer_ptr = dbias_buffer.has_value() ? dbias_buffer.value().get<float>() : nullptr;
        matmul_backward(dinp.get<nv_bfloat16>(), dweight.get<nv_bfloat16>(), d_bias_ptr, dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                        bias_buffer_ptr, dinp_scale, dweight_scale, accumulate_gradient, handle, ws, ws_size, B, T, C, OC, dp, stream);
    } else {
        throw std::logic_error("matmul_forward: invalid DType combination");
    }
}

void matmul_backward_fp8(Tensor dinp, Tensor dweight, std::optional<Tensor> dbias,
                     const Tensor& dout, const Tensor& dout_t, const Tensor& inp, const Tensor& weight,
                     std::optional<Tensor> dbias_buffer, const float* dinp_scale, const float* dweight_scale, const float* dout_scale,
                     bool accumulate_gradient,
                     cublasLtHandle_t handle, Tensor& workspace,
                     int B, int T, int C, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    std::size_t ws_size = workspace.bytes();
    if(dinp.DType == ETensorDType::BF16 && inp.DType == ETensorDType::FP8_E4M3) {
        nv_bfloat16* d_bias_ptr = dbias.has_value() ? dbias.value().get<nv_bfloat16>() : nullptr;
        float* bias_buffer_ptr = dbias_buffer.has_value() ? dbias_buffer.value().get<float>() : nullptr;
        matmul_backward_fp8_imp(dinp.get<nv_bfloat16>(), dweight.get<nv_bfloat16>(), d_bias_ptr,
            dout.get<__nv_fp8_e4m3>(), dout_t.get<__nv_fp8_e4m3>(), inp.get<__nv_fp8_e4m3>(), weight.get<__nv_fp8_e4m3>(),
                                bias_buffer_ptr, dinp_scale, dweight_scale, dout_scale, accumulate_gradient, handle, ws, ws_size, B, T, C, OC, dp, stream);
    }  else {
        throw std::logic_error("matmul_backward_fp8: invalid DType combination");
    }
}