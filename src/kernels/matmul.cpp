// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cublasLt.h>
#include <fmt/core.h>

#include "kernels.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;

EMatmulBackend& get_matmul_backend() {
    static EMatmulBackend backend = EMatmulBackend::CuBLAS;
    return backend;
}

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
inline void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(fmt::format("cuBLAS ERROR ({}) at {}:{}", (int)status, file, line));
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

void destroy_cublaslt_handle(cublasLtHandle_t handle) {
    CUBLAS_CHECK(cublasLtDestroy(handle));
}

// ----------------------------------------------------------------------------
// kernel launchers

// Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
template<class FloatC, class FloatA, class FloatB, class FloatBias>
void matmul_cublaslt(FloatC* d, const FloatA* a, const FloatB* b, const FloatBias* bias,
                     std::byte* workspace, std::size_t workspace_size,
                     int m, int n, int k, cudaStream_t stream, cublasLtHandle_t handle,
                     const float* scale_a, const float* scale_b, EMMTranspose mode, bool accumulate)
{
    bool has_bias = (bias != nullptr);

    // check alignment (some modes work unaligned, but it is always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        throw std::runtime_error("All cuBLASLt pointers must be aligned!");
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    bool transA = mode == EMMTranspose::TN || mode == EMMTranspose::TT;
    bool transB = mode == EMMTranspose::NT || mode == EMMTranspose::TT;

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
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<FloatA>, k, m, k));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<FloatA>, m, k, m));
    }
    if (transB) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<FloatB>, n, k, n));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<FloatB>, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&CLayout, to_cuda_lib_type_enum<FloatC>, m, n, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&DLayout, to_cuda_lib_type_enum<FloatC>, m, n, m));

    // create a preference handle with specified max workspace
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_size, sizeof(workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if(has_bias){
        epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = to_cuda_lib_type_enum<FloatBias>; // force BF16 bias for FP8 mode
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    if(scale_a) {
        if(sizeof(FloatA) != 1) {
            throw std::runtime_error("Scaling A is only supported for FP8");
        }
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(&scale_a)));
    }
    if(scale_b) {
        if(sizeof(FloatB) != 1) {
            throw std::runtime_error("Scaling B is only supported for FP8");
        }
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(&scale_b)));
    }

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        throw std::runtime_error(fmt::format("No cuBLASLt algorithm: m: {}, n: {}, k: {}, bias: {}", n, m, k, has_bias));
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    float one = 1.f;
    float zero = 0.f;
    float* alpha = &one;
    float* beta = accumulate ? &one : &zero;

    // call the matmul
    CUBLAS_CHECK(cublasLtMatmul(handle, operationDesc,
                               alpha, a, ALayout, b, BLayout, beta, d, CLayout, d, DLayout,
                               &heuristic.algo, workspace, workspace_size, stream));
    CUDA_CHECK(cudaGetLastError());

    // cleanups
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(ALayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(BLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(CLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(DLayout));
    CUDA_CHECK(cudaGetLastError());
}

// custom matmuls
void gemm_mma_tn(nv_bfloat16* out, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, int m, int n, int k, const float* scale, const nv_bfloat16* bias, bool accumulate, cudaStream_t stream);
void gemm_mma_tn(nv_bfloat16* out, const nv_bfloat16* a, const nv_bfloat16* b, int m, int n, int k, const float* scale, const nv_bfloat16* bias, bool accumulate, cudaStream_t stream);


template<class floatO, class floatX, class floatB>
void matmul_dispatch(floatO* d, const floatX* a, const floatX* b, const floatB* bias,
                     std::byte* workspace, std::size_t workspace_size,
                     int m, int n, int k, cudaStream_t stream, cublasLtHandle_t handle,
                     const float* scale, EMMTranspose mode, bool accumulate)
{
    if(get_matmul_backend() == EMatmulBackend::CuBLAS || mode != EMMTranspose::TN) {
        matmul_cublaslt(d, a, b, bias, workspace, workspace_size, m, n, k, stream, handle, scale, mode, accumulate);
    } else if constexpr (std::is_same_v<floatO, nv_bfloat16> && std::is_same_v<floatB, nv_bfloat16>){
        gemm_mma_tn(d, a, b, m, n, k, scale, bias, accumulate, stream);
    } else {
        matmul_cublaslt(d, a, b, bias, workspace, workspace_size, m, n, k, stream, handle, scale, mode, accumulate);
    }
}

void matmul(float* c, const float* a, const float* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_dispatch(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

void matmul(float* c, const nv_bfloat16* a, const nv_bfloat16* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_dispatch(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_dispatch(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_dispatch(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

void matmul(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_dispatch(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_dispatch(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e5m2* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

/*
void matmul(float* c, const std::int8_t* a, const std::int8_t* b, const nv_bfloat16* bias, const float* scale,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, b, a, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
    if(bias) {
        add_bias(c, bias, B, T, OC, stream);
    }
}
*/
