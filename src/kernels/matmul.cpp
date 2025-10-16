// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cstdio>
#include <optional>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <fmt/core.h>

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
                     const float* scale, EMMTranspose mode, bool accumulate)
{
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
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if(has_bias){
        epilogue = CUBLASLT_EPILOGUE_BIAS;
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
        throw std::runtime_error(fmt::format("No cuBLASLt algorithm: m: {}, n: {}, k: {}, bias: {}", n, m, k, has_bias));
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

void matmul(float* c, const float* a, const float* b, const float* bias, const float* scale,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int B, int T, int C, int OC, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, OC, B*T, C, stream, handle, scale, mode, accumulate);
}

void matmul(float* c, const nv_bfloat16* a, const nv_bfloat16* b, const float* bias, const float* scale,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int B, int T, int C, int OC, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, OC, B*T, C, stream, handle, scale, mode, accumulate);
}

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const float* bias, const float* scale,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int B, int T, int C, int OC, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, OC, B*T, C, stream, handle, scale, mode, accumulate);
}

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int B, int T, int C, int OC, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, OC, B*T, C, stream, handle, scale, mode, accumulate);
}

void matmul(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int B, int T, int C, int OC, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, OC, B*T, C, stream, handle, scale, mode, accumulate);
}

void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int B, int T, int C, int OC, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, OC, B*T, C, stream, handle, scale, mode, accumulate);
}

/*
void matmul(float* c, const std::int8_t* a, const std::int8_t* b, const nv_bfloat16* bias, const float* scale,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int B, int T, int C, int OC, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, b, a, bias, workspace, workspace_size, OC, B*T, C, stream, handle, scale, mode, accumulate);
    if(bias) {
        add_bias(c, bias, B, T, OC, stream);
    }
}
*/

template<class floatX>
void matmul_backward_imp(floatX* dinp, floatX* dweight, floatX* dbias,
                         const floatX* dout, const floatX* inp, const floatX* weight,
                         float* dbias_buffer, const float* dinp_scale, const float* dweight_scale,
                         bool accumulate_gradient,
                         cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                         int B, int T, int C, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        backward_bias(dbias, dout, nullptr, dbias_buffer, B, T, OC, dp, stream);
    }

    // backward to input, uses = in the backward pass (set the gradient)
    matmul_cublaslt(dinp, weight, dout, (float*)nullptr, workspace, workspace_size, C, B*T, OC, stream, handle, device_one, EMMTranspose::NN, false);

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    matmul_cublaslt(dweight, inp, dout, (float*)nullptr, workspace, workspace_size, C, OC, B*T, stream, handle, device_one, EMMTranspose::NT,
                    accumulate_gradient);
}

template<class floatX, class Float8>
void matmul_backward_fp8_imp(floatX* dinp, floatX* dweight, floatX* dbias,
                             const Float8* dout, const Float8* dout_t, const Float8* inp, const Float8* weight,
                             float* dbias_buffer, const float* dinp_scale, const float* dweight_scale, const float* dout_abs_max,
                             bool accumulate_gradient,
                             cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                             int B, int T, int C, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        backward_bias(dbias, dout, dout_abs_max, dbias_buffer, B, T, OC, dp, stream);
    }

    // backward to input, uses = in the backward pass (set the gradient)
    matmul_cublaslt(dinp, weight, dout, (float*)nullptr, workspace, workspace_size, C, B*T, OC, stream, handle, dinp_scale, EMMTranspose::TN, false);

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    matmul_cublaslt(dweight, inp, dout_t, (float*)nullptr, workspace, workspace_size, C, OC, B*T, stream, handle, dweight_scale, EMMTranspose::TN,
                    accumulate_gradient);
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