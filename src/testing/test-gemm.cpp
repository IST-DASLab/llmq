// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels/kernels.h"
#include "utilities/utils.h"
#include <cstdlib>
#include <cstdio>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "test_config.h"

template<class floatO, class floatX, class floatB>
extern void matmul_cublaslt(floatO* d, const floatX* a, const floatX* b, const floatB* bias,
                     std::byte* workspace, std::size_t workspace_size,
                     int m, int n, int k, cudaStream_t stream, cublasLtHandle_t handle,
                     const float* scale_a, const float* scale_b, EMMTranspose mode, bool accumulate);

extern cublasLtHandle_t create_cublaslt_handle();

template<typename Atype, typename Btype, typename Ctype>
void run_test(int m, int n, int k, float scale = 1.f, bool accumulate = false, bool use_bias=false, bool check=true) {
    auto saved_backend = get_matmul_backend();
    Atype* a;
    Btype* b;
    Ctype* c;
    Ctype* bias;
    CUDA_CHECK(cudaMallocManaged(&a, m * k * sizeof(Atype)));
    CUDA_CHECK(cudaMallocManaged(&b, n * k * sizeof(Btype)));
    CUDA_CHECK(cudaMallocManaged(&c, m * n * sizeof(Ctype)));
    CUDA_CHECK(cudaMallocManaged(&bias, n * sizeof(Ctype)));
    CUDA_CHECK(cudaMemset(a, 0, m * k * sizeof(Atype)));
    CUDA_CHECK(cudaMemset(c, 0, m * n * sizeof(Ctype)));
    CUDA_CHECK(cudaMemset(b, 0, n * k * sizeof(Btype)));
    CUDA_CHECK(cudaMemset(bias, 0, n * sizeof(Ctype)));

    float* a_float = nullptr;
    float* b_float = nullptr;
    float* c_float = nullptr;
    float* bias_float = nullptr;
    CUDA_CHECK(cudaMallocManaged(&a_float, m * k * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b_float, n * k * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&c_float, m * n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&bias_float, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(a_float, 0, m * k * sizeof(float)));
    CUDA_CHECK(cudaMemset(b_float, 0, n * k * sizeof(float)));
    CUDA_CHECK(cudaMemset(c_float, 0, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(bias_float, 0, n * sizeof(float)));

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-15, 15);

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < k; ++j) {
            auto val = static_cast<Atype>(dist(rng));
            a[i*k+j] = val;
            a_float[i*k+j] = static_cast<float>(val);
        }
    }

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < k; ++j) {
            auto val = static_cast<Btype>(dist(rng));
            b[i*k+j] = val;
            b_float[i*k+j] = static_cast<float>(val);
        }
    }

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            auto val = static_cast<Ctype>(dist(rng));
            c[i*n+j] = val;
            c_float[i*n+j] = static_cast<float>(val);
        }
    }

    for(int i = 0; i < n; ++i) {
        auto val = static_cast<Ctype>(dist(rng));
        bias[i] = val;
        bias_float[i] = static_cast<float>(val);
    }

    float* scale_a_ptr, *scale_b_ptr;
    CUDA_CHECK(cudaMallocManaged(&scale_a_ptr, sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&scale_b_ptr, sizeof(float)));
    *scale_a_ptr = sqrtf(scale);
    *scale_b_ptr = sqrtf(scale);

    CUDA_CHECK(cudaMemPrefetchAsync(a, m*k * sizeof(Atype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(b, n*k * sizeof(Btype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(c, m*n * sizeof(Ctype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(scale_a_ptr, 4, cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(scale_b_ptr, 4, cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));

    cublasLtHandle_t handle = create_cublaslt_handle();
    std::byte* workspace;
    size_t workspace_size = 128 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    get_matmul_backend() = EMatmulBackend::Custom;

    CUDA_CHECK(cudaDeviceSynchronize());
    matmul(c, a, b, use_bias ? bias : nullptr, scale_a_ptr, scale_b_ptr, handle, workspace, workspace_size, m, n, k, EMMTranspose::TN, accumulate, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(check) {
        CUDA_CHECK(cudaMemPrefetchAsync(a_float, m*k * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(b_float, n*k * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(c_float, m*n * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(bias_float, n * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0));
        get_matmul_backend() = EMatmulBackend::CuBLAS;
        CUDA_CHECK(cudaDeviceSynchronize());
        matmul(c_float, a_float, b_float,  use_bias ? bias_float : nullptr, nullptr, nullptr,
            handle, workspace, workspace_size, m, n, k , EMMTranspose::TN, accumulate, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        double r_tol = 1e-2;
        int approx_count = 0;
        int far_count = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float expected = c_float[j * m + i] * (*scale_a_ptr) * (*scale_b_ptr);
                float received = (float) c[j * m + i];
                float r_tol_max = r_tol * std::max(fabsf((float) c[j * m + i]), fabsf(expected));
                float err = fabsf(expected - received);
                float tol = std::max(r_tol_max, 1e-4f);
                if (err > 10 * tol) {
                    printf(" %d %d: %f != %f\n", i, j, expected, received);
                    ++far_count;
                } else if (err > tol) {
                    ++approx_count;
                }
            }
            if (far_count > 0) {
                break;
            }
        }

        if (far_count == 0 && approx_count == 0) {
            SUCCEED();
            printf("PASS\n");
        } else if(far_count < m * n / 100 && approx_count < m * n / 10) {
            SUCCEED();
            printf("CLOSE %d%%  [%d+%d]\n", 100 - (approx_count + far_count) * 100 / (m * n), far_count, approx_count);
        } else {
            FAIL();
        }
    }

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(a_float));
    CUDA_CHECK(cudaFree(b_float));
    CUDA_CHECK(cudaFree(c_float));
    CUDA_CHECK(cudaFree(bias_float));
    CUDA_CHECK(cudaFree(scale_a_ptr));
    CUDA_CHECK(cudaFree(scale_b_ptr));
    CUDA_CHECK(cudaFree(workspace));
    cublasLtDestroy(handle);
    get_matmul_backend() = saved_backend;
}

TEST_CASE("tiny matmul bfloat16 x bfloat16 -> bfloat16", "[gemm][bf16]") {
    bool accumulate = false;
    bool bias = false;
    SECTION("set-nobias") {
        accumulate = false;
        bias = false;
    }
    SECTION("set-bias") {
        accumulate = false;
        bias = true;
    }
    SECTION("accumulate-nobias") {
        accumulate = true;
        bias = false;
    }
    SECTION("accumulate-bias") {
        accumulate = true;
        bias = true;
    }
    run_test<nv_bfloat16, nv_bfloat16, nv_bfloat16>(128, 128, 128, 1.0f, accumulate, bias);
}

TEST_CASE("tiny matmul fp8 x fp8 -> bfloat16", "[gemm][fp8]") {
    bool accumulate = false;
    bool bias = false;
    SECTION("set-nobias") {
        accumulate = false;
        bias = false;
    }
    SECTION("set-bias") {
        accumulate = false;
        bias = true;
    }
    SECTION("accumulate-nobias") {
        accumulate = true;
        bias = false;
    }
    SECTION("accumulate-bias") {
        accumulate = true;
        bias = true;
    }

    run_test<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>(128, 128, 128, 4.0f / 128, accumulate, bias);
}

TEST_CASE("matmul bfloat16 x bfloat16 -> bfloat16", "[gemm][bf16]") {
    const auto& cfg = testing_config::get_test_config();
    int m = cfg.B * cfg.T;
    int k = cfg.C;
    int n = div_ceil(2 * m / 3, 128) * 128;

    bool accumulate = false;
    bool bias = false;
    SECTION("set-nobias") {
        accumulate = false;
        bias = false;
    }
    SECTION("set-bias") {
        accumulate = false;
        bias = true;
    }
    SECTION("accumulate-nobias") {
        accumulate = true;
        bias = false;
    }
    SECTION("accumulate-bias") {
        accumulate = true;
        bias = true;
    }
    run_test<nv_bfloat16, nv_bfloat16, nv_bfloat16>(m, n, k, 1.0f, accumulate, bias);
}

TEST_CASE("matmul fp8 x fp8 -> bfloat16", "[gemm][fp8]") {
    const auto& cfg = testing_config::get_test_config();
    int m = cfg.B * cfg.T;
    int k = cfg.C;
    int n = div_ceil(2 * m / 3, 128) * 128;

    bool accumulate = false;
    bool bias = false;
    SECTION("set-nobias") {
        accumulate = false;
        bias = false;
    }
    SECTION("set-bias") {
        accumulate = false;
        bias = true;
    }
    SECTION("accumulate-nobias") {
        accumulate = true;
        bias = false;
    }
    SECTION("accumulate-bias") {
        accumulate = true;
        bias = true;
    }
    run_test<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>(m, n, k, 4.0f / k, accumulate, bias);
}
