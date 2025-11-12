#include "kernels/kernels.h"
#include "utilities/utils.h"
#include <cstdlib>
#include <cstdio>
#include <random>
#include <cublasLt.h>
#include <cublas_v2.h>

template<class floatO, class floatX, class floatB>
extern void matmul_cublaslt(floatO* d, const floatX* a, const floatX* b, const floatB* bias,
                     std::byte* workspace, std::size_t workspace_size,
                     int m, int n, int k, cudaStream_t stream, cublasLtHandle_t handle,
                     const float* scale, EMMTranspose mode, bool accumulate);


template<typename Atype, typename Btype, typename Ctype>
void run_test(int m, int n, int k, float scale = 1.f, bool accumulate = false, bool use_bias=false, bool check=true) {
    Atype* a;
    Btype* b;
    Ctype* c;
    Ctype* bias;
    cudaMallocManaged(&a, m * k * sizeof(Atype));
    cudaMallocManaged(&b, n * k * sizeof(Btype));
    cudaMallocManaged(&c, m * n * sizeof(Ctype));
    cudaMallocManaged(&bias, m * sizeof(Ctype));
    cudaMemset(a, 0, m * k * sizeof(Atype));
    cudaMemset(c, 0, m * n * sizeof(Ctype));
    cudaMemset(b, 0, n * k * sizeof(Btype));
    cudaMemset(bias, 0, m * sizeof(Ctype));

    float* a_float = nullptr;
    float* b_float = nullptr;
    float* c_float = nullptr;
    float* bias_float = nullptr;
    cudaMallocManaged(&a_float, m * k * sizeof(float));
    cudaMallocManaged(&b_float, n * k * sizeof(float));
    cudaMallocManaged(&c_float, m * n * sizeof(float));
    cudaMallocManaged(&bias_float, m* sizeof(float));
    cudaMemset(a_float, 0, m * k * sizeof(float));
    cudaMemset(b_float, 0, n * k * sizeof(float));
    cudaMemset(c_float, 0, m * n * sizeof(float));
    cudaMemset(bias_float, 0, m * sizeof(float));

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < k; ++j) {
            auto val = static_cast<Atype>(rand() % 31 - 15);
            a[i*k+j] = val;
            a_float[i*k+j] = static_cast<float>(val);
        }
    }

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < k; ++j) {
            auto val = static_cast<Btype>(rand() % 31 - 15);
            b[i*k+j] = val;
            b_float[i*k+j] = static_cast<float>(val);
        }
    }

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            auto val = static_cast<Ctype>(rand() % 31 - 15);
            c[i*n+j] = val;
            c_float[i*n+j] = static_cast<float>(val);
        }
    }

    for(int i = 0; i < m; ++i) {
        auto val = static_cast<Ctype>(rand() % 31 - 15);
        bias[i] = val;
        bias_float[i] = static_cast<float>(val);
    }

    float* scale_ptr;
    cudaMallocManaged(&scale_ptr, sizeof(float));
    *scale_ptr = scale;

    cudaMemPrefetchAsync(a, m*k * sizeof(Atype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync(b, n*k * sizeof(Btype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync(c, m*n * sizeof(Ctype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync(scale_ptr, 4, cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);

    cublasLtHandle_t handle;
    std::byte* workspace;
    size_t workspace_size = 128 * 1024 * 1024;
    assert(cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    cudaMalloc(&workspace, workspace_size);
    setup_cublas();
    get_matmul_backend() = EMatmulBackend::Custom;

    CUDA_CHECK(cudaDeviceSynchronize());
    matmul(c, a, b, use_bias ? bias : nullptr, scale_ptr, handle, workspace, workspace_size, m, n, k , EMMTranspose::TN, accumulate, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(check) {
        cudaMemPrefetchAsync(a_float, m*k * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
        cudaMemPrefetchAsync(b_float, n*k * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
        cudaMemPrefetchAsync(c_float, m*n * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
        cudaMemPrefetchAsync(bias_float, m * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
        get_matmul_backend() = EMatmulBackend::CuBLAS;
        CUDA_CHECK(cudaDeviceSynchronize());
        matmul(c_float, a_float, b_float,  use_bias ? bias_float : nullptr, scale_ptr, handle, workspace, workspace_size, m, n, k , EMMTranspose::TN, accumulate, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        double r_tol = 1e-2;
        bool equal = true;
        int approx_count = 0;
        int far_count = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float r_tol_max = r_tol * std::max(fabsf((float) c[j * m + i]), fabsf((float) c_float[j * m + i]));
                float err = fabsf(c_float[j * m + i] - (float) c[j * m + i]);
                float tol = std::max(r_tol_max, 1e-4f);
                if (err > 10 * tol) {
                    printf(" %d %d: %f != %f\n", i, j, (float) c_float[j * m + i], (float) c[j * m + i]);
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
            printf("PASS\n");
        } else if(far_count < m * n / 100 && approx_count < m * n / 10) {
            printf("CLOSE %d%%  [%d+%d]\n", 100 - (approx_count + far_count) * 100 / (m * n), far_count, approx_count);
        } else {
            printf("FAIL\n");
        }
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(bias);
    cudaFree(a_float);
    cudaFree(b_float);
    cudaFree(c_float);
    cudaFree(bias_float);
    cudaFree(scale_ptr);
    cudaFree(workspace);
    cublasLtDestroy(handle);
}

int main() {
    int m = 1536;
    int n = 1024;
    int k = 1664;

    // larger shape for benchmarking
    if (true) {
        m = 2*4864;
        n = 1024*8;
        k = 896;
    }

    std::swap(m, n);

    run_test<nv_bfloat16, nv_bfloat16, nv_bfloat16>(m, n, k, 1.f, false);
    run_test<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>(m, n, k, 4.0/k, false);
/*
    run_test<nv_bfloat16, nv_bfloat16, nv_bfloat16>(m, n, k, 1.f, true);
    run_test<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>(m, n, k, 4.0/k, true);

    run_test<nv_bfloat16, nv_bfloat16, nv_bfloat16>(m, n, k, 1.f, false, true);
    run_test<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>(m, n, k, 4.0/k, false, true);*/
}
