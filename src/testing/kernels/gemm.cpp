#include "kernels/kernels.h"
#include "utilities/utils.h"
#include <cstdlib>
#include <cstdio>
#include <random>
#include <cublasLt.h>
#include <cublas_v2.h>

void gemm_mma_tn(nv_bfloat16* out, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, int m, int n, int k, const float* scale, bool accumulate, cudaStream_t stream);
void gemm_mma_tn(nv_bfloat16* out, const nv_bfloat16* a, const nv_bfloat16* b, int m, int n, int k, const float* scale, bool accumulate, cudaStream_t stream);

template<class floatO, class floatX, class floatB>
extern void matmul_cublaslt(floatO* d, const floatX* a, const floatX* b, const floatB* bias,
                     std::byte* workspace, std::size_t workspace_size,
                     int m, int n, int k, cudaStream_t stream, cublasLtHandle_t handle,
                     const float* scale, EMMTranspose mode, bool accumulate);


template<typename Atype, typename Btype, typename Ctype>
void run_test(int m, int n, int k, float scale = 1.f, bool accumulate = false) {
    Atype* a;
    Btype* b;
    Ctype* c;
    cudaMallocManaged(&a, m * k * sizeof(Atype));
    cudaMallocManaged(&b, n * k * sizeof(Btype));
    cudaMallocManaged(&c, m * n * sizeof(Ctype));
    cudaMemset(a, 0, m * k * sizeof(Atype));
    cudaMemset(c, 0, m * n * sizeof(Ctype));
    cudaMemset(b, 0, n * k * sizeof(Btype));

    float* a_float = nullptr;
    float* b_float = nullptr;
    float* c_float = nullptr;
    cudaMallocManaged(&a_float, m * k * sizeof(float));
    cudaMallocManaged(&b_float, n * k * sizeof(float));
    cudaMallocManaged(&c_float, m * n * sizeof(float));
    cudaMemset(a_float, 0, m * k * sizeof(float));
    cudaMemset(b_float, 0, n * k * sizeof(float));
    cudaMemset(c_float, 0, m * n * sizeof(float));

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

    float* scale_ptr;
    cudaMallocManaged(&scale_ptr, sizeof(float));
    *scale_ptr = scale;

    cudaMemPrefetchAsync(a, m*k * sizeof(Atype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync(b, n*k * sizeof(Btype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync(c, m*n * sizeof(Ctype), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync(scale_ptr, 4, cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);

    CUDA_CHECK(cudaDeviceSynchronize());
    gemm_mma_tn(c, a, b, m, n, k, scale_ptr, accumulate, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasLtHandle_t handle;
    std::byte* workspace;
    size_t workspace_size = 128 * 1024 * 1024;
    assert(cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    cudaMalloc(&workspace, workspace_size);
    setup_cublas();


    cudaMemPrefetchAsync(a_float, m*k * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync(b_float, n*k * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync(c_float, m*n * sizeof(float), cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    matmul_cublaslt(c_float, a_float, b_float, (float*)nullptr, workspace, workspace_size, m, n, k, nullptr, handle, scale_ptr, EMMTranspose::TN, accumulate);
    CUDA_CHECK(cudaDeviceSynchronize());

    double r_tol = std::is_same_v<Atype, nv_bfloat16> ? 1e-2 : 0.125;

    bool equal = true;
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            if(fabsf(c_float[j * m + i] - (float)c[j * m + i]) > std::max(r_tol * fabsf((float)c[j * m + i]), 1e-4)) {
                printf("%d %d: %f != %f\n", i, j, (float) c_float[j * m + i], (float) c[j * m + i]);
                equal = false;
            }
        }
        if(!equal) {
            break;
        }
    }

    if(equal) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(a_float);
    cudaFree(b_float);
    cudaFree(c_float);
    cudaFree(scale_ptr);
    cudaFree(workspace);
    cublasLtDestroy(handle);
}

int main() {
    int m = 1536;
    int n = 1024;
    int k = 1664;

    // larger shape for benchmarking
    if (false) {
        m = 2*4864;
        n = 1024*8;
        k = 896;
    }

    run_test<nv_bfloat16, nv_bfloat16, nv_bfloat16>(m, n, k, 1.f, false);
    run_test<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>(m, n, k, 4.0/k, false);

    run_test<nv_bfloat16, nv_bfloat16, nv_bfloat16>(m, n, k, 1.f, true);
    run_test<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>(m, n, k, 4.0/k, true);
}
