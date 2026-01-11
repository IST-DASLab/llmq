// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "test_config.h"
#include "test_utils.h"

using namespace testing_utils;

namespace {

// CPU baseline for RMSNorm backward matching the CUDA implementation in rmsnorm.cu
// Inputs shapes: inp(B,T,C), weight(C), dout(B,T,C), rstd(B*T)
// dresidual is added to dinp (identity path), here we allow passing nullptr meaning zeros.
static void rmsnorm_backward_cpu(
    float* dinp, float* dweight,
    const float* dresidual,
    const float* dout, const float* inp, const float* weight,
    const float* rstd,
    int B, int T, int C)
{
    // initialize outputs
    std::fill(dinp, dinp + (size_t)B * T * C, 0.0f);
    std::fill(dweight, dweight + C, 0.0f);

    // dweight[c] = sum_{bt} (inp[bt,c] * rstd[bt]) * dout[bt,c]
    for (int bt = 0; bt < B * T; ++bt) {
        const float r = rstd[bt];
        const float* x = inp + (size_t)bt * C;
        const float* g = dout + (size_t)bt * C;
        for (int c = 0; c < C; ++c) {
            dweight[c] += (x[c] * r) * g[c];
        }
    }

    // dinp: for each bt, compute
    // dnorm_i = weight_i * dout_i
    // dnorm_norm_mean = (sum_i dnorm_i * inp_i) / C * rstd_bt
    // dinp_i += (weight_i * dout_i - (inp_i * rstd_bt) * dnorm_norm_mean) * rstd_bt
    for (int bt = 0; bt < B * T; ++bt) {
        const float r = rstd[bt];
        const float* x = inp + (size_t)bt * C;
        const float* g = dout + (size_t)bt * C;
        float* dx = dinp + (size_t)bt * C;
        // start with residual gradient if provided
        if (dresidual) {
            const float* dr = dresidual + (size_t)bt * C;
            for (int c = 0; c < C; ++c) dx[c] = dr[c];
        }
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
            float dnorm = weight[c] * g[c];
            acc += dnorm * x[c];
        }
        float dnorm_norm_mean = (acc / C) * r;
        for (int c = 0; c < C; ++c) {
            float term = weight[c] * g[c] - (x[c] * r) * dnorm_norm_mean;
            dx[c] += term * r;
        }
    }
}

static float max_abs(const float* data, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(data[i]));
    return m;
}

} // namespace

TEST_CASE("rmsnorm backward fp32 matches CPU", "[rmsnorm][backward][fp32]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int C = cfg.C; // kernel prefers multiples of 4 for vectorization

    if (C % 4 != 0) {
        INFO("Invalid sizes for fp32: require C % 4 == 0");
        FAIL("Aborting fp32 rmsnorm backward test due to invalid size configuration");
    }

    const size_t N = static_cast<size_t>(B) * T * C;
    const size_t BT = static_cast<size_t>(B) * T;

    // Random but deterministic inputs
    std::vector<float> h_inp = uniform_host(N, -1.0f, 1.0f, 1337ULL);
    std::vector<float> h_weight = uniform_host(C, 0.5f, 1.5f, 42ULL);
    std::vector<float> h_dout = uniform_host(N, -1.0f, 1.0f, 9001ULL);
    std::vector<float> h_rstd(BT);

    // Build rstd by running forward on CPU-like logic: compute rms per token
    // rstd = 1/sqrt(mean(x^2) + eps). Use epsilon that the kernel likely uses in training code.
    // The backward API expects rstd given externally, so we synthesize it here.
    const float eps = 1e-6f;
    for (int bt = 0; bt < B * T; ++bt) {
        const float* x = h_inp.data() + (size_t)bt * C;
        float acc = 0.f;
        for (int c = 0; c < C; ++c) acc += x[c] * x[c];
        acc /= C;
        h_rstd[bt] = 1.0f / std::sqrt(acc + eps);
    }

    // Prepare CPU reference
    std::vector<float> h_dinp_cpu(N);
    std::vector<float> h_dweight_cpu(C);
    rmsnorm_backward_cpu(h_dinp_cpu.data(), h_dweight_cpu.data(), /*dresidual*/ nullptr,
                         h_dout.data(), h_inp.data(), h_weight.data(), h_rstd.data(), B, T, C);

    // Device buffers
    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<float> d_weight = to_device(h_weight);
    thrust::device_vector<float> d_dout = to_device(h_dout);
    thrust::device_vector<float> d_rstd = to_device(h_rstd);
    thrust::device_vector<float> d_dinp(N, 0.0f);
    thrust::device_vector<float> d_dweight(C, 0.0f);

    // Scratch sizing
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp dp{}; cudaGetDeviceProperties(&dp, dev);
    int scratch_size = get_rmsnorm_backward_scratch_size(C, dp);
    thrust::device_vector<std::byte> d_scratch(scratch_size);

    // Launch kernel (dresidual = dinp to initialize zeros efficiently)
    rmsnorm_backward(thrust::raw_pointer_cast(d_dinp.data()),
                     thrust::raw_pointer_cast(d_dweight.data()),
                     thrust::raw_pointer_cast(d_scratch.data()),
                     /*dresidual*/ thrust::raw_pointer_cast(d_dinp.data()),
                     thrust::raw_pointer_cast(d_dout.data()),
                     thrust::raw_pointer_cast(d_inp.data()),
                     thrust::raw_pointer_cast(d_weight.data()),
                     thrust::raw_pointer_cast(d_rstd.data()),
                     /*abs_max*/ nullptr,
                     B, T, C, dp, /*stream*/ 0);

    cudaDeviceSynchronize();

    // Copy back
    std::vector<float> h_dinp = from_device(d_dinp);
    std::vector<float> h_dweight = from_device(d_dweight);

    // Validate
    const float atol = 5e-5f;
    const float rtol = 5e-4f;
    for (size_t i = 0; i < N; ++i) {
        REQUIRE(h_dinp[i] == Catch::Approx(h_dinp_cpu[i]).margin(atol).epsilon(rtol));
    }
    for (int c = 0; c < C; ++c) {
        REQUIRE(h_dweight[c] == Catch::Approx(h_dweight_cpu[c]).margin(atol).epsilon(rtol));
    }
}

TEST_CASE("rmsnorm backward bf16 matches CPU within tolerance", "[rmsnorm][backward][bf16]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int C = cfg.C;

    if (C % 8 != 0) { // bf16 vectorization prefers 8 (since x128::size=8 for bf16)
        INFO("Invalid sizes for bf16: require C % 8 == 0");
        FAIL("Aborting bf16 rmsnorm backward test due to invalid size configuration");
    }

    const size_t N = static_cast<size_t>(B) * T * C;
    const size_t BT = static_cast<size_t>(B) * T;

    // Generate float host data then convert to bf16 for device
    std::vector<float> h_inp_f = uniform_host(N, -1.0f, 1.0f, 1337ULL);
    std::vector<float> h_weight_f = uniform_host(C, 0.5f, 1.5f, 42ULL);
    std::vector<float> h_dout_f = uniform_host(N, -1.0f, 1.0f, 9001ULL);
    std::vector<float> h_rstd(BT);
    const float eps = 1e-6f;
    for (int bt = 0; bt < B * T; ++bt) {
        const float* x = h_inp_f.data() + (size_t)bt * C;
        float acc = 0.f;
        for (int c = 0; c < C; ++c) acc += x[c] * x[c];
        acc /= C;
        h_rstd[bt] = 1.0f / std::sqrt(acc + eps);
    }

    // CPU reference in float
    std::vector<float> h_dinp_cpu(N);
    std::vector<float> h_dweight_cpu(C);
    rmsnorm_backward_cpu(h_dinp_cpu.data(), h_dweight_cpu.data(), /*dresidual*/ nullptr,
                         h_dout_f.data(), h_inp_f.data(), h_weight_f.data(), h_rstd.data(), B, T, C);

    // Convert to bf16 for device execution
    std::vector<nv_bfloat16> h_inp_bf16 = to_bf16(h_inp_f);
    std::vector<nv_bfloat16> h_weight_bf16 = to_bf16(h_weight_f);
    std::vector<nv_bfloat16> h_dout_bf16 = to_bf16(h_dout_f);

    thrust::device_vector<nv_bfloat16> d_inp = to_device(h_inp_bf16);
    thrust::device_vector<nv_bfloat16> d_weight = to_device(h_weight_bf16);
    thrust::device_vector<nv_bfloat16> d_dout = to_device(h_dout_bf16);
    thrust::device_vector<float> d_rstd = to_device(h_rstd);
    const nv_bfloat16 bf16_zero = testing_utils::make_nvbf16_from_float(0.0f);
    thrust::device_vector<nv_bfloat16> d_dinp(N, bf16_zero);
    thrust::device_vector<nv_bfloat16> d_dweight(C, bf16_zero);
    thrust::device_vector<nv_bfloat16> d_dresidual(N, bf16_zero);

    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp dp{}; cudaGetDeviceProperties(&dp, dev);
    int scratch_size = get_rmsnorm_backward_scratch_size(C, dp);
    thrust::device_vector<std::byte> d_scratch(scratch_size);

    // Use explicit zero dresidual buffer to ensure deterministic behavior across kernel variants
    rmsnorm_backward(thrust::raw_pointer_cast(d_dinp.data()),
                     thrust::raw_pointer_cast(d_dweight.data()),
                     thrust::raw_pointer_cast(d_scratch.data()),
                     /*dresidual*/ thrust::raw_pointer_cast(d_dresidual.data()),
                     thrust::raw_pointer_cast(d_dout.data()),
                     thrust::raw_pointer_cast(d_inp.data()),
                     thrust::raw_pointer_cast(d_weight.data()),
                     thrust::raw_pointer_cast(d_rstd.data()),
                     /*abs_max*/ nullptr,
                     B, T, C, dp, /*stream*/ 0);
    cudaDeviceSynchronize();

    // Copy back to host as bf16 and convert to float for comparison
    std::vector<nv_bfloat16> h_dinp_bf16 = from_device(d_dinp);
    std::vector<nv_bfloat16> h_dweight_bf16 = from_device(d_dweight);

    std::vector<float> h_dinp(h_dinp_bf16.size());
    std::vector<float> h_dweight(h_dweight_bf16.size());
    // Properly convert bf16 values to float using helper (avoid invalid memcpy casting)
    for (size_t i = 0; i < h_dinp_bf16.size(); ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_dinp_bf16[i], sizeof(bits));
        h_dinp[i] = testing_utils::bf16_bits_to_float(bits);
    }
    for (size_t i = 0; i < h_dweight_bf16.size(); ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_dweight_bf16[i], sizeof(bits));
        h_dweight[i] = testing_utils::bf16_bits_to_float(bits);
    }

    // Tolerances looser due to bf16 quantization. dweight accumulates over B*T and may incur larger rounding error.
    const float atol_inp = 2e-2f;
    const float rtol_inp = 2e-2f;
    const float atol_w = 4e-2f;
    const float rtol_w = 3e-2f;
    for (size_t i = 0; i < h_dinp.size(); ++i) {
        REQUIRE(h_dinp[i] == Catch::Approx(h_dinp_cpu[i]).margin(atol_inp).epsilon(rtol_inp));
    }
    for (int c = 0; c < C; ++c) {
        REQUIRE(h_dweight[c] == Catch::Approx(h_dweight_cpu[c]).margin(atol_w).epsilon(rtol_w));
    }
}
