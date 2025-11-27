// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#include <cuda_bf16.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "test_config.h"

namespace {

// BF16 helper: round-to-nearest-even conversion (emulated on CPU)
static inline uint16_t float_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // round to nearest even on the cut at 16 LSBs
    uint32_t lsb = (u >> 16) & 1u;                // last bit that will remain
    uint32_t rounding_bias = 0x7FFFu + lsb;       // RN-even
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

static inline float bf16_bits_to_float(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static inline nv_bfloat16 make_nvbf16_from_float(float f) {
    uint16_t b = float_to_bf16_bits(f);
    nv_bfloat16 v{};
    std::memcpy(&v, &b, sizeof(b));
    return v;
}

// CPU baseline SWIGLU forward: out = (x1 * x2) / (1 + exp(-x2))
static void swiglu_forward_cpu(float* out, const float* inp, int B, int T, int C) {
    const int C2 = 2 * C;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* row = inp + (b * T + t) * C2;
            const float* up = row;
            const float* gate = row + C;
            float* o = out + (b * T + t) * C;
            for (int c = 0; c < C; ++c) {
                float x1 = up[c];
                float x2 = gate[c];
                float s = 1.0f / (1.0f + std::exp(-x2));
                o[c] = x1 * x2 * s;
            }
        }
    }
}

// CPU baseline SWIGLU backward: given dout(B,T,C), inp(B,T,2C) -> dinp(B,T,2C)
static void swiglu_backward_cpu(float* dinp, const float* dout, const float* inp, int B, int T, int C) {
    const int C2 = 2 * C;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* row = inp + (b * T + t) * C2;
            const float* up = row;
            const float* gate = row + C;
            float* di = dinp + (b * T + t) * C2;
            float* di1 = di;
            float* di2 = di + C;
            const float* d = dout + (b * T + t) * C;
            for (int c = 0; c < C; ++c) {
                float x1 = up[c];
                float x2 = gate[c];
                float g = d[c];
                float s = 1.0f / (1.0f + std::exp(-x2));
                float dx1 = g * x2 * s;
                float dx2 = g * x1 * s * (1.0f + x2 * (1.0f - s));
                di1[c] = dx1;
                di2[c] = dx2;
            }
        }
    }
}

static float max_abs(const float* data, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(data[i]));
    return m;
}

static void fill_inputs(std::vector<float>& inp, int B, int T, int C2) {
    // Deterministic but varied pattern
    for (int i = 0; i < B * T * C2; ++i) {
        float x = std::sin(0.001f * (i + 1)) * 0.9f + std::cos(0.013f * (i + 7)) * 0.1f;
        inp[i] = x;
    }
}

static void fill_dout(std::vector<float>& dout, int B, int T, int C) {
    for (int i = 0; i < B * T * C; ++i) {
        float x = std::cos(0.002f * (i + 5)) * 0.5f;
        dout[i] = x;
    }
}

} // namespace

TEST_CASE("swiglu forward/backward fp32 matches CPU", "[swiglu][fp32]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T; // ensure B*T*C divisible by kernel block requirements
    const int C = cfg.C; // multiple of 4
    // Validate constraints for the fp32 kernels
    long long prod = 1LL * B * T * C;
    bool ok = (C % 4 == 0) && (prod % 1024 == 0);
    if (!ok) {
        INFO("Invalid sizes for fp32: require C % 4 == 0 and (B*T*C) % 1024 == 0");
        INFO("Provided: B=" << B << ", T=" << T << ", C=" << C << ", B*T*C=" << prod);
        FAIL("Aborting fp32 test due to invalid size configuration");
    }
    const int C2 = 2 * C;
    const size_t n_inp = static_cast<size_t>(B) * T * C2;
    const size_t n_out = static_cast<size_t>(B) * T * C;

    std::vector<float> h_inp(n_inp);
    fill_inputs(h_inp, B, T, C2);

    std::vector<float> h_out_cpu(n_out);
    swiglu_forward_cpu(h_out_cpu.data(), h_inp.data(), B, T, C);
    float cpu_absmax_fwd = max_abs(h_out_cpu.data(), n_out);

    // Device buffers
    thrust::device_vector<float> d_inp(n_inp);
    thrust::device_vector<float> d_out(n_out);
    thrust::device_vector<float> d_dout(n_out);
    thrust::device_vector<float> d_dinp(n_inp);
    thrust::device_vector<float> d_absmax(1);

    thrust::copy(h_inp.begin(), h_inp.end(), d_inp.begin());

    // Forward without absmax
    swiglu_forward(thrust::raw_pointer_cast(d_out.data()),
                   thrust::raw_pointer_cast(d_inp.data()),
                   nullptr, B, T, C, /*stream*/0);
    std::vector<float> h_out(n_out, 0.f);
    thrust::copy(d_out.begin(), d_out.end(), h_out.begin());

    for (size_t i = 0; i < n_out; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(h_out_cpu[i]).margin(1e-6f));
    }

    // Forward with absmax
    swiglu_forward(thrust::raw_pointer_cast(d_out.data()),
                   thrust::raw_pointer_cast(d_inp.data()),
                   thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);
    float h_absmax = 0.f;
    thrust::copy(d_absmax.begin(), d_absmax.end(), &h_absmax);
    REQUIRE(h_absmax == Catch::Approx(cpu_absmax_fwd).margin(1e-6f));

    // Backward: prepare dout
    std::vector<float> h_dout(n_out);
    fill_dout(h_dout, B, T, C);
    thrust::copy(h_dout.begin(), h_dout.end(), d_dout.begin());

    // CPU backward
    std::vector<float> h_dinp_cpu(n_inp);
    swiglu_backward_cpu(h_dinp_cpu.data(), h_dout.data(), h_inp.data(), B, T, C);
    float cpu_absmax_bwd = max_abs(h_dinp_cpu.data(), n_inp);

    // GPU backward without absmax
    swiglu_backward(thrust::raw_pointer_cast(d_dinp.data()),
                    thrust::raw_pointer_cast(d_dout.data()),
                    thrust::raw_pointer_cast(d_inp.data()),
                    nullptr, B, T, C, 0);
    std::vector<float> h_dinp(n_inp, 0.f);
    thrust::copy(d_dinp.begin(), d_dinp.end(), h_dinp.begin());
    for (size_t i = 0; i < n_inp; ++i) {
        REQUIRE(h_dinp[i] == Catch::Approx(h_dinp_cpu[i]).margin(1e-5f));
    }

    // GPU backward with absmax
    swiglu_backward(thrust::raw_pointer_cast(d_dinp.data()),
                    thrust::raw_pointer_cast(d_dout.data()),
                    thrust::raw_pointer_cast(d_inp.data()),
                    thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);
    thrust::copy(d_absmax.begin(), d_absmax.end(), &h_absmax);
    REQUIRE(h_absmax == Catch::Approx(cpu_absmax_bwd).margin(1e-6f));
}

TEST_CASE("swiglu forward/backward bfloat16 matches CPU (emulated)", "[swiglu][bf16]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T; // ensure divisibility
    const int C = cfg.C; // multiple of 8 for bf16 path
    // Validate constraints for the bf16 kernels
    long long prod = 1LL * B * T * C;
    bool ok = (C % 8 == 0) && (prod % 2048 == 0);
    if (!ok) {
        INFO("Invalid sizes for bf16: require C % 8 == 0 and (B*T*C) % 2048 == 0");
        INFO("Provided: B=" << B << ", T=" << T << ", C=" << C << ", B*T*C=" << prod);
        FAIL("Aborting bf16 test due to invalid size configuration");
    }
    const int C2 = 2 * C;
    const size_t n_inp = static_cast<size_t>(B) * T * C2;
    const size_t n_out = static_cast<size_t>(B) * T * C;

    // Prepare float inputs and convert to bf16 storage for GPU
    std::vector<float> h_inp_f(n_inp);
    fill_inputs(h_inp_f, B, T, C2);

    std::vector<nv_bfloat16> h_inp_bf16(n_inp);
    for (size_t i = 0; i < n_inp; ++i) h_inp_bf16[i] = make_nvbf16_from_float(h_inp_f[i]);

    // CPU forward but quantized to bf16 first (to match kernel math in bf16)
    // Convert inputs to bf16 -> back to float to emulate bf16 arithmetic for x1, x2
    std::vector<float> h_inp_q(n_inp);
    for (size_t i = 0; i < n_inp; ++i) h_inp_q[i] = bf16_bits_to_float(float_to_bf16_bits(h_inp_f[i]));

    std::vector<float> h_out_cpu(n_out);
    swiglu_forward_cpu(h_out_cpu.data(), h_inp_q.data(), B, T, C);

    // Quantize output to bf16 too, since kernel stores bf16
    std::vector<float> h_out_cpu_q(n_out);
    for (size_t i = 0; i < n_out; ++i) h_out_cpu_q[i] = bf16_bits_to_float(float_to_bf16_bits(h_out_cpu[i]));
    float cpu_absmax_fwd = max_abs(h_out_cpu_q.data(), n_out);

    // Device allocations via Thrust
    thrust::device_vector<nv_bfloat16> d_inp(n_inp);
    thrust::device_vector<nv_bfloat16> d_out(n_out);
    thrust::device_vector<nv_bfloat16> d_dout(n_out);
    thrust::device_vector<nv_bfloat16> d_dinp(n_inp);
    thrust::device_vector<float> d_absmax(1);

    thrust::copy(h_inp_bf16.begin(), h_inp_bf16.end(), d_inp.begin());

    // Forward with absmax (also validates without by comparing values)
    swiglu_forward(thrust::raw_pointer_cast(d_out.data()),
                   thrust::raw_pointer_cast(d_inp.data()),
                   thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);
    std::vector<nv_bfloat16> h_out_bf16(n_out);
    thrust::copy(d_out.begin(), d_out.end(), h_out_bf16.begin());

    std::vector<float> h_out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_out_bf16[i], sizeof(bits));
        h_out[i] = bf16_bits_to_float(bits);
        REQUIRE(h_out[i] == Catch::Approx(h_out_cpu_q[i]).margin(3e-2f));
    }
    float h_absmax = 0.f;
    thrust::copy(d_absmax.begin(), d_absmax.end(), &h_absmax);
    REQUIRE(h_absmax == Catch::Approx(cpu_absmax_fwd).margin(5e-3f));

    // Backward
    std::vector<float> h_dout_f(n_out);
    fill_dout(h_dout_f, B, T, C);
    // Quantize dout to bf16 for GPU input and for CPU emulation
    std::vector<nv_bfloat16> h_dout_bf16(n_out);
    std::vector<float> h_dout_q(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        h_dout_bf16[i] = make_nvbf16_from_float(h_dout_f[i]);
        h_dout_q[i] = bf16_bits_to_float(float_to_bf16_bits(h_dout_f[i]));
    }
    thrust::copy(h_dout_bf16.begin(), h_dout_bf16.end(), d_dout.begin());

    // CPU backward on quantized inputs/grad
    std::vector<float> h_dinp_cpu(n_inp);
    swiglu_backward_cpu(h_dinp_cpu.data(), h_dout_q.data(), h_inp_q.data(), B, T, C);
    // Quantize gradients as kernel outputs bf16
    for (size_t i = 0; i < n_inp; ++i) h_dinp_cpu[i] = bf16_bits_to_float(float_to_bf16_bits(h_dinp_cpu[i]));
    float cpu_absmax_bwd = max_abs(h_dinp_cpu.data(), n_inp);

    swiglu_backward(thrust::raw_pointer_cast(d_dinp.data()),
                    thrust::raw_pointer_cast(d_dout.data()),
                    thrust::raw_pointer_cast(d_inp.data()),
                    thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);
    std::vector<nv_bfloat16> h_dinp_bf16(n_inp);
    thrust::copy(d_dinp.begin(), d_dinp.end(), h_dinp_bf16.begin());

    for (size_t i = 0; i < n_inp; ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_dinp_bf16[i], sizeof(bits));
        float v = bf16_bits_to_float(bits);
        REQUIRE(v == Catch::Approx(h_dinp_cpu[i]).margin(3e-2f));
    }
    thrust::copy(d_absmax.begin(), d_absmax.end(), &h_absmax);
    REQUIRE(h_absmax == Catch::Approx(cpu_absmax_bwd).margin(5e-3f));
}

