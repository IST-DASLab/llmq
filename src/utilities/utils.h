// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_UTILS_UTILS_H
#define LLMQ_SRC_UTILS_UTILS_H

#include <concepts>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <driver_types.h>
#include <library_types.h>

#ifndef __CUDACC__
#define HOST_DEVICE
#else
#define HOST_DEVICE __host__ __device__
#endif

/// This exception will be thrown for reported cuda errors
class cuda_error : public std::runtime_error {
public:
    cuda_error(cudaError_t err, const std::string& arg) :
            std::runtime_error(arg), code(err){};

    cudaError_t code;
};

/// Check `status`; if it isn't `cudaSuccess`, throw the corresponding `cuda_error`
void cuda_throw_on_error(cudaError_t status, const char* statement, const char* file, int line);

#define CUDA_CHECK(status) cuda_throw_on_error(status, #status, __FILE__, __LINE__)

template<std::integral T>
constexpr T HOST_DEVICE div_ceil(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

[[noreturn]] void throw_not_divisible(long long dividend, long long divisor);

template<std::integral T>
constexpr T div_exact(T dividend, T divisor) {
    if(dividend % divisor != 0) {
        throw_not_divisible(dividend, divisor);
    }
    return dividend / divisor;
}

template<std::integral Dst, std::integral Src>
constexpr Dst narrow(Src input) {
    if constexpr (std::is_signed_v<Src>) {
        if (std::is_unsigned_v<Dst> && input < 0) {
            throw std::out_of_range("Cannot convert negative number to unsigned");
        }
        if (std::is_signed_v<Dst> && input < std::numeric_limits<Dst>::min())
        {
            throw std::out_of_range("Out of range in integer conversion: underflow");
        }
    }

    if (input > std::numeric_limits<Dst>::max())
    {
        throw std::out_of_range("Out of range in integer conversion: overflow");
    }

    return static_cast<Dst>(input);
}

// ----------------------------------------------------------------------------
template<typename Scalar>
inline cudaDataType to_cuda_lib_type_enum;

template<> inline constexpr cudaDataType to_cuda_lib_type_enum<float> = cudaDataType::CUDA_R_32F;
template<> inline constexpr cudaDataType to_cuda_lib_type_enum<nv_bfloat16> = cudaDataType::CUDA_R_16BF;
template<> inline constexpr cudaDataType to_cuda_lib_type_enum<std::int8_t> = cudaDataType::CUDA_R_8I;
template<> inline constexpr cudaDataType to_cuda_lib_type_enum<__nv_fp8_e4m3> = cudaDataType::CUDA_R_8F_E4M3;
template<> inline constexpr cudaDataType to_cuda_lib_type_enum<__nv_fp8_e5m2> = cudaDataType::CUDA_R_8F_E5M2;

template<typename FP8Type>
inline __nv_fp8_interpretation_t fp8_interpretation_v;
template<> inline constexpr __nv_fp8_interpretation_t fp8_interpretation_v<__nv_fp8_e4m3> = __nv_fp8_interpretation_t::__NV_E4M3;
template<> inline constexpr __nv_fp8_interpretation_t fp8_interpretation_v<__nv_fp8_e5m2> = __nv_fp8_interpretation_t::__NV_E5M2;

// ----------------------------------------------------------------------------
// NVTX utils

class NvtxRange {
public:
    explicit NvtxRange(const char* s) noexcept;
    NvtxRange(const std::string& base_str, int number);
    ~NvtxRange() noexcept;
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range_##__COUNTER__ (__FUNCTION__)

cudaStream_t create_named_stream(const char* name);
cudaEvent_t create_named_event(const char* name, bool timing=false);


// ----------------------------------------------------------------------------
bool iequals(std::string_view lhs, std::string_view rhs);

#endif //LLMQ_SRC_UTILS_UTILS_H
