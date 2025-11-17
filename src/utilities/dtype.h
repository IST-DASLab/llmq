// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_UTILITIES_DTYPE_H
#define LLMQ_SRC_UTILITIES_DTYPE_H

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

enum class ETensorDType : int {
    FP32,
    BF16,
    FP16,
    INT32,
    INT8,
    FP8_E4M3,
    FP8_E5M2,
    BYTE,               // use for generic buffers
};

template<class T>
consteval ETensorDType dtype_from_pointer(const T*) {
    if constexpr (std::is_same_v<T, float>)  {
        return ETensorDType::FP32;
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return ETensorDType::BF16;
    } else if constexpr (std::is_same_v<T, half>) {
        return ETensorDType::FP16;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return ETensorDType::INT32;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return ETensorDType::INT8;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return ETensorDType::FP8_E4M3;
    }  else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
        return ETensorDType::FP8_E5M2;
    } else if constexpr (std::is_same_v<T, std::byte>) {
        return ETensorDType::BYTE;
    }
    throw std::runtime_error("Invalid dtype");
}

template<typename T>
constexpr ETensorDType dtype_from_type = dtype_from_pointer((T*) nullptr);

ETensorDType dtype_from_str(std::string_view dtype);
const char* dtype_to_str(ETensorDType dtype);
const char* dtype_to_torch_str(ETensorDType dtype);

constexpr int get_dtype_size(const ETensorDType type)  {
    switch (type) {
        case ETensorDType::FP32:
        case ETensorDType::INT32:
            return 4;
        case ETensorDType::BF16:
        case ETensorDType::FP16:
            return 2;
        case ETensorDType::INT8:
        case ETensorDType::FP8_E4M3:
        case ETensorDType::FP8_E5M2:
        case ETensorDType::BYTE:
            return 1;
    }
    throw std::logic_error("Invalid dtype");
}

#endif //LLMQ_SRC_UTILITIES_DTYPE_H
