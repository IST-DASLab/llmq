// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "binding_utils.h"

#include <fmt/format.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using nb::dlpack::dtype_code;

nb::dlpack::dtype to_dlpack_dtype(ETensorDType dtype) {
    switch (dtype) {
        case ETensorDType::FP32:
            return {static_cast<std::uint8_t>(dtype_code::Float), 32, 1};
        case ETensorDType::BF16:
            return {static_cast<std::uint8_t>(dtype_code::Bfloat), 16, 1};
        case ETensorDType::INT8:
            return {static_cast<std::uint8_t>(dtype_code::Int), 8, 1};
        case ETensorDType::BYTE:
            return {static_cast<std::uint8_t>(dtype_code::UInt), 8, 1};
        case ETensorDType::FP16:
            return {static_cast<std::uint8_t>(dtype_code::Float), 16, 1};
        case ETensorDType::INT32:
            return {static_cast<std::uint8_t>(dtype_code::Int), 32, 1};
        case ETensorDType::FP8_E4M3:
            return {static_cast<std::uint8_t>(dtype_code::Float8_E4M3FN), 8, 1};
        case ETensorDType::FP8_E5M2:
            return {static_cast<std::uint8_t>(dtype_code::Float8_E5M2), 8, 1};
        default:
            throw std::runtime_error("Invalid dtype");
    }
}


ETensorDType from_dlpack_dtype(nb::dlpack::dtype dtype) {
    nb::dlpack::dtype_code code = static_cast<nb::dlpack::dtype_code>(dtype.code);
    using nb::dlpack::dtype_code;
    switch (code) {
        case dtype_code::Float: {
            if (dtype.bits == 32) {
                return ETensorDType::FP32;
            } else if (dtype.bits == 16) {
                return ETensorDType::FP16;
            } else {
                throw std::invalid_argument("Unsupported Float dtype: bit width must be 32 or 16");
            }
        }
        case dtype_code::Bfloat:
            if (dtype.bits == 16) {
                return ETensorDType::BF16;
            } else {
                throw std::invalid_argument("Unsupported BFloat dtype: bit width must be 16");
            }
        case dtype_code::Int:
            if (dtype.bits == 8) {
                return ETensorDType::INT8;
            } else if (dtype.bits == 32) {
                return ETensorDType::INT32;
            } else {
                throw std::invalid_argument("Unsupported Int dtype: bit width must be 8 or 32, got " + std::to_string(dtype.bits));
            }
        case dtype_code::UInt:
            if (dtype.bits == 8) {
                return ETensorDType::BYTE;
            } else {
                throw std::invalid_argument("Unsupported UInt dtype: bit width must be 8");
            }
        case dtype_code::Float8_E4M3FN:
            if (dtype.bits == 8) {
                return ETensorDType::FP8_E4M3;
            } else {
                throw std::invalid_argument("Unsupported E4M3 dtype: bit width must be 8");
            }
        case dtype_code::Float8_E5M2:
            if (dtype.bits == 8) {
                return ETensorDType::FP8_E5M2;
            } else {
                throw std::invalid_argument("Unsupported E5M2 dtype: bit width must be 8");
            }
        case dtype_code::Float8_E3M4:
        case dtype_code::Float8_E5M2FNUZ:
        case dtype_code::Float8_E4M3:
        case dtype_code::Float8_E4M3FNUZ:
        case dtype_code::Float8_E4M3B11FNUZ:
        case dtype_code::Float8_E8M0FNU:
            throw std::invalid_argument("Unsupported Float8 dtype");
        default:
            throw std::invalid_argument("Unsupported dtype");
    }
}

std::optional<ETensorDType> opt_dtype_from_str(const std::string& dtype_str) {
    if (dtype_str.empty()) {
        return std::nullopt;
    }
    return dtype_from_str(dtype_str);
}

nb::object cast_opt_dtype(std::optional<ETensorDType> dtype) {
    if (dtype.has_value()) {
        return nb::cast(dtype_to_str(dtype.value()));
    }
    return nb::none();
}

void check_ndims(int dims, std::string_view name, int expected) {
    if(dims != expected) {
        throw std::invalid_argument(fmt::format("Expected {} to have {} dimensions, but got {}", name, expected, dims));
    }
}
