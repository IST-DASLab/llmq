// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "binding_utils.h"
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
