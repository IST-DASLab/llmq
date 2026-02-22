// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_BINDING_UTILS_H
#define LLMQ_BINDING_UTILS_H

#include "utilities/allocator.h"
#include <optional>
#include <string>
#include <nanobind/ndarray.h>

nanobind::dlpack::dtype to_dlpack_dtype(ETensorDType dtype);
ETensorDType from_dlpack_dtype(nanobind::dlpack::dtype dtype);

std::optional<ETensorDType> opt_dtype_from_str(const std::string& dtype_str) ;
nanobind::object cast_opt_dtype(std::optional<ETensorDType> dtype);

void check_ndims(int dims, std::string_view name, int expected);
#define NB_CHECK_NDIMS(arr, ndims) check_ndims(arr.ndim(), #arr, ndims)

#endif //LLMQ_BINDING_UTILS_H
