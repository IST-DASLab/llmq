// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//
// SPDX-License-Identifier: MIT

#ifndef LLMQ_SRC_UTILITIES_MFU_H
#define LLMQ_SRC_UTILITIES_MFU_H

#include <utility>
#include <vector>

enum class ETensorDType : int;

// Estimate how many ops of each type a transformer will need
std::vector<std::pair<ETensorDType, long>> get_transformer_ops(long non_embedding_params, ETensorDType non_embedding_dtype, long embedding_params, ETensorDType embedding_dtype, long d_att, long n_layers, long ctx);

// For a given list of operations, how many microseconds would it take the device if it ran at
// peak speed
long estimate_speed_of_light(const char* device, const std::vector<std::pair<ETensorDType, long>>& ops);

#endif // LLMQ_SRC_UTILITIES_MFU_H
