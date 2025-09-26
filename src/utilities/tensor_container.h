// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//
// SPDX-License-Identifier: MIT

#ifndef LLMQ_SRC_MODELS_MODEL_H
#define LLMQ_SRC_MODELS_MODEL_H

#include <functional>
#include <string>

#include <nlohmann/json_fwd.hpp>

class TensorShard;

class ITensorContainer {
  public:
    virtual void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) = 0;

  protected:
    ~ITensorContainer() = default;
};

#endif //LLMQ_SRC_MODELS_MODEL_H
