// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#ifndef LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H
#define LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H

#include "llama_weights.h"
#include "training/adamw_optimizer.h"

class LLamaOptimizerStateManager : public AdamWStateManager {
public:
    LLamaOptimizerStateManager(TransformerConfig cfg, IModel& model, LLamaOptions options, NCCLCommunicator& comm);

    void safe_to_checkpoint(const std::string& checkpoint_dir) override;
    void load_from_checkpoint(const std::string& checkpoint_dir) override;
};

#endif //LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H
