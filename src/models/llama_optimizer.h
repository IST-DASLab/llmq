// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#ifndef LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H
#define LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H

#include "llama_weights.h"
#include "training/adamw_optimizer.h"
#include <array>

class LLamaOptimizerStateManager : public AdamWStateManager {
public:
    LLamaOptimizerStateManager(TransformerConfig cfg, LLamaOptions options, cudaStream_t stream, NCCLCommunicator& comm, TensorAllocator& alloc);
    SimpleTensorContainer& non_block_m() override;
    SimpleTensorContainer& non_block_v() override;

    void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream) override;

    ITensorContainer& full_m() { return mOptM; }
    ITensorContainer& full_v() { return mOptV; }
    sLLamaWeights& scales_m() { return mOptMScales; }

    SimpleTensorContainer& get_block_m(int layer_idx, cudaStream_t stream) override;
    SimpleTensorContainer& get_block_v(int layer_idx, cudaStream_t stream) override;
    SimpleTensorContainer& get_block_scales_m(int layer_idx) override;
private:
    // mOptM.Blocks[i] and mMBlockStorage[i] alias the same memory.
    // mOptM provides convenient access to the individual tensors of a block, whereas
    // mMBlockStorage has just one large, byte-typed buffer for bulk transfers.
    sLLamaWeights mOptM;
    sLLamaWeights mOptV;
    sLLamaWeights mOptMScales;

    std::array<sLLamaBlockWeights<TensorShard>, 2> mOptMBuffer;
    std::array<sLLamaBlockWeights<TensorShard>, 2> mOptVBuffer;
};

#endif //LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H
