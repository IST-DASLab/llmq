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
    sLLamaNonBlockWeights<TensorShard>& non_block_m();
    sLLamaNonBlockWeights<TensorShard>& non_block_v();

    void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream) override;
    void end_optimizer(DeviceMemoryStack& memory) override;

    sLLamaWeights& full_m() { return mOptM; }
    sLLamaWeights& scales_m() { return mOptMScales; }
    sLLamaWeights& full_v() { return mOptV; }

    void fetch_block(int layer_idx, cudaStream_t fetch_stream) override;
    SimpleTensorContainer& get_block_m(int layer_idx, cudaStream_t stream) override;
    SimpleTensorContainer& get_block_v(int layer_idx, cudaStream_t stream) override;
    SimpleTensorContainer& get_block_scales_m(int layer_idx) override;
    void store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream) override;
private:
    // mOptM.Blocks[i] and mOptMBlockStorage[i] alias the same memory.
    // mOptM provides convenient access to the individual tensors of a block, whereas
    // mOptMBlockStorage has just one large, byte-typed buffer for bulk transfers.
    sLLamaWeights mOptM;
    std::vector<Tensor> mOptMBlockStorage;
    sLLamaWeights mOptV;
    std::vector<Tensor> mOptVBlockStorage;

    std::array<sLLamaBlockWeights<TensorShard>, 2> mOptMBuffer;
    std::array<Tensor, 2> mOptMBufferStorage;
    std::array<sLLamaBlockWeights<TensorShard>, 2> mOptVBuffer;
    std::array<Tensor, 2> mOptVBufferStorage;

    sLLamaWeights mOptMScales;

    struct sBufferStatus {
        int LayerIdx = -1;
        cudaEvent_t DoneEvent = nullptr;
        bool Fetch = false;
        bool Done = true;
    };

    std::array<sBufferStatus, 2> mStatus;

    sLLamaBlockWeights<TensorShard>& get_block_from(int layer_idx, cudaStream_t stream, sLLamaBlockWeights<TensorShard>& buf);
};

#endif //LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H
