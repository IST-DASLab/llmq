// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#ifndef LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H
#define LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H

#include "llama_weights.h"
#include <array>

class LLamaOptimizerStateManager {
public:
    LLamaOptimizerStateManager(LLamaConfig cfg, LLamaOptions options, cudaStream_t stream, NCCLCommunicator& comm, TensorAllocator& alloc);
    sLLamaNonBlockWeights<TensorShard>& non_block_m();
    sLLamaNonBlockWeights<TensorShard>& non_block_v();

    void begin_optimizer(DeviceMemoryStack& memory);
    void end_optimizer(DeviceMemoryStack& memory);

    sLLamaWeights& full_m() { return mOptM; }
    sLLamaWeights& scales_m() { return mOptMScales; }
    sLLamaWeights& full_v() { return mOptV; }

    void fetch_block(int layer_idx, cudaStream_t fetch_stream);
    sLLamaBlockWeights<TensorShard>& get_block_m(int layer_idx, cudaStream_t stream);
    sLLamaBlockWeights<TensorShard>& get_block_v(int layer_idx, cudaStream_t stream);
    void store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream);
private:
    LLamaConfig mConfig;
    sLLamaWeights mOptM;
    sLLamaWeights mOptV;
    std::array<sLLamaBlockWeights<TensorShard>, 2> mOptMBuffer;
    std::array<sLLamaBlockWeights<TensorShard>, 2> mOptVBuffer;

    sLLamaWeights mOptMScales;

    struct sBufferStatus {
        int LayerIdx = -1;
        cudaEvent_t DoneEvent = nullptr;
        bool Fetch = false;
        bool Done = true;
    };

    std::array<sBufferStatus, 2> mStatus;

    bool mOffloadM;
    bool mOffloadV;
    bool mUseZeroCopy;

    int mRank;
    int mWorld;

    sLLamaBlockWeights<TensorShard>& get_block_from(int layer_idx, cudaStream_t stream, sLLamaBlockWeights<TensorShard>& buf);
    void store_one_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream, sLLamaBlockWeights<TensorShard>& buf, sLLamaBlockWeights<TensorShard>& dst);
};

#endif //LLMQ_SRC_MODELS_LLAMA_OPTIMIZER_H
