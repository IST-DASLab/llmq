// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#ifndef LLMQ_ADAMW_OPTIMIZER_H
#define LLMQ_ADAMW_OPTIMIZER_H

#include "transformer_config.h"
#include "utilities/tensor_container.h"

typedef struct CUstream_st *cudaStream_t;

class DeviceMemoryStack;

class AdamWStateManager {
public:
    AdamWStateManager(TransformerConfig cfg, bool offload_m, bool offload_v, bool zero_copy, int rank, int world) :
        mConfig(cfg), mOffloadM(offload_m), mOffloadV(offload_v), mUseZeroCopy(zero_copy), mRank(rank), mWorld(world) {}
    virtual ~AdamWStateManager() = default;
    virtual void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream) = 0;
    virtual void end_optimizer(DeviceMemoryStack& memory) = 0;

    virtual void fetch_block(int layer_idx, cudaStream_t fetch_stream) = 0;
    virtual SimpleTensorContainer& get_block_m(int layer_idx, cudaStream_t stream) = 0;
    virtual SimpleTensorContainer& get_block_v(int layer_idx, cudaStream_t stream) = 0;
    virtual SimpleTensorContainer& get_block_scales_m(int layer_idx) = 0;
    virtual void store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream) = 0;

protected:
    TransformerConfig mConfig;

    bool mOffloadM;
    bool mOffloadV;
    bool mUseZeroCopy;

    int mRank;
    int mWorld;
};

#endif //LLMQ_ADAMW_OPTIMIZER_H
