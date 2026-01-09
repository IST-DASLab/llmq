// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#ifndef LLMQ_ADAMW_OPTIMIZER_H
#define LLMQ_ADAMW_OPTIMIZER_H

#include "transformer_config.h"
#include "utilities/tensor_container.h"
#include "utilities/tensor.h"

class IModel;
typedef struct CUstream_st *cudaStream_t;

class DeviceMemoryStack;

class AdamWStateManager {
public:
    AdamWStateManager(TransformerConfig cfg, IModel& model, bool offload_m, bool offload_v, ETensorDType type_m, ETensorDType type_v, bool zero_copy, int rank, int world);
    virtual ~AdamWStateManager() = default;
    virtual void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream);
    virtual void end_optimizer(DeviceMemoryStack& memory);

    void fetch_block(int layer_idx, cudaStream_t fetch_stream);
    virtual SimpleTensorContainer& get_block_m(int layer_idx, cudaStream_t stream) = 0;
    virtual SimpleTensorContainer& get_block_v(int layer_idx, cudaStream_t stream) = 0;
    virtual SimpleTensorContainer& get_block_scales_m(int layer_idx) = 0;
    void store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream);

    virtual SimpleTensorContainer& non_block_m() = 0;
    virtual SimpleTensorContainer& non_block_v() = 0;

protected:
    SimpleTensorContainer& get_block_from(int layer_idx, cudaStream_t stream, SimpleTensorContainer& buf);

    TransformerConfig mConfig;

    bool mOffloadM;
    bool mOffloadV;
    bool mUseZeroCopy;

    ETensorDType mMType;
    ETensorDType mVType;

    int mRank;
    int mWorld;

    struct sBufferStatus {
        int LayerIdx = -1;
        cudaEvent_t DoneEvent = nullptr;
        bool Fetch = false;
        bool Done = true;
    };

    std::vector<Tensor> mMBlockStorage;
    std::vector<Tensor> mVBlockStorage;

    std::array<Tensor, 2> mMBufferStorage;
    std::array<Tensor, 2> mVBufferStorage;
    std::array<sBufferStatus, 2> mStatus;
    std::array<GenericTensorContainer, 2> mOptMBuffer;
    std::array<GenericTensorContainer, 2> mOptVBuffer;
};

#endif //LLMQ_ADAMW_OPTIMIZER_H
