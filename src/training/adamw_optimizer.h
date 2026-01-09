// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#ifndef LLMQ_ADAMW_OPTIMIZER_H
#define LLMQ_ADAMW_OPTIMIZER_H

#include "transformer_config.h"
#include "utilities/tensor_container.h"
#include "utilities/tensor.h"

enum class EAllocationType : int;
class IModel;
class TensorAllocator;
typedef struct CUstream_st *cudaStream_t;

class DeviceMemoryStack;

class AdamWStateManager {
public:
    AdamWStateManager(TransformerConfig cfg, IModel& model, bool offload_m, bool offload_v, ETensorDType type_m, ETensorDType type_v, bool zero_copy, int rank, int world);
    virtual ~AdamWStateManager() = default;
    void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream);
    void end_optimizer(DeviceMemoryStack& memory);

    void fetch_block(int layer_idx, cudaStream_t fetch_stream);
    SimpleTensorContainer& get_block_m(int layer_idx, cudaStream_t stream);
    SimpleTensorContainer& get_block_v(int layer_idx, cudaStream_t stream);
    SimpleTensorContainer& get_block_scales_m(int layer_idx);
    void store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream);

    SimpleTensorContainer& non_block_m();
    SimpleTensorContainer& non_block_m_scales();
    SimpleTensorContainer& non_block_v();

    void allocate_state(IModel& model, cudaStream_t stream, EAllocationType kind, TensorAllocator& alloc);

    virtual void safe_to_checkpoint(const std::string& checkpoint_dir) = 0;
    virtual void load_from_checkpoint(const std::string& checkpoint_dir) = 0;

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

    std::vector<Tensor> mStorageM;
    std::vector<Tensor> mStorageV;
    std::vector<GenericTensorContainer> mBlocksM;
    std::vector<GenericTensorContainer> mBlocksV;
    std::vector<GenericTensorContainer> mBlocksMScales;
    GenericTensorContainer mNonBlockM;
    GenericTensorContainer mNonBlockMScales;
    GenericTensorContainer mNonBlockV;

    std::array<Tensor, 2> mMBufferStorage;
    std::array<Tensor, 2> mVBufferStorage;
    std::array<sBufferStatus, 2> mStatus;
    std::array<GenericTensorContainer, 2> mMDeviceBuffer;
    std::array<GenericTensorContainer, 2> mVDeviceBuffer;
};

#endif //LLMQ_ADAMW_OPTIMIZER_H
