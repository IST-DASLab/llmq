// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_TRAINING_GRADIENTS_H
#define LLMQ_TRAINING_GRADIENTS_H

#include <cstdint>
#include <memory>
#include <vector>

#include "utilities/philox.h"
#include "utilities/tensor_container.h"

class IModel;
class TensorAllocator;
struct TransformerConfig;
struct Tensor;
class SimpleTensorContainer;
class NCCLCommunicator;

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

class IGradientManager {
public:
    virtual ~IGradientManager() = default;

    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);
    virtual void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) = 0;

    // Get references to full gradient accumulators for use in the backward pass
    virtual Tensor& get_non_block_full(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) = 0;
    virtual SimpleTensorContainer& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) = 0;

    // Get references to sharded gradients for use in the optimizer
    virtual Tensor& get_non_block_shard(std::size_t index, cudaStream_t stream) = 0;
    virtual SimpleTensorContainer& get_block_shard(int layer_idx, cudaStream_t stream) = 0;

    // notify that gradient calculations have been completed
    virtual void notify_non_block(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm) = 0;
    virtual void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) = 0;

    int get_step_counter() const { return mStepCounter; }
protected:
    IGradientManager(std::uint64_t seed, int step);

    bool is_first_micro_step() const { return mIsFirstMicroStep; }
    bool is_last_micro_step() const { return mIsLastMicroStep; }

    template<std::size_t S>
    void generate_rng(std::span<std::uint32_t, S> dst, std::uint32_t x, std::uint32_t y) {
        mRng.generate(dst, x, y);
    }

private:
    virtual void on_first_micro_step(cudaStream_t stream) = 0;

    int mStepCounter = -1;
    bool mIsFirstMicroStep = true;
    bool mIsLastMicroStep = false;
    Philox4x32 mRng;
};

class UnshardedGradientManager : public IGradientManager {
public:
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) override;

    Tensor& get_non_block_full(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    SimpleTensorContainer& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;

    Tensor& get_non_block_shard(std::size_t index, cudaStream_t stream) override;
    SimpleTensorContainer& get_block_shard(int layer_idx, cudaStream_t stream) override;

    void notify_non_block(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm) override;
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) override;

protected:
    UnshardedGradientManager(const TransformerConfig& cfg, IModel& model, std::uint64_t seed, int step, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc);

    void on_first_micro_step(cudaStream_t stream) override = 0;

    cudaEvent_t mGradEvent;
    std::vector<GenericTensorContainer> mBlockGradients;
    GenericTensorContainer mNonBlockGradients;

    std::vector<GenericTensorContainer> mBlockShards;
    GenericTensorContainer mNonBlockShards;
};

#endif //LLMQ_TRAINING_GRADIENTS_H
