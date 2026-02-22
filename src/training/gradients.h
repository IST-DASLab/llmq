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

/// Manages gradient accumulation and distribution across micro-batches and distributed workers.
/// Subclasses support different gradient sharding strategies (e.g., unsharded, block-sharded) to trade off
/// memory consumption and communication cost.
///
/// There are two sets of methods that allow querying gradient buffers. The `_full` methods return a view
/// to an unsharded tensor to which the weight gradient of the current step needs to be added.
/// The `_shard` methods return a view to a shard of the gradient tensor, e.g., for use in the optimizer.
/// Shards may alias into the original full tensors, or they may use independent memory.
/// These functions may block the supplied stream until the requested buffers are ready.
class IGradientManager {
public:
    virtual ~IGradientManager() = default;

    /// Must be called at the start of each micro-batch step.
    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);

    /// Must be called at the end of each micro-batch step. Finalizes gradient operations and may trigger
    /// communication to reduce/scatter gradients across ranks.
    virtual void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) = 0;

    /// Get full (non-sharded) gradient tensor for non-block parameters (embeddings, LM head, etc.).
    /// After the tensor has been updated, `notify_non_block` must be called to ensure proper synchronization.
    /// \param index Weight index in the non-block parameter set
    /// \param stream CUDA stream for potential synchronization
    /// \param comm Communicator for potential gradient fetching
    /// \param accumulate [out] true = add to existing gradients, false = overwrite
    virtual Tensor& get_non_block_full(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) = 0;

    /// Get full (non-sharded) gradient container for a transformer block.
    /// After the tensors have been updated, `notify_block` must be called to ensure proper synchronization.
    /// \param layer_idx Index of the transformer layer
    /// \param stream CUDA stream for potential synchronization
    /// \param comm Communicator for potential gradient fetching
    /// \param accumulate [out] true = add to existing gradients, false = overwrite
    virtual SimpleTensorContainer& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) = 0;

    /// Get local rank's shard of non-block parameter gradients.
    /// Used by optimizer to update the portion of parameters this rank is responsible for (ZeRO-1 style).
    virtual Tensor& get_non_block_shard(std::size_t index, cudaStream_t stream) = 0;

    /// Get local rank's shard of block parameter gradients.
    /// Used by optimizer to update the portion of parameters this rank is responsible for (ZeRO-1 style).
    virtual SimpleTensorContainer& get_block_shard(int layer_idx, cudaStream_t stream) = 0;

    /// Notify that gradient computation is complete for a non-block parameter.
    /// May trigger async communication (reduce-scatter/all-reduce) depending on sharding strategy.
    virtual void notify_non_block(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm) = 0;

    /// Notify that gradient computation is complete for a transformer block.
    /// May trigger async communication (reduce-scatter/all-reduce) depending on sharding strategy.
    virtual void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) = 0;

    /// Gets the current step number
    int get_step_counter() const { return mStepCounter; }

protected:
    IGradientManager(std::uint64_t seed, int step);

    /// Check if this is the first micro-batch in the gradient accumulation loop.
    /// Useful for determining whether to zero or accumulate gradients.
    bool is_first_micro_step() const { return mIsFirstMicroStep; }

    /// Check if this is the last micro-batch in the gradient accumulation loop.
    /// Useful for determining when to finalize gradient communication.
    bool is_last_micro_step() const { return mIsLastMicroStep; }

    /// Generate deterministic random numbers, e.g., for stochastic rounding, using the philox generator.
    template<std::size_t S>
    void generate_rng(std::span<std::uint32_t, S> dst, std::uint32_t x, std::uint32_t y) {
        mRng.generate(dst, x, y);
    }

private:
    /// Hook called on the first micro-batch step for gradient initialization (e.g., zeroing).
    virtual void on_first_micro_step(cudaStream_t stream) = 0;

    int mStepCounter = -1;
    bool mIsFirstMicroStep = true;
    bool mIsLastMicroStep = false;
    Philox4x32 mRng;
};

/// Gradient manager where each worker maintains full gradient copies.
/// Implements ZeRO-1 optimization: gradients are replicated across all ranks,
/// but each rank's optimizer is responsible for updating only its assigned shard.
/// This reduces optimizer memory while keeping gradient computation simple.
class UnshardedGradientManager : public IGradientManager {
public:
    ~UnshardedGradientManager();
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) override;

    Tensor& get_non_block_full(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    SimpleTensorContainer& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;

    Tensor& get_non_block_shard(std::size_t index, cudaStream_t stream) override;
    SimpleTensorContainer& get_block_shard(int layer_idx, cudaStream_t stream) override;

    void notify_non_block(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm) override;
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) override;

protected:
    UnshardedGradientManager(const TransformerConfig& cfg, IModel& model, std::uint64_t seed, int step, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc);

    cudaEvent_t mGradEvent = nullptr;  ///< Synchronization event for backward pass completion

    /// Full gradient accumulators for transformer blocks (replicated across all ranks)
    std::vector<GenericTensorContainer> mBlockGradients;
    /// Full gradient accumulators for non-block parameters (replicated across all ranks)
    GenericTensorContainer mNonBlockGradients;

    /// Views into mBlockGradients representing this rank's shard
    std::vector<GenericTensorContainer> mBlockShards;
    /// Views into mNonBlockGradients representing this rank's shard
    GenericTensorContainer mNonBlockShards;
};

/// Gradient manager where the transformer blocks are sharded across
/// workers, but the non-block weights are replicated.
class ShardedBlocksGradientManager : public IGradientManager {
public:
    ShardedBlocksGradientManager(const TransformerConfig& cfg, IModel& model, std::uint64_t seed, int step, int rank, int world, bool offload, const std::shared_ptr<TensorAllocator>& alloc);
    ~ShardedBlocksGradientManager();

    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) override;

    Tensor& get_non_block_full(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    SimpleTensorContainer& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;

    Tensor& get_non_block_shard(std::size_t index, cudaStream_t stream) override;
    SimpleTensorContainer& get_block_shard(int layer_idx, cudaStream_t stream) override;

    void notify_non_block(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm) override;
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) override;
protected:
    virtual void sr_accumulate_layer(int layer_idx,
                                     SimpleTensorContainer& dw,
                                     SimpleTensorContainer& sw,
                                     cudaStream_t stream,
                                     NCCLCommunicator& comm) = 0;

    virtual void on_get_block(SimpleTensorContainer& block, cudaStream_t stream) = 0;
    virtual void on_notify_block(int layer_idx, SimpleTensorContainer& block, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) = 0;

    GenericTensorContainer mFullNonBlock;
    GenericTensorContainer mNonBlockShards;

    std::array<GenericTensorContainer, 2> mGradBuffers;
    struct sBlockState {
        cudaEvent_t Event;
        int LayerIdx = -1;
        bool NeedsAccumulation = false;
    };
    std::array<sBlockState, 2> mGradStates;
    std::vector<GenericTensorContainer> mGradShards;
    cudaEvent_t mNonBlockEvent;
};


#endif //LLMQ_TRAINING_GRADIENTS_H
