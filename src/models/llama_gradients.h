// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H
#define LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H

#include "llama_weights.h"
#include "training/gradients.h"

class LLamaModel;

class LLamaGradientsUnsharded : public UnshardedGradientManager {
public:
    LLamaGradientsUnsharded(const TransformerConfig& cfg, IModel& model, std::uint64_t seed, int step, int rank,
        int world, const std::shared_ptr<TensorAllocator>& alloc)
        : UnshardedGradientManager(cfg, model, seed, step, rank, world, alloc) {
    }

    void on_first_micro_step(cudaStream_t stream) override;
};

class LLamaGradientsBlockShardedBase : public ShardedBlocksGradientManager {
public:
    using ShardedBlocksGradientManager::ShardedBlocksGradientManager;

    void on_first_micro_step(cudaStream_t stream) override;
    void on_get_block(SimpleTensorContainer& block, cudaStream_t stream) override;
};

class LLamaGradientsBlockSharded_ScatterReduce : public LLamaGradientsBlockShardedBase {
public:
    using LLamaGradientsBlockShardedBase::LLamaGradientsBlockShardedBase;
private:
    void on_notify_block(int layer_idx, SimpleTensorContainer& block, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) override;
    void sr_accumulate_layer(int layer_idx,
                             SimpleTensorContainer& dw,
                             SimpleTensorContainer& sw,
                             cudaStream_t stream,
                             NCCLCommunicator& comm) override;
};

class LLamaGradientsBlockSharded_AllToAll : public LLamaGradientsBlockShardedBase {
public:
    using LLamaGradientsBlockShardedBase::LLamaGradientsBlockShardedBase;
private:
    void on_notify_block(int layer_idx, SimpleTensorContainer& block, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) override;
    void sr_accumulate_layer(int layer_idx,
                             SimpleTensorContainer& dw,
                             SimpleTensorContainer& sw,
                             cudaStream_t stream,
                             NCCLCommunicator& comm) override;
};

std::unique_ptr<IGradientManager> create_grads_manager(
    std::uint64_t seed, int step, LLamaModel& model,
    const TransformerConfig& config, const LLamaOptions& options,
    int rank, int world, const std::shared_ptr<TensorAllocator>& alloc);
#endif //LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H
