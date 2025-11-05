// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H
#define LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H

#include "llama_weights.h"
#include "utilities/philox.h"

class LLamaGradsManager {
public:
    virtual ~LLamaGradsManager() = default;

    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);
    virtual void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) = 0;

    // Get references to full gradient accumulators for use in the backward pass
    virtual Tensor& get_embeddings_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) = 0;
    virtual Tensor& get_lmhead_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) = 0;
    virtual Tensor& get_lnf_w_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) = 0;
    virtual sLLamaBlockWeights<Tensor>& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) = 0;

    // Get references to sharded gradients for use in the optimizer
    virtual TensorShard& get_embeddings_shard(cudaStream_t stream) = 0;
    virtual TensorShard& get_lmhead_shard(cudaStream_t stream) = 0;
    virtual TensorShard& get_lnf_w_shard(cudaStream_t stream) = 0;
    virtual sLLamaBlockWeights<TensorShard>& get_block_shard(int layer_idx, cudaStream_t stream) = 0;

    // notify that gradient calculations have been completed
    virtual void notify_embeddings(cudaStream_t stream, NCCLCommunicator& comm) = 0;
    virtual void notify_lmhead(cudaStream_t stream, NCCLCommunicator& comm) = 0;
    virtual void notify_lnf_w(cudaStream_t stream, NCCLCommunicator& comm) = 0;
    virtual void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) = 0;

    static std::unique_ptr<LLamaGradsManager> create(std::uint64_t seed, int step, const LLamaConfig& config,
                                                     const LLamaOptions& options, int rank, int world,
                                                     const std::shared_ptr<TensorAllocator>& alloc);

protected:
    LLamaGradsManager(std::uint64_t seed, int step);
    virtual void on_first_micro_step(cudaStream_t stream) = 0;

    void scatter_reduce(Tensor& tensor, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm);
    virtual void scatter_reduce(int layer_idx, sLLamaBlockWeights<Tensor>& block, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm);

    Philox4x32 mRng;
    int mStepCounter = -1;
    bool mIsFirstMicroStep = true;
    bool mIsLastMicroStep = false;
};

#endif //LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H
