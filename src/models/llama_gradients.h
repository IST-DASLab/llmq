// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H
#define LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H

#include "llama_weights.h"
#include "training/gradients.h"

class LLamaModel;

class LLamaGradsManager : public IGradientManager {
public:
    virtual ~LLamaGradsManager() = default;

    static std::unique_ptr<IGradientManager> create(std::uint64_t seed, int step, LLamaModel& model, const TransformerConfig& config,
                                                     const LLamaOptions& options, int rank, int world,
                                                     const std::shared_ptr<TensorAllocator>& alloc);

protected:
    LLamaGradsManager(std::uint64_t seed, int step);
    virtual void on_first_micro_step(cudaStream_t stream) = 0;

    void scatter_reduce(Tensor& tensor, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm);
    virtual void scatter_reduce(int layer_idx, SimpleTensorContainer& block, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm);
};

#endif //LLMQ_SRC_MODELS_LLAMA_GRADIENTS_H
