// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_TRAINING_GRADIENTS_H
#define LLMQ_TRAINING_GRADIENTS_H

#include <cstddef>

struct Tensor;
class SimpleTensorContainer;
class NCCLCommunicator;
typedef struct CUstream_st *cudaStream_t;

class IGradientManager {
public:
    virtual ~IGradientManager() = default;

    virtual void start_micro_step(cudaStream_t stream, int micro_step, int total_steps) = 0;
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
};

#endif //LLMQ_TRAINING_GRADIENTS_H
