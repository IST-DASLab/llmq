// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "gradients.h"

#include "model.h"
#include "models/llama_weights.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/lazy_allocator.h"

IGradientManager::IGradientManager(std::uint64_t seed, int step) : mStepCounter(step), mRng(seed) {

}

void IGradientManager::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    mIsFirstMicroStep = micro_step == 0;
    mIsLastMicroStep = micro_step == total_steps - 1;
    if (micro_step == 0) {
        ++mStepCounter;
        on_first_micro_step(stream);
    }
}

UnshardedGradientManager::UnshardedGradientManager(const TransformerConfig& cfg, IModel& model, std::uint64_t seed, int step, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc) : IGradientManager(seed, step){
    LazyAllocator alloc_lazy;

    mBlockGradients.resize(cfg.NumLayers);
    mBlockShards.resize(cfg.NumLayers);
    for (int i = 0; i < cfg.NumLayers; ++i) {
        mBlockGradients[i] = model.create_block_container(cfg, cfg.DType, cfg.DType);
        alloc_lazy.allocate(mBlockGradients[i]);
        alloc_lazy.commit(*alloc, EAllocationType::ON_DEVICE, "block_grad");
        mBlockShards[i] = shard_container(GenericTensorContainer(mBlockGradients[i]), world);
    }

    mNonBlockGradients = model.create_non_block_container(cfg, cfg.DType, cfg.DType);
    alloc_lazy.allocate(mNonBlockGradients);
    alloc_lazy.commit(*alloc, EAllocationType::ON_DEVICE, "nonblock_grad");
    mNonBlockShards = shard_container(GenericTensorContainer(mNonBlockGradients), world);

    mGradEvent = create_named_event("grad_event");
}

void UnshardedGradientManager::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    if (is_last_micro_step()) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, mGradEvent, 0));
    }
}

Tensor& UnshardedGradientManager::get_non_block_full(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm,
    bool& accumulate) {
    accumulate = !is_first_micro_step();
    return mNonBlockGradients.get_tensor(index);
}

SimpleTensorContainer& UnshardedGradientManager::get_block_full(int layer_idx, cudaStream_t stream,
    NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !is_first_micro_step();
    return mBlockGradients.at(layer_idx);
}

Tensor& UnshardedGradientManager::get_non_block_shard(std::size_t index, cudaStream_t stream) {
    return mNonBlockShards.get_tensor(index);
}

SimpleTensorContainer& UnshardedGradientManager::get_block_shard(int layer_idx, cudaStream_t stream) {
    return mBlockShards.at(layer_idx);
}

static void scatter_reduce(Tensor& tensor, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) {
    comm.begin_transaction(stream);
    comm.schedule_reduce_scatter(tensor);
    comm.execute_transaction(signal);
}

static void scatter_reduce(SimpleTensorContainer& block, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) {
    comm.begin_transaction(stream);
    visit([&](Tensor& t){ comm.schedule_reduce_scatter(t); }, block);
    comm.execute_transaction(signal);
}

void UnshardedGradientManager::notify_non_block(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm) {
    if(!is_last_micro_step()) return;
    if (comm.world_size() != 1) {
        NvtxRange r{"notify_non_block"};
        scatter_reduce(mNonBlockGradients.get_tensor(index), stream, mGradEvent, comm);
    }
}

void UnshardedGradientManager::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    if(!is_last_micro_step()) return;
    auto& dw = mBlockGradients.at(layer_idx);
    if (comm.world_size() != 1) {
        scatter_reduce(dw, stream, mGradEvent, comm);
    }
}
