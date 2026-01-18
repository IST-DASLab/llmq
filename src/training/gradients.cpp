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

void UnshardedGradientManager::notify_non_block(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm) {
    if(!is_last_micro_step()) return;
    if (comm.world_size() != 1) {
        NvtxRange r{"notify_non_block"};
        comm.reduce_scatter(mNonBlockGradients.get_tensor(index), stream, mGradEvent);
    }
}

void UnshardedGradientManager::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    if(!is_last_micro_step()) return;
    auto& dw = mBlockGradients.at(layer_idx);
    if (comm.world_size() != 1) {
        comm.reduce_scatter(dw, stream, mGradEvent);
    }
}

ShardedBlocksGradientManager::ShardedBlocksGradientManager(const TransformerConfig& cfg, IModel& model, std::uint64_t seed, int step,
    int rank, int world, bool offload, const std::shared_ptr<TensorAllocator>& alloc) : IGradientManager(seed, step)
{
    LazyAllocator alloc_lazy;
    mFullNonBlock = model.create_non_block_container(cfg, cfg.DType, cfg.DType);
    alloc_lazy.allocate(mFullNonBlock);
    alloc_lazy.commit(*alloc, EAllocationType::ON_DEVICE, "nonblock_grad");
    mNonBlockShards = shard_container(GenericTensorContainer(mFullNonBlock), world);

    mGradBuffers[0] = model.create_block_container(cfg, cfg.DType, cfg.DType);
    mGradBuffers[1] = model.create_block_container(cfg, cfg.DType, cfg.DType);
    alloc_lazy.allocate(mGradBuffers[0]);
    alloc_lazy.allocate(mGradBuffers[1]);
    alloc_lazy.commit(*alloc, EAllocationType::ON_DEVICE, "block_grad_buffers");

    mGradStates[0].Event = create_named_event("grad_event_0");
    mGradStates[1].Event = create_named_event("grad_event_1");
    mNonBlockEvent = create_named_event("grad_nonblock_event");
    mGradShards.resize(cfg.NumLayers);
    for(int i = 0; i < cfg.NumLayers; ++i) {
        EAllocationType kind = offload ? EAllocationType::PINNED : EAllocationType::ON_DEVICE;
        mGradShards[i] = shard_container(model.create_block_container(cfg, cfg.DType, cfg.DType), world);
        alloc_lazy.allocate(mGradShards[i]);
        alloc_lazy.commit(*alloc, kind, "block_grad_shards");
    }
}

void ShardedBlocksGradientManager::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    for (int i = 0; i < 2; ++i) {
        auto& state = mGradStates[i];
        int layer_idx = state.LayerIdx;
        if (state.NeedsAccumulation) {
            // we need to wait for the previous accumulation to finish
            CUDA_CHECK(cudaStreamWaitEvent(stream, state.Event, 0));
            auto& dw = mGradBuffers.at(layer_idx % 2);
            auto& sw = mGradShards.at(layer_idx);
            sr_accumulate_layer(layer_idx, dw, sw, stream, comm);
            state.NeedsAccumulation = false;
        }
    }
    if (is_last_micro_step())
        CUDA_CHECK(cudaStreamWaitEvent(stream, mNonBlockEvent, 0));
}


Tensor& ShardedBlocksGradientManager::get_non_block_full(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !is_first_micro_step();
    return mFullNonBlock.get_tensor(index);
}

SimpleTensorContainer& ShardedBlocksGradientManager::get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = false;

    auto& state = mGradStates.at(layer_idx % 2);
    auto& dw = mGradBuffers.at(layer_idx % 2);
    CUDA_CHECK(cudaStreamWaitEvent(stream, state.Event, 0)); // make sure the previous copy has finished
    if (state.NeedsAccumulation) {
        // already used; this means we need to schedule the accumulation first
        sr_accumulate_layer(state.LayerIdx, mGradBuffers.at(state.LayerIdx % 2), mGradShards.at(state.LayerIdx), stream, comm);
        state.NeedsAccumulation = false;
    }
    state.LayerIdx = layer_idx;
    // reset local gradient buffers
    on_get_block(dw, stream);
    return dw;
}

void ShardedBlocksGradientManager::notify_non_block(std::size_t index, cudaStream_t stream, NCCLCommunicator& comm) {
    if(!is_last_micro_step()) return;
    NvtxRange r{"notify"};
    comm.reduce_scatter(mFullNonBlock.get_tensor(index), stream, mNonBlockEvent);
}

void ShardedBlocksGradientManager::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    auto& state = mGradStates[layer_idx % 2];
    if(state.LayerIdx != layer_idx) {
        throw std::logic_error("notify_block called with wrong layer index");
    }
    if (state.NeedsAccumulation) {
        throw std::logic_error("notify_block called before accumulation has finished");
    }

    auto& dw = mGradBuffers.at(layer_idx % 2);
    on_notify_block(layer_idx, dw, stream, state.Event, comm);
    state.NeedsAccumulation = true;
}


Tensor& ShardedBlocksGradientManager::get_non_block_shard(std::size_t index, cudaStream_t stream) {
    return mNonBlockShards.get_tensor(index);
}

SimpleTensorContainer& ShardedBlocksGradientManager::get_block_shard(int layer_idx, cudaStream_t stream) {
    return mGradShards.at(layer_idx);
}
