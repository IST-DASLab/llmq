// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "llama_gradients.h"

#include "kernels/kernels.h"
#include "llama_model.h"
#include "utilities/comm.h"

LLamaGradsManager::LLamaGradsManager(std::uint64_t seed, int step) : mRng(seed), mStepCounter(step) {

}

void LLamaGradsManager::scatter_reduce(Tensor& tensor, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) {
    comm.begin_transaction(stream);
    comm.schedule_reduce_scatter(tensor);
    comm.execute_transaction(signal);
}

void LLamaGradsManager::scatter_reduce(int layer_idx, sLLamaBlockWeights<Tensor>& block, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) {
    comm.begin_transaction(stream);
    comm.schedule_reduce_scatter(block.LN1_w);
    comm.schedule_reduce_scatter(block.Attn_QKV_w);
    comm.schedule_reduce_scatter(block.Attn_Out_w);
    comm.schedule_reduce_scatter(block.LN2_w);
    comm.schedule_reduce_scatter(block.MLP_Up_w);
    comm.schedule_reduce_scatter(block.MLP_Down_w);
    if(block.Attn_QKV_b.has_value()) {
        comm.schedule_reduce_scatter(block.Attn_QKV_b.value());
    }
    comm.execute_transaction(signal);
}

void LLamaGradsManager::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    mIsFirstMicroStep = micro_step == 0;
    mIsLastMicroStep = micro_step == total_steps - 1;
    if (micro_step == 0) {
        ++mStepCounter;
        on_first_micro_step(stream);
    }
}

class LLamaGradientsUnsharded : public LLamaGradsManager {
public:
    LLamaGradientsUnsharded(std::uint64_t seed, int step, const LLamaConfig& config, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc);

    void on_first_micro_step(cudaStream_t stream) override;
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) override;

    Tensor& get_embeddings_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    Tensor& get_lmhead_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    Tensor& get_lnf_w_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    sLLamaBlockWeights<Tensor>& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;

    TensorShard& get_embeddings_shard(cudaStream_t stream) override;
    TensorShard& get_lmhead_shard(cudaStream_t stream) override;
    TensorShard& get_lnf_w_shard(cudaStream_t stream) override;
    sLLamaBlockWeights<TensorShard>& get_block_shard(int layer_idx, cudaStream_t stream) override;

    void notify_embeddings(cudaStream_t stream, NCCLCommunicator& comm) override;
    void notify_lmhead(cudaStream_t stream, NCCLCommunicator& comm) override;
    void notify_lnf_w(cudaStream_t stream, NCCLCommunicator& comm) override;

    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) override;
private:
    sLLamaWeightsSet<Tensor> mFullGradient;
    sLLamaWeightsSet<TensorShard> mShardView;
    cudaEvent_t mGradEvent;
};

LLamaGradientsUnsharded::LLamaGradientsUnsharded(std::uint64_t seed, int step, const LLamaConfig& config, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc) :
    LLamaGradsManager(seed, step)
{
    mFullGradient = allocate_full_weights(config, EAllocationType::ON_DEVICE, *alloc);
    mShardView.NonBlocks = shard_non_block(mFullGradient.NonBlocks, rank, world);
    mShardView.Blocks.reserve(config.NumLayers);
    for(int i = 0; i < config.NumLayers; ++i) {
        mShardView.Blocks.push_back(shard_block(mFullGradient.Blocks[i], rank, world));
    }
    mGradEvent = create_named_event("grad_event");
}

void LLamaGradientsUnsharded::on_first_micro_step(cudaStream_t stream) {
    fill_zero(mFullGradient.NonBlocks.LNF_w, stream);
    if(mFullGradient.NonBlocks.LMHead.Data != mFullGradient.NonBlocks.Embeddings.Data) {
        fill_zero(mFullGradient.NonBlocks.Embeddings, stream);
        fill_zero(mFullGradient.NonBlocks.LMHead, stream);
    } else {
        // embedding backward comes after LMHead backward; and LMHead backward *sets* the gradient
        // on the first backward call, so no need to zero anything.
    }
    for(auto& layer: mFullGradient.Blocks) {
        fill_zero(layer.LN1_w, stream);
        fill_zero(layer.LN2_w, stream);

        if(auto& qkv_b = layer.Attn_QKV_b; qkv_b.has_value()) {
            fill_zero(qkv_b.value(), stream);
        }
        // no need to zero out the matrix weights, we'll just overwrite them on the first
        // grad accumulation step
    }
}

void LLamaGradientsUnsharded::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    if (mIsLastMicroStep) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, mGradEvent, 0));
    }
}


Tensor& LLamaGradientsUnsharded::get_embeddings_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullGradient.NonBlocks.Embeddings;
}
Tensor& LLamaGradientsUnsharded::get_lmhead_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullGradient.NonBlocks.LMHead;
}
Tensor& LLamaGradientsUnsharded::get_lnf_w_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullGradient.NonBlocks.LNF_w;
}
sLLamaBlockWeights<Tensor>& LLamaGradientsUnsharded::get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullGradient.Blocks.at(layer_idx);
}

TensorShard& LLamaGradientsUnsharded::get_embeddings_shard(cudaStream_t stream) {
    return mShardView.NonBlocks.Embeddings;
}
TensorShard& LLamaGradientsUnsharded::get_lmhead_shard(cudaStream_t stream) {
    return mShardView.NonBlocks.LMHead;
}
TensorShard& LLamaGradientsUnsharded::get_lnf_w_shard(cudaStream_t stream) {
    return mShardView.NonBlocks.LNF_w;
}
sLLamaBlockWeights<TensorShard>& LLamaGradientsUnsharded::get_block_shard(int layer_idx, cudaStream_t stream) {
    return mShardView.Blocks.at(layer_idx);
}

void LLamaGradientsUnsharded::notify_embeddings(cudaStream_t stream, NCCLCommunicator& comm) {
    if(!mIsLastMicroStep) return;
    scatter_reduce(mFullGradient.NonBlocks.Embeddings, stream, mGradEvent, comm);
}

void LLamaGradientsUnsharded::notify_lmhead(cudaStream_t stream, NCCLCommunicator& comm) {
    if(!mIsLastMicroStep) return;
    if(mFullGradient.NonBlocks.LMHead.Data == mFullGradient.NonBlocks.Embeddings.Data) return;    // sync lmhead with embeddings
    NvtxRange r{"notify_lmhead"};
    scatter_reduce(mFullGradient.NonBlocks.LMHead, stream, mGradEvent, comm);
}

void LLamaGradientsUnsharded::notify_lnf_w(cudaStream_t stream, NCCLCommunicator& comm) {
    if(!mIsLastMicroStep) return;
    scatter_reduce(mFullGradient.NonBlocks.LNF_w, stream, mGradEvent, comm);
}

void LLamaGradientsUnsharded::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    if(!mIsLastMicroStep) return;
    auto& dw = mFullGradient.Blocks[layer_idx];
    scatter_reduce(layer_idx, dw, stream, mGradEvent, comm);
}

// ---------------------------------------------------------------------------------------------------------------------

// shard the transformer blocks, but not the embeddings and lmhead.
class LLamaGradientsBlockShardedBase : public LLamaGradsManager {
public:
    LLamaGradientsBlockShardedBase(std::uint64_t seed, int step, const LLamaConfig& config, const LLamaOptions& options, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc);

    void on_first_micro_step(cudaStream_t stream) override;
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) override;

    Tensor& get_embeddings_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    Tensor& get_lmhead_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    Tensor& get_lnf_w_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;
    sLLamaBlockWeights<Tensor>& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) override;

    void notify_embeddings(cudaStream_t stream, NCCLCommunicator& comm) override;
    void notify_lmhead(cudaStream_t stream, NCCLCommunicator& comm) override;
    void notify_lnf_w(cudaStream_t stream, NCCLCommunicator& comm) override;
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) override;

    TensorShard& get_embeddings_shard(cudaStream_t stream) override;
    TensorShard& get_lmhead_shard(cudaStream_t stream) override;
    TensorShard& get_lnf_w_shard(cudaStream_t stream) override;
    sLLamaBlockWeights<TensorShard>& get_block_shard(int layer_idx, cudaStream_t stream) override;
private:
    virtual void sr_accumulate_layer(int layer_idx,
                                     sLLamaBlockWeights<Tensor>& dw,
                                     sLLamaBlockWeights<TensorShard> sw,
                                     cudaStream_t stream,
                                     NCCLCommunicator& comm) = 0;

    sLLamaNonBlockWeights<Tensor> mFullNonBlock;
    sLLamaNonBlockWeights<TensorShard> mNonBlockShards;

    std::array<sLLamaBlockWeights<Tensor>, 2> mGradBuffers;
    struct sBlockState {
        cudaEvent_t Event;
        int LayerIdx = -1;
        bool NeedsAccumulation = false;
    };
    std::array<sBlockState, 2> mGradStates;
    std::vector<sLLamaBlockWeights<TensorShard>> mGradShards;
    cudaEvent_t mNonBlockEvent;
};

LLamaGradientsBlockShardedBase::LLamaGradientsBlockShardedBase(std::uint64_t seed, int step, const LLamaConfig& config, const LLamaOptions& options, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc):
    LLamaGradsManager(seed, step),
    mFullNonBlock( allocate_non_block_full(config, config.DType, EAllocationType::ON_DEVICE, *alloc))
{
    mGradBuffers[0] = allocate_block_full(config, config.DType, config.DType, EAllocationType::ON_DEVICE, *alloc);
    mGradBuffers[1] = allocate_block_full(config, config.DType, config.DType, EAllocationType::ON_DEVICE, *alloc);
    mGradStates[0].Event = create_named_event("grad_event_0");
    mGradStates[1].Event = create_named_event("grad_event_1");
    mNonBlockEvent = create_named_event("grad_nonblock_event");
    mGradShards.reserve(config.NumLayers);
    for(int i = 0; i < config.NumLayers; ++i) {
        EAllocationType kind = options.OffloadGrads ? EAllocationType::PINNED : EAllocationType::ON_DEVICE;
        mGradShards.push_back(allocate_block_shard(config, config.DType, config.DType, kind, rank, world, *alloc));
    }

    mNonBlockShards = shard_non_block(mFullNonBlock, rank, world);
}

void LLamaGradientsBlockShardedBase::on_first_micro_step(cudaStream_t stream) {
    // if we have untied embeddings, we need to zero them out
    if(mFullNonBlock.Embeddings.Data != mFullNonBlock.LMHead.Data) {
        fill_zero(mFullNonBlock.Embeddings, stream);
    }
    fill_zero(mFullNonBlock.LNF_w, stream);
}

void LLamaGradientsBlockShardedBase::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
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
    if (mIsLastMicroStep)
        CUDA_CHECK(cudaStreamWaitEvent(stream, mNonBlockEvent, 0));
}

Tensor& LLamaGradientsBlockShardedBase::get_embeddings_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullNonBlock.Embeddings;
}

Tensor& LLamaGradientsBlockShardedBase::get_lmhead_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullNonBlock.LMHead;
}

Tensor& LLamaGradientsBlockShardedBase::get_lnf_w_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullNonBlock.LNF_w;
}

sLLamaBlockWeights<Tensor>& LLamaGradientsBlockShardedBase::get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
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
    fill_zero(dw.LN1_w, stream);
    fill_zero(dw.LN2_w, stream);
    if (dw.Attn_QKV_b.has_value()) {
        fill_zero(dw.Attn_QKV_b.value(), stream);
    }
    return dw;
}

void LLamaGradientsBlockShardedBase::notify_embeddings(cudaStream_t stream, NCCLCommunicator& comm) {
    if(!mIsLastMicroStep) return;
    scatter_reduce(mFullNonBlock.Embeddings, stream, mNonBlockEvent, comm);
}

void LLamaGradientsBlockShardedBase::notify_lmhead(cudaStream_t stream, NCCLCommunicator& comm) {
    if(!mIsLastMicroStep) return;
    if(mFullNonBlock.LMHead.Data == mFullNonBlock.Embeddings.Data) return;    // sync lmhead with embeddings
    NvtxRange r{"notify_lmhead"};
    scatter_reduce(mFullNonBlock.LMHead, stream, mNonBlockEvent, comm);
}

void LLamaGradientsBlockShardedBase::notify_lnf_w(cudaStream_t stream, NCCLCommunicator& comm) {
    if(!mIsLastMicroStep) return;
    scatter_reduce(mFullNonBlock.LNF_w, stream, mNonBlockEvent, comm);
}

void LLamaGradientsBlockShardedBase::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    auto& state = mGradStates[layer_idx % 2];
    if(state.LayerIdx != layer_idx) {
        throw std::logic_error("notify_block called with wrong layer index");
    }
    if (state.NeedsAccumulation) {
        throw std::logic_error("notify_block called before accumulation has finished");
    }

    auto& dw = mGradBuffers.at(layer_idx % 2);
    scatter_reduce(layer_idx, dw, stream, state.Event, comm);
    state.NeedsAccumulation = true;
}

TensorShard& LLamaGradientsBlockShardedBase::get_embeddings_shard(cudaStream_t stream) {
    return mNonBlockShards.Embeddings;
}

TensorShard& LLamaGradientsBlockShardedBase::get_lmhead_shard(cudaStream_t stream) {
    return mNonBlockShards.LMHead;
}

TensorShard& LLamaGradientsBlockShardedBase::get_lnf_w_shard(cudaStream_t stream) {
    return mNonBlockShards.LNF_w;
}

sLLamaBlockWeights<TensorShard>& LLamaGradientsBlockShardedBase::get_block_shard(int layer_idx, cudaStream_t stream) {
    return mGradShards.at(layer_idx);
}

// ---------------------------------------------------------------------------------------------------------------------

class LLamaGradientsBlockSharded_ScatterReduce : public LLamaGradientsBlockShardedBase {
public:
    using LLamaGradientsBlockShardedBase::LLamaGradientsBlockShardedBase;
private:
    void sr_accumulate_tensor(TensorShard& dst, Tensor& src, cudaStream_t stream, unsigned seed);
    void sr_accumulate_layer(int layer_idx,
                             sLLamaBlockWeights<Tensor>& dw,
                             sLLamaBlockWeights<TensorShard> sw,
                             cudaStream_t stream,
                             NCCLCommunicator& comm) override;
};

void LLamaGradientsBlockSharded_ScatterReduce::sr_accumulate_tensor(TensorShard& dst, Tensor& src, cudaStream_t stream, unsigned seed) {
    Tensor local_slice = shard_view(src, dst.ShardIndex, dst.NumShards);
    if(mIsFirstMicroStep) {
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, local_slice.Data, local_slice.bytes(), cudaMemcpyDeviceToDevice, stream));
    } else {
        vector_add_sr(dst, dst, local_slice, 1.f, local_slice.nelem(), seed, stream);
    }
}

void LLamaGradientsBlockSharded_ScatterReduce::sr_accumulate_layer(int layer_idx,
                                                                   sLLamaBlockWeights<Tensor>& dw,
                                                                   sLLamaBlockWeights<TensorShard> sw,
                                                                   cudaStream_t stream,
                                                                   NCCLCommunicator& comm) {
    NvtxRange range("accumulate_layer", layer_idx);
    auto rng_1 = mRng.generate(2*mStepCounter + 0, layer_idx);
    auto rng_2 = mRng.generate(2*mStepCounter + 1, layer_idx);

    sr_accumulate_tensor(sw.LN1_w, dw.LN1_w, stream, rng_1[0]);
    sr_accumulate_tensor(sw.LN2_w, dw.LN2_w, stream, rng_1[1]);
    sr_accumulate_tensor(sw.MLP_Up_w, dw.MLP_Up_w, stream, rng_1[2]);
    sr_accumulate_tensor(sw.MLP_Down_w, dw.MLP_Down_w, stream, rng_1[3]);
    sr_accumulate_tensor(sw.Attn_QKV_w, dw.Attn_QKV_w, stream, rng_2[0]);
    sr_accumulate_tensor(sw.Attn_Out_w, dw.Attn_Out_w, stream, rng_2[1]);
    if(sw.Attn_QKV_b.has_value()) {
        sr_accumulate_tensor(sw.Attn_QKV_b.value(), dw.Attn_QKV_b.value(), stream, rng_2[2]);
    }
}

// ---------------------------------------------------------------------------------------------------------------------

class LLamaGradientsBlockSharded_AllToAll : public LLamaGradientsBlockShardedBase {
public:
    using LLamaGradientsBlockShardedBase::LLamaGradientsBlockShardedBase;
private:
    void scatter_reduce(int layer_idx, sLLamaBlockWeights<Tensor>& block, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) override;
    void sr_accumulate_tensor(TensorShard& dst, Tensor& src, cudaStream_t stream, bool first, float scale, int shard, unsigned seed);
    void sr_accumulate_layer(int layer_idx,
                             sLLamaBlockWeights<Tensor>& dw,
                             sLLamaBlockWeights<TensorShard> sw,
                             cudaStream_t stream,
                             NCCLCommunicator& comm) override;
};

void LLamaGradientsBlockSharded_AllToAll::scatter_reduce(int layer_idx, sLLamaBlockWeights<Tensor>& dw, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) {
    auto& sw = get_block_shard(layer_idx, stream);
    int rank = sw.LN1_w.ShardIndex;

    // accumulate local slice of block to local gradient
    {
        NvtxRange range("accumulate-own-shard", layer_idx);
        auto rng_1 = mRng.generate(2 * mStepCounter + 0, layer_idx);
        auto rng_2 = mRng.generate(2 * mStepCounter + 1, layer_idx);
        sr_accumulate_tensor(sw.LN1_w, dw.LN1_w, stream, mIsFirstMicroStep, 1.f, rank, rng_1[0]);
        sr_accumulate_tensor(sw.LN2_w, dw.LN2_w, stream, mIsFirstMicroStep, 1.f, rank, rng_1[1]);
        sr_accumulate_tensor(sw.Attn_QKV_w, dw.Attn_QKV_w, stream, mIsFirstMicroStep, 1.f, rank, rng_1[2]);
        sr_accumulate_tensor(sw.Attn_Out_w, dw.Attn_Out_w, stream, mIsFirstMicroStep, 1.f, rank, rng_1[3]);
        sr_accumulate_tensor(sw.MLP_Up_w, dw.MLP_Up_w, stream, mIsFirstMicroStep, 1.f, rank, rng_2[0]);
        sr_accumulate_tensor(sw.MLP_Down_w, dw.MLP_Down_w, stream, mIsFirstMicroStep, 1.f, rank, rng_2[1]);
        if (sw.Attn_QKV_b.has_value()) {
            sr_accumulate_tensor(sw.Attn_QKV_b.value(), dw.Attn_QKV_b.value(), stream, mIsFirstMicroStep, 1.f, rank,
                                 rng_2[2]);
        }
    }

    // make sure we've done the local accumulation before we allow communication to begin.

    CUDA_CHECK(cudaEventRecord(signal, stream));
    NvtxRange range("all-to-all-gradients", layer_idx);

    comm.begin_transaction(signal);
    comm.schedule_destructive_all_to_all(dw.LN1_w);
    comm.schedule_destructive_all_to_all(dw.Attn_QKV_w);
    comm.schedule_destructive_all_to_all(dw.Attn_Out_w);
    comm.schedule_destructive_all_to_all(dw.LN2_w);
    comm.schedule_destructive_all_to_all(dw.MLP_Up_w);
    comm.schedule_destructive_all_to_all(dw.MLP_Down_w);
    if(dw.Attn_QKV_b.has_value()) {
        comm.schedule_destructive_all_to_all(dw.Attn_QKV_b.value());
    }
    comm.execute_transaction(signal);
}

void LLamaGradientsBlockSharded_AllToAll::sr_accumulate_tensor(TensorShard& dst, Tensor& src, cudaStream_t stream, bool first, float scale, int shard, unsigned seed) {
    Tensor local_slice = shard_view(src, shard, dst.NumShards);
    if(first) {
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, local_slice.Data, local_slice.bytes(), cudaMemcpyDeviceToDevice, stream));
    } else {
        vector_add_sr(dst, dst, local_slice, scale, local_slice.nelem(), seed, stream);
    }
}

void LLamaGradientsBlockSharded_AllToAll::sr_accumulate_layer(int layer_idx,
                                                 sLLamaBlockWeights<Tensor>& dw,
                                                 sLLamaBlockWeights<TensorShard> sw,
                                                 cudaStream_t stream,
                                                 NCCLCommunicator& comm) {
    NvtxRange range("accumulate_layer", layer_idx);

    int rank = comm.rank();
    int world = comm.world_size();
    float scale = 1.f;
    if (mIsLastMicroStep) {
        scale = 1.f / world;
    }

    auto rng_1 = mRng.generate(2*mStepCounter + 0, layer_idx);
    auto rng_2 = mRng.generate(2*mStepCounter + 1, layer_idx + 12345);

    vector_reduce_sr(sw.LN1_w, dw.LN1_w, scale, world, (rank + world - 1) % world, sw.LN1_w.nelem(), true, rng_1[0], stream);
    vector_reduce_sr(sw.LN2_w, dw.LN2_w, scale, world, (rank + world - 1) % world, sw.LN2_w.nelem(), true, rng_1[1], stream);
    vector_reduce_sr(sw.MLP_Up_w, dw.MLP_Up_w, scale, world, (rank + world - 1) % world, sw.MLP_Up_w.nelem(), true, rng_1[2], stream);
    vector_reduce_sr(sw.MLP_Down_w, dw.MLP_Down_w, scale, world, (rank + world - 1) % world, sw.MLP_Down_w.nelem(), true, rng_1[3], stream);
    vector_reduce_sr(sw.Attn_QKV_w, dw.Attn_QKV_w, scale, world, (rank + world - 1) % world, sw.Attn_QKV_w.nelem(), true, rng_2[0], stream);
    vector_reduce_sr(sw.Attn_Out_w, dw.Attn_Out_w, scale, world, (rank + world - 1) % world, sw.Attn_Out_w.nelem(), true, rng_2[1], stream);
    if(sw.Attn_QKV_b.has_value()) {
        vector_reduce_sr(sw.Attn_QKV_b.value(), dw.Attn_QKV_b.value(), scale, world, (rank + world - 1) % world, sw.Attn_QKV_b->nelem(), true, rng_2[2], stream);
    }
}

std::unique_ptr<LLamaGradsManager> LLamaGradsManager::create(std::uint64_t seed, int step, const LLamaConfig& config, const LLamaOptions& options, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc) {
    if (options.ShardGradients) {
        if(options.UseAllToAllReduce) {
            return std::make_unique<LLamaGradientsBlockSharded_AllToAll>(seed, step, config, options, rank, world, alloc);
        } else {
            return std::make_unique<LLamaGradientsBlockSharded_ScatterReduce>(seed, step, config, options, rank, world, alloc);
        }

    } else {
        if(options.OffloadGrads) {
            throw std::logic_error("Offloading gradients is not supported for unsharded gradients");
        }
        return std::make_unique<LLamaGradientsUnsharded>(seed, step, config, rank, world, alloc);
    }
}
