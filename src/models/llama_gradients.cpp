// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "llama_gradients.h"

#include "kernels/kernels.h"
#include "llama_model.h"
#include "utilities/comm.h"

void LLamaGradientsUnsharded::on_first_micro_step(cudaStream_t stream) {
    using namespace LLamaWeightID;
    fill_zero(mNonBlockGradients.get_tensor(LNF_W), stream);
    if(mNonBlockGradients.get_tensor(LM_HEAD).Data != nullptr) {
        fill_zero(mNonBlockGradients.get_tensor(EMBEDDING), stream);
    } else {
        // embedding backward comes after LMHead backward; and LMHead backward *sets* the gradient
        // on the first backward call, so no need to zero anything.
    }
    for(auto& layer: mBlockGradients) {
        fill_zero(layer.get_tensor(LN1_W), stream);
        fill_zero(layer.get_tensor(LN2_W), stream);
        fill_zero(layer.get_tensor(QKV_B), stream);
        // no need to zero out the matrix weights, we'll just overwrite them on the first
        // grad accumulation step
    }
}

// ---------------------------------------------------------------------------------------------------------------------

// shard the transformer blocks, but not the embeddings and lmhead.

void LLamaGradientsBlockShardedBase::on_first_micro_step(cudaStream_t stream) {
    // if we have untied embeddings, we need to zero them out, same as above
    if(mFullNonBlock.get_tensor(LLamaWeightID::LM_HEAD).Data != nullptr) {
        fill_zero(mFullNonBlock.get_tensor(LLamaWeightID::EMBEDDING), stream);
    }
    fill_zero(mFullNonBlock.get_tensor(LLamaWeightID::LNF_W), stream);
}

void LLamaGradientsBlockShardedBase::on_get_block(SimpleTensorContainer& block, cudaStream_t stream) {
    fill_zero(block.get_tensor(LLamaWeightID::LN1_W), stream);
    fill_zero(block.get_tensor(LLamaWeightID::LN2_W), stream);
    fill_zero(block.get_tensor(LLamaWeightID::QKV_B), stream);
}


// ---------------------------------------------------------------------------------------------------------------------

void LLamaGradientsBlockSharded_ScatterReduce::on_notify_block(int layer_idx, SimpleTensorContainer& block,
    cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) {
    comm.reduce_scatter(block, stream, signal);
}

void LLamaGradientsBlockSharded_ScatterReduce::sr_accumulate_layer(int layer_idx,
                                                                   SimpleTensorContainer& dw,
                                                                   SimpleTensorContainer& sw,
                                                                   cudaStream_t stream,
                                                                   NCCLCommunicator& comm) {
    NvtxRange range("accumulate_layer", layer_idx);
    std::array<std::uint32_t, 8> rng;
    generate_rng(std::span(rng), 2*get_step_counter(), layer_idx);

    int rank = comm.rank();
    int world = comm.world_size();

    int i = 0;
    visit([&](Tensor& dst, Tensor& src){
        Tensor local_slice = shard_view(src, rank, world);
        if(is_first_micro_step()) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, local_slice.Data, local_slice.bytes(), cudaMemcpyDeviceToDevice, stream));
        } else {
            vector_add_sr(dst, dst, local_slice, 1.f, local_slice.nelem(), rng.at(i), stream);
        }
        ++i;
    }, sw, dw);
}

// ---------------------------------------------------------------------------------------------------------------------

void LLamaGradientsBlockSharded_AllToAll::on_notify_block(int layer_idx, SimpleTensorContainer& dw, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm) {
    auto& sw = get_block_shard(layer_idx, stream);
    int rank = comm.rank();
    int world = comm.world_size();

    // accumulate local slice of block to local gradient
    {
        NvtxRange range("accumulate-own-shard", layer_idx);
        std::array<std::uint32_t, 8> rng;
        generate_rng(std::span(rng), 2*get_step_counter(), layer_idx);
        int i = 0;
        visit([&](Tensor& dst, Tensor& src){
            Tensor local_slice = shard_view(src, rank, world);
            if(is_first_micro_step()) {
                CUDA_CHECK(cudaMemcpyAsync(dst.Data, local_slice.Data, local_slice.bytes(), cudaMemcpyDeviceToDevice, stream));
            } else {
                vector_add_sr(dst, dst, local_slice, 1.f, local_slice.nelem(), rng.at(i), stream);
            }
            ++i;
        }, sw, dw);
    }

    // make sure we've done the local accumulation before we allow communication to begin.
    CUDA_CHECK(cudaEventRecord(signal, stream));
    NvtxRange range("all-to-all-gradients", layer_idx);

    comm.begin_transaction(signal);
    visit([&](Tensor& t){ comm.schedule_destructive_all_to_all(t); }, dw);
    comm.execute_transaction(signal);
}

void LLamaGradientsBlockSharded_AllToAll::sr_accumulate_layer(int layer_idx,
                                                              SimpleTensorContainer& dw,
                                                              SimpleTensorContainer& sw,
                                                              cudaStream_t stream,
                                                              NCCLCommunicator& comm) {
    NvtxRange range("accumulate_layer", layer_idx);

    int rank = comm.rank();
    int world = comm.world_size();
    float scale = 1.f;
    if (is_last_micro_step()) {
        scale = 1.f / world;
    }

    std::array<std::uint32_t, 8> rng;
    generate_rng(std::span(rng), 2*get_step_counter(), layer_idx);

    int i = 0;
    visit([&](Tensor& s, Tensor& d){
        vector_reduce_sr(s, d, scale, world, (rank + world - 1) % world, s.nelem(), true, rng.at(i), stream);
        ++i;
    }, sw, dw);
}

std::unique_ptr<IGradientManager> create_grads_manager(std::uint64_t seed, int step, LLamaModel& model, const TransformerConfig& config,
    const LLamaOptions& options, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc)
{
    if (options.ShardGradients) {
        if(options.UseAllToAllReduce) {
            return std::make_unique<LLamaGradientsBlockSharded_AllToAll>(config, model, seed, step, rank, world, options.OffloadGrads, alloc);
        } else {
            return std::make_unique<LLamaGradientsBlockSharded_ScatterReduce>(config, model, seed, step, rank, world, options.OffloadGrads, alloc);
        }

    } else {
        if(options.OffloadGrads) {
            throw std::logic_error("Offloading gradients is not supported for unsharded gradients");
        }
        return std::make_unique<LLamaGradientsUnsharded>(config, model, seed, step, rank, world, alloc);
    }
}
