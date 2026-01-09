// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#include "llama_optimizer.h"
#include "training/transformer_config.h"
#include "llama_model.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"
#include "utilities/stack.h"
#include "utilities/lazy_allocator.h"

SimpleTensorContainer& LLamaOptimizerStateManager::get_block_m(int layer_idx, cudaStream_t stream) {
    if(!mOffloadM || mUseZeroCopy) return mOptM.Blocks[layer_idx];
    return get_block_from(layer_idx, stream, mOptMBuffer.at(layer_idx % 2));
}

SimpleTensorContainer& LLamaOptimizerStateManager::get_block_v(int layer_idx, cudaStream_t stream) {
    if(!mOffloadV || mUseZeroCopy) return mOptV.Blocks[layer_idx];
    return get_block_from(layer_idx, stream, mOptVBuffer.at(layer_idx % 2));
}

SimpleTensorContainer& LLamaOptimizerStateManager::non_block_m() {
    return mOptM.NonBlocks;
}

SimpleTensorContainer& LLamaOptimizerStateManager::non_block_v() {
    return mOptV.NonBlocks;
}

void zero_opt_non_block(sLLamaWeights& weights, cudaStream_t stream) {
    // here's the first disadvantage of having individual buffers: We need to make a ton of memset calls
    fill_zero(weights.NonBlocks.Embeddings, stream);
    fill_zero(weights.NonBlocks.LNF_w, stream);
    if(weights.NonBlocks.LMHead.Data != weights.NonBlocks.Embeddings.Data) {
        fill_zero(weights.NonBlocks.LMHead, stream);
    }
}

sLLamaWeights allocate_scales(TransformerConfig config, int shard_idx, int num_shards, TensorAllocator& alloc) {
    long C = config.HiddenSize;
    long V = config.VocabSize;
    long H = config.IntermediateSize;
    long head_size = C / config.NumQueryHeads;
    long attn_intermediate_size = (config.NumQueryHeads + 2 * config.NumKeyValHeads) * head_size;

    sLLamaWeights result;
    result.Blocks.resize(config.NumLayers);
    for(auto& block : result.Blocks) {
        block.Attn_QKV_w = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "att_qkv_w", {div_exact(attn_intermediate_size * C, 128l)}, EAllocationType::ON_DEVICE);
        block.Attn_Out_w = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "attproj_w", {div_exact(C * C, 128l)}, EAllocationType::ON_DEVICE);
        block.MLP_Up_w = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "mlp_up_w", {div_exact(2 * H * C, 128l)}, EAllocationType::ON_DEVICE);
        block.MLP_Down_w = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "mlp_down_w", {div_exact(C * H, 128l)}, EAllocationType::ON_DEVICE);

        block.LN1_w = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "ln1_w", {div_exact(C, 128l)}, EAllocationType::ON_DEVICE);
        block.LN2_w = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "ln2_w", {div_exact(C, 128l)}, EAllocationType::ON_DEVICE);
        if(config.UseQKVBias) {
            block.Attn_QKV_b = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "att_qkv_b", {div_exact(attn_intermediate_size, 128l)}, EAllocationType::ON_DEVICE);
        } else {
            block.Attn_QKV_b = Tensor{};
        }

        visit([](Tensor& t){
            fill_constant(t, 1.f, t.nelem(), nullptr);
        }, block);
    }
    result.NonBlocks.Embeddings = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "embeddings", {div_exact(V * C, 128l)}, EAllocationType::ON_DEVICE);
    result.NonBlocks.LNF_w      = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards,"lnf_w", {div_exact(C, 128l)}, EAllocationType::ON_DEVICE);
    fill_constant(result.NonBlocks.Embeddings, 1.f, result.NonBlocks.Embeddings.nelem(), nullptr);
    fill_constant(result.NonBlocks.LNF_w, 1.f, result.NonBlocks.LNF_w.nelem(), nullptr);
    if(config.TiedWordEmbeddings) {
        result.NonBlocks.LMHead = result.NonBlocks.Embeddings;
    } else {
        result.NonBlocks.LMHead = alloc.allocate_shard(ETensorDType::FP32, shard_idx, num_shards, "lmhead", {div_exact(V * C, 128l)}, EAllocationType::ON_DEVICE);
        fill_constant(result.NonBlocks.LMHead, 1.f, result.NonBlocks.LMHead.nelem(), nullptr);
    }
    return result;
}

std::vector<Tensor> allocate_weights_opt(sLLamaWeights& weights, const TransformerConfig& config, ETensorDType dtype, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc) {
    std::vector<Tensor> result;
    weights.Blocks.resize(config.NumLayers);
    LazyAllocator alloc_lazy;
    for(auto& block : weights.Blocks) {
        fill_matrix_shapes(block, config, dtype, shard_idx, num_shards);
        fill_non_matrix_shapes(block, config, dtype, shard_idx, num_shards);
        alloc_lazy.allocate(block);
        result.push_back(alloc_lazy.commit(alloc, kind, "block_shard"));
    }
    weights.NonBlocks = allocate_non_block_shard(config, dtype, kind, shard_idx, num_shards, alloc);
    return result;
}


LLamaOptimizerStateManager::LLamaOptimizerStateManager(TransformerConfig cfg, IModel& model, LLamaOptions options, cudaStream_t stream, NCCLCommunicator& comm, TensorAllocator& alloc):
        AdamWStateManager(cfg, model, options.OffloadOptM, options.OffloadOptV, options.OptMomentumType, options.OptVarianceType, options.UseZeroCopy, comm.rank(), comm.world_size())
{
    {
        auto ctx = alloc.with_context("Adam M");
        EAllocationType alloc_type = options.OffloadOptM ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        mMBlockStorage = allocate_weights_opt(mOptM, cfg, mMType, alloc_type, comm.rank(), comm.world_size(), alloc);
        for(auto& block : mMBlockStorage) {
            fill_zero(block, stream);
        }
        zero_opt_non_block(mOptM, stream);

        if(mMType == ETensorDType::FP8_E4M3) {
            mOptMScales = allocate_scales(cfg, comm.rank(), comm.world_size(), alloc);
        } else {
            mOptMScales.Blocks.resize(cfg.NumLayers);
        }
    }

    {
        auto ctx = alloc.with_context("Adam V");
        EAllocationType alloc_type = options.OffloadOptV ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        mVBlockStorage = allocate_weights_opt(mOptV, cfg, mVType, alloc_type, comm.rank(), comm.world_size(), alloc);
        for(auto& block : mVBlockStorage) {
            fill_zero(block, stream);
        }
        zero_opt_non_block(mOptV, stream);
    }
}

SimpleTensorContainer& LLamaOptimizerStateManager::get_block_scales_m(int layer_idx) {
    return mOptMScales.Blocks.at(layer_idx);
}
