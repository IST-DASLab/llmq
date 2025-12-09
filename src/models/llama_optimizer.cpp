// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#include "llama_optimizer.h"
#include "llama_config.h"
#include "llama_model.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"
#include "utilities/stack.h"
#include "utilities/lazy_allocator.h"

void LLamaOptimizerStateManager::fetch_block(int layer_idx, cudaStream_t fetch_stream) {
    if((!mOffloadM && !mOffloadV) || mUseZeroCopy) return;

    NvtxRange range("fetch_opt_block", layer_idx);
    int buffer = layer_idx % 2;
    auto& stat = mStatus.at(buffer);
    stat.LayerIdx = layer_idx;
    stat.Fetch = true;

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, stat.DoneEvent, 0));

    auto fetch = [fetch_stream, &stat](TensorShard& dst, TensorShard& src) {
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyHostToDevice, fetch_stream));
        dst.Stats = src.Stats;
    };

    auto fetch_all = [&](sLLamaBlockWeights<TensorShard>& buf, sLLamaBlockWeights<TensorShard>& ref){
        fetch(buf.LN1_w, ref.LN1_w);
        fetch(buf.LN2_w, ref.LN2_w);
        fetch(buf.Attn_QKV_w, ref.Attn_QKV_w);
        fetch(buf.Attn_Out_w, ref.Attn_Out_w);
        fetch(buf.MLP_Up_w, ref.MLP_Up_w);
        fetch(buf.MLP_Down_w, ref.MLP_Down_w);
        if (ref.Attn_QKV_b.has_value()) {
            fetch(buf.Attn_QKV_b.value(), ref.Attn_QKV_b.value());
        }
    };

    if(mOffloadM) {
        auto& buf = mOptMBuffer.at(buffer);
        auto& ref = mOptM.Blocks[layer_idx];

        fetch_all(buf, ref);
    }

    if(mOffloadV) {
        auto& buf = mOptVBuffer.at(buffer);
        auto& ref = mOptV.Blocks[layer_idx];

        fetch_all(buf, ref);
    }

    CUDA_CHECK(cudaEventRecord(stat.DoneEvent, fetch_stream));
}

void LLamaOptimizerStateManager::begin_optimizer(DeviceMemoryStack& memory) {
    LazyAllocator alloc;
    if(mOffloadM &&! mUseZeroCopy) {
        ETensorDType m_type = mOptM.Blocks[0].LN1_w.DType;
        matrix_params_lazy(mOptMBuffer[0], mConfig, m_type, mRank, mWorld, alloc);
        non_matrix_params_lazy(mOptMBuffer[0], mConfig, m_type, mRank, mWorld, alloc);
        mOptMBufferStorage[0] = alloc.commit(memory, "opt_m_a");
        matrix_params_lazy(mOptMBuffer[1], mConfig, m_type, mRank, mWorld, alloc);
        non_matrix_params_lazy(mOptMBuffer[1], mConfig, m_type, mRank, mWorld, alloc);
        mOptMBufferStorage[1] = alloc.commit(memory, "opt_m_b");
    }

    if(mOffloadV &&! mUseZeroCopy) {
        ETensorDType v_type = mOptV.Blocks[0].LN1_w.DType;
        matrix_params_lazy(mOptVBuffer[0], mConfig, v_type, mRank, mWorld, alloc);
        non_matrix_params_lazy(mOptVBuffer[0], mConfig, v_type, mRank, mWorld, alloc);
        mOptVBufferStorage[0] = alloc.commit(memory, "opt_v_a");
        matrix_params_lazy(mOptVBuffer[1], mConfig, v_type, mRank, mWorld, alloc);
        non_matrix_params_lazy(mOptVBuffer[1], mConfig, v_type, mRank, mWorld, alloc);
        mOptVBufferStorage[1] = alloc.commit(memory, "opt_v_b");

    }
}
void LLamaOptimizerStateManager::end_optimizer(DeviceMemoryStack& memory) {
    if(mOffloadV &&! mUseZeroCopy) {
        memory.free(mOptVBufferStorage[1]);
        memory.free(mOptVBufferStorage[0]);
    }

    if(mOffloadM &&! mUseZeroCopy) {
        memory.free(mOptMBufferStorage[1]);
        memory.free(mOptMBufferStorage[0]);
    }
}

sLLamaBlockWeights<TensorShard>& LLamaOptimizerStateManager::get_block_m(int layer_idx, cudaStream_t stream) {
    if(!mOffloadM || mUseZeroCopy) return mOptM.Blocks[layer_idx];
    return get_block_from(layer_idx, stream, mOptMBuffer.at(layer_idx % 2));
}

sLLamaBlockWeights<TensorShard>& LLamaOptimizerStateManager::get_block_v(int layer_idx, cudaStream_t stream) {
    if(!mOffloadV || mUseZeroCopy) return mOptV.Blocks[layer_idx];
    return get_block_from(layer_idx, stream, mOptVBuffer.at(layer_idx % 2));
}

sLLamaBlockWeights<TensorShard>& LLamaOptimizerStateManager::get_block_from(int layer_idx, cudaStream_t stream, sLLamaBlockWeights<TensorShard>& buf) {
    int buffer = layer_idx % 2;
    auto& stat = mStatus.at(buffer);

    if(stat.LayerIdx != layer_idx) {
        throw std::logic_error("Layer index mismatch in get_block_from");
    }

    stat.Done = false;
    // if we needed to fetch, we need to wait
    if(stat.Fetch) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, stat.DoneEvent, 0));
    }
    stat.Fetch = false;

    return buf;
}

void LLamaOptimizerStateManager::store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream)  {
    if (mUseZeroCopy) return;

    NvtxRange range("store_opt_block", layer_idx);
    int buffer = layer_idx % 2;
    if(mOffloadM) {
        store_one_block(layer_idx, stream, put_stream, mOptMBuffer[buffer], mOptM.Blocks[layer_idx]);
    }

    if(mOffloadV) {
        store_one_block(layer_idx, stream, put_stream, mOptVBuffer[buffer], mOptV.Blocks[layer_idx]);
    }

    if(mOffloadM || mOffloadV) {
        auto& stat = mStatus.at(layer_idx % 2);
        if(stat.LayerIdx != layer_idx) {
            throw std::logic_error("layer index mismatch in store_block");
        }
        CUDA_CHECK(cudaEventRecord(stat.DoneEvent, put_stream));
        stat.Done = true;
    }
}

void LLamaOptimizerStateManager::store_one_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream, sLLamaBlockWeights<TensorShard>& buf, sLLamaBlockWeights<TensorShard>& dst) {
    auto& stat = mStatus.at(layer_idx % 2);

    auto send = [put_stream](TensorShard& dst, TensorShard& src) {
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDeviceToHost, put_stream));
    };

    // put stream can start as soon as the new master weights are ready
    CUDA_CHECK(cudaEventRecord(stat.DoneEvent, stream));
    CUDA_CHECK(cudaStreamWaitEvent(put_stream, stat.DoneEvent, 0));

    CUDA_CHECK(cudaEventRecord(stat.DoneEvent, stream));

    send(dst.LN1_w, buf.LN1_w);
    send(dst.LN2_w, buf.LN2_w);
    send(dst.Attn_QKV_w, buf.Attn_QKV_w);
    send(dst.Attn_Out_w, buf.Attn_Out_w);
    send(dst.MLP_Up_w, buf.MLP_Up_w);
    send(dst.MLP_Down_w, buf.MLP_Down_w);
    if (dst.Attn_QKV_b.has_value()) {
        send(dst.Attn_QKV_b.value(), buf.Attn_QKV_b.value());
    }
}

sLLamaNonBlockWeights<TensorShard>& LLamaOptimizerStateManager::non_block_m() {
    return mOptM.NonBlocks;
}

sLLamaNonBlockWeights<TensorShard>& LLamaOptimizerStateManager::non_block_v() {
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

sLLamaWeights allocate_scales(LLamaConfig config, int shard_idx, int num_shards, TensorAllocator& alloc) {
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
            fill_constant(block.Attn_QKV_b.value(), 1.f, block.Attn_QKV_b.value().nelem(), nullptr);
        } else {
            block.Attn_QKV_b = std::nullopt;
        }

        fill_constant(block.Attn_QKV_w, 1.f, block.Attn_QKV_w.nelem(), nullptr);
        fill_constant(block.Attn_Out_w, 1.f, block.Attn_Out_w.nelem(), nullptr);
        fill_constant(block.MLP_Up_w, 1.f, block.MLP_Up_w.nelem(), nullptr);
        fill_constant(block.MLP_Down_w, 1.f, block.MLP_Down_w.nelem(), nullptr);
        fill_constant(block.LN1_w, 1.f, block.LN1_w.nelem(), nullptr);
        fill_constant(block.LN2_w, 1.f, block.LN2_w.nelem(), nullptr);
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

std::vector<Tensor> allocate_weights_opt(sLLamaWeights& weights, const LLamaConfig& config, ETensorDType dtype, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc) {
    std::vector<Tensor> result;
    weights.Blocks.resize(config.NumLayers);
    LazyAllocator alloc_lazy;
    for(auto& block : weights.Blocks) {
        matrix_params_lazy(block, config, dtype, shard_idx, num_shards, alloc_lazy);
        non_matrix_params_lazy(block, config, dtype, shard_idx, num_shards, alloc_lazy);
        result.push_back(alloc_lazy.commit(alloc, kind, "block_shard"));
    }
    weights.NonBlocks = allocate_non_block_shard(config, dtype, kind, shard_idx, num_shards, alloc);
    return result;
}


LLamaOptimizerStateManager::LLamaOptimizerStateManager(LLamaConfig cfg, LLamaOptions options, cudaStream_t stream, NCCLCommunicator& comm, TensorAllocator& alloc):
    mOffloadM(options.OffloadOptM), mOffloadV(options.OffloadOptV), mUseZeroCopy(options.UseZeroCopy), mConfig(cfg), mRank(comm.rank()), mWorld(comm.world_size())
{
    {
        auto ctx = alloc.with_context("Adam M");
        EAllocationType alloc_type = options.OffloadOptM ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        mOptMBlockStorage = allocate_weights_opt(mOptM, cfg, options.OptMomentumType, alloc_type, comm.rank(), comm.world_size(), alloc);
        for(auto& block : mOptMBlockStorage) {
            fill_zero(block, stream);
        }
        zero_opt_non_block(mOptM, stream);

        if(options.OptMomentumType == ETensorDType::FP8_E4M3) {
            mOptMScales = allocate_scales(cfg, comm.rank(), comm.world_size(), alloc);
        } else {
            mOptMScales.Blocks.resize(cfg.NumLayers);
        }
    }

    {
        auto ctx = alloc.with_context("Adam V");
        EAllocationType alloc_type = options.OffloadOptV ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        mOptVBlockStorage = allocate_weights_opt(mOptV, cfg, options.OptVarianceType, alloc_type, comm.rank(), comm.world_size(), alloc);
        for(auto& block : mOptVBlockStorage) {
            fill_zero(block, stream);
        }
        zero_opt_non_block(mOptV, stream);
    }

    if((options.OffloadOptM || options.OffloadOptV) && !mUseZeroCopy) {
        mStatus[0] = sBufferStatus{-1, create_named_event("opt_fetch_0"), false, true};
        mStatus[1] = sBufferStatus{-1, create_named_event("opt_fetch_1"), false, true};
    }
}
