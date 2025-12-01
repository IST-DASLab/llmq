// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#include "llama_optimizer.h"
#include "llama_config.h"
#include "llama_model.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"
#include "utilities/stack.h"

void LLamaOptimizerStateManager::fetch_block(int layer_idx, cudaStream_t fetch_stream) {
    if((!mOffloadM && !mOffloadV) || mUseZeroCopy) return;

    int buffer = layer_idx % 2;
    auto& stat = mStatus.at(buffer);
    stat.LayerIdx = layer_idx;
    stat.Fetch = false;

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, stat.DoneEvent, 0));

    auto fetch = [fetch_stream, &stat](TensorShard& dst, TensorShard& src) {
        // tensors on the same device are handled by pointer assignment
        if(dst.Device == src.Device) {
            dst.Data = src.Data;
            dst.Stats = src.Stats;
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyHostToDevice, fetch_stream));
            dst.Stats = src.Stats;
            stat.Fetch = true;
        }
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

    if(stat.Fetch) {
        CUDA_CHECK(cudaEventRecord(stat.DoneEvent, fetch_stream));
    }
}

void LLamaOptimizerStateManager::begin_optimizer(DeviceMemoryStack& memory) {
    if(mOffloadM &&! mUseZeroCopy) {
        ETensorDType m_type = mOptM.Blocks[0].LN1_w.DType;
        matrix_params_from_stack(mOptMBuffer[0], mConfig, m_type, mRank, mWorld, memory);
        non_matrix_params_from_stack(mOptMBuffer[0], mConfig, m_type, mRank, mWorld, memory);
        matrix_params_from_stack(mOptMBuffer[1], mConfig, m_type, mRank, mWorld, memory);
        non_matrix_params_from_stack(mOptMBuffer[1], mConfig, m_type, mRank, mWorld, memory);
    }

    if(mOffloadV &&! mUseZeroCopy) {
        ETensorDType v_type = mOptV.Blocks[0].LN1_w.DType;
        matrix_params_from_stack(mOptVBuffer[0], mConfig, v_type, mRank, mWorld, memory);
        non_matrix_params_from_stack(mOptVBuffer[0], mConfig, v_type, mRank, mWorld, memory);
        matrix_params_from_stack(mOptVBuffer[1], mConfig, v_type, mRank, mWorld, memory);
        non_matrix_params_from_stack(mOptVBuffer[1], mConfig, v_type, mRank, mWorld, memory);
    }
}
void LLamaOptimizerStateManager::end_optimizer(DeviceMemoryStack& memory) {
    auto free_non_matrix = [&](sLLamaBlockWeights<TensorShard>& buf) {
        if(buf.Attn_QKV_b.has_value()) {
            memory.free(buf.Attn_QKV_b.value());
        }
        memory.free(buf.LN2_w);
        memory.free(buf.LN1_w);
    };

     auto free_matrix = [&](sLLamaBlockWeights<TensorShard>& buf) {
         memory.free(buf.MLP_Down_w);
         memory.free(buf.MLP_Up_w);
         memory.free(buf.Attn_Out_w);
         memory.free(buf.Attn_QKV_w);
    };

    if(mOffloadV &&! mUseZeroCopy) {
        free_non_matrix(mOptVBuffer[1]);
        free_matrix(mOptVBuffer[1]);
        free_non_matrix(mOptVBuffer[0]);
        free_matrix(mOptVBuffer[0]);
    }

    if(mOffloadM &&! mUseZeroCopy) {
        free_non_matrix(mOptMBuffer[1]);
        free_matrix(mOptMBuffer[1]);
        free_non_matrix(mOptMBuffer[0]);
        free_matrix(mOptMBuffer[0]);
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

void zero_opt_state(sLLamaWeights& weights, cudaStream_t stream) {
    // here's the first disadvantage of having individual buffers: We need to make a ton of memset calls
    fill_zero(weights.NonBlocks.Embeddings, stream);
    fill_zero(weights.NonBlocks.LNF_w, stream);
    if(weights.NonBlocks.LMHead.Data != weights.NonBlocks.Embeddings.Data) {
        fill_zero(weights.NonBlocks.LMHead, stream);
    }
    for(auto& layer: weights.Blocks) {
        fill_zero(layer.LN1_w, stream);
        fill_zero(layer.LN2_w, stream);
        fill_zero(layer.Attn_QKV_w, stream);
        if(auto& qkv_b = layer.Attn_QKV_b; qkv_b.has_value()) {
            fill_zero(qkv_b.value(), stream);
        }
        fill_zero(layer.Attn_Out_w, stream);
        fill_zero(layer.MLP_Up_w, stream);
        fill_zero(layer.MLP_Down_w, stream);
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

LLamaOptimizerStateManager::LLamaOptimizerStateManager(LLamaConfig cfg, LLamaOptions options, cudaStream_t stream, NCCLCommunicator& comm, TensorAllocator& alloc):
    mOffloadM(options.OffloadOptM), mOffloadV(options.OffloadOptV), mUseZeroCopy(options.UseZeroCopy), mConfig(cfg), mRank(comm.rank()), mWorld(comm.world_size())
{
    {
        auto ctx = alloc.with_context("Adam M");
        EAllocationType alloc_type = options.OffloadOptM ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        LLamaConfig c = cfg;
        c.DType = options.OptMomentumType;
        mOptM = allocate_weights(c, alloc_type, comm.rank(), comm.world_size(), alloc);
        zero_opt_state(mOptM, stream);

        if(options.OptMomentumType == ETensorDType::FP8_E4M3) {
            mOptMScales = allocate_scales(c, comm.rank(), comm.world_size(), alloc);
        } else {
            mOptMScales.Blocks.resize(c.NumLayers);
        }
    }

    {
        auto ctx = alloc.with_context("Adam V");
        EAllocationType alloc_type = options.OffloadOptV ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        LLamaConfig c = cfg;
        c.DType = options.OptVarianceType;
        mOptV = allocate_weights(c, alloc_type, comm.rank(), comm.world_size(), alloc);
        zero_opt_state(mOptV, stream);
    }

    if((options.OffloadOptM || options.OffloadOptV) && !mUseZeroCopy) {
        mStatus[0] = sBufferStatus{-1, create_named_event("opt_fetch_0"), false, true};
        mStatus[1] = sBufferStatus{-1, create_named_event("opt_fetch_1"), false, true};
    }
}
