// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "llama_weights.h"

#include "kernels/kernels.h"
#include "llama_model.h"
#include "llama_run_state.h"
#include "utilities/comm.h"
#include "utilities/philox.h"
#include "utilities/safetensors.h"

template<class T>
void allocate_non_matrix_params(sLLamaBlockWeights<T>& target, const LLamaConfig& config, ETensorDType dtype, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc) {
    long C = config.HiddenSize;
    long HS = config.head_size();

    target.LN1_w = alloc.allocate_shard(dtype, shard_idx, num_shards, "ln1_w", {C}, kind);
    target.LN2_w = alloc.allocate_shard(dtype, shard_idx, num_shards, "ln2_w", {C}, kind);
    long attn_intermediate_size = (config.NumQueryHeads + 2 * config.NumKeyValHeads) * HS;
    if(config.UseQKVBias) {
        target.Attn_QKV_b = alloc.allocate_shard(dtype, shard_idx, num_shards, "att_qkv_b", {attn_intermediate_size}, kind);
    } else {
        target.Attn_QKV_b = std::nullopt;
    }
}

template<class T>
void allocate_matrix_params(sLLamaBlockWeights<T>& target, const LLamaConfig& config, ETensorDType dtype, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc) {
    long C = config.HiddenSize;
    long H = config.IntermediateSize;

    long head_size = C / config.NumQueryHeads;
    long attn_intermediate_size = (config.NumQueryHeads + 2 * config.NumKeyValHeads) * head_size;
    target.Attn_QKV_w = alloc.allocate_shard(dtype, shard_idx, num_shards, "att_qkv_w", {attn_intermediate_size, C}, kind);
    target.Attn_Out_w = alloc.allocate_shard(dtype, shard_idx, num_shards, "attproj_w", {C, C}, kind);
    target.MLP_Up_w = alloc.allocate_shard(dtype, shard_idx, num_shards, "mlp_up_w", {2 * H, C}, kind);
    target.MLP_Down_w = alloc.allocate_shard(dtype, shard_idx, num_shards, "mlp_down_w", {C, H}, kind);
}

void matrix_params_from_stack(sLLamaBlockWeights<TensorShard>& target, const LLamaConfig& config, ETensorDType dtype, int shard_idx, int num_shards, DeviceMemoryStack& memory) {
    long C = config.HiddenSize;
    long H = config.IntermediateSize;

    auto create_matrix_shard = [&](long rows, long cols, const char* name) {
        Tensor raw = memory.allocate(dtype, {div_exact(rows, (long)num_shards), cols}, name);
        return TensorShard{raw, shard_idx, num_shards, std::vector<long>{rows, cols}};
    };

    long head_size = C / config.NumQueryHeads;
    long attn_intermediate_size = (config.NumQueryHeads + 2 * config.NumKeyValHeads) * head_size;
    target.Attn_QKV_w = create_matrix_shard(attn_intermediate_size, C, "Attn_QKV_w");
    target.Attn_Out_w = create_matrix_shard(C, C, "Attn_Out_w");
    target.MLP_Up_w = create_matrix_shard(2 * H, C, "MLP_Up_w");
    target.MLP_Down_w = create_matrix_shard(C, H, "MLP_Down_w");
}

void non_matrix_params_from_stack(sLLamaBlockWeights<TensorShard>& target, const LLamaConfig& config, ETensorDType dtype, int shard_idx, int num_shards, DeviceMemoryStack& memory) {
    long C = config.HiddenSize;
    long HS = config.head_size();

    auto create_vector_shard = [&](long elems, const char* name) {
        Tensor raw = memory.allocate(dtype, {div_exact(elems, (long)num_shards)}, name);
        return TensorShard{raw, shard_idx, num_shards, std::vector<long>{elems}};
    };

    target.LN1_w = create_vector_shard(C, "LN1_w");
    target.LN2_w = create_vector_shard(C, "LN2_w");
    long attn_intermediate_size = (config.NumQueryHeads + 2 * config.NumKeyValHeads) * HS;
    if(config.UseQKVBias) {
        target.Attn_QKV_b = create_vector_shard(attn_intermediate_size, "Attn_QKV_b");
    } else {
        target.Attn_QKV_b = std::nullopt;
    }
}

std::size_t aligned_size(std::size_t raw, int num_shards) {
    return div_ceil(div_exact(raw, static_cast<std::size_t>(num_shards)), static_cast<std::size_t>(4096)) * 4096;
}

std::size_t bytes_for_block_matrices(const LLamaConfig& config, ETensorDType dtype, int num_shards) {
    std::size_t C = config.HiddenSize;
    std::size_t HS = config.head_size();

    std::size_t total = 2 * aligned_size(C * get_dtype_size(dtype), num_shards);          // norms
    long attn_intermediate_size = (config.NumQueryHeads + 2 * config.NumKeyValHeads) * HS;
    if(config.UseQKVBias) {
        total += aligned_size(attn_intermediate_size * get_dtype_size(dtype), num_shards); // QKV bias
    }
    return total;
}

std::size_t bytes_for_block_non_matrix(const LLamaConfig& config, ETensorDType dtype, int num_shards) {
    std::size_t C = config.HiddenSize;
    long H = config.IntermediateSize;
    long HS = C / config.NumQueryHeads;
    long attn_intermediate_size = (config.NumQueryHeads + 2 * config.NumKeyValHeads) * HS;

    std::size_t total = 0;
    total += aligned_size(attn_intermediate_size * C * get_dtype_size(dtype), num_shards); // QKV
    total += aligned_size(C * C * get_dtype_size(dtype), num_shards); // out
    total += aligned_size(2 * C * H * get_dtype_size(dtype), num_shards); // up
    total += aligned_size(H * C * get_dtype_size(dtype), num_shards); // down
    return total;
}

std::size_t bytes_for_block(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, int num_shards) {
    return bytes_for_block_non_matrix(config, other_dtype, num_shards) + bytes_for_block_matrices(config, matrix_dtype, num_shards);
}

sLLamaBlockWeights<Tensor> allocate_block_full(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, EAllocationType kind, TensorAllocator& alloc) {
    sLLamaBlockWeights<Tensor> layer;
    allocate_matrix_params(layer, config, matrix_dtype, kind, 0, 1, alloc);
    allocate_non_matrix_params(layer, config, other_dtype, kind, 0, 1, alloc);
    return layer;
}

sLLamaBlockWeights<TensorShard> allocate_block_shard(const LLamaConfig& config, ETensorDType matrix_dtype, ETensorDType other_dtype, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc) {
    sLLamaBlockWeights<TensorShard> layer;
    allocate_matrix_params(layer, config, matrix_dtype, kind, shard_idx, num_shards, alloc);
    allocate_non_matrix_params(layer, config, other_dtype, kind, shard_idx, num_shards, alloc);
    return layer;
}

sLLamaBlockWeights<TensorShard> shard_block(const sLLamaBlockWeights<Tensor>& block, int shard_idx, int num_shards) {
    sLLamaBlockWeights<TensorShard> result;
    result.Attn_QKV_w = shard_view(block.Attn_QKV_w, shard_idx, num_shards);
    result.Attn_Out_w = shard_view(block.Attn_Out_w, shard_idx, num_shards);
    result.MLP_Up_w = shard_view(block.MLP_Up_w, shard_idx, num_shards);
    result.MLP_Down_w = shard_view(block.MLP_Down_w, shard_idx, num_shards);
    if(block.Attn_QKV_b.has_value()) {
        result.Attn_QKV_b = shard_view(block.Attn_QKV_b.value(), shard_idx, num_shards);
    }
    result.LN1_w = shard_view(block.LN1_w, shard_idx, num_shards);
    result.LN2_w = shard_view(block.LN2_w, shard_idx, num_shards);
    return result;
}

sLLamaNonBlockWeights<Tensor> allocate_non_block_full(LLamaConfig config, ETensorDType dtype, EAllocationType kind, TensorAllocator& alloc) {
    long V = config.VocabSize;
    long C = config.HiddenSize;

    sLLamaNonBlockWeights<Tensor> w;
    w.Embeddings = alloc.allocate(dtype, "embeddings", kind, {V, C});
    w.LNF_w      = alloc.allocate(dtype, "lnf_w", kind, {C});
    if(config.TiedWordEmbeddings) {
        w.LMHead = w.Embeddings;
    } else {
        w.LMHead = alloc.allocate(dtype, "lmhead", kind, {V, C});
    }

    return w;
}

sLLamaNonBlockWeights<TensorShard> allocate_non_block_shard(LLamaConfig config, ETensorDType dtype, EAllocationType kind, int shard_idx, int num_shard, TensorAllocator& alloc) {
    long V = config.VocabSize;
    long C = config.HiddenSize;

    sLLamaNonBlockWeights<TensorShard> w;
    w.Embeddings = alloc.allocate_shard(dtype, shard_idx, num_shard, "embeddings", {V, C}, kind);
    w.LNF_w      = alloc.allocate_shard(dtype, shard_idx, num_shard,"lnf_w", {C}, kind);
    if(config.TiedWordEmbeddings) {
        w.LMHead = w.Embeddings;
    } else {
        w.LMHead = alloc.allocate_shard(dtype, shard_idx, num_shard, "lmhead", {V, C}, kind);
    }

    return w;
}

sLLamaNonBlockWeights<TensorShard> shard_non_block(const sLLamaNonBlockWeights<Tensor>& block, int shard_idx, int num_shards) {
    sLLamaNonBlockWeights<TensorShard> result;
    result.Embeddings = shard_view(block.Embeddings, shard_idx, num_shards);
    result.LNF_w      = shard_view(block.LNF_w, shard_idx, num_shards);
    result.LMHead     = shard_view(block.LMHead, shard_idx, num_shards);
    return result;
}

sLLamaWeightsSet<Tensor> allocate_full_weights(const LLamaConfig& config, EAllocationType kind, TensorAllocator& alloc) {
    sLLamaWeightsSet<Tensor> result;
    result.Blocks.resize(config.NumLayers);
    for(auto& block : result.Blocks) {
        block = allocate_block_full(config, config.DType, config.DType, kind, alloc);
    }
    result.NonBlocks = allocate_non_block_full(config, config.DType, kind, alloc);
    return result;
}

sLLamaWeights allocate_weights(const LLamaConfig& config, EAllocationType kind, int shard_idx, int num_shards, TensorAllocator& alloc) {
    sLLamaWeights result;
    result.Blocks.resize(config.NumLayers);
    for(auto& block : result.Blocks) {
        block = allocate_block_shard(config, config.DType, config.DType, kind, shard_idx, num_shards, alloc);
    }
    result.NonBlocks = allocate_non_block_shard(config, config.DType, kind, shard_idx, num_shards, alloc);
    return result;
}

LLamaWeightsManager::LLamaWeightsManager(const LLamaConfig& config, const LLamaOptions& options, int rank, int world) :
    mMasterDType(options.MasterDType.value_or(config.DType)), mWorkMatDType(options.matmul_dtype()),
    mShardIdx(rank), mNumShards(world), mConfig(config)
{
    mEmbStatus.DoneEvent = create_named_event("emb_done");
    mLnfStatus.DoneEvent = create_named_event("lnf_done");
    HQ = config.NumQueryHeads;
    HKV = config.NumKeyValHeads;

    mMaster.Blocks.reserve(config.NumLayers);
    mHeadID = config.TiedWordEmbeddings ? 0 : 1;

    mOffloadMaster = options.OffloadMaster;
    mUseZeroCopy = options.UseZeroCopy;
}

LLamaWeightsManager::~LLamaWeightsManager() {
    CUDA_CHECK(cudaEventDestroy(mEmbStatus.DoneEvent));
    CUDA_CHECK(cudaEventDestroy(mLnfStatus.DoneEvent));
    for(auto& d : mBlockStatus) {
        CUDA_CHECK(cudaEventDestroy(d.DoneEvent));
    }
}

void LLamaWeightsManager::setup_scales(TensorAllocator& alloc) {
    int layers = mMaster.Blocks.size();
    mAbsMaxes = alloc.allocate(ETensorDType::FP32, "abs_maxes", EAllocationType::ON_DEVICE, {6 + layers * 14});
    float* abs_maxes = mAbsMaxes.get<float>();
    mMaster.NonBlocks.Embeddings.Stats = abs_maxes + 0;
    mMaster.NonBlocks.LNF_w.Stats = abs_maxes + 2;
    mMaster.NonBlocks.LMHead.Stats = abs_maxes + 4;
    for(int i = 0; i < layers; ++i) {
        float* a = abs_maxes + 6 + i * 14;
        mMaster.Blocks[i].Attn_QKV_w.Stats = a + 0;
        mMaster.Blocks[i].Attn_Out_w.Stats = a + 2;
        mMaster.Blocks[i].MLP_Up_w.Stats = a + 4;
        mMaster.Blocks[i].MLP_Down_w.Stats = a + 6;
        if(mMaster.Blocks[i].Attn_QKV_b.has_value()) {
            mMaster.Blocks[i].Attn_QKV_b.value().Stats = a + 8;
        }
        mMaster.Blocks[i].LN1_w.Stats = a + 10;
        mMaster.Blocks[i].LN2_w.Stats = a + 12;
    }
}


std::pair<float*, float*> LLamaWeightsManager::get_scales_for_block(int layer_idx) {
    float* abs_maxes = mAbsMaxes.get<float>();
    float* begin = abs_maxes + 6 + layer_idx * 14;
    return {begin + 0, begin + 14};
}


void LLamaWeightsManager::setup_master_buffers(const LLamaConfig& config, TensorAllocator& alloc) {
    if (mOffloadMaster && !mUseZeroCopy) {
        for(int i = 0; i < 2; ++i) {
            mMasterDeviceBufferStatus.at(i) = sGatherData{i, create_named_event(("master_event_" + std::to_string(i)).c_str())};
        }
    }
}

void LLamaWeightsManager::invalidate() {
    ++mVersion;
}

void LLamaWeightsManager::reset_scales(cudaStream_t stream) {
    fill_zero(mAbsMaxes, stream);
}

// Weight shards that get updated by the optimizer
TensorShard& LLamaWeightsManager::get_master_embeddings() {
    return mMaster.NonBlocks.Embeddings;
}

TensorShard& LLamaWeightsManager::get_master_lmhead() {
    return mMaster.NonBlocks.LMHead;
}

TensorShard& LLamaWeightsManager::get_master_lnf_w() {
    return mMaster.NonBlocks.LNF_w;
}

void LLamaWeightsManager::begin_optimizer(DeviceMemoryStack& memory, cudaStream_t stream) {
    reset_scales(stream);
    if (mOffloadMaster && !mUseZeroCopy) {
        // wait for all work on main stream to finished before the buffers can be used.
        // otherwise, we might start H2D copies while the stack memory is still in use
        // for activations.
        CUDA_CHECK(cudaEventRecord(mMasterDeviceBufferStatus.at(0).DoneEvent, stream));
        CUDA_CHECK(cudaEventRecord(mMasterDeviceBufferStatus.at(1).DoneEvent, stream));

        for (int i = 0; i < 2; ++i) {
            auto& buf = mMasterDeviceDoubleBuffer.at(i);
            if (mMaster.Blocks[0].Attn_QKV_w.Device == -1) {
                matrix_params_from_stack(buf, mConfig, mMasterDType, mShardIdx, mNumShards, memory);
            } else {
                // note: the actual data pointers will be overwritten before use, so this is safe
                buf.Attn_QKV_w = mMaster.Blocks[0].Attn_QKV_w;
                buf.Attn_Out_w = mMaster.Blocks[0].Attn_Out_w;
                buf.MLP_Up_w = mMaster.Blocks[0].MLP_Up_w;
                buf.MLP_Down_w = mMaster.Blocks[0].MLP_Down_w;
            }

            if (mMaster.Blocks[0].LN1_w.Device == -1) {
                non_matrix_params_from_stack(buf, mConfig, mMasterDType, mShardIdx, mNumShards, memory);
            } else {
                buf.LN1_w = mMaster.Blocks[0].LN1_w;
                buf.LN2_w = mMaster.Blocks[0].LN2_w;
                buf.Attn_QKV_b = mMaster.Blocks[0].Attn_QKV_b;
            }
        }
    }
}

void LLamaWeightsManager::end_optimizer(DeviceMemoryStack& memory) {
    if (mOffloadMaster && !mUseZeroCopy) {
        // it's a stack, so we need to free in reverse order
        for (int i = 1; i >= 0; --i) {
            auto& buf = mMasterDeviceDoubleBuffer.at(i);
            if (mMaster.Blocks[0].LN1_w.Device == -1) {
                if(buf.Attn_QKV_b.has_value()) {
                    memory.free(buf.Attn_QKV_b.value());
                }
                memory.free(buf.LN2_w);
                memory.free(buf.LN1_w);
            }

            if (mMaster.Blocks[0].Attn_QKV_w.Device == -1) {
                memory.free(buf.MLP_Down_w);
                memory.free(buf.MLP_Up_w);
                memory.free(buf.Attn_Out_w);
                memory.free(buf.Attn_QKV_w);
            }
        }
    }
}

void LLamaWeightsManager::fetch_master_block(int layer_idx, cudaStream_t fetch_stream) {
    if(!mOffloadMaster || mUseZeroCopy) return;

    NvtxRange range("fetch_master_block", layer_idx);
    int buffer = layer_idx % 2;
    auto& buf = mMasterDeviceDoubleBuffer.at(buffer);
    auto& stat = mMasterDeviceBufferStatus.at(buffer);
    auto& ref = mMaster.Blocks[layer_idx];
    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, stat.DoneEvent, 0));
    stat.LayerIdx = layer_idx;
    stat.Fetch = false;
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

    fetch(buf.LN1_w, ref.LN1_w);
    fetch(buf.LN2_w, ref.LN2_w);
    fetch(buf.Attn_QKV_w, ref.Attn_QKV_w);
    fetch(buf.Attn_Out_w, ref.Attn_Out_w);
    fetch(buf.MLP_Up_w, ref.MLP_Up_w);
    fetch(buf.MLP_Down_w, ref.MLP_Down_w);
    if (ref.Attn_QKV_b.has_value()) {
        fetch(buf.Attn_QKV_b.value(), ref.Attn_QKV_b.value());
    }

    if(stat.Fetch) {
        CUDA_CHECK(cudaEventRecord(stat.DoneEvent, fetch_stream));
    }
}

sLLamaBlockWeights<TensorShard>& LLamaWeightsManager::get_master_block(int layer_idx, cudaStream_t stream) {
    if(!mOffloadMaster || mUseZeroCopy) return mMaster.Blocks[layer_idx];

    int buffer = layer_idx % 2;
    auto& buf = mMasterDeviceDoubleBuffer.at(buffer);
    auto& stat = mMasterDeviceBufferStatus.at(buffer);
    update_get_status(stat, layer_idx, stream);
    return buf;
}

void LLamaWeightsManager::release_master_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream, LLamaRunState& run_state) {
    if(!mOffloadMaster || mUseZeroCopy) return;

    NvtxRange range("release_master_block", layer_idx);
    int buffer = layer_idx % 2;
    auto& buf = mMasterDeviceDoubleBuffer.at(buffer);
    auto& stat = mMasterDeviceBufferStatus.at(buffer);
    auto& ref = mMaster.Blocks[layer_idx];

    auto send = [put_stream](TensorShard& dst, TensorShard& src) {
        // tensors on the same device are handled by pointer assignment
        if(dst.Device != src.Device) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDeviceToHost, put_stream));
        }
    };

    auto& src = mMasterDeviceDoubleBuffer.at(buffer);
    auto& qnt = lookup_block_quants(layer_idx);

    // put stream can start as soon as the new master weights are ready
    CUDA_CHECK(cudaEventRecord(stat.DoneEvent, stream));
    CUDA_CHECK(cudaStreamWaitEvent(put_stream, stat.DoneEvent, 0));

    bool convert_any = false;
    if (qnt.LayerIdx == layer_idx) {
        NvtxRange q_rng("quantize");
        convert_dtype_for_gather(src.LN1_w, qnt.Block.LN1_w, convert_any, !mOffloadMaster, run_state);
        convert_dtype_for_gather(src.LN2_w, qnt.Block.LN2_w, convert_any, !mOffloadMaster, run_state);
        convert_dtype_for_gather(src.Attn_QKV_w, qnt.Block.Attn_QKV_w, convert_any, !mOffloadMaster, run_state);
        convert_dtype_for_gather(src.Attn_Out_w, qnt.Block.Attn_Out_w, convert_any, !mOffloadMaster, run_state);
        convert_dtype_for_gather(src.MLP_Up_w, qnt.Block.MLP_Up_w, convert_any, !mOffloadMaster, run_state);
        convert_dtype_for_gather(src.MLP_Down_w, qnt.Block.MLP_Down_w, convert_any, !mOffloadMaster, run_state);
        if (src.Attn_QKV_b.has_value()) {
            convert_dtype_for_gather(src.Attn_QKV_b.value(), qnt.Block.Attn_QKV_b.value(), convert_any, !mOffloadMaster, run_state);
        }
        // indicate that this is already the version for the next step
        qnt.Version = mVersion + 1;
    }
    CUDA_CHECK(cudaEventRecord(stat.DoneEvent, stream));

    send(ref.LN1_w, buf.LN1_w);
    send(ref.LN2_w, buf.LN2_w);
    send(ref.Attn_QKV_w, buf.Attn_QKV_w);
    send(ref.Attn_Out_w, buf.Attn_Out_w);
    send(ref.MLP_Up_w, buf.MLP_Up_w);
    send(ref.MLP_Down_w, buf.MLP_Down_w);
    if (ref.Attn_QKV_b.has_value()) {
        send(ref.Attn_QKV_b.value(), buf.Attn_QKV_b.value());
    }

    // put is only considered complete once *both* master weights *and* quants
    // are finished.
    CUDA_CHECK(cudaStreamWaitEvent(put_stream, stat.DoneEvent, 0));
    release_status(stat, layer_idx, put_stream);
}

bool LLamaWeightsManager::is_in_cache(sGatherData& data, int expected) const {
    if(!data.Done) {
        throw std::logic_error("still in use");
    }

    if(data.LayerIdx == expected && data.Version == mVersion) {
        data.Fetch = false;
        return true;
    }

    data.LayerIdx = expected;
    data.Fetch = true;
    return false;
}

void LLamaWeightsManager::update_get_status(sGatherData& data, int expected, cudaStream_t stream) const {
    data.Done = false;

    cudaEvent_t done_event = data.DoneEvent;
    if(data.LayerIdx != expected) {
        throw std::logic_error("Gather data is not for the requested layer");
    }

    // if we needed to fetch, we need to wait
    if(data.Fetch) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, done_event, 0));
    }
    data.Version = mVersion;
}

void LLamaWeightsManager::release_status(sGatherData& data, int expected, cudaStream_t stream) {
    if(data.LayerIdx != expected) {
        throw std::logic_error("Gather data is not for the requested layer");
    }
    CUDA_CHECK(cudaEventRecord(data.DoneEvent, stream));
    data.Done = true;
}

void LLamaWeightsManager::convert_dtype_for_gather(TensorShard& src, TensorShard& qnt, bool& convert, bool src_is_persistent, LLamaRunState& run_state) {
    qnt.Stats = src.Stats;
    if (qnt.DType == src.DType) {
        // Identical tensors
        if(qnt.Device == src.Device && src_is_persistent) {
            qnt.Data = src.Data;
            return;
        } else {    // transfer to other device? (should just be GPU -> CPU)
            CUDA_CHECK(cudaMemcpyAsync(qnt.Data, src.Data, qnt.bytes(), cudaMemcpyDefault, run_state.MainStream));
            convert = true;
            return;
        }
    }

    quantize_with_abs_max(qnt, src.scale(), src, src.abs_max(), qnt.nelem(), run_state.DeviceProp, run_state.MainStream);
    convert = true;
}

void LLamaWeightsManager::gather_block(int layer_idx, NCCLCommunicator& comm, LLamaRunState& run_state) {
    auto& src = mMaster.Blocks[layer_idx];
    auto& qnt = lookup_block_quants(layer_idx);
    auto& dst = lookup_block_weights(layer_idx);
    auto& gather_data = lookup_block_status(layer_idx);

    // Check if data is still in cache
    if(is_in_cache(gather_data, layer_idx)) {
        return;
    }

    NvtxRange range("gather_block", layer_idx);

    bool convert_any = false;
    if (qnt.Version != mVersion || qnt.LayerIdx != layer_idx) {
        NvtxRange q_rng("quantize");
        convert_dtype_for_gather(src.LN1_w, qnt.Block.LN1_w, convert_any, true, run_state);
        convert_dtype_for_gather(src.LN2_w, qnt.Block.LN2_w, convert_any, true, run_state);
        convert_dtype_for_gather(src.Attn_QKV_w, qnt.Block.Attn_QKV_w, convert_any, true, run_state);
        convert_dtype_for_gather(src.Attn_Out_w, qnt.Block.Attn_Out_w, convert_any, true, run_state);
        convert_dtype_for_gather(src.MLP_Up_w, qnt.Block.MLP_Up_w, convert_any, true, run_state);
        convert_dtype_for_gather(src.MLP_Down_w, qnt.Block.MLP_Down_w, convert_any, true, run_state);
        if (src.Attn_QKV_b.has_value()) {
            convert_dtype_for_gather(src.Attn_QKV_b.value(), qnt.Block.Attn_QKV_b.value(), convert_any, true, run_state);
        }

        qnt.Version = mVersion;
        qnt.LayerIdx = layer_idx;
    }

    // make sure the target scales are set up correctly
    dst.LN1_w.Stats = qnt.Block.LN1_w.Stats;
    dst.LN2_w.Stats = qnt.Block.LN2_w.Stats;
    dst.Attn_QKV_w.Stats = qnt.Block.Attn_QKV_w.Stats;
    if (src.Attn_QKV_b.has_value()) {
        dst.Attn_QKV_b.value().Stats = qnt.Block.Attn_QKV_b.value().Stats;
    }
    dst.Attn_Out_w.Stats = qnt.Block.Attn_Out_w.Stats;
    dst.MLP_Up_w.Stats = qnt.Block.MLP_Up_w.Stats;
    dst.MLP_Down_w.Stats = qnt.Block.MLP_Down_w.Stats;

    if (convert_any) {
        CUDA_CHECK(cudaEventRecord(gather_data.DoneEvent, run_state.MainStream));
    }

    comm.begin_transaction(gather_data.DoneEvent);
    comm.schedule_all_gather(qnt.Block.LN1_w, dst.LN1_w);
    comm.schedule_all_gather(qnt.Block.LN2_w, dst.LN2_w);
    comm.schedule_all_gather(qnt.Block.Attn_QKV_w, dst.Attn_QKV_w);
    if (src.Attn_QKV_b.has_value()) {
        comm.schedule_all_gather(qnt.Block.Attn_QKV_b.value(), dst.Attn_QKV_b.value());
    }
    comm.schedule_all_gather(qnt.Block.Attn_Out_w, dst.Attn_Out_w);
    comm.schedule_all_gather(qnt.Block.MLP_Up_w, dst.MLP_Up_w);
    comm.schedule_all_gather(qnt.Block.MLP_Down_w, dst.MLP_Down_w);
    comm.execute_transaction(gather_data.DoneEvent);
}


sLLamaBlockWeights<Tensor>& LLamaWeightsManager::get_block(int layer_idx, cudaStream_t stream) {
    auto& gather_data = lookup_block_status(layer_idx);
    update_get_status(gather_data, layer_idx, stream);
    return lookup_block_weights(layer_idx);
}

void LLamaWeightsManager::release_block(int layer_idx, cudaStream_t stream) {
    auto& gather_data = lookup_block_status(layer_idx);
    release_status(gather_data, layer_idx, stream);
}

void LLamaWeightsManager::gather_embeddings(NCCLCommunicator& comm) {
    if(is_in_cache(mEmbStatus, 0)) {
        return;
    }

    comm.begin_transaction(mEmbStatus.DoneEvent);
    comm.schedule_all_gather(get_master_embeddings(), mWork.NonBlocks.Embeddings);
    comm.execute_transaction(mEmbStatus.DoneEvent);
}

Tensor& LLamaWeightsManager::get_embeddings(cudaStream_t stream) {
    update_get_status(mEmbStatus, 0, stream);
    return mWork.NonBlocks.Embeddings;
}

void LLamaWeightsManager::release_embeddings(cudaStream_t stream) {
    release_status(mEmbStatus, 0, stream);
}

void LLamaWeightsManager::gather_lnf(NCCLCommunicator& comm) {
    if(is_in_cache(mLnfStatus, 0)) {
        return;
    }
    mLnfStatus.LayerIdx = 0;
    comm.begin_transaction(mLnfStatus.DoneEvent);
    comm.schedule_all_gather(get_master_lnf_w(), mWork.NonBlocks.LNF_w);
    comm.execute_transaction(mLnfStatus.DoneEvent);
}

Tensor& LLamaWeightsManager::get_lnf(cudaStream_t stream) {
    update_get_status(mLnfStatus, 0, stream);
    return mWork.NonBlocks.LNF_w;
}

void LLamaWeightsManager::release_lnf(cudaStream_t stream) {
    release_status(mLnfStatus, 0, stream);
}

void LLamaWeightsManager::gather_head(NCCLCommunicator& comm) {
    if(is_in_cache(mEmbStatus, mHeadID)) {
        return;
    }

    comm.begin_transaction(mEmbStatus.DoneEvent);
    comm.schedule_all_gather(get_master_lmhead(), mWork.NonBlocks.LMHead);
    comm.execute_transaction(mEmbStatus.DoneEvent);
}

Tensor& LLamaWeightsManager::get_head(cudaStream_t stream) {
    update_get_status(mEmbStatus, mHeadID, stream);
    return mWork.NonBlocks.LMHead;
}

void LLamaWeightsManager::release_head(cudaStream_t stream) {
    release_status(mEmbStatus, mHeadID, stream);
}

void sLLamaWeights::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    callback("model.embed_tokens.weight", NonBlocks.Embeddings);
    if(NonBlocks.LMHead.Data != NonBlocks.Embeddings.Data) {
        callback("lm_head.weight", NonBlocks.LMHead);
    }
    callback("model.norm.weight", NonBlocks.LNF_w);

    const auto& Layers = Blocks;
    for(int i = 0; i < Layers.size(); i++) {
        auto& layer = Layers[i];
        const Tensor& qkv_w = layer.Attn_QKV_w;
        const Tensor& up_proj = layer.MLP_Up_w;
        std::string prefix = "model.layers." + std::to_string(i);
        callback(prefix + ".self_attn.qkv.weight", qkv_w);
        if (layer.Attn_QKV_b) {
            callback(prefix + ".self_attn.qkv.bias", layer.Attn_QKV_b.value());
        }

        callback(prefix + ".self_attn.o_proj.weight", layer.Attn_Out_w);
        callback(prefix + ".mlp.up.weight", up_proj);
        callback(prefix + ".mlp.down_proj.weight", layer.MLP_Down_w);
        callback(prefix + ".input_layernorm.weight", layer.LN1_w);
        callback(prefix + ".post_attention_layernorm.weight", layer.LN2_w);
    }
}

void LLamaWeightsManager::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    mMaster.iterate_tensors(callback);
}

/*!
 * \brief Weights unsharded, i.e., full copy on each device.
 * \details Optionally allows offloading master copies to the CPU.
 * For tensors where the "work" dtype equals the master dtype, however,
 * the master copy is just the local shard, aliasing the work tensor.
 *
 * Quantized weights are trivial in this setup: the work copy *is* the
 * quantized version, so the quantized shard is just a view into the work
 * copy, and no additional memory is needed.
 */
class WeightsMgrUnsharded final: public LLamaWeightsManager {
public:
    WeightsMgrUnsharded(const LLamaConfig& config, const LLamaOptions& options, int rank, int world, TensorAllocator& alloc);
private:
    sLLamaBlockWeights<Tensor>& lookup_block_weights(int layer_idx) override;
    sQuantBlock& lookup_block_quants(int layer_idx) override;
    sGatherData& lookup_block_status(int layer_idx) override;

    std::vector<sQuantBlock> mQuants;
};

WeightsMgrUnsharded::WeightsMgrUnsharded(const LLamaConfig& config, const LLamaOptions& options, int rank, int world, TensorAllocator& alloc) : LLamaWeightsManager(config, options, rank, world) {
    auto ctx = alloc.with_context("Weights");
    mOffloadMaster = options.OffloadMaster;
    EAllocationType master_alloc = mOffloadMaster ? options.offload_alloc() : EAllocationType::ON_DEVICE;
    mWork.Blocks.reserve(config.NumLayers);
    mBlockStatus.reserve(config.NumLayers);

    for(int i = 0; i < config.NumLayers; ++i) {
        mWork.Blocks.push_back(
            allocate_block_full(config, mWorkMatDType, config.DType, EAllocationType::ON_DEVICE, alloc));
        mMaster.Blocks.push_back(shard_block(mWork.Blocks.back(), mShardIdx, mNumShards));
        if (mWorkMatDType != mMasterDType) {
            auto c = alloc.with_context("Master");
            allocate_matrix_params(mMaster.Blocks.back(), config, mMasterDType, master_alloc, mShardIdx, mNumShards, alloc);
        }
        if (config.DType != mMasterDType) {
            auto c = alloc.with_context("Master");
            allocate_non_matrix_params(mMaster.Blocks.back(), config, mMasterDType, master_alloc, mShardIdx, mNumShards, alloc);
        }
        mBlockStatus.push_back(sGatherData{i, create_named_event(("gather_done_" + std::to_string(i)).c_str())});
        mQuants.push_back(sQuantBlock{shard_block(mWork.Blocks.back(), mShardIdx, mNumShards)});
    }

    mWork.NonBlocks = allocate_non_block_full(config, config.DType, EAllocationType::ON_DEVICE, alloc);
    if (config.DType != mMasterDType) {
        auto c = alloc.with_context("Master");
        mMaster.NonBlocks = allocate_non_block_shard(config, mMasterDType, master_alloc, mShardIdx, mNumShards, alloc);
    } else {
        mMaster.NonBlocks = shard_non_block(mWork.NonBlocks, mShardIdx, mNumShards);
    }

    setup_scales(alloc);
    setup_master_buffers(config, alloc);
}

sLLamaBlockWeights<Tensor>& WeightsMgrUnsharded::lookup_block_weights(int layer_idx) {
    return mWork.Blocks[layer_idx];
}

WeightsMgrUnsharded::sQuantBlock& WeightsMgrUnsharded::lookup_block_quants(int layer_idx) {
    return mQuants[layer_idx];
}

WeightsMgrUnsharded::sGatherData& WeightsMgrUnsharded::lookup_block_status(int layer_idx) {
    return mBlockStatus[layer_idx];
}

/*!
 * \brief Weights manager that shards weights across the gpu, gathering them only for calculations.
 * \details ZeRO-3 / FSDP. We allocate a shard of master weights on each worker, and double-buffers
 * for work weights. This leaves us with two options regarding quantized weights:
 *  1) keep a quantized copy along with the master copy, and transfer that on each request
 *  2) re-quantize for each request, less efficient but also less memory.
 *  Note that if we re-quantize, we actually do not need _any_ additional memory, since we
 *  can use the local shard of the work copy as temporary space.
 */
class WeightsMgrSharded final: public LLamaWeightsManager {
public:
    WeightsMgrSharded(const LLamaConfig& config, const LLamaOptions& options, int rank, int world, TensorAllocator& alloc);
private:
    sLLamaBlockWeights<Tensor>& lookup_block_weights(int layer_idx) override;
    sQuantBlock& lookup_block_quants(int layer_idx) override;
    sGatherData& lookup_block_status(int layer_idx) override;

    std::vector<sQuantBlock> mQuants;
    bool mPersistentQuants = false;     // whether to keep a quantized copy of the master shards
    bool mOffloadQuants = false;
};

WeightsMgrSharded::WeightsMgrSharded(const LLamaConfig& config, const LLamaOptions& options, int rank, int world, TensorAllocator& alloc) : LLamaWeightsManager(config, options, rank, world) {
    mOffloadMaster = options.OffloadMaster;
    mPersistentQuants = options.PersistentQuants;
    mOffloadQuants = options.OffloadQuants;
    {
        auto ctx = alloc.with_context("Master");
        EAllocationType master_alloc = mOffloadMaster ? options.offload_alloc() : EAllocationType::ON_DEVICE;

        // master params are just fully separate sharded params
        for (int i = 0; i < config.NumLayers; ++i) {
            mMaster.Blocks.push_back(
                allocate_block_shard(config, mMasterDType, config.DType, master_alloc, mShardIdx, mNumShards, alloc));
        }
        mMaster.NonBlocks = allocate_non_block_shard(config, mMasterDType, master_alloc, mShardIdx, mNumShards, alloc);
    }

    {
        auto ctx2 = alloc.with_context("Weights");
        mWork.Blocks.reserve(2);
        // work params use double buffering
        for (int i = 0; i < 2; ++i) {
            mWork.Blocks.push_back(
                allocate_block_full(config, mWorkMatDType, mMasterDType, EAllocationType::ON_DEVICE, alloc));
            mBlockStatus.push_back({i, create_named_event(("gather_done_" + std::to_string(i)).c_str())});
        }

        // ensure there's just one buffer for Emb and LMHead
        LLamaConfig cpy{config};
        cpy.TiedWordEmbeddings = true;
        mWork.NonBlocks = allocate_non_block_full(cpy, config.DType, EAllocationType::ON_DEVICE, alloc);
    }

    if(mPersistentQuants) {
        auto ctx2 = alloc.with_context("Quants");
        EAllocationType quant_alloc = mOffloadQuants ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        for (int i = 0; i < config.NumLayers; ++i) {
            mQuants.push_back(sQuantBlock{allocate_block_shard(config, mWorkMatDType, config.DType, quant_alloc, mShardIdx, mNumShards, alloc), i});
        }
    } else {
        // TODO this should be more fine-grained; taking into account matrix and non-matrix parameters separately
        if (mWorkMatDType == config.DType && mMasterDType == config.DType) {
            for (int i = 0; i < config.NumLayers; ++i) {
                mQuants.push_back(sQuantBlock{mMaster.Blocks[i], i, -1});
            }
        } else {
            for (int i = 0; i < 2; ++i) {
                mQuants.push_back(sQuantBlock{shard_block(mWork.Blocks[i], mShardIdx, mNumShards)});
            }
        }
    }

    setup_scales(alloc);
    setup_master_buffers(config, alloc);
}

sLLamaBlockWeights<Tensor>& WeightsMgrSharded::lookup_block_weights(int layer_idx) {
    return mWork.Blocks[layer_idx % 2];
}

WeightsMgrSharded::sQuantBlock& WeightsMgrSharded::lookup_block_quants(int layer_idx) {
    if (mPersistentQuants) {
        return mQuants[layer_idx];
    } else {
        return mQuants[layer_idx % 2];
    }
}

WeightsMgrSharded::sGatherData& WeightsMgrSharded::lookup_block_status(int layer_idx) {
    return mBlockStatus[layer_idx % 2];
}

std::unique_ptr<LLamaWeightsManager> LLamaWeightsManager::create(const LLamaConfig& config, const LLamaOptions& options, int rank, int world, TensorAllocator& alloc) {
    if (options.ShardWeights) {
        return std::make_unique<WeightsMgrSharded>(config, options, rank, world, alloc);
    } else {
        return std::make_unique<WeightsMgrUnsharded>(config, options, rank, world, alloc);
    }
}

void LLamaWeightsManager::random_init(int seed, const LLamaOptions& options, NCCLCommunicator& comm) {
    Philox4x32 rng(seed);

    float scale = 0.02f;
    float residual_scale = 1.0f / sqrtf(2.0f * mMaster.Blocks.size());

    for (int l = 0; l < mMaster.Blocks.size(); l++) {
        auto local_seeds = rng.generate(comm.rank(), l);
        auto& layer = mMaster.Blocks[l];
        auto& qkv_w = layer.Attn_QKV_w;
        auto& up_proj = layer.MLP_Up_w;
        auto& down_proj = layer.MLP_Down_w;
        auto& qkv_b = layer.Attn_QKV_b;
        auto& out_w = layer.Attn_Out_w;

        fill_constant(layer.LN1_w, 1.f, layer.LN1_w.nelem(), nullptr);
        fill_constant(layer.LN2_w, 1.f, layer.LN2_w.nelem(), nullptr);

        fill_normal(qkv_w, qkv_w.nelem(), 0.f, scale, seed, local_seeds[0], nullptr);
        fill_normal(up_proj, up_proj.nelem(), 0.f, scale, seed, local_seeds[1], nullptr);

        if (options.InitProjectionsToZero) {
            fill_zero(out_w, nullptr);
            fill_zero(down_proj, nullptr);
        } else {
            fill_normal(out_w, out_w.nelem(), 0.f, scale * residual_scale, seed, local_seeds[3], nullptr);
            fill_normal(down_proj, down_proj.nelem(), 0.f, scale * residual_scale, seed, local_seeds[2], nullptr);
        }
        if (qkv_b) {
            fill_zero(qkv_b.value(), nullptr);
        }
    }

    auto local_seeds = rng.generate(comm.rank(), mMaster.Blocks.size());
    fill_normal(mMaster.NonBlocks.Embeddings, mMaster.NonBlocks.Embeddings.nelem(), 0.f, scale, seed, local_seeds[0], nullptr);
    if (mMaster.NonBlocks.LMHead.Data != mMaster.NonBlocks.Embeddings.Data) {
        fill_normal(mMaster.NonBlocks.LMHead, mMaster.NonBlocks.LMHead.nelem(), 0.f, scale, seed, local_seeds[1], nullptr);
    }

    fill_constant(mMaster.NonBlocks.LNF_w, 1.f, mMaster.NonBlocks.LNF_w.nelem(), nullptr);

    synchronize_absmax(comm);
    comm.barrier();     // make sure all import is done before any process proceeds.
}

void LLamaWeightsManager::synchronize_absmax(NCCLCommunicator& comm) {
    cudaDeviceProp dp;
    CUDA_CHECK(cudaGetDeviceProperties(&dp, mShardIdx));

    // in order to reach a consistent state, like after an optimizer step, we need to calculate the abs-maxes
    for (auto& layer : mMaster.Blocks) {
        abs_max(layer.LN1_w.abs_max(), layer.LN1_w, layer.LN1_w.nelem(), dp, nullptr);
        abs_max(layer.LN2_w.abs_max(), layer.LN2_w, layer.LN2_w.nelem(), dp, nullptr);
        abs_max(layer.Attn_QKV_w.abs_max(), layer.Attn_QKV_w, layer.Attn_QKV_w.nelem(), dp, nullptr);
        abs_max(layer.Attn_Out_w.abs_max(), layer.Attn_Out_w, layer.Attn_Out_w.nelem(), dp, nullptr);
        abs_max(layer.MLP_Up_w.abs_max(), layer.MLP_Up_w, layer.MLP_Up_w.nelem(), dp, nullptr);
        abs_max(layer.MLP_Down_w.abs_max(), layer.MLP_Down_w, layer.MLP_Down_w.nelem(), dp, nullptr);
        if (layer.Attn_QKV_b.has_value()) {
            abs_max(layer.Attn_QKV_b.value().abs_max(), layer.Attn_QKV_b.value(), layer.Attn_QKV_b.value().nelem(), dp, nullptr);
        }
        comm.reduce_max(layer.LN1_w.abs_max());
        comm.reduce_max(layer.LN2_w.abs_max());
        comm.reduce_max(layer.Attn_QKV_w.abs_max());
        comm.reduce_max(layer.Attn_Out_w.abs_max());
        comm.reduce_max(layer.MLP_Up_w.abs_max());
        comm.reduce_max(layer.MLP_Down_w.abs_max());
        if (layer.Attn_QKV_b.has_value()) {
            comm.reduce_max(layer.Attn_QKV_b.value().abs_max());
        }
        comm.wait_on_comms(nullptr);
    }
    comm.barrier();     // make sure all import is done before any process proceeds.
}

namespace {
    void load_intersect(TensorShard& dst, const SafeTensorEntry& src,
                        std::ptrdiff_t src_begin, std::ptrdiff_t src_end,
                        bool allow_cast) {
        std::ptrdiff_t dst_begin = dst.shard_offset();
        std::ptrdiff_t dst_end = dst.shard_offset() + dst.nelem();

        // no overlap?
        if (dst_begin >= src_end) return;
        if (dst_end <= src_begin) return;

        std::ptrdiff_t dst_slice_begin = dst_begin;
        std::ptrdiff_t dst_slice_end = dst_end;
        if (dst_begin < src_begin) {
            dst_slice_begin = src_begin;
        }
        if (dst_end > src_end) {
            dst_slice_end = src_end;
        }

        Tensor dst_slice = dst;
        dst_slice.Sizes.fill(1);
        dst_slice.Sizes[0] = dst_slice_end - dst_slice_begin;
        dst_slice.Data = dst.Data + (dst_slice_begin - dst_begin) * get_dtype_size(dst.DType);

        src.read_raw(dst_slice, dst_slice_begin - src_begin, dst_slice_end - dst_slice_begin, allow_cast);
    }

    void write_intersect(SafeTensorWriter& writer, const std::string& tensor_name, const TensorShard& src,
                        std::ptrdiff_t dst_begin, std::ptrdiff_t dst_end) {
        std::ptrdiff_t src_begin = src.shard_offset();
        std::ptrdiff_t src_end = src.shard_offset() + src.nelem();

        writer.mark_done(tensor_name);

        // no overlap?
        if (src_begin >= dst_end) return;
        if (src_end <= dst_begin) return;

        std::ptrdiff_t src_slice_begin = src_begin;
        std::ptrdiff_t src_slice_end = src_end;
        if (src_begin < dst_begin) {
            src_slice_begin = dst_begin;
        }
        if (src_end > dst_end) {
            src_slice_end = dst_end;
        }

        std::ptrdiff_t dst_offset = src_slice_begin - dst_begin;
        std::ptrdiff_t elements = src_slice_end - src_slice_begin;

        Tensor src_slice = src;
        src_slice.Sizes.fill(1);
        src_slice.Sizes[0] = src_slice_end - src_slice_begin;
        src_slice.Data = src.Data + (src_slice_begin - src_begin) * get_dtype_size(src.DType);
        writer.write_raw(tensor_name, dst_offset, elements, src_slice);
    }
}

void LLamaWeightsManager::import_from_file(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    SafeTensorsReader reader{file_name};

    std::unordered_map<std::string, TensorShard> named_tensors;
    this->iterate_tensors([&named_tensors](std::string name, const TensorShard& tensor) {
        named_tensors.emplace(std::move(name), tensor);
    });

    long C =  mMaster.Blocks[0].Attn_QKV_w.GlobalShape[1];
    long HS = mMaster.Blocks[0].Attn_QKV_w.GlobalShape[0] / (HQ + 2 * HKV);
    long H =  mMaster.Blocks[0].MLP_Up_w.GlobalShape[0] / 2;

    for (const auto& entry : reader.entries()) {
        if (auto found = named_tensors.find(entry.name()); found != named_tensors.end()) {
            load_intersect(found->second, entry, 0, found->second.global_nelem(), allow_cast);
        } else if (entry.name().starts_with("model.layers.")) {
            // convert QKV and UpGate
            std::size_t chars = 0;
            auto layer_idx = std::stoi(entry.name().c_str() + 13, &chars);
            std::string suffix = entry.name().substr(13 + chars);
            auto& layer = mMaster.Blocks.at(layer_idx);

            // split positions in global tensor
            std::ptrdiff_t q_end = HS * HQ;
            std::ptrdiff_t k_end = HS * (HQ + HKV);
            std::ptrdiff_t v_end = HS * (HQ + 2 * HKV);

            if (suffix == ".self_attn.q_proj.weight") {
                load_intersect(layer.Attn_QKV_w, entry, 0, q_end * C, allow_cast);
            } else if (suffix == ".self_attn.k_proj.weight") {
                load_intersect(layer.Attn_QKV_w, entry, q_end * C, k_end * C, allow_cast);
            } else if (suffix == ".self_attn.v_proj.weight") {
                load_intersect(layer.Attn_QKV_w, entry, k_end * C, v_end * C, allow_cast);
            } else if (suffix == ".self_attn.q_proj.bias") {
                load_intersect(layer.Attn_QKV_b.value(), entry, 0, q_end, allow_cast);
            } else if (suffix == ".self_attn.k_proj.bias") {
                load_intersect(layer.Attn_QKV_b.value(), entry, q_end, k_end, allow_cast);
            } else if (suffix == ".self_attn.v_proj.bias") {
                load_intersect(layer.Attn_QKV_b.value(), entry, k_end, v_end, allow_cast);
            } else if (suffix == ".mlp.up_proj.weight") {
                load_intersect(layer.MLP_Up_w, entry, 0, H * C, allow_cast);
            } else if (suffix == ".mlp.gate_proj.weight") {
                load_intersect(layer.MLP_Up_w, entry, H * C, 2 * H * C, allow_cast);
            } else {
                throw std::runtime_error("Unexpected tensor name: " + entry.name());
            }
        } else {
            throw std::runtime_error("Unexpected tensor name: " + entry.name());
        }
    }

    synchronize_absmax(comm);
    comm.barrier();     // make sure all import is done before any process proceeds.
}

void LLamaWeightsManager::export_to_file(const std::string& file_name, NCCLCommunicator& comm) const {
    SafeTensorWriter writer{file_name};
    const_cast<LLamaWeightsManager*>(this)->iterate_tensors([&](const std::string& name, const TensorShard& tensor) {
        if (name.find(".self_attn.qkv.") == std::string::npos &&
            name.find(".mlp.up.") == std::string::npos) {
            writer.register_tensor(name, tensor);
        }
    });

    long C = mMaster.Blocks[0].Attn_QKV_w.GlobalShape[1];
    long HS = mMaster.Blocks[0].Attn_QKV_w.GlobalShape[0] / (HQ + 2 * HKV);
    long H = mMaster.Blocks[0].MLP_Up_w.GlobalShape[0] / 2;

    // Register QKV and MLP splits
    for (int i = 0; i < mMaster.Blocks.size(); ++i) {
        const auto& layer = mMaster.Blocks[i];
        std::string prefix = "model.layers." + std::to_string(i);

        TensorShard q_proj_w = layer.Attn_QKV_w;
        q_proj_w.GlobalShape[0] = HS * HQ;
        TensorShard k_proj_w = layer.Attn_QKV_w;
        k_proj_w.GlobalShape[0] = HS * HKV;
        TensorShard v_proj_w = layer.Attn_QKV_w;
        v_proj_w.GlobalShape[0] = HS * HKV;

        writer.register_tensor(prefix + ".self_attn.q_proj.weight", q_proj_w);
        writer.register_tensor(prefix + ".self_attn.k_proj.weight", k_proj_w);
        writer.register_tensor(prefix + ".self_attn.v_proj.weight", v_proj_w);

        TensorShard up_proj_w = layer.MLP_Up_w;
        up_proj_w.GlobalShape[0] = H;
        TensorShard gate_proj_w = layer.MLP_Up_w;
        gate_proj_w.GlobalShape[0] = H;

        writer.register_tensor(prefix + ".mlp.up_proj.weight", up_proj_w);
        writer.register_tensor(prefix + ".mlp.gate_proj.weight", gate_proj_w);

        // Handle bias if present
        if (layer.Attn_QKV_b.has_value()) {
            TensorShard q_proj_b = layer.Attn_QKV_b.value();
            q_proj_b.GlobalShape[0] = HS * HQ;
            TensorShard k_proj_b = layer.Attn_QKV_b.value();
            k_proj_b.GlobalShape[0] = HS * HKV;
            TensorShard v_proj_b = layer.Attn_QKV_b.value();
            v_proj_b.GlobalShape[0] = HS * HKV;

            writer.register_tensor(prefix + ".self_attn.q_proj.bias", q_proj_b);
            writer.register_tensor(prefix + ".self_attn.k_proj.bias", k_proj_b);
            writer.register_tensor(prefix + ".self_attn.v_proj.bias", v_proj_b);
        }
    }

    writer.prepare_metadata(&comm);

    const_cast<LLamaWeightsManager*>(this)->iterate_tensors([&](const std::string& name, const TensorShard& tensor) {
        if (name.find(".self_attn.qkv.") == std::string::npos &&
           name.find(".mlp.up.") == std::string::npos) {
           writer.write_tensor(name, tensor, &comm);
       }
    });

    for (int i = 0; i < mMaster.Blocks.size(); ++i) {
        const auto& layer = mMaster.Blocks[i];
        std::string prefix = "model.layers." + std::to_string(i);

        // Split positions in global tensor for QKV
        std::ptrdiff_t q_end = HS * HQ;
        std::ptrdiff_t k_end = HS * (HQ + HKV);
        std::ptrdiff_t v_end = HS * (HQ + 2 * HKV);

        write_intersect(writer, prefix + ".self_attn.q_proj.weight",
                       layer.Attn_QKV_w, 0, q_end * C);
        write_intersect(writer, prefix + ".self_attn.k_proj.weight",
                       layer.Attn_QKV_w, q_end * C, k_end * C);
        write_intersect(writer, prefix + ".self_attn.v_proj.weight",
                       layer.Attn_QKV_w, k_end * C, v_end * C);

        write_intersect(writer, prefix + ".mlp.up_proj.weight",
                       layer.MLP_Up_w, 0, H * C);
        write_intersect(writer, prefix + ".mlp.gate_proj.weight",
                       layer.MLP_Up_w, H * C, 2 * H * C);

        // Write Q, K, V projection biases if present
        if (layer.Attn_QKV_b.has_value()) {
            const auto& qkv_bias = layer.Attn_QKV_b.value();
            write_intersect(writer, prefix + ".self_attn.q_proj.bias",
                           qkv_bias, 0, q_end);
            write_intersect(writer, prefix + ".self_attn.k_proj.bias",
                           qkv_bias, q_end, k_end);
            write_intersect(writer, prefix + ".self_attn.v_proj.bias",
                           qkv_bias, k_end, v_end);
        }
    }

    writer.finalize(&comm);
}
