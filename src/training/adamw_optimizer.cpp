// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "adamw_optimizer.h"

#include "model.h"
#include "kernels/kernels.h"
#include "utilities/allocator.h"
#include "utilities/utils.h"
#include "utilities/tensor.h"
#include "utilities/stack.h"
#include "utilities/lazy_allocator.h"


AdamWStateManager::AdamWStateManager(TransformerConfig cfg, IModel& model, bool offload_m, bool offload_v,
    ETensorDType type_m, ETensorDType type_v, bool zero_copy, int rank, int world):
    mConfig(cfg), mOffloadM(offload_m), mOffloadV(offload_v), mUseZeroCopy(zero_copy), mRank(rank), mWorld(world), mMType(type_m), mVType(type_v) {

    if(mOffloadM && !mUseZeroCopy) {
        mMDeviceBuffer[0] = shard_empty_container(model.create_block_container(mConfig, mMType, mMType), mWorld);
        mMDeviceBuffer[1] = shard_empty_container(model.create_block_container(mConfig, mMType, mMType), mWorld);
    }

    if(mOffloadV && !mUseZeroCopy) {
        mVDeviceBuffer[0] = shard_empty_container(model.create_block_container(mConfig, mVType, mVType), mWorld);
        mVDeviceBuffer[1] = shard_empty_container(model.create_block_container(mConfig, mVType, mVType), mWorld);
    }

    if((mOffloadM || mOffloadV) && !mUseZeroCopy) {
        mStatus[0] = sBufferStatus{-1, create_named_event("opt_fetch_0"), false, true};
        mStatus[1] = sBufferStatus{-1, create_named_event("opt_fetch_1"), false, true};
    }
}

void AdamWStateManager::begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream) {
    LazyAllocator alloc;
    if (!mUseZeroCopy && (mOffloadM || mOffloadV)) {
        // double buffering needs to block on main_stream, so we can be sure that the stack memory can be reused safely
        CUDA_CHECK(cudaEventRecord(mStatus.at(0).DoneEvent, main_stream));
        CUDA_CHECK(cudaEventRecord(mStatus.at(1).DoneEvent, main_stream));
    }

    if(mOffloadM && !mUseZeroCopy) {
        alloc.allocate(mMDeviceBuffer.at(0));
        mMBufferStorage[0] = alloc.commit(memory, "opt_m_a");
        alloc.allocate(mMDeviceBuffer.at(1));
        mMBufferStorage[1] = alloc.commit(memory, "opt_m_b");
    }

    if(mOffloadV && !mUseZeroCopy) {
        alloc.allocate(mVDeviceBuffer.at(0));
        mVBufferStorage[0] = alloc.commit(memory, "opt_v_a");
        alloc.allocate(mVDeviceBuffer.at(1));
        mVBufferStorage[1] = alloc.commit(memory, "opt_v_b");
    }
}


void AdamWStateManager::end_optimizer(DeviceMemoryStack& memory) {
    if(mOffloadV && !mUseZeroCopy) {
        memory.free(mVBufferStorage[1]);
        memory.free(mVBufferStorage[0]);
    }

    if(mOffloadM && !mUseZeroCopy) {
        memory.free(mMBufferStorage[1]);
        memory.free(mMBufferStorage[0]);
    }
}


void AdamWStateManager::fetch_block(int layer_idx, cudaStream_t fetch_stream) {
    if((!mOffloadM && !mOffloadV) || mUseZeroCopy) return;

    NvtxRange range("fetch_opt_block", layer_idx);
    int buffer = layer_idx % 2;
    auto& stat = mStatus.at(buffer);
    stat.LayerIdx = layer_idx;
    stat.Fetch = true;

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, stat.DoneEvent, 0));

    auto fetch = [fetch_stream](Tensor& dst, Tensor& src) {
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyHostToDevice, fetch_stream));
        dst.Stats = src.Stats;
    };

    if(mOffloadM) {
        auto& buf = mMBufferStorage.at(buffer);
        auto& ref = mStorageM.at(layer_idx);

        fetch(buf, ref);
    }

    if(mOffloadV) {
        auto& buf = mVBufferStorage.at(buffer);
        auto& ref = mStorageV.at(layer_idx);

        fetch(buf, ref);
    }

    CUDA_CHECK(cudaEventRecord(stat.DoneEvent, fetch_stream));
}

SimpleTensorContainer& AdamWStateManager::get_block_m(int layer_idx, cudaStream_t stream) {
    if(!mOffloadM || mUseZeroCopy) return mBlocksM.at(layer_idx);
    return get_block_from(layer_idx, stream, mMDeviceBuffer.at(layer_idx % 2));
}

SimpleTensorContainer& AdamWStateManager::get_block_v(int layer_idx, cudaStream_t stream) {
    if(!mOffloadV || mUseZeroCopy) return mBlocksV.at(layer_idx);
    return get_block_from(layer_idx, stream, mVDeviceBuffer.at(layer_idx % 2));
}

SimpleTensorContainer& AdamWStateManager::get_block_scales_m(int layer_idx) {
    return mBlocksMScales.at(layer_idx);
}

SimpleTensorContainer& AdamWStateManager::non_block_m() {
    return mNonBlockM;
}

SimpleTensorContainer& AdamWStateManager::non_block_m_scales() {
    return mNonBlockMScales;
}

SimpleTensorContainer& AdamWStateManager::non_block_v() {
    return mNonBlockV;
}


void AdamWStateManager::store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream)  {
    if (mUseZeroCopy) return;

    NvtxRange range("store_opt_block", layer_idx);
    int buffer = layer_idx % 2;
    auto& stat = mStatus.at(layer_idx % 2);
    if(mOffloadM || mOffloadV) {
        CUDA_CHECK(cudaEventRecord(stat.DoneEvent, stream));
        CUDA_CHECK(cudaStreamWaitEvent(put_stream, stat.DoneEvent, 0));
    }

    if(mOffloadM) {
        CUDA_CHECK(cudaMemcpyAsync(mStorageM.at(layer_idx).Data, mMBufferStorage.at(buffer).Data, mStorageM.at(layer_idx).bytes(), cudaMemcpyDeviceToHost, put_stream));
    }

    if(mOffloadV) {
        CUDA_CHECK(cudaMemcpyAsync(mStorageV.at(layer_idx).Data, mVBufferStorage.at(buffer).Data, mStorageV.at(layer_idx).bytes(), cudaMemcpyDeviceToHost, put_stream));
    }

    if(mOffloadM || mOffloadV) {
        if(stat.LayerIdx != layer_idx) {
            throw std::logic_error("layer index mismatch in store_block");
        }
        CUDA_CHECK(cudaEventRecord(stat.DoneEvent, put_stream));
        stat.Done = true;
    }
}

void AdamWStateManager::allocate_state(IModel& model, cudaStream_t stream, EAllocationType kind, TensorAllocator& alloc) {
    {
        auto ctx = alloc.with_context("Adam M");
        LazyAllocator alloc_lazy;
        mBlocksM.resize(mConfig.NumLayers);
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            mBlocksM[i] = shard_empty_container(model.create_block_container(mConfig, mMType, mMType), mWorld);
            alloc_lazy.allocate(mBlocksM[i]);
            mStorageM.push_back(alloc_lazy.commit(alloc, mOffloadM ? kind : EAllocationType::ON_DEVICE, "m_block_shard"));
        }
        mNonBlockM = shard_empty_container(model.create_non_block_container(mConfig, mMType, mMType), mWorld);
        alloc_lazy.allocate(mNonBlockM);
        mStorageM.push_back(alloc_lazy.commit(alloc, mOffloadM ? kind : EAllocationType::ON_DEVICE, "m_nonblock_shard"));

        for (auto& t : mStorageM) {
            fill_zero(t, stream);
        }

        mBlocksMScales.resize(mConfig.NumLayers);

        if(mMType == ETensorDType::FP8_E4M3) {
            auto prepare_shape_for_scales = [&](auto&& c) {
                // creates shards same as main weight
                auto sharded = shard_empty_container(flattened_view(c), mWorld);
                // flatten the local shard
                auto flattened = flattened_view(sharded);
                // and group into scaling groups
                auto grouped = shard_empty_container(std::move(flattened), 128);
                return grouped;
            };
            // we "shard" for 128 as many GPUs, so that we get 1 scale per 128 weights.
            for (int i = 0; i < mConfig.NumLayers; ++i) {
                mBlocksMScales[i] = prepare_shape_for_scales(model.create_block_container(mConfig, ETensorDType::FP32, ETensorDType::FP32));
                alloc_lazy.allocate(mBlocksMScales[i]);
                alloc_lazy.commit(alloc, EAllocationType::ON_DEVICE, "m_block_scales");
                visit([stream](Tensor& t){
                    fill_constant(t, 1.f, t.nelem(), stream);
                }, mBlocksMScales[i]);
            }
            mNonBlockMScales = prepare_shape_for_scales(model.create_non_block_container(mConfig, ETensorDType::FP32, ETensorDType::FP32));
            alloc_lazy.allocate(mNonBlockMScales);
            alloc_lazy.commit(alloc, EAllocationType::ON_DEVICE, "m_nonblock_scales");
            visit([stream](Tensor& t){
                    fill_constant(t, 1.f, t.nelem(), stream);
                }, mNonBlockMScales);
        } else {
            for (int i = 0; i < mConfig.NumLayers; ++i) {
                mBlocksMScales[i] = GenericTensorContainer(std::vector<Tensor>(model.num_block_tensors()));
            }
            mNonBlockMScales = GenericTensorContainer(std::vector<Tensor>(model.num_non_block_tensors()));
        }
    }

    {
        auto ctx = alloc.with_context("Adam V");
        LazyAllocator alloc_lazy;
        mBlocksV.resize(mConfig.NumLayers);
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            mBlocksV[i] = shard_empty_container(model.create_block_container(mConfig, mVType, mVType), mWorld);
            alloc_lazy.allocate(mBlocksV[i]);
            mStorageV.push_back(alloc_lazy.commit(alloc, mOffloadV ? kind : EAllocationType::ON_DEVICE, "v_block_shard"));
        }
        mNonBlockV = shard_empty_container(model.create_non_block_container(mConfig, mVType, mVType), mWorld);
        alloc_lazy.allocate(mNonBlockV);
        mStorageV.push_back(alloc_lazy.commit(alloc, mOffloadV ? kind : EAllocationType::ON_DEVICE, "v_nonblock_shard"));

        for (auto& t : mStorageV) {
            fill_zero(t, stream);
        }
    }
}

SimpleTensorContainer& AdamWStateManager::get_block_from(int layer_idx, cudaStream_t stream, SimpleTensorContainer &buf) {
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
