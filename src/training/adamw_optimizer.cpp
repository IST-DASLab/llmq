// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "adamw_optimizer.h"
#include "utilities/utils.h"
#include "utilities/tensor.h"
#include "utilities/stack.h"
#include "utilities/lazy_allocator.h"


void AdamWStateManager::begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream) {
    LazyAllocator alloc;
    if (!mUseZeroCopy && (mOffloadM || mOffloadV)) {
        // double buffering needs to block on main_stream, so we can be sure that the stack memory can be reused safely
        CUDA_CHECK(cudaEventRecord(mStatus.at(0).DoneEvent, main_stream));
        CUDA_CHECK(cudaEventRecord(mStatus.at(1).DoneEvent, main_stream));
    }

    if(mOffloadM && !mUseZeroCopy) {
        alloc.allocate(get_m_buffer(0));
        mMBufferStorage[0] = alloc.commit(memory, "opt_m_a");
        alloc.allocate(get_m_buffer(1));
        mMBufferStorage[1] = alloc.commit(memory, "opt_m_b");
    }

    if(mOffloadV && !mUseZeroCopy) {
        alloc.allocate(get_v_buffer(0));
        mVBufferStorage[0] = alloc.commit(memory, "opt_v_a");
        alloc.allocate(get_v_buffer(1));
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
        auto& ref = mMBlockStorage.at(layer_idx);

        fetch(buf, ref);
    }

    if(mOffloadV) {
        auto& buf = mVBufferStorage.at(buffer);
        auto& ref = mVBlockStorage.at(layer_idx);

        fetch(buf, ref);
    }

    CUDA_CHECK(cudaEventRecord(stat.DoneEvent, fetch_stream));
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
        CUDA_CHECK(cudaMemcpyAsync(mMBlockStorage.at(layer_idx).Data, mMBufferStorage.at(buffer).Data, mMBlockStorage.at(layer_idx).bytes(), cudaMemcpyDeviceToHost, put_stream));
    }

    if(mOffloadV) {
        CUDA_CHECK(cudaMemcpyAsync(mVBlockStorage.at(layer_idx).Data, mVBufferStorage.at(buffer).Data, mVBlockStorage.at(layer_idx).bytes(), cudaMemcpyDeviceToHost, put_stream));
    }

    if(mOffloadM || mOffloadV) {
        if(stat.LayerIdx != layer_idx) {
            throw std::logic_error("layer index mismatch in store_block");
        }
        CUDA_CHECK(cudaEventRecord(stat.DoneEvent, put_stream));
        stat.Done = true;
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
