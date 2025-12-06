// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "stack.h"
#include "utilities/utils.h"

DeviceMemoryStack::DeviceMemoryStack(std::byte* memory, std::size_t amount, int device_id) :
    mBackingMemory(memory), mTop(memory), mDeviceID(device_id), mCapacity(amount) {

}

std::byte* DeviceMemoryStack::allocate(std::size_t amount, const char* name) {
    constexpr size_t alignment = 4096;
    std::size_t aligned_amount = div_ceil(amount, alignment) * alignment;
    std::byte* new_top = mTop + aligned_amount;
    if(new_top > mBackingMemory + mCapacity) {
        throw std::bad_alloc();
    }

    mAlloc.emplace_back(mTop, aligned_amount, name);
    mTop = new_top;
    _track_max();
    return mAlloc.back().Pointer;
}

Tensor DeviceMemoryStack::allocate(ETensorDType dtype, const std::vector<long>& shape, const char* name) {
    std::size_t total = std::accumulate(std::begin(shape), std::end(shape), (long)get_dtype_size(dtype), std::multiplies<>());
    return Tensor::from_pointer(allocate(total, name), mDeviceID, dtype, shape);
}

void DeviceMemoryStack::free(std::byte* ptr) {
    if(mAlloc.empty()) {
        throw std::logic_error("DeviceMemoryStack::free_left called with empty allocation list");
    }
    if(mAlloc.back().Pointer != ptr) {
        throw std::logic_error("DeviceMemoryStack::free_left called with wrong pointer");
    }
    mTop = mAlloc.back().Pointer;
    mAlloc.pop_back();
}

std::vector<std::pair<std::string, long>> DeviceMemoryStack::get_allocation_stats() const {
    std::vector<std::pair<std::string, long>> result;
    for (auto& [ptr, amount, name]: get_high_mark()) {
        result.emplace_back(name, amount);
    }
    return result;
}

void DeviceMemoryStack::_track_max() {
    if(bytes_used() > mMaxUtilization) {
        mMaxUtilization = bytes_used();
        mHighMark = mAlloc;
    }
}

std::size_t DeviceMemoryStack::unused_capacity() const {
    return mCapacity - (mTop - mBackingMemory);
}

std::size_t DeviceMemoryStack::bytes_used() const {
    return mCapacity - unused_capacity();
}

std::size_t DeviceMemoryStack::max_utilization() const {
    return mMaxUtilization;
}

void DeviceMemoryStack::free(Tensor& tensor) {
    free(tensor.Data);
    tensor.Data = nullptr;
}

int DeviceMemoryStack::device_id() const {
    return mDeviceID;
}
