// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "stack.h"
#include "utilities/utils.h"

DeviceMemoryStack::DeviceMemoryStack(std::byte* memory, std::size_t amount, int device_id) :
    mBackingMemory(memory), mTop(memory), mDeviceID(device_id), mCapacity(amount) {

}

std::byte* DeviceMemoryStack::allocate(std::size_t amount) {
    constexpr size_t alignment = 4096;
    std::size_t aligned_amount = div_ceil(amount, alignment) * alignment;
    std::byte* new_top = mTop + aligned_amount;
    if(new_top > mBackingMemory + mCapacity) {
        throw std::bad_alloc();
    }

    mAlloc.emplace_back(mTop, aligned_amount);
    mTop = new_top;
    mMaxUtilization = std::max(mMaxUtilization, bytes_used());
    return mAlloc.back().first;
}

Tensor DeviceMemoryStack::allocate(ETensorDType dtype, const std::vector<long>& shape) {
    std::size_t total = std::accumulate(std::begin(shape), std::end(shape), (long)get_dtype_size(dtype), std::multiplies<>());
    return Tensor::from_pointer(allocate(total), mDeviceID, dtype, shape);
}

void DeviceMemoryStack::free(std::byte* ptr) {
    if(mAlloc.empty()) {
        throw std::logic_error("DeviceMemoryStack::free_left called with empty allocation list");
    }
    if(mAlloc.back().first != ptr) {
        throw std::logic_error("DeviceMemoryStack::free_left called with wrong pointer");
    }
    mTop = mAlloc.back().first;
    mAlloc.pop_back();
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
