// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_UTILITIES_STACK_H
#define LLMQ_SRC_UTILITIES_STACK_H

#include <cstddef>
#include <vector>
#include "utilities/tensor.h"

class DeviceMemoryStack {
public:
    DeviceMemoryStack() = default;
    DeviceMemoryStack(std::byte* memory, std::size_t amount, int device_id);

    std::byte* allocate(std::size_t amount);
    Tensor allocate(ETensorDType dtype, const std::vector<long>& shape);

    void free(std::byte* ptr);
    void free(Tensor& tensor);

    std::size_t unused_capacity() const;
    std::size_t max_utilization() const;
    int device_id() const;
private:
    int mDeviceID;
    std::byte* mBackingMemory;
    std::byte* mTop;
    std::size_t mCapacity;

    using AllocationList = std::vector<std::pair<std::byte*, std::size_t>>;
    AllocationList mAlloc;

    std::size_t mMaxUtilization = 0;
};

#endif //LLMQ_SRC_UTILITIES_STACK_H
