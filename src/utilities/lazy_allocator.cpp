// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_allocator.h"
#include "utilities/tensor.h"
#include "utilities/allocator.h"
#include "utilities/stack.h"

void LazyAllocator::allocate(Tensor* target, ETensorDType dtype, const std::vector<long>& shape) {
    target->Data = nullptr;
    target->Rank = shape.size();
    target->DType = dtype;
    std::copy(shape.begin(), shape.end(), target->Sizes.begin());
    mTargets.push_back(target);
}

Tensor LazyAllocator::commit(TensorAllocator& storage, EAllocationType type, const char* name) {
    std::size_t total_size = 0;
    constexpr std::size_t page_size = 4096;
    for(auto& target: mTargets) {
        std::size_t tgt_size = div_ceil(target->bytes(), page_size) * page_size;
        total_size += tgt_size;
    }

    Tensor backing = storage.allocate(ETensorDType::BYTE, name, type, {(long)total_size});
    auto* ptr = backing.get<std::byte>();
    for(auto& target: mTargets) {
        target->Data = ptr;
        target->Device = backing.Device;
        ptr += div_ceil(target->bytes(), page_size) * page_size;
    }

    mTargets.clear();

    return backing;
}

Tensor LazyAllocator::commit(DeviceMemoryStack& storage, const char* name) {
    std::size_t total_size = 0;
    constexpr std::size_t page_size = 4096;
    for(auto& target: mTargets) {
        std::size_t tgt_size = div_ceil(target->bytes(), page_size) * page_size;
        total_size += tgt_size;
    }

    Tensor backing = storage.allocate(ETensorDType::BYTE, {(long)total_size}, name);
    // we may run the allocator in "tracing" mode, where we don't get any memory allocated from `storage`.
    // in that case, `backing` may be nullptr, and we should leave target tensors empty
    if (backing) {
        auto* ptr = backing.get<std::byte>();
        for(auto& target: mTargets) {
            target->Data = ptr;
            target->Device = backing.Device;
            ptr += div_ceil(target->bytes(), page_size) * page_size;
        }
    }

    mTargets.clear();

    return backing;
}
