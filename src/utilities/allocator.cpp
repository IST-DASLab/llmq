// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "allocator.h"

#include <iostream>
#include <unordered_map>

#include <fmt/core.h>

#include "utilities/gpu_info.h"

TensorAllocator::TensorAllocator(TensorAllocator&&) noexcept = default;
TensorAllocator& TensorAllocator::operator=(TensorAllocator&&) noexcept = default;

struct sTotalAllocations {
    long ON_DEVICE = 0;
    long MANAGED = 0;
    long PINNED = 0;
    long ON_HOST = 0;
    long WRITE_CMB = 0;

    long& operator[](EAllocationType kind)
    {
        switch (kind) {
            case EAllocationType::ON_DEVICE: return ON_DEVICE;
            case EAllocationType::MANAGED: return MANAGED;
            case EAllocationType::PINNED: return PINNED;
            case EAllocationType::WRITE_CMB: return WRITE_CMB;
            case EAllocationType::ON_HOST: return ON_HOST;
            default: throw std::logic_error("Unknown allocation type");
        }
    }
};

struct TensorAllocator::sAllocStats
{
    std::string Context = "";
    std::unordered_map<std::string, sTotalAllocations> TensorStats;
    std::unordered_map<std::string, sTotalAllocations> ContextStats;
};

template<typename Container>
Tensor allocate_tensor(ETensorDType dtype, EAllocationType kind, const Container& shape)
{
    if(shape.size() > MAX_TENSOR_DIM) {
        throw std::runtime_error("Tensor rank too large");
    }

    int did;
    CUDA_CHECK(cudaGetDevice(&did));

    int rank = narrow<int>(shape.size());
    std::size_t total = std::accumulate(std::begin(shape), std::end(shape), 1l, std::multiplies<>());
    std::byte* ptr;
    if(kind == EAllocationType::ON_DEVICE) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), total * get_dtype_size(dtype)));
    } else if(kind == EAllocationType::MANAGED) {
        CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&ptr), total * get_dtype_size(dtype)));
    } else if(kind == EAllocationType::PINNED) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr), total * get_dtype_size(dtype)));
        did = -1;
    }  else if(kind == EAllocationType::WRITE_CMB) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr), total * get_dtype_size(dtype), cudaHostAllocWriteCombined | cudaHostAllocMapped));
        did = -1;
    } else {
        ptr = new std::byte[total * get_dtype_size(dtype)];
        did = -1;
    }
    std::array<long, MAX_TENSOR_DIM> sizes{};
    std::copy(shape.begin(), shape.end(), sizes.begin());
    std::fill(sizes.begin() + shape.size(), sizes.end(), 1);

    return Tensor{dtype, sizes, ptr, nullptr, rank, did};
}

void record_stats(std::unordered_map<std::string, sTotalAllocations>& target, std::string name, EAllocationType kind, long bytes) {
    target[name][kind] += narrow<long>(bytes);
}

Tensor TensorAllocator::allocate(ETensorDType dtype, const char* name, EAllocationType kind, const std::initializer_list<long>& shape) {
    return allocate_impl(dtype, name, kind, shape);
}

Tensor TensorAllocator::allocate(ETensorDType dtype, const char* name, EAllocationType kind, const std::vector<long>& shape) {
    return allocate_impl(dtype, name, kind, shape);
}

Tensor TensorAllocator::allocate(ETensorDType dtype, const char* name, const std::initializer_list<long>& shape) {
    return allocate_impl(dtype, name, EAllocationType::ON_DEVICE, shape);
}

Tensor TensorAllocator::allocate(ETensorDType dtype, const char* name, const std::vector<long>& shape) {
    return allocate_impl(dtype, name, EAllocationType::ON_DEVICE, shape);
}

TensorShard TensorAllocator::allocate_shard(ETensorDType dtype, int shard_idx, int num_shards, const char* name, const std::vector<long>& shape,  EAllocationType kind) {
    std::vector<long> shard_shape(shape);
    shard_shape[0] = div_exact(shape[0], (long)num_shards);
    return TensorShard(allocate(dtype, name, kind, shard_shape), shard_idx, num_shards, shape);
}

template<typename Container>
Tensor TensorAllocator::allocate_impl(ETensorDType dtype, const char* name, EAllocationType kind, const Container& shape) {
    try {
        Tensor allocated = allocate_tensor(dtype, kind, shape);
        m_Pointers.emplace_back(kind, allocated.Data, allocated.bytes());
        record_stats(m_Stats->TensorStats, name, kind, allocated.bytes());
        if (!m_Stats->Context.empty()){
            record_stats(m_Stats->ContextStats, m_Stats->Context, kind, allocated.bytes());
        }
        if(mCallback) {
            mCallback(m_Stats->Context, name, kind, allocated.bytes());
        }
        return allocated;
    } catch (const cuda_error& error) {
        if(error.code == cudaErrorMemoryAllocation) {
            print_stats();
            std::string shape_str = "[";
            for(auto s: shape) {
                shape_str += std::to_string(s) + ", ";
            }
            shape_str.pop_back();
            shape_str.pop_back();
            shape_str.push_back(']');
            std::string message = fmt::format("Cuda OOM when allocating tensor {} of shape {} with dtype {} in context {}",
                                              name, shape_str, dtype_to_str(dtype), m_Stats->Context);
            throw std::runtime_error(message);
        }
        throw;
    }
}

TensorAllocator::TensorAllocator() : m_Stats(std::make_unique<sAllocStats>()) {
}

TensorAllocator::~TensorAllocator() noexcept {
    CUDA_CHECK(cudaDeviceSynchronize());
    int did;
    CUDA_CHECK(cudaGetDevice(&did));
    for (auto& ptr: m_Pointers) {
        try {
            switch (ptr.Kind) {
                case EAllocationType::ON_DEVICE:
                case EAllocationType::MANAGED:
                    CUDA_CHECK(cudaFree(ptr.Pointer));
                    break;
                case EAllocationType::WRITE_CMB:
                case EAllocationType::PINNED:
                    CUDA_CHECK(cudaFreeHost(ptr.Pointer));
                    break;
                case EAllocationType::ON_HOST:
                    delete[] ptr.Pointer;
                    break;
            }
        } catch (const cuda_error& error) {
            const char* kind_str;
            switch (ptr.Kind) {
                case EAllocationType::ON_DEVICE: kind_str = "device"; break;
                case EAllocationType::MANAGED: kind_str = "managed"; break;
                case EAllocationType::PINNED: kind_str = "pinned"; break;
                case EAllocationType::ON_HOST: kind_str = "host"; break;
                default: kind_str = "unknown"; break;
            }
            fprintf(stderr, "Cuda error on device %d when deleting allocation %p [%s of size %ld]: %s\n", did, ptr.Pointer, kind_str, ptr.Size, error.what());
            fflush(stderr);
            std::terminate();
        }
    }
}

void TensorAllocator::print_stats() const {
    for (auto& [name, amount]: m_Stats->TensorStats)
    {
        if (amount.ON_DEVICE < total_allocation() / 1024) continue;       // skip tiny tensors
        if (amount.ON_DEVICE >= 1024 * 1024 * 20) {
            std::cerr << name << ": " << amount.ON_DEVICE / 1024 / 1024 << " MiB\n";
        } else {
            std::cerr << name << ": " << amount.ON_DEVICE / 1024 << " KiB\n";
        }
    }
}

std::size_t TensorAllocator::total_allocation() const {
    std::size_t total = 0;
    for(const auto& ptr: m_Pointers) {
        total += ptr.Size;
    }
    return total;
}

std::size_t TensorAllocator::total_allocation(EAllocationType kind) const {
    std::size_t total = 0;
    for(const auto& ptr: m_Pointers) {
        if(ptr.Kind == kind) {
            total += ptr.Size;
        }
    }
    return total;
}

void TensorAllocator::set_context(const std::string& ctx) {
    m_Stats->Context = ctx;
}

const std::string& TensorAllocator::get_context() const {
    return m_Stats->Context;
}

TensorAllocator::AllocationMonitor::AllocationMonitor(const std::string& name, TensorAllocator* alloc) :
    mName(name), mAllocator(alloc), mParent(alloc->get_context()) {
    alloc->set_context(mName);
}

TensorAllocator::AllocationMonitor::~AllocationMonitor() {
    if (mAllocator->get_context() != mName) {
        throw std::runtime_error("AllocationMonitor: Improper nesting");
    }
    mAllocator->set_context(mParent);
}

std::vector<std::pair<std::string, sSegmentMemory>> TensorAllocator::get_allocation_segments() const {
    long sum = 0;
    std::vector<std::pair<std::string, sSegmentMemory>> segments;
    for (const auto& [name, amount]: m_Stats->ContextStats) {
        segments.emplace_back(name, sSegmentMemory{amount.ON_DEVICE, amount.MANAGED, amount.PINNED + amount.WRITE_CMB, amount.ON_HOST});
        sum += amount.ON_DEVICE;
    }
    std::size_t free = 0;
    std::size_t total = 0;
    long reserved = get_mem_reserved();
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    segments.emplace_back("Free", sSegmentMemory{(long)free, 0, 0, 0});
    if(reserved > 0) {
        segments.emplace_back("Reserved", sSegmentMemory{reserved, 0, 0, 0});
    }
    segments.emplace_back("Other", sSegmentMemory{(long)total - (long)free - sum - reserved, 0, 0, 0});
    return segments;
}

void TensorAllocator::set_callback(std::function<void(const std::string&, const std::string&, EAllocationType, std::size_t)> cb) {
    mCallback = std::move(cb);
}
