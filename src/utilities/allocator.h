// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_UTILITIES_ALLOCATOR_H
#define LLMQ_SRC_UTILITIES_ALLOCATOR_H

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>
#include <functional>

#include "tensor.h"

enum class ETensorDType : int;
class Tensor;

enum class EAllocationType : int {
    ON_DEVICE,  // cudaMalloc
    MANAGED,    // cudaMallocManaged
    PINNED,     // cudaMallocHost,
    WRITE_CMB,  // cudaHostAlloc(WriteCombined)
    ON_HOST     // new[]
};

struct sSegmentMemory {
    long OnDevice;
    long Managed;
    long PinnedHost;
    long PageableHost;
};

class TensorAllocator {
public:
    TensorAllocator();
    ~TensorAllocator() noexcept;
    TensorAllocator(TensorAllocator&&) noexcept ;
    TensorAllocator(const TensorAllocator&) = delete;
    TensorAllocator& operator=(TensorAllocator&&) noexcept ;
    TensorAllocator& operator=(const TensorAllocator&) = delete;

    void print_stats() const;
    void set_callback(std::function<void(const std::string& ctx, const std::string& name, EAllocationType kind, std::size_t amount)>);

    Tensor allocate(ETensorDType dtype, const char* name, EAllocationType kind, const std::vector<long>& shape);
    Tensor allocate(ETensorDType dtype, const char* name, EAllocationType kind, const std::initializer_list<long>& shape);

    Tensor allocate(ETensorDType dtype, const char* name, const std::vector<long>& shape);
    Tensor allocate(ETensorDType dtype, const char* name, const std::initializer_list<long>& shape);

    TensorShard allocate_shard(ETensorDType dtype, int shard_idx, int num_shards, const char* name, const std::vector<long>& shape,  EAllocationType kind=EAllocationType::ON_DEVICE);

    std::size_t total_allocation() const;
    std::size_t total_allocation(EAllocationType kind) const;

    void set_context(const std::string& ctx);
    const std::string& get_context() const;

    class AllocationMonitor {
    public:
        AllocationMonitor(const std::string& name, TensorAllocator*);
        ~AllocationMonitor();
    private:
        std::string mName;
        std::string mParent;
        TensorAllocator* mAllocator;
    };

    [[nodiscard]] AllocationMonitor with_context(const std::string& ctx) { return AllocationMonitor(ctx, this); }

    std::vector<std::pair<std::string, sSegmentMemory>> get_allocation_segments() const;
private:

    template<typename Container>
    Tensor allocate_impl(ETensorDType dtype, const char* name, EAllocationType kind, const Container& shape);

    struct sAllocStats;

    struct sAllocationData {
        EAllocationType Kind;
        std::byte* Pointer;
        long Size;
    };

    std::vector<sAllocationData> m_Pointers;
    std::unique_ptr<sAllocStats> m_Stats;

    std::function<void(const std::string& ctx, const std::string& name, EAllocationType kind, std::size_t amount)> mCallback;
};

#endif //LLMQ_SRC_UTILITIES_ALLOCATOR_H
