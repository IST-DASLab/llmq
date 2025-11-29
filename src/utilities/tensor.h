// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_UTILS_TENSOR_H
#define LLMQ_SRC_UTILS_TENSOR_H

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "dtype.h"
#include "utils.h"

constexpr int MAX_TENSOR_DIM = 5;

//! \brief The Tensor class represents a contiguous view on memory that is associated
//! with a specific data type and shape.
struct Tensor {
    ETensorDType DType;
    std::array<long, MAX_TENSOR_DIM> Sizes;
    std::byte* Data = nullptr;
    float* Stats = nullptr;
    int Rank = 0;
    int Device = -1;

    [[nodiscard]] constexpr std::size_t bytes() const {
        return nelem() * get_dtype_size(DType);
    }

    [[nodiscard]] constexpr std::size_t nelem() const {
        std::size_t sz = 1;
        for(int i = 0; i < Rank; ++i) {
            sz *= Sizes[i];
        }
        return sz;
    }

    //! this is a debugging function, copying the requested element from the GPU to the CPU
    //! for easier printing. Do **not** use it for anything else!
    template<class TargetType>
    TargetType at(long index) const;

    //! this is a debugging function, printing a few consecutive elements of the tensor.
    //! Do **not** use it for anything else!
    void print_sample(long offset, long count=10) const;

    template<class TargetType>
    [[nodiscard]] constexpr const TargetType* get() const {
        if(dtype_from_type<TargetType> != DType) {
            throw std::logic_error("DType mismatch");
        }

        return reinterpret_cast<const TargetType*>(Data);
    }

    template<class TargetType>
    [[nodiscard]] constexpr TargetType* get() {
        if(dtype_from_type<TargetType> != DType) {
            throw std::logic_error("DType mismatch");
        }

        return reinterpret_cast<TargetType*>(Data);
    }


    template<typename Container>
    static Tensor from_pointer(std::byte* ptr, int device, ETensorDType dtype, const Container& shape)
    {
        if(shape.size() > MAX_TENSOR_DIM) {
            throw std::runtime_error("Tensor rank too large");
        }

        int rank = narrow<int>(shape.size());
        std::array<long, MAX_TENSOR_DIM> sizes{};
        std::copy(shape.begin(), shape.end(), sizes.begin());
        std::fill(sizes.begin() + shape.size(), sizes.end(), 1);

        return Tensor{dtype, sizes, ptr, nullptr, rank, device};
    }

    float* abs_max() {
        return Stats;
    }

    float* scale() {
        if(Stats == nullptr)
            return nullptr;
        return Stats + 1;
    }
};

void fill_zero(Tensor& dst, cudaStream_t stream);
Tensor slice(const Tensor& src, int dim, long start, long end);

class TensorShard : public Tensor {
public:
    TensorShard() = default;
    TensorShard(const Tensor& src);   // implicit

    template<typename Container>
    TensorShard(const Tensor& src, int idx, int num, const Container& global_shape)
        : Tensor(src), GlobalShape{}, ShardIndex(idx), NumShards(num) {
        std::copy(global_shape.begin(), global_shape.end(), GlobalShape.begin());
        std::fill(GlobalShape.begin() + global_shape.size(), GlobalShape.end(), 1);

        if(global_nelem() != src.nelem() * NumShards) {
            throw std::logic_error("Invalid global shape");
        }
    }


    std::size_t global_nelem() const;
    std::ptrdiff_t shard_offset() const;

    std::array<long, MAX_TENSOR_DIM> GlobalShape;
    int ShardIndex;
    int NumShards;
};

TensorShard shard_view(const Tensor& src, int idx, int num);
#endif //LLMQ_SRC_UTILS_TENSOR_H
