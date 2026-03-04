// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor.h"
#include "tensor_container.h"

#include <iostream>
#include <vector>
#include <fmt/core.h>

#include <cuda_fp8.h>

Tensor slice(const Tensor& src, int dim, long start, long end) {
    if (dim != 0)
        throw std::logic_error("Slices must be contiguous, so only the first dimension can be sliced.");

    if (start >= src.Sizes[dim] || end > src.Sizes[dim])
        throw std::logic_error("Slice out of bounds.");

    std::array<long, MAX_TENSOR_DIM> strides{};

    for (int i = src.Rank; i < MAX_TENSOR_DIM; ++i)
        strides[i] = 0;

    strides[src.Rank - 1] = 1;
    for (int i = src.Rank - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * src.Sizes[i + 1];

    Tensor dst = src;
    dst.Sizes[dim] = end - start;
    std::ptrdiff_t offset = start * strides[dim] * get_dtype_size(src.DType);
    dst.Data = src.Data + offset;
    return dst;
}

void fill_zero(Tensor& dst, cudaStream_t stream) {
    if(dst.empty()) return;
    CUDA_CHECK(cudaMemsetAsync(dst.Data, 0, dst.bytes(), stream));
}

template <class TargetType>
TargetType Tensor::at(long index) const {
    TargetType result;
    CUDA_CHECK(cudaMemcpy(&result, get<TargetType>() + index, sizeof(TargetType), cudaMemcpyDeviceToHost));
    return result;
}

namespace {
template <class TrueType, class PrintType>
void do_print(const Tensor& tensor, long offset, long count) {
    std::ios_base::fmtflags old_flags{std::cout.flags()};

    auto sz = get_dtype_size(tensor.DType);
    std::vector<TrueType> host_buffer(count);
    CUDA_CHECK(cudaMemcpy(host_buffer.data(), tensor.Data + offset * sz, count * sz, cudaMemcpyDeviceToHost));
    if constexpr (std::is_same_v<TrueType, std::byte>)
        std::cout << std::hex;
    for (long i = 0; i < count; ++i)
        std::cout << (PrintType)host_buffer[i] << " ";
    std::cout << std::endl;
    std::cout.flags(old_flags);
}
} // namespace

void throw_dtype_mismatch(ETensorDType expected, ETensorDType actual) {
    throw std::invalid_argument(fmt::format("Expected dtype {}, got {}", dtype_to_str(expected), dtype_to_str(actual)));
}

void Tensor::print_sample(long offset, long count) const {
    switch (DType) {
    case ETensorDType::FP32:
        do_print<float, float>(*this, offset, count);
        break;
    case ETensorDType::BF16:
        do_print<nv_bfloat16, float>(*this, offset, count);
        break;
    case ETensorDType::FP16:
        do_print<half, float>(*this, offset, count);
        break;
    case ETensorDType::FP8_E4M3:
        do_print<__nv_fp8_e4m3, float>(*this, offset, count);
        break;
    case ETensorDType::FP8_E5M2:
        do_print<__nv_fp8_e5m2, float>(*this, offset, count);
        break;
    case ETensorDType::INT32:
        do_print<int, int>(*this, offset, count);
        break;
    case ETensorDType::INT8:
        do_print<int8_t, int>(*this, offset, count);
        break;
    case ETensorDType::BYTE:
        do_print<std::byte, int>(*this, offset, count);
        break;
    }
}

TensorShard::TensorShard(const Tensor& src) : Tensor(src), GlobalShape(src.Sizes), ShardIndex(0), NumShards(1) {
}

std::size_t TensorShard::global_nelem() const {
    std::size_t sz = 1;
    for (int i = 0; i < Rank; ++i)
        sz *= GlobalShape[i];
    return sz;
}

std::ptrdiff_t TensorShard::shard_offset() const {
    return nelem() * ShardIndex;
}

TensorShard shard_view(const Tensor& src, int idx, int num) {
    Tensor shard{src};
    shard.Sizes[0] = div_exact(src.Sizes[0], static_cast<long>(num));
    if (!src.empty()) {
        shard.Data = src.Data + div_exact(src.bytes(), static_cast<std::size_t>(num)) * idx;
    }
    return TensorShard{shard, idx, num, src.Sizes};
}

void visit(const std::function<void(Tensor&)>& func, SimpleTensorContainer& container) {
    auto cs = container.num_tensors();
    for(std::size_t i = 0; i < cs; ++i) {
        auto& t = container.get_tensor(i);
        if(t) {
            func(t);
        }
    }
}

void visit(const std::function<void(Tensor&, Tensor&)>& func, SimpleTensorContainer& a, SimpleTensorContainer& b) {
    if(a.num_tensors() != b.num_tensors()) {
        throw std::invalid_argument(fmt::format("TensorContainer size mismatch: {} != {}", a.num_tensors(), b.num_tensors()));
    }

    auto cs = a.num_tensors();
    for(std::size_t i = 0; i < cs; ++i) {
        auto& t1 = a.get_tensor(i);
        auto& t2 = b.get_tensor(i);
        if(t1.empty() != t2.empty()) {
            throw std::invalid_argument(fmt::format("TensorContainer structure mismatch at tensor {}", i));
        }
        if(t1 && t2) {
            func(t1, t2);
        }
    }
}

GenericTensorContainer::GenericTensorContainer(std::vector<Tensor> t): mTensors( std::move(t) ) {
}

std::size_t GenericTensorContainer::num_tensors() const noexcept {
    return mTensors.size();
}

const Tensor& GenericTensorContainer::get_tensor(std::size_t idx) const {
    return mTensors.at(idx);
}

GenericTensorContainer shard_empty_container(GenericTensorContainer&& c, int world) {
    // can't use visit here, because we explicitly want to iterate over empty tensors
    for (std::size_t i = 0; i < c.num_tensors(); ++i) {
        auto& t = c.get_tensor(i);
        if (!t.empty()) { throw std::logic_error("shard_empty_container called with non-empty tensor"); }
        t.Sizes[0] = div_exact(t.Sizes[0], static_cast<long>(world));
    }
    return std::move(c);
}

GenericTensorContainer shard_view(const GenericTensorContainer& c, int rank, int world) {
    std::vector<Tensor> shards(c.num_tensors());
    for (std::size_t i = 0; i < c.num_tensors(); ++i) {
        shards.at(i) = static_cast<Tensor>(shard_view(c.get_tensor(i), rank, world));
    }
    return GenericTensorContainer{shards};
}
