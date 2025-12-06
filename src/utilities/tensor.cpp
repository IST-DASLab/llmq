// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor.h"

#include <iostream>
#include <vector>

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
    shard.Data = src.Data + div_exact(src.bytes(), static_cast<std::size_t>(num)) * idx;
    return TensorShard{shard, idx, num, src.Sizes};
}
