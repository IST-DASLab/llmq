// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// This file implements a fallback path, to be used in cases in which cuFile is not available.
//

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string_view>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <fmt/core.h>

#include "cu_file.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"


cuFileRef open_cufile(std::string file_name) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(fmt::format("posix open error ({}) for file {}: {}", errno, file_name, strerror(errno)));
    }

    return {nullptr, fd, std::move(file_name)};
}

void cufile_read_bytes(int fd, std::byte* d_target, std::ptrdiff_t begin, std::ptrdiff_t end, std::string_view file_name) {
    if(end < begin) {
        throw std::logic_error(fmt::format("Invalid range {} - {} in cufile_read_bytes for {}", begin, end, file_name));
    }

    const size_t nbytes = static_cast<size_t>(end - begin);

    constexpr size_t CHUNK = 1 << 20;
    void* hbuf = nullptr;
    CUDA_CHECK(cudaMallocHost(&hbuf, CHUNK));

    size_t done = 0;
    while (done < nbytes) {
        const size_t want = std::min(CHUNK, nbytes - done);
        const off_t off = static_cast<off_t>(begin + done);
        ssize_t r = ::pread(fd, hbuf, want, off);
        if (r < 0) {
            cudaFreeHost(hbuf);
            throw std::runtime_error(fmt::format("posix pread error ({}) for {}, range {} - {}",
                                                 errno, file_name, off, off + want));
        }
        if (r == 0) break;

        auto ce = cudaMemcpy(reinterpret_cast<void*>(d_target + done),
                        hbuf, static_cast<size_t>(r),
                        cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            cudaFreeHost(hbuf);
            throw std::runtime_error(fmt::format("cudaMemcpy failed: {}",
                                                 cudaGetErrorString(ce)));
        }
        done += static_cast<size_t>(r);
    }

    cudaFreeHost(hbuf);

    if (done != nbytes) {
        throw std::runtime_error(fmt::format("posix read short: expected {} bytes, got {}",
                                             nbytes, done));
    }
}

void convert_tensor_dispatch(std::byte* target, const std::byte* source, std::size_t size, ETensorDType t_type, ETensorDType s_type) {
    if(t_type == ETensorDType::FP32 && s_type == ETensorDType::BF16) {
        convert_dtype(reinterpret_cast<float*>(target), reinterpret_cast<const nv_bfloat16*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP32) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const float*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP16) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const half*>(source), size);
    } else {
        throw std::runtime_error("Unsupported conversion");
    }
}

void cufile_convert_tensor(int fd, std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end,
                           std::string_view file_name, ETensorDType t_type, ETensorDType s_type,
                           std::byte* d_buffer, std::size_t buffer_size) {
    for(std::ptrdiff_t p = 0; p < end - begin; p += buffer_size) {
        std::ptrdiff_t amount = std::min(end - begin - p, (std::ptrdiff_t)buffer_size);
        cufile_read_bytes(fd, d_buffer, begin + p, begin + p + amount, file_name);
        convert_tensor_dispatch(target + p * get_dtype_size(t_type) / get_dtype_size(s_type),
                                d_buffer,
                                amount / get_dtype_size(s_type),
                                t_type, s_type);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}



cuFileRef::cuFileRef(std::string file_name) : cuFileRef(open_cufile(std::move(file_name)))
{

}

cuFileRef::~cuFileRef() noexcept
{
    if (mFileDescriptor >= 0) {
        close(mFileDescriptor);
        mFileDescriptor = -1;
    }
}

void cuFileRef::read_bytes(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end)
{
    cufile_read_bytes(mFileDescriptor, target, begin, end, mFileName);
}

void cuFileRef::read_and_convert(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end,
        std::string_view file_name, ETensorDType t_type, ETensorDType s_type,
        std::byte* d_buffer, std::size_t buffer_size)
{
    cufile_convert_tensor(mFileDescriptor, target, begin, end, file_name, t_type, s_type, d_buffer, buffer_size);
}
