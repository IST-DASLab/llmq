// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <fmt/core.h>

#include "cu_file.h"

#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

cuFileRef open_cufile(std::string file_name) {
    auto status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cufile driver open error");
    }

    int fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        throw std::runtime_error(fmt::format("cufile file open error ({}) for file {}: {}", errno, file_name, strerror(errno)));
    }

    // create cufile handle
    CUfileDescr_t descr;
    CUfileHandle_t handle;
    std::memset(&descr, 0, sizeof(CUfileDescr_t));
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cufile register error");
    }
    return {handle, fd, std::move(file_name)};
}

void cufile_read_bytes(CUfileHandle_t handle, std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end, std::string_view file_name) {
    if(end < begin) {
        throw std::logic_error(fmt::format("Invalid range {} - {} in cufile_read_bytes for {}", begin, end, file_name));
    }
    ssize_t ret = cuFileRead(handle, target, end - begin, begin, 0);
    if (ret < 0) {
        if (ret == -1) {
            throw std::runtime_error(
                    fmt::format("cufile read error ({}) for file {}, range {} - {}: {}", errno, file_name,
                                begin, end, strerror(errno)));
        } else {
            throw std::runtime_error(
                    fmt::format("cufile read error ({}) for file {}, range {} - {}", -ret, file_name, begin,
                                end));
        }
    } else if (ret != end - begin) {
        throw std::runtime_error(fmt::format("cufile read error for file {}: expected {} bytes, got {}", file_name, end-begin, ret));
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

void cufile_convert_tensor(CUfileHandle_t handle, std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end,
                           std::string_view file_name, ETensorDType t_type, ETensorDType s_type,
                           std::byte* d_buffer, std::size_t buffer_size) {
    for(std::ptrdiff_t p = 0; p < end - begin; p += buffer_size) {
        std::ptrdiff_t amount = std::min(end - begin - p, (std::ptrdiff_t)buffer_size);
        cufile_read_bytes(handle, d_buffer, begin + p, begin + p + amount, file_name);
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
    close(mFileDescriptor);
    cuFileHandleDeregister(mHandle);
}

void cuFileRef::read_bytes(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end)
{
    cufile_read_bytes(mHandle, target, begin, end, mFileName);
}

void cuFileRef::read_and_convert(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end,
        std::string_view file_name, ETensorDType t_type, ETensorDType s_type,
        std::byte* d_buffer, std::size_t buffer_size)
{
    cufile_convert_tensor(mHandle, target, begin, end, file_name, t_type, s_type, d_buffer, buffer_size);
}
