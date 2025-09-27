// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <format>
#include <string_view>

#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include "cu_file.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

static inline bool cufile_disabled_env() {
    const char* e = std::getenv("CUFILE_DISABLE");
    return e && *e && std::string(e) != "0";
}

static inline void posix_pread_to_device(std::string_view file_name,
                                         std::byte* d_target,
                                         std::ptrdiff_t begin,
                                         std::ptrdiff_t end)
{
    if (end < begin) {
        throw std::logic_error(std::format("Invalid range {} - {} in cufile_read_bytes for {}",
                                           begin, end, file_name));
    }

    const size_t nbytes = static_cast<size_t>(end - begin);

    int fd = ::open(std::string(file_name).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(std::format("posix open error ({}) for {}: {}",
                                             errno, file_name, strerror(errno)));
    }

    constexpr size_t CHUNK = 1 << 20;
    void* hbuf = nullptr;
    cudaError_t ce = cudaMallocHost(&hbuf, CHUNK);
    if (ce != cudaSuccess) {
        ::close(fd);
        throw std::runtime_error(std::format("cudaMallocHost failed: {}",
                                             cudaGetErrorString(ce)));
    }

    size_t done = 0;
    while (done < nbytes) {
        const size_t want = std::min(CHUNK, nbytes - done);
        const off_t off = static_cast<off_t>(begin + done);
        ssize_t r = ::pread(fd, hbuf, want, off);
        if (r < 0) {
            cudaFreeHost(hbuf);
            ::close(fd);
            throw std::runtime_error(std::format("posix pread error ({}) for {}, range {} - {}",
                                                 errno, file_name, off, off + want));
        }
        if (r == 0) break;

        ce = cudaMemcpy(reinterpret_cast<void*>(d_target + done),
                        hbuf, static_cast<size_t>(r),
                        cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            cudaFreeHost(hbuf);
            ::close(fd);
            throw std::runtime_error(std::format("cudaMemcpy failed: {}",
                                                 cudaGetErrorString(ce)));
        }
        done += static_cast<size_t>(r);
    }

    cudaFreeHost(hbuf);
    ::close(fd);

    if (done != nbytes) {
        throw std::runtime_error(std::format("posix read short: expected {} bytes, got {}",
                                             nbytes, done));
    }
}

static inline CUfileHandle_t try_register_cufile_handle(int fd) {
    if (cufile_disabled_env()) return CUfileHandle_t{nullptr};

    (void) cuFileDriverOpen();

    CUfileDescr_t descr{};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t h{};
    CUfileError_t st = cuFileHandleRegister(&h, &descr);
    if (st.err != CU_FILE_SUCCESS) {
        return CUfileHandle_t{nullptr};
    }
    return h;
}

cuFileRef open_cufile(std::string file_name) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(std::format("cufile file open error ({}) for file {}: {}", errno, file_name, strerror(errno)));
    }

    CUfileHandle_t handle = try_register_cufile_handle(fd);

    return {handle, fd, std::move(file_name)};
}

void cufile_read_bytes(CUfileHandle_t handle, std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end, std::string_view file_name) {
    if(end < begin) {
        throw std::logic_error(std::format("Invalid range {} - {} in cufile_read_bytes for {}", begin, end, file_name));
    }

    if (cufile_disabled_env() || handle == nullptr) {
        posix_pread_to_device(file_name, target, begin, end);
        return;
    }

    ssize_t ret = cuFileRead(handle, target, end - begin, begin, 0);
    if (ret < 0) {
        if (ret == -1) {
            throw std::runtime_error(
                    std::format("cufile read error ({}) for file {}, range {} - {}: {}", errno, file_name,
                                begin, end, strerror(errno)));
        } else {
            throw std::runtime_error(
                    std::format("cufile read error ({}) for file {}, range {} - {}", -ret, file_name, begin,
                                end));
        }
    } else if (ret != end - begin) {
        throw std::runtime_error(std::format("cufile read error for file {}: expected {} bytes, got {}", file_name, end-begin, ret));
    }
}


template<typename Src, typename Dst>
__global__ void convert_tensor_kernel(Dst* target, const Src* source, std::size_t size) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= size) {
        return;
    }
    target[tid] = static_cast<Dst>(source[tid]);
}

template<typename Src, typename Dst>
void convert_tensor_launcher(Dst* target, const Src* source, std::size_t size) {
    unsigned long n_blocks = div_ceil(size, 128ul);
    convert_tensor_kernel<Src, Dst><<<n_blocks, 128>>>(target, source, size);
}

void convert_tensor_dispatch(std::byte* target, const std::byte* source, std::size_t size, ETensorDType t_type, ETensorDType s_type) {
    if(t_type == ETensorDType::FP32 && s_type == ETensorDType::BF16) {
        convert_tensor_launcher(reinterpret_cast<float*>(target), reinterpret_cast<const nv_bfloat16*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP32) {
        convert_tensor_launcher(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const float*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP16) {
        convert_tensor_launcher(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const half*>(source), size);
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
    if (mFileDescriptor >= 0) {
        close(mFileDescriptor);
        mFileDescriptor = -1;
    }
    if (mHandle) {
        cuFileHandleDeregister(mHandle);
        mHandle = nullptr;
    }
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
