// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//

#include "utils.h"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>

void cuda_throw_on_error(cudaError_t status, const char* statement, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("Cuda Error in ") + file + ":" + std::to_string(line) + " (" + std::string(statement) + "): " + cudaGetErrorName(status) + ": ";
        msg += cudaGetErrorString(status);
        // make sure we have a clean cuda error state before launching the exception
        // otherwise, if there are calls to the CUDA API in the exception handler,
        // they will return the old error.
        [[maybe_unused]] cudaError_t clear_error = cudaGetLastError();
        throw cuda_error(status, msg);
    }
}

NvtxRange::NvtxRange(const char* s) noexcept { nvtxRangePush(s); }
NvtxRange::NvtxRange(const std::string& base_str, int number) {
    std::string range_string = base_str + " " + std::to_string(number);
    nvtxRangePush(range_string.c_str());
}
NvtxRange::~NvtxRange() noexcept { nvtxRangePop(); }

cudaStream_t create_named_stream(const char* name) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    nvtxNameCudaStreamA(stream, name);
    return stream;
}

cudaEvent_t create_named_event(const char* name) {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    nvtxNameCudaEventA(event, name);
    return event;
}
