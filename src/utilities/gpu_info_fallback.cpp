// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_info.h"

#include "utils.h"

class GPUUtilTrackerFallback : public IGPUUtilTracker {
public:
    GPUUtilTrackerFallback();
    ~GPUUtilTrackerFallback();
    const GPUUtilInfo& update() override;
private:
    GPUUtilInfo mInfo;
    int mDeviceID;
};

std::unique_ptr<IGPUUtilTracker> IGPUUtilTracker::create() {
    return std::make_unique<GPUUtilTrackerFallback>();
}

GPUUtilTrackerFallback::GPUUtilTrackerFallback() {
    CUDA_CHECK(cudaGetDevice(&mDeviceID));
}

GPUUtilTrackerFallback::~GPUUtilTrackerFallback() {
}

const GPUUtilInfo& GPUUtilTrackerFallback::update() {
    // just return fallback values
    mInfo.clock = 0;
    mInfo.max_clock = 0;
    mInfo.power_limit = 0;
    mInfo.temperature = 0;
    mInfo.temp_slowdown = 0;
    mInfo.fan = 0;
    mInfo.gpu_utilization = -1.f;
    mInfo.mem_utilization = -1.f;
    mInfo.throttle_reason = "not supported";
    mInfo.pcie_rx = -1;
    mInfo.pcie_tx = -1;
    mInfo.power = -1;

    CUDA_CHECK(cudaMemGetInfo(&mInfo.mem_free, &mInfo.mem_total));
    mInfo.mem_reserved = -1;
    int clockRateKHz;
    CUDA_CHECK(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, mDeviceID));
    mInfo.max_clock = clockRateKHz / 1000;
    return mInfo;
}

std::size_t get_mem_reserved() {
    return 0;
}

std::string get_gpu_name() {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    return prop.name;
}

bool set_cpu_affinity() {
    return false;
}
