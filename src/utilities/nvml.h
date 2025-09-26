// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//

#ifndef LLMQ_SRC_UTILITIES_NVML_H
#define LLMQ_SRC_UTILITIES_NVML_H

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include <nvml.h>

struct GPUUtilInfo {
    unsigned int clock;
    unsigned int max_clock;
    unsigned int power;
    unsigned int power_limit;
    unsigned int fan;
    unsigned int temperature;
    unsigned int temp_slowdown;

    std::size_t mem_free;
    std::size_t mem_total;
    std::size_t mem_reserved;

    float gpu_utilization;
    float mem_utilization;
    const char* throttle_reason;

    std::size_t pcie_rx;    // in bytes/µs
    std::size_t pcie_tx;
};

std::size_t get_mem_reserved();
std::string get_gpu_name();

class GPUUtilTracker {
public:
    GPUUtilTracker();
    ~GPUUtilTracker();
    const GPUUtilInfo& update();
private:
    GPUUtilInfo mInfo;
    nvmlDevice_t mDevice;

    long long mLastTimestamp;       // µs
    std::size_t mLastPCIeRX;
    std::size_t mLastPCIeTX;
    unsigned long long mLastEnergy;

    std::atomic<std::size_t> mIntervalPCIeRX{0};
    std::atomic<std::size_t> mIntervalPCIeTX{0};
    std::atomic<std::size_t> mIntervalEnergy{0};

    std::size_t mTotalPCIeRX;
    std::size_t mTotalPCIeTX;
    std::size_t mTotalEnergy;

    std::jthread mThread;
};

#endif //LLMQ_SRC_UTILITIES_NVML_H
