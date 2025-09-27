// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "nvml.h"

#include <format>
#include <type_traits>
#include <cstdlib>

#include "utils.h"

static inline bool halo_disable_nvml() {
    const char* e = std::getenv("HALO_DISABLE_NVML");
    return e && *e && std::string(e) != "0";
}

inline void nvml_check(nvmlReturn_t status, const char *file, int line) {
    if (status == NVML_SUCCESS) return;
    if (status == NVML_ERROR_NOT_SUPPORTED || halo_disable_nvml()) return;
    throw std::runtime_error(std::format("[NVML ERROR] at file {}:{}:\n{}\n", file, line, nvmlErrorString(status)));
}
#define NVML_CHECK(err) (nvml_check(err, __FILE__, __LINE__))

inline nvmlDevice_t nvml_get_device() {
    thread_local bool needs_init = true;
    thread_local nvmlDevice_t device{};
    if (needs_init) {
        needs_init = false;

        if (halo_disable_nvml()) {
            return nullptr;
        }
        nvmlReturn_t st = nvmlInit();
        if (st == NVML_ERROR_NOT_SUPPORTED) {
            return nullptr;
        }
        NVML_CHECK(st);

        char bus_id[256];
        int did;
        CUDA_CHECK(cudaGetDevice(&did));
        CUDA_CHECK(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), did));
        nvmlReturn_t hst = nvmlDeviceGetHandleByPciBusId(bus_id, &device);
        if (hst == NVML_ERROR_NOT_SUPPORTED) {
            return nullptr;
        }
        NVML_CHECK(hst);
    }
    return device;
}

inline const char* get_throttle_reason(unsigned long long bits) {
    if(bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        return "power cap";
    } else if (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        return "thermal cap";
    } else if (bits & (nvmlClocksThrottleReasonGpuIdle)) {
        return "idle";
    } else if (bits & (nvmlClocksThrottleReasonAll)) {
        return "other cap";
    } else {
        return "no cap";
    }
}

GPUUtilTracker::GPUUtilTracker() : mDevice(nvml_get_device()) {
    if (!mDevice) return;

    nvmlFieldValue_t fields[] = {{NVML_FI_DEV_PCIE_COUNT_RX_BYTES}, {NVML_FI_DEV_PCIE_COUNT_TX_BYTES}, {NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION, 0}};
    NVML_CHECK(nvmlDeviceGetFieldValues(mDevice, 3, fields));
    NVML_CHECK(fields[0].nvmlReturn);
    NVML_CHECK(fields[1].nvmlReturn);
    NVML_CHECK(fields[2].nvmlReturn);

    mLastPCIeRX = fields[0].value.uiVal;
    mLastPCIeTX = fields[1].value.uiVal;
    mLastEnergy = fields[2].value.ullVal;

    mLastTimestamp = std::chrono::steady_clock::now().time_since_epoch().count();

    // TODO should this be one thread for all devices?
    mThread = std::jthread([this](std::stop_token stop_token)
    {
        NVML_CHECK(nvmlDeviceSetCpuAffinity(mDevice));

        nvmlFieldValue_t fields[] = {{NVML_FI_DEV_PCIE_COUNT_RX_BYTES}, {NVML_FI_DEV_PCIE_COUNT_TX_BYTES}, {NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION, 0}};
        while (true) {
            NVML_CHECK(nvmlDeviceGetFieldValues(mDevice, 3, fields));
            NVML_CHECK(fields[0].nvmlReturn);
            NVML_CHECK(fields[1].nvmlReturn);
            NVML_CHECK(fields[2].nvmlReturn);
            unsigned pcie_rx, pcie_tx;
            unsigned long long energy;
            if (mLastPCIeRX <= fields[0].value.uiVal) {
               pcie_rx = fields[0].value.uiVal - mLastPCIeRX;
            } else {
               pcie_rx = std::numeric_limits<unsigned>::max() - mLastPCIeRX + fields[0].value.uiVal;
            }

            if (mLastPCIeTX <= fields[1].value.uiVal) {
               pcie_tx = fields[1].value.uiVal - mLastPCIeTX;
            } else {
               pcie_tx = std::numeric_limits<unsigned>::max() - mLastPCIeTX + fields[1].value.uiVal;
            }

            if (mLastEnergy <= fields[2].value.ullVal) {
               energy = fields[2].value.ullVal - mLastEnergy;
            } else {
               energy = (std::numeric_limits<unsigned long long>::max() - mLastEnergy) + fields[2].value.ullVal;
            }

            mIntervalPCIeRX += pcie_rx;
            mIntervalPCIeTX += pcie_tx;
            mIntervalEnergy += energy;
            mLastPCIeRX = fields[0].value.uiVal;
            mLastPCIeTX = fields[1].value.uiVal;
            mLastEnergy = fields[2].value.ullVal;

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (stop_token.stop_requested()) {
                break;
            }
        }
    });
}

GPUUtilTracker::~GPUUtilTracker() {
    mThread.request_stop();
}

const GPUUtilInfo& GPUUtilTracker::update() {
    if (!mDevice) return mInfo;

    // query different infos directly
    NVML_CHECK(nvmlDeviceGetClockInfo(mDevice, NVML_CLOCK_SM, &mInfo.clock));
    NVML_CHECK(nvmlDeviceGetMaxClockInfo(mDevice, NVML_CLOCK_SM, &mInfo.max_clock));
    if (const nvmlReturn_t status = nvmlDeviceGetPowerManagementLimit(mDevice, &mInfo.power_limit); status == NVML_ERROR_NOT_SUPPORTED) {
        mInfo.power_limit = 0;
    } else if (status != NVML_SUCCESS) {
        NVML_CHECK(status);
    }

    NVML_CHECK(nvmlDeviceGetTemperature(mDevice, NVML_TEMPERATURE_GPU, &mInfo.temperature));
    NVML_CHECK(nvmlDeviceGetTemperatureThreshold(mDevice, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &mInfo.temp_slowdown));
    unsigned long long throttle;
    NVML_CHECK(nvmlDeviceGetCurrentClocksThrottleReasons(mDevice, &throttle));
    mInfo.throttle_reason = get_throttle_reason(throttle);
    if (const nvmlReturn_t status = nvmlDeviceGetFanSpeed(mDevice, &mInfo.fan); status == NVML_ERROR_NOT_SUPPORTED) {
        mInfo.fan = 0;
    } else if (status != NVML_SUCCESS) {
        NVML_CHECK(status);
    }

    // For "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr const int BUFFER_LIMIT = 128;
    nvmlSample_t buffer[BUFFER_LIMIT];
    nvmlValueType_t v_type;
    unsigned int sample_count = BUFFER_LIMIT;
    NVML_CHECK(nvmlDeviceGetSamples(mDevice, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer));
    float gpu_utilization = 0.f;
    for(unsigned i = 0; i < sample_count; ++i) {
        gpu_utilization += (float)buffer[i].sampleValue.uiVal;
    }
    gpu_utilization /= (float)sample_count;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_count = BUFFER_LIMIT;
    NVML_CHECK(nvmlDeviceGetSamples(mDevice, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer));
    float mem_utilization = 0.f;
    for(unsigned i = 0; i < sample_count; ++i) {
        mem_utilization += (float)buffer[i].sampleValue.uiVal;
    }
    mem_utilization /= (float)sample_count;

    mInfo.gpu_utilization = gpu_utilization;
    mInfo.mem_utilization = mem_utilization;
    nvmlMemory_v2_t mem_info;
    mem_info.version = nvmlMemory_v2;
    NVML_CHECK(nvmlDeviceGetMemoryInfo_v2(mDevice, &mem_info));
    mInfo.mem_free = mem_info.free;
    mInfo.mem_total = mem_info.total;
    mInfo.mem_reserved = mem_info.reserved;

    // query PCIe info
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto interval = std::chrono::duration_cast<std::chrono::microseconds>(now - std::chrono::steady_clock::duration{mLastTimestamp}).count();

    std::size_t int_rx = mIntervalPCIeRX.exchange(0);
    std::size_t int_tx = mIntervalPCIeTX.exchange(0);
    std::size_t int_eg = mIntervalEnergy.exchange(0);

    mInfo.pcie_rx = (1'000'000ull * int_rx) / interval;
    mInfo.pcie_tx = (1'000'000ull * int_tx) / interval;
    // not using nvmlDeviceGetPowerUsage, because that is the past 1sec average, so it might not be representative
    mInfo.power = (1'000'000ull * int_eg) / interval;

    mTotalPCIeRX += int_rx;
    mTotalPCIeTX += int_tx;
    mTotalEnergy += int_eg;
    mLastTimestamp = now.count();

    return mInfo;
}

std::size_t get_mem_reserved() {
    if (halo_disable_nvml()) return 0;
    nvmlDevice_t dev = nvml_get_device();
    if (!dev) return 0;
    nvmlMemory_v2_t mem_info;
    mem_info.version = nvmlMemory_v2;
    NVML_CHECK(nvmlDeviceGetMemoryInfo_v2(dev, &mem_info));
    return mem_info.reserved;
}

std::string get_gpu_name() {
    if (halo_disable_nvml()) {
        cudaDeviceProp prop;
        int deviceId;
        if (cudaGetDevice(&deviceId) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
            return prop.name;
        }
        return "Unknown GPU";
    }
    nvmlDevice_t dev = nvml_get_device();
    if (!dev) {
        cudaDeviceProp prop;
        int deviceId;
        if (cudaGetDevice(&deviceId) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
            return prop.name;
        }
        return "Unknown GPU";
    }
    char name[256];
    NVML_CHECK(nvmlDeviceGetName(dev, name, 256));
    return name;
}
