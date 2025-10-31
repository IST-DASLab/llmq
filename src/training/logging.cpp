// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//

#include "logging.h"

#include <chrono>
#include <cmath>
#include <filesystem>

#include <fmt/core.h>
#include <fmt/chrono.h>

#include "dataloader.h"
#include "utilities/comm.h"
#include "utilities/gpu_info.h"
#include "utilities/utils.h"
#include <iostream>

TrainingRunLogger::TrainingRunLogger(const std::string& file_name, int rank, EVerbosity verbosity) :
    mFileName(std::move(file_name)), mRank(rank), mVerbosity(verbosity)
{
    if(mRank == 0) {
        auto log_path = std::filesystem::path(mFileName).parent_path();
        std::cout << log_path << "\n";
        if (!log_path.empty()) {
            std::filesystem::create_directories(log_path);
        }
        mLogFile.open(mFileName, std::fstream::out);
        mLogFile << "[\n";
        mLogFile << "\n]\n";
    }
}

TrainingRunLogger::~TrainingRunLogger()
{
    if(mLogFile.is_open()) mLogFile.close();
}

void TrainingRunLogger::set_expected_time_per_token(long nanoseconds) {
    mExpectedTimePerToken = nanoseconds;
}

std::string fmt_token_count(long num_tokens) {
    if(num_tokens < 1'000'000 ) {
        return fmt::format("{:4d}k", num_tokens / 1'000);
    } else if(num_tokens < 20'000'000 ) {
        return fmt::format("{:4.1f}M", float(num_tokens / 1'000) / 1000.f);
    } else if(num_tokens < 1'000'000'000 ) {
        return fmt::format("{:4d}M", num_tokens / 1'000'000);
    } else if(num_tokens < 20'000'000'000 ) {
        return fmt::format("{:4.1f}B", float(num_tokens / 1'000'000) / 1000.f);
    } else if(num_tokens < 1'000'000'000'000 ) {
        return fmt::format("{:4d}B", num_tokens / 1'000'000'000);
    } else if(num_tokens < 20'000'000'000'000 ) {
        return fmt::format("{:4.1f}T", float(num_tokens / 1'000'000'000) / 1000.f);
    } else {
        return fmt::format("{:4d}T", num_tokens / 1'000'000'000'000);
    }
}

std::string format_data_loader(const DataLoader& loader, const char* split) {
    return fmt::format(R"(  {{"log": "dataset", "split": "{}", "time": "{}", "step": 0, "files": {}, "tokens": {}, "file_index": {}, "chunk_index": {}, "seed": {}}})",
        split, std::chrono::system_clock::now(), loader.num_files(), loader.num_tokens(), loader.file_index(), loader.chunk_index(), loader.seed());
}

void TrainingRunLogger::log_dataset(const DataLoader& train_loader, const DataLoader& eval_loader) {
    if(mRank != 0) return;
    log_line(format_data_loader(train_loader, "train"));
    log_line(format_data_loader(eval_loader, "eval"));

    if (mVerbosity >= 0) {
        printf("[Dataset]\n");
        
        printf(" train: %s tokens\n", fmt_token_count(train_loader.num_tokens()).c_str());
        for (int i = 0; i < train_loader.num_files(); ++i) {
            if (i < 10 || mVerbosity >= 1) {
                printf("   %s : %10d\n", train_loader.file_name(i).c_str(), train_loader.file_tokens(i));
            }
        }
        printf(" eval: %s tokens\n", fmt_token_count(eval_loader.num_tokens()).c_str());
        for (int i = 0; i < eval_loader.num_files(); ++i) {
            if (i < 10 || mVerbosity >= 1) {
                printf("   %s : %10d\n", eval_loader.file_name(i).c_str(), eval_loader.file_tokens(i));
            }
        }
        printf("\n");
    }
}

void TrainingRunLogger::log_options(const std::vector<std::pair<std::string_view, std::variant<bool, int, float, std::string>>>& options) {
    if(mRank != 0) return;

    int option_length = 0;
    for(auto& [name, value]: options) {
        auto log = [&](auto&& v){
            if(std::is_same_v<std::remove_cvref_t<decltype(v)>, std::string>) {
                log_line(fmt::format(R"(  {{"log": "option", "time": "{}", "step": 0, "name": "{}", "value": "{}"}})",
                                     std::chrono::system_clock::now(), name, v));
            } else {
                log_line(fmt::format(R"(  {{"log": "option", "time": "{}", "step": 0, "name": "{}", "value": {}}})",
                                     std::chrono::system_clock::now(), name, v));
            }
        };
        option_length = std::max(option_length, static_cast<int>(name.size()));
        std::visit(log, value);
    }
    if(mVerbosity >= 0) {
        printf("[Options]\n");
        for(auto& [name, value]: options) {
            printf("  %-*s: ", option_length, name.data());
            std::visit([](auto&& v){ printf("%s\n", fmt::format("{}", v).c_str()); }, value);
        }
        printf("\n");
    }
}

std::string format_tps(long eval_tokens, long duration_ms) {
    if (duration_ms == 0) {
        return "-";
    }

    long tps = 1000ll * eval_tokens / duration_ms;
    if(tps < 100'000) {
        return fmt::format("{:5}", tps);
    } else {
        return fmt::format("{:4}k", tps / 1000);
    }
}

std::string format_time(int duration_ms) {
    if (duration_ms >= 100'000) {
        return fmt::format("{:5d}  s", duration_ms / 1000);
    } else {
        return fmt::format("{:5d} ms", duration_ms);
    }
}

void TrainingRunLogger::log_step(int step, float epoch, int step_tokens, int duration_ms, float norm, float loss, float lr)
{
    if(mRank != 0) return;
    mTotalTrainingLoss += loss;
    ++mTotalTrainingSteps;

    if(mVerbosity >= 0) {
        float iptr;
        float progress = 100.f * std::modf(epoch,  &iptr);
        std::string tps_msg = format_tps(step_tokens, duration_ms);
        std::string time_str = format_time(duration_ms);
        std::string sol_msg = "";

        // speed-of-light
        if (mExpectedTimePerToken > 0) {
            long peak = mExpectedTimePerToken * step_tokens / 1'000'000;
            double ratio = static_cast<double>(peak) / static_cast<double>(duration_ms);
            sol_msg = fmt::format(" | sol {:.1f}%", ratio * 100.0);
        }

        printf("[T] step %5d [%5.1f%%] | time: %s | norm %10f | loss %10f | tps %s%s\n", step, progress, time_str.c_str(), norm, loss, tps_msg.c_str(), sol_msg.c_str());
        fflush(stdout);
    }
    log_line(fmt::format(R"(  {{"log": "step", "time": "{}", "step": {}, "epoch": {}, "step_tokens": {}, "duration_ms": {}, "norm": {}, "loss": {}, "lr": {}}})",
        std::chrono::system_clock::now(), step, epoch, step_tokens, duration_ms, norm, loss, lr ));
}

void TrainingRunLogger::log_eval(int step, float epoch, int eval_tokens, int duration_ms, float loss)
{
    if(mRank != 0) return;
    if(mVerbosity >= -1) {
        float iptr;
        float progress = 100.f * std::modf(epoch,  &iptr);
        float train_avg = static_cast<float>(mTotalTrainingLoss / std::max(mTotalTrainingSteps, 1));
        std::string tps_msg = format_tps(eval_tokens, duration_ms);
        std::string time_str = format_time(duration_ms);
        printf("\x1b[1m[V] step %5d [%5.1f%%] | time: %s | eval %10f | train %9f | tps %s\x1b[22m\n", step, progress, time_str.c_str(), loss, train_avg, tps_msg.c_str());
        fflush(stdout);
    }
    mTotalTrainingLoss = 0;
    mTotalTrainingSteps = 0;
    log_line(fmt::format(R"(  {{"log": "eval", "time": "{}", "step": {}, "epoch": {}, "eval_tokens": {}, "duration_ms": {}, "loss": {}}})",
        std::chrono::system_clock::now(), step, epoch, eval_tokens, duration_ms, loss ));
}

void TrainingRunLogger::log_gpu_state(int step, int gpu_id, const GPUUtilInfo& gpu_util)
{
    log_line(fmt::format(R"(  {{"log": "gpu", "time": "{}", "step": {}, "id": {}, "clock": {}, "max_clock": {}, "fan": {}, "power": {}, "power_limit": {}, "temperature": {}, "temp_slowdown": {}, "gpu_util": {}, "mem_util": {}, "throttle": "{}", "dram_free": {}, "pcie_rx": {}, "pcie_tx": {}}})",
       std::chrono::system_clock::now(), step, gpu_id, gpu_util.clock, gpu_util.max_clock, gpu_util.fan,
       gpu_util.power, gpu_util.power_limit, gpu_util.temperature, gpu_util.temp_slowdown, gpu_util.gpu_utilization,
       gpu_util.mem_utilization, gpu_util.throttle_reason, gpu_util.mem_free, gpu_util.pcie_rx, gpu_util.pcie_tx ));
    if(mVerbosity >= 1) {
        printf("[G] step %5d [gpu %2d] | power %4d W   | temp %3d /%3d°C | clock %4d MHz\n",
               step, gpu_id, gpu_util.power / 1000, gpu_util.temperature, gpu_util.temp_slowdown, gpu_util.clock);
        printf("                        | PCI↓%5d MiB/s| PCI↑%5d MiB/s\n",
               static_cast<int>(gpu_util.pcie_rx / 1024 / 1024),  static_cast<int>(gpu_util.pcie_tx / 1024 / 1024));
    }
}

void TrainingRunLogger::log_gpu_model(NCCLCommunicator& comm)
{
    struct sGPUInfoMessage {
        std::chrono::system_clock::time_point time;
        int rank;
        int device_id;
        cudaDeviceProp prop;
        int driver_version;
        int runtime_version;
        std::size_t mem_free;
        std::size_t mem_total;
        std::size_t mem_reserved;
    } msg;

    msg.time = std::chrono::system_clock::now();
    msg.rank = comm.rank();
    CUDA_CHECK(cudaGetDevice(&msg.device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&msg.prop, msg.device_id));
    CUDA_CHECK(cudaDriverGetVersion( &msg.driver_version ));
    CUDA_CHECK(cudaRuntimeGetVersion( &msg.runtime_version ));
    CUDA_CHECK(cudaMemGetInfo(&msg.mem_free, &msg.mem_total));
    msg.mem_reserved = get_mem_reserved();

    auto all_gpus = comm.host_gather(msg);
    if(mRank == 0) {
        for (auto& d: all_gpus) {
            std::string uuid;
            for (char& byte: d.prop.uuid.bytes) {
                uuid += fmt::format("{:02x}", byte);
            }
            std::string line =
                fmt::format(
                    R"(  {{"log": "gpu-model", "time": "{}", "rank": {}, "step": 0, "id": {}, "name": "{}", "l2_size": {}, "sm_count": {}, "major": {}, "minor": {}, "memory": {}, "free": {}, "reserved": {}, "uuid": "{}", "ecc": {}, "shared_mem": {}, "cuda_driver": {}, "cuda_runtime": {}}})",
                    d.time, d.rank, d.device_id, d.prop.name, d.prop.l2CacheSize, d.prop.multiProcessorCount, d.prop.major,
                    d.prop.minor, d.prop.totalGlobalMem, d.mem_free, d.mem_reserved, uuid,
                    d.prop.ECCEnabled, d.prop.sharedMemPerMultiprocessor, d.driver_version, d.runtime_version
                );
            log_line(line);

            if(mVerbosity >= 1 || (mVerbosity >= 0 && d.rank == 0)) {
                printf("[System %d]\n", d.rank);
                printf("  Device %d: %s\n", d.device_id, d.prop.name);
                printf("  CUDA version: driver %d, runtime %d\n", d.driver_version, d.runtime_version);
                printf("  Memory: %zu MiB / %zu MiB\n", (d.mem_total-d.mem_free) / 1024 / 1024, d.mem_total / 1024 / 1024);
                printf("\n");
            }
        }
    }
}

void TrainingRunLogger::log_cmd(int argc, const char** argv)
{
    if(mRank == 0) return;
    std::string cmd = fmt::format(R"(  {{"log": "cmd", "time": "{}", "step": 0, "cmd": [)", std::chrono::system_clock::now());
    for (int i = 0; i < argc; i++)
    {
        if (i != 0) cmd += ", ";
        cmd += fmt::format("\"{}\"", argv[i]);
    }
    cmd += "]}";
    log_line(cmd);
}

void TrainingRunLogger::log_checkpoint(int step, std::string path, int duration_ms) {
    if(mRank != 0) return;
    log_line(fmt::format(R"(  {{"log": "checkpoint", "time": "{}", "step": {}, "path": "{}", "duration_ms": {}}})",
        std::chrono::system_clock::now(), step, path, duration_ms ));
}

void TrainingRunLogger::log_line(std::string_view line) {
    if(mCallback)
        mCallback(line);

    mLogFile.seekp(-3, std::ios::end);  // overwrite the array closing part
    if (!mFirst)
    {
        mLogFile << ",\n";
    }
    mLogFile << line << "\n]" << std::endl;
    mFirst = false;
}

void TrainingRunLogger::log_allocator(const std::vector<std::pair<std::string, std::size_t>>& stats) {
    if (mRank != 0) return;
    std::string stat_str = "[";
    bool first = true;
    for (auto& [name, amount]: stats) {
        if (!first) stat_str += ", ";
        first = false;
        stat_str += fmt::format("{{\"name\": \"{}\", \"amount\": {}}}", name, amount);
    }
    stat_str += "]";
    std::string line = fmt::format(R"(  {{"log": "allocator", "time": "{}", "step": 0, "stats": {}}})", std::chrono::system_clock::now(), stat_str);
    log_line(line);

    if (mVerbosity >= 0) {
        printf("[Allocator State]\n");
        for (auto& [name, amount]: stats) {
            printf("  %16s: %5zu MiB\n", name.c_str(), amount / 1024 / 1024);
        }
    }
}

void TrainingRunLogger::set_callback(std::function<void(std::string_view)> cb) {
    mCallback = std::move(cb);
}
