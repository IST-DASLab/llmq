// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_TRAINING_LOGGING_H
#define LLMQ_TRAINING_LOGGING_H

#include <fstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <functional>

struct GPUUtilInfo;
struct sSegmentMemory;
class NCCLCommunicator;
class DataLoader;

class TrainingRunLogger
{
public:
    enum EVerbosity {
        SILENT = -2,
        QUIET = -1,
        DEFAULT = 0,
        VERBOSE = 1
    };

    TrainingRunLogger(const std::string& file_name, int rank, EVerbosity verbosity);
    ~TrainingRunLogger();

    void set_expected_time_per_token(long nanoseconds);
    void set_callback(std::function<void(std::string_view)> cb);

    void log_cmd(int argc, const char** argv);
    void log_options(const std::vector<std::pair<std::string_view, std::variant<bool, int, float, std::string>>>& options);
    void log_gpu_model(NCCLCommunicator& comm);
    void log_dataset(const DataLoader& train_loader, const DataLoader& eval_loader);
    void log_step(int step, float epoch, int step_tokens, int duration_ms, float norm, float loss, float lr);
    void log_eval(int step, float epoch, int eval_tokens, int duration_ms, float loss);
    void log_gpu_state(int step, int gpu_id, const GPUUtilInfo& gpu_util);
    void log_allocator(const std::vector<std::pair<std::string, sSegmentMemory>>& stats);

    void log_checkpoint(int step, std::string path, int duration_ms);
private:
    void log_line(std::string_view line);
    std::string mFileName;
    std::fstream mLogFile;
    bool mFirst = true;

    int mRank;
    EVerbosity mVerbosity;

    // running mean for training loss
    double mTotalTrainingLoss = 0.0;
    int mTotalTrainingSteps = 0;

    // to estimate ETA
    int mRemainingTokens = -1;

    // to estimate MFU
    long mExpectedTimePerToken = -1;

    // arbitrary callback for log lines
    std::function<void(std::string_view)> mCallback;
};

#endif //LLMQ_TRAINING_LOGGING_H
