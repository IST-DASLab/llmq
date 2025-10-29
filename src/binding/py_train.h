#ifndef HALO_CPP_SRC_BINDING_PY_TRAIN_H
#define HALO_CPP_SRC_BINDING_PY_TRAIN_H

#include <string>
#include <utility>
#include <thread>
#include <functional>

#include "models/llama_config.h"
#include "models/llama_model.h"

class DataLoader;
class IGPUUtilTracker;
struct GPUUtilInfo;

class MultiGPUPyTrainer
{
public:
    MultiGPUPyTrainer(int ngpus, LLamaConfig config, LLamaOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv);
    ~MultiGPUPyTrainer();

    void import_weights(std::string path);
    void init_weights();
    void load_checkpoint(std::string directory, int step);
    void save_checkpoint(std::string directory, int step);
    void step(const std::int32_t* inputs, const std::int32_t* targets);
    float validate(const std::int32_t* inputs, const std::int32_t* targets);
    std::pair<float, float> update(float lr, float beta1, float beta2, int step, float weight_decay, float grad_clip);
    void stop();

    std::vector<GPUUtilInfo> get_gpu_info();

    int world_size() const;
    int batch_size() const { return B; }
    int seq_length() const { return T; }

private:
    LLamaConfig mConfig;
    LLamaOptions mOptions;
    int B;
    int T;

    int mMicroStep = 0;
    int mGradAccumulation = 1;

    std::vector<std::jthread> mThreads;
    struct sThreadContext {
        NCCLCommunicator* Communicator;
        std::unique_ptr<LLamaModel> Model;
        std::unique_ptr<IGPUUtilTracker> GPUUtil;
        std::function<void(sThreadContext& ctx)> Work;
    };
    std::vector<sThreadContext> mContexts;
    std::mutex mGlobalMutex;
    std::atomic<bool> mIsRunning = false;
    std::atomic<int> mIsReady = 0;
    std::atomic<int> mWorkDone = 0;

    std::function<void(sThreadContext& ctx)> fetch_work(sThreadContext& ctx);
    void run_work(std::function<void(sThreadContext& ctx)> work);
    void main_loop(NCCLCommunicator& comm);
};


#endif //HALO_CPP_SRC_BINDING_PY_TRAIN_H
