#include "py_train.h"

#include <filesystem>

#include <fmt/format.h>

#include "utilities/gpu_info.h"
#include "training/checkpoint.h"
#include "training/dataloader.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"

MultiGPUPyTrainer::MultiGPUPyTrainer(int ngpus, LLamaConfig config, LLamaOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv) :
    mConfig(config), mOptions(options), B(batch_size), T(seq_len), mGradAccumulation(grad_accum)
{
    mContexts.resize(ngpus);
    mThreads = NCCLCommunicator::launch_threads_communicators(
       ngpus, memcpy_all_gather, memcpy_send_recv,
       [&](NCCLCommunicator& comm) {
           this->main_loop(comm);
       });

    while(!mIsRunning) {
        std::this_thread::yield();
    }
}

MultiGPUPyTrainer::~MultiGPUPyTrainer() {
    mIsRunning = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void MultiGPUPyTrainer::import_weights(std::string path) {
    run_work([path](sThreadContext& ctx) {
        ctx.Model->import_weights(path, true, *ctx.Communicator);
    });
}


void MultiGPUPyTrainer::export_model(std::string path) {
    run_work([path](sThreadContext& ctx) {
        std::filesystem::path p(path);
        std::filesystem::create_directories(p);
        save_llama_config(ctx.Model->config(), (p / "config.json").c_str());
        ctx.Model->export_weights((p / "model.safetensors").c_str(), *ctx.Communicator);
    });
}

void MultiGPUPyTrainer::init_weights() {
    run_work([](sThreadContext& ctx) {
        ctx.Model->init_weights(*ctx.Communicator);
    });
}

void MultiGPUPyTrainer::load_checkpoint(std::string directory, int step) {
    run_work([directory, step](sThreadContext& ctx) {
        ::load_checkpoint(directory, step, *ctx.Model, nullptr, *ctx.Communicator);
    });
}

void MultiGPUPyTrainer::save_checkpoint(std::string directory, int step) {
    run_work([directory, step](sThreadContext& ctx) {
        ::save_checkpoint(directory, step, *ctx.Model, nullptr, *ctx.Communicator);
    });
}

void MultiGPUPyTrainer::step(const std::int32_t* inputs, const std::int32_t* targets) {
    for(int i = 0; i < mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
    }

    if(mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(fmt::format("step: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }

    run_work([micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation](sThreadContext& ctx) {
        Tensor inputs = ctx.Model->get_input_buffer();
        Tensor targets = ctx.Model->get_target_buffer();
        ctx.Model->forward(inputs, *ctx.Communicator, micro_idx);
        ctx.Model->backward(inputs, targets, *ctx.Communicator, micro_batches, micro_idx);
    });
    ++mTrainMicroStep;
}

float MultiGPUPyTrainer::validate(const std::int32_t* inputs, const std::int32_t* targets) {
    for(int i = 0; i < mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
    }

    float loss;

    run_work([micro_idx = mEvalStep, &loss](sThreadContext& ctx) {
        Tensor inputs = ctx.Model->get_input_buffer();
        Tensor targets = ctx.Model->get_target_buffer();
        float calc_loss = ctx.Model->validate(inputs, targets, *ctx.Communicator, micro_idx);
        if (ctx.Communicator->rank() == 0) {
            loss = calc_loss;
        }
    });

    ++mEvalStep;

    return loss;
}



std::pair<float, float> MultiGPUPyTrainer::update(float lr, float beta1, float beta2, int step, float weight_decay, float grad_clip) {
    run_work([=](sThreadContext& ctx) {
        ctx.Model->update(*ctx.Communicator, lr, beta1, beta2, step + 1, 1e-8f, weight_decay, grad_clip);
        CUDA_CHECK(cudaDeviceSynchronize());

    });
    float step_loss, step_norm;

    auto& ctx = mContexts.at(0);
    step_loss = ctx.Model->get_loss();
    step_norm = ctx.Model->get_norm();

    // ensure we're re-gathering on next forward for eval and train
    mTrainMicroStep = 0;
    mEvalStep = 0;

    return {step_loss / B / T / mGradAccumulation, step_norm};
}


std::vector<GPUUtilInfo> MultiGPUPyTrainer::get_gpu_info() {
    std::vector<GPUUtilInfo> infos(mContexts.size());
    run_work([&](sThreadContext& ctx) {
         infos[ctx.Communicator->rank()] = ctx.GPUUtil->update();
    });
    return infos;
}


void MultiGPUPyTrainer::stop() {
    mIsRunning = false;
}

auto MultiGPUPyTrainer::fetch_work(sThreadContext& ctx) -> std::function<void(sThreadContext & ctx)> {
    std::lock_guard<std::mutex> lock(mGlobalMutex);
    if (!ctx.Work) {
        std::this_thread::yield();
        return {};
    } else {
        auto work = std::move(ctx.Work);
        return work;
    }
}

void MultiGPUPyTrainer::run_work(std::function<void(sThreadContext & ctx)> work) {
    {
        std::lock_guard<std::mutex> lock(mGlobalMutex);
        mWorkDone = 0;

        for (auto& ctx: mContexts) {
            ctx.Work = work;
        }
    }

    while(mWorkDone.load() < mContexts.size()) {
        std::this_thread::yield();
    }
}

void MultiGPUPyTrainer::main_loop(NCCLCommunicator& comm) {
    sThreadContext& ctx = mContexts.at(comm.rank());

    ctx.Communicator = &comm;
    ctx.GPUUtil = IGPUUtilTracker::create();
    setup_cublas();
    ctx.Model = std::make_unique<LLamaModel>(mConfig, mOptions, comm.rank(), comm.world_size());
    ctx.Model->allocate_run_state(mOptions, comm, B, T);

    if(mIsReady.fetch_add(1) == comm.world_size() - 1) {
        mIsRunning = true;
    };

    while (!mIsRunning) {
        std::this_thread::yield();
    }

    while (mIsRunning) {
        if (auto work = fetch_work(ctx); work) {
            work(ctx);
            mWorkDone.fetch_add(1);
        } else {
            std::this_thread::yield();
        }
    }
}

int MultiGPUPyTrainer::world_size() const {
    return mContexts.at(0).Communicator->world_size();
}

