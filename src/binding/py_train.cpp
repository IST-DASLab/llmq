// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "py_train.h"

#include <filesystem>

#include <fmt/format.h>

#include "utilities/gpu_info.h"
#include "training/checkpoint.h"
#include "training/dataloader.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"
#include "models/llama_gradients.h"
#include "models/llama_run_state.h"

MultiGPUPyTrainer::MultiGPUPyTrainer(int ngpus, TransformerConfig config, LLamaOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv) :
    mConfig(config), mOptions(options), B(batch_size), T(seq_len), mGradAccumulation(grad_accum)
{
    int gpus_available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpus_available));
    if (ngpus == 0) {
        ngpus = gpus_available;
    }

    if (ngpus > gpus_available) {
        throw std::runtime_error(fmt::format("Requested {} GPUs, only {} available", ngpus, gpus_available));
    }
    mContexts.resize(ngpus);
    mThreads = NCCLCommunicator::launch_threads_communicators(
       ngpus, memcpy_all_gather, memcpy_send_recv,
       [&](NCCLCommunicator& comm) {
           try {
               this->main_loop(comm);
           } catch (...) {
               mHasCrashed = true;
               throw;
           }
       });

    while(!mIsRunning && !mHasCrashed) {
        std::this_thread::yield();
    }
}

MultiGPUPyTrainer::~MultiGPUPyTrainer() {
    mIsRunning = false;

    // make sure all work has finished
    for(auto& ctx : mContexts) {
        if(ctx.Communicator) {
            CUDA_CHECK(cudaSetDevice(ctx.Communicator->rank()));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mThreads->join();
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
        save_transformer_config(ctx.Model->config(), (p / "config.json").c_str());
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

void MultiGPUPyTrainer::step(const std::int32_t* inputs, const std::int32_t* targets, float z_loss) {
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

    run_work([micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation, z_loss](sThreadContext& ctx) {
        Tensor inputs = ctx.Model->get_input_buffer();
        Tensor targets = ctx.Model->get_target_buffer();
        ctx.Model->forward(inputs, *ctx.Communicator, micro_idx);
        ctx.Model->backward(inputs, targets, *ctx.Communicator, z_loss, micro_batches, micro_idx);
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
        auto calc_loss = ctx.Model->validate(inputs, targets, *ctx.Communicator, micro_idx);
        if (ctx.Communicator->rank() == 0) {
            loss = calc_loss.first;
        }
    });

    ++mEvalStep;

    return loss;
}



std::tuple<float, float, float, float> MultiGPUPyTrainer::update(float lr, float beta1, float beta2, int step, float weight_decay, float grad_clip) {
    run_work([=](sThreadContext& ctx) {
        ctx.Model->update(*ctx.Communicator, lr, beta1, beta2, step + 1, 1e-8f, weight_decay, grad_clip);
        CUDA_CHECK(cudaDeviceSynchronize());

    });
    float step_loss, step_norm;

    auto& ctx = mContexts.at(0);
    step_loss = ctx.Model->get_loss();
    step_norm = ctx.Model->get_norm();
    float logit_lse_max = ctx.Model->get_run_state().get_lse_max();
    float logit_lse_mean = ctx.Model->get_run_state().get_lse_sum() / ( B * T * mGradAccumulation );

    // ensure we're re-gathering on next forward for eval and train
    mTrainMicroStep = 0;
    mEvalStep = 0;

    return {step_loss / B / T / mGradAccumulation, step_norm, logit_lse_max, logit_lse_mean};
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

void MultiGPUPyTrainer::run_work(std::function<void(sThreadContext & ctx)> work, int idx) {
    {
        std::lock_guard<std::mutex> lock(mGlobalMutex);

        if (idx >= 0) {
            mWorkDone = mContexts.size() - 1;
            mContexts.at(idx).Work = work;
        } else {
            mWorkDone = 0;
            for (auto& ctx: mContexts) {
                ctx.Work = work;
            }
        }
    }

    while(mWorkDone.load() < mContexts.size()) {
        if(mThreads->has_exception()) {
            stop();
            mThreads->join(); // will throw, ending the loop
        }
        std::this_thread::yield();
    }
}

void MultiGPUPyTrainer::main_loop(NCCLCommunicator& comm) {
    sThreadContext& ctx = mContexts.at(comm.rank());

    ctx.Communicator = &comm;
    ctx.GPUUtil = IGPUUtilTracker::create();
    ctx.Model = std::make_unique<LLamaModel>(mConfig, mOptions, comm.rank(), comm.world_size());
    ctx.Model->allocate_run_state(mOptions, comm, B, T);

    if (mIsReady.fetch_add(1) == comm.world_size() - 1) {
        mIsRunning = true;
    };

    while (!mIsRunning.load()) {
        std::this_thread::yield();
        if(mHasCrashed.load()) throw std::runtime_error("Another worker has crashed, exiting.");
    }

    while (mIsRunning.load()) {
        if (auto work = fetch_work(ctx); work) {
            work(ctx);
            mWorkDone.fetch_add(1);
        } else {
            std::this_thread::yield();
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();

    // free resources
    ctx.Model.reset();
    ctx.GPUUtil.reset();
    CUDA_CHECK(cudaDeviceSynchronize());
}

int MultiGPUPyTrainer::world_size() const {
    return mContexts.at(0).Communicator->world_size();
}

std::vector<std::pair<std::string, sSegmentMemory>> MultiGPUPyTrainer::get_allocations(int gpu_id) {
    std::vector<std::pair<std::string, sSegmentMemory>> result;
    run_work([&result](sThreadContext& ctx) {
        result = ctx.Model->get_allocator().get_allocation_segments();
    }, gpu_id);
    return result;
}

std::vector<std::pair<std::string, long>> MultiGPUPyTrainer::get_stack_info(int gpu_id) {
    std::vector<std::pair<std::string, long>> result;
    run_work([&result](sThreadContext& ctx) {
        result = ctx.Model->run_state().Stack.get_allocation_stats();
    }, gpu_id);
    return result;
}

std::vector<std::pair<std::string, Tensor>> MultiGPUPyTrainer::get_gradients(int gpu_id) {
    std::vector<std::pair<std::string, Tensor>> result;
    run_work([&result](sThreadContext& ctx) {
        const auto& config = ctx.Model->config();
        auto& grads = ctx.Model->grads();
        CUDA_CHECK(cudaDeviceSynchronize());
        result.emplace_back("model.embed_tokens.weight", grads.get_embeddings_shard(nullptr));
        if (!config.TiedWordEmbeddings) {
            result.emplace_back("lm_head.weight", grads.get_lmhead_shard(nullptr));
        }
        result.emplace_back("model.norm.weight", grads.get_lnf_w_shard(nullptr));
        for (int l = 0; l < config.NumLayers; l++) {
            using namespace LLamaWeightID;
            std::string prefix = "model.layers." + std::to_string(l);
            auto& block = grads.get_block_shard(l, nullptr);
            result.emplace_back(prefix + ".self_attn.qkv.weight", block.get_tensor(QKV_W));
            if (block.get_tensor(QKV_B))
                result.emplace_back(prefix + ".self_attn.qkv.bias", block.get_tensor(QKV_B));
            result.emplace_back(prefix + ".self_attn.o_proj.weight", block.get_tensor(ATTO_W));
            result.emplace_back(prefix + ".mlp.up.weight", block.get_tensor(UP_W));
            result.emplace_back(prefix + ".mlp.down_proj.weight", block.get_tensor(DOWN_W));
            result.emplace_back(prefix + ".input_layernorm.weight", block.get_tensor(LN1_W));
            result.emplace_back(prefix + ".post_attention_layernorm.weight", block.get_tensor(LN2_W));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }, gpu_id);
    return result;
}
