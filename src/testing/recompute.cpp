// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//


#include "utilities/safetensors.h"
#include "utilities/gpu_info.h"
#include "utilities/sol.h"
#include "utilities/comm.h"

#include "kernels/kernels.h"

#include "training/logging.h"
#include "training/dataloader.h"
#include "training/schedule.h"
#include "training/checkpoint.h"

#include "models/llama_model.h"
#include <chrono>
#include <CLI/CLI.hpp>

bool lexical_cast(const std::string& input, ETensorDType& output) {
    output = dtype_from_str(input);
    return true;
}

namespace CLI::detail {
    template<>
    constexpr const char* type_name<ETensorDType>() {
        return "DTYPE";
    }
}

struct TrainingRunner {
    int B = 2;
    int T = 1024;

    int MaxSteps = 10;

    float Beta1 = 0.9f;
    float Beta2 = 0.95f;
    float GradClip = 1.0f;
    float WeightDecay = 1.0f;
    int GradAccSteps = 4;

    ETensorDType ModelDType = ETensorDType::BF16;
    std::string ModelRootPath = "Qwen/Qwen2.5-0.5B";

    std::string TrainFile = "data/tiny-shakespeare-qwen/train.bin";

    bool MemcpyAllGather = false;
    bool MemcpySendRecv = false;

    LLamaOptions Options;

    void load_training_config(int argc, const char** argv);
    void launch_training();
    void run_training(NCCLCommunicator& comm);

    std::vector<float> Norms;
    std::vector<float> Losses;
};

void TrainingRunner::load_training_config(int argc, const char** argv) {
    CLI::App app;

    std::string matmul_dtype = "";

    Options.KeepAllActivations = false;
    Options.RecomputeSwiGLu = false;
    Options.RecomputeRMSNorm = false;
    Options.RecomputeFFN = false;
    Options.UseCudaGraphs = false;

    // config

    app.add_option("--model", ModelRootPath, "Path to the huggingface model directory or name of a HF model that is cache locally.");
    app.add_option("--matmul-dtype", matmul_dtype, "Which dtype to use for matmuls. Defaults to model-dtype.");
    app.add_option("--model-dtype", ModelDType, "Which dtype to use for model");
    app.add_option("--train-file", TrainFile, "Tokens for training");
    app.add_option("--grad-accumulation", GradAccSteps, "number of micro-batches per optimizer step");

    // The following options should all result in bit-perfectly identical losses
    app.add_flag("--recompute-swiglu", Options.RecomputeSwiGLu, "Recompute swiglu during the backward pass to save activation memory");
    app.add_flag("--recompute-norm", Options.RecomputeRMSNorm, "Recompute rms-norms during the backward pass to save activation memory");
    app.add_flag("--recompute-ffn", Options.RecomputeFFN, "Recompute the feed-forward block during the backward pass to save activation memory");
    app.add_flag("--recompute-qkv", Options.RecomputeQKV, "Recompute the qkv projections during the backward pass");
    app.add_flag("--recompute-att", Options.RecomputeAtt, "Recompute the attention block during the backward pass");
    app.add_flag("--recompute-block", Options.RecomputeBlock, "Recompute the entire transformer block");
    app.add_flag("--use-cuda-graphs,!--no-use-cuda-graphs", Options.UseCudaGraphs, "Enable/disable use of cuda graphs");

    // These should normally stay at their default values
    app.add_option("--batch,--batch-size", B, "micro-batch size");
    app.add_option("--seq-len,--seq-length", T, "sequence length");
    app.add_option("--steps", MaxSteps, "Number of training steps");
    app.add_option("--beta-1", Beta1, "Beta 1 for Adam");
    app.add_option("--beta-2", Beta2, "Beta 2 for Adam");
    app.add_option("--opt-m-dtype", Options.OptMomentumType, "DType for first-order momentum. FP32 or BF16");
    app.add_option("--opt-v-dtype", Options.OptVarianceType, "DType for second-order momentum. FP32 or BF16");
    app.add_option("--grad-clip", GradClip, "Gradient clipping");
    app.add_option("--weight-decay", WeightDecay, "Weight decay for matrix parameters");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (!std::filesystem::exists(ModelRootPath)) {
        std::string hf_path = get_hf_model_files(ModelRootPath);
        if (hf_path.empty()) {
            throw std::runtime_error("Could not find model files for " + ModelRootPath);
        }
        ModelRootPath = hf_path;
    }

    Options.MatmulType = matmul_dtype.empty() ? std::optional<ETensorDType>{} : dtype_from_str(matmul_dtype);

    if (Options.RecomputeAtt) {
        Options.RecomputeQKV = true;
    }
    if (Options.RecomputeFFN) {
        Options.RecomputeSwiGLu = true;
    }
}

void TrainingRunner::launch_training() {
    // Threads code path -- launch one thread per GPU
    NCCLCommunicator::launch_threads_communicators(
        1, MemcpyAllGather, MemcpySendRecv,
        [&](NCCLCommunicator& comm) {
            run_training(comm);
        });
}

void TrainingRunner::run_training(NCCLCommunicator& comm) {
    std::string config_path = ModelRootPath + "/config.json";
    std::string model_path = ModelRootPath + "/model.safetensors";
    if (!std::filesystem::exists(model_path)) {
        model_path = ModelRootPath + "/model.safetensors.index.json";
    }
    LLamaConfig config = load_llama_config(config_path.c_str(), ModelDType);
    setup_cublas();

    DataLoader train_loader({TrainFile}, B*T, comm.rank(), comm.world_size(), 0x83b45442ull);
    LLamaModel model{config, Options, comm.rank(), comm.world_size()};

    model.allocate_run_state(Options, comm, B, T);
    model.import_weights(model_path, true, comm);

    Tensor inputs = model.get_input_buffer();
    Tensor targets = model.get_target_buffer();

    for (int step = 0; step < 10; ++step) {
        for (int j = 0; j < GradAccSteps; ++j) {
            train_loader.load_batch(inputs, targets);
            model.forward(inputs, comm, j);
            model.backward(inputs, targets, comm, GradAccSteps, j);
        }

        model.update(comm, 1e-5, Beta1, Beta2, step + 1, 1e-8f, WeightDecay, GradClip);
        CUDA_CHECK(cudaDeviceSynchronize());
        Norms.push_back(model.get_norm());
        Losses.push_back(model.get_loss() / (B*T*GradAccSteps));
    }
}

int main(int argc, const char** argv) {
    TrainingRunner runner;
    runner.load_training_config(argc, argv);
    runner.launch_training();

    std::vector<float> losses = runner.Losses;
    std::vector<float> norms = runner.Norms;

    // reset runner to baseline
    runner.Norms.clear();
    runner.Losses.clear();

    runner.Options.RecomputeSwiGLu = false;
    runner.Options.RecomputeRMSNorm = false;
    runner.Options.RecomputeFFN = false;
    runner.Options.RecomputeQKV = false;
    runner.Options.RecomputeAtt = false;
    runner.Options.UseCudaGraphs = false;

    runner.launch_training();

    std::vector<float> ref_losses = runner.Losses;
    std::vector<float> ref_norms = runner.Norms;
    bool all_ok = true;

    printf("losses:\n");
    for (int i = 0; i < losses.size(); ++i) {
        if (ref_losses[i] != losses[i]) {
            printf(" \033[1;31m✗\033[0m step %d: %.10f ≠ %.10f\n", i, losses[i], ref_losses[i]);
            all_ok = false;
        } else {
            printf(" \033[1;32m✓\033[0m step %d: %.10f = %.10f\n", i, losses[i], ref_losses[i]);
        }
    }
    printf("\n");
    printf("norms:\n");
    for (int i = 0; i < norms.size(); ++i) {
        if (ref_norms[i] != norms[i]) {
            printf(" \033[1;31m✗\033[0m step %d: %.10f ≠ %.10f\n", i, norms[i], ref_norms[i]);
            all_ok = false;           
        } else {
            printf(" \033[1;32m✓\033[0m step %d: %.10f = %.10f\n", i, norms[i], ref_norms[i]);
        }
    }

    if (all_ok) {
        printf("\n\033[1;32mPASS\033[0m\n");
        fflush(stdout);
        exit(EXIT_SUCCESS);
    } else {
        printf("\n\033[1;31mFAIL\033[0m\n");
        fflush(stdout);
        exit(EXIT_FAILURE);
    }
}
