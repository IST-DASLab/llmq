// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
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
#include "models/llama_run_state.h"     // only needed for debug timing
#include <chrono>
#include <CLI/CLI.hpp>
#include <fmt/chrono.h>

float run_evaluation(DataLoader& test_loader, LLamaModel& model, TrainingRunLogger& logger, float epoch, int step,
                     NCCLCommunicator& comm, int max_steps, Tensor& inputs, Tensor& targets);

bool lexical_cast(const std::string& input, ETensorDType& output) {
    output = dtype_from_str(input);
    return true;
}

std::string replace(std::string haystack, const std::string& needle, const std::string& replacement) {
    size_t pos = haystack.find(needle);
    while (pos != std::string::npos) {
        haystack.replace(pos, needle.size(), replacement);
        pos = haystack.find(needle, pos + needle.size());
    }
    return std::move(haystack);
}

namespace CLI::detail {
    template<>
    constexpr const char* type_name<ETensorDType>() {
        return "DTYPE";
    }
}

struct TrainingRunner {
    int B = 4;
    int T = 1024;

    int MaxSteps = -1;
    int NumEpochs = 1;

    float LearningRate = 1e-5f;
    int WarmupSteps = 0;
    int CoolDownSteps = 0;
    float FinalLrFraction = 1.f;
    std::string LrScheduleType = "cosine";

    float Beta1 = 0.9f;
    float Beta2 = 0.95f;
    float GradClip = 1.0f;
    float WeightDecay = 0.1f;
    float Epsilon = 1e-8f;
    int GradAccSteps = 4;

    bool FromScratch = false;

    ETensorDType ModelDType = ETensorDType::BF16;
    std::string ModelRootPath = ".";
    std::string RunName = "llmq";

    std::string TrainFile = "tiny-shakespeare-train.bin";
    std::uint64_t TrainLoaderSeed = 0x83b45442ull;
    std::string EvalFile = "tiny-shakespeare-test.bin";
    std::uint64_t EvalLoaderSeed = 0x384b4524ull;
    std::string OutDir = "output/%n";
    std::string CkptDir = "ckpt/%n";
    std::string LogFile = fmt::format("logs/%n-{:%FT%H_%M}.json", std::chrono::system_clock::now());
    int EvalEvery = 100;
    int EvalNumSteps = 100;
    int CkptEvery = 100;
    int CkptToKeep = -1;
    int MajorCkptEvery = -1;
    int LogGPUEvery = 25;

    std::optional<int> ContinueFromCheckpoint;

    int NGPUs = 0;
    bool MemcpyAllGather = false;
    bool MemcpySendRecv = false;

    LLamaOptions Options;

    int LogAllocations = -1;
    int DebugLogAbsMaxes = -1;
    std::chrono::steady_clock::time_point BeginStartup;

    void load_training_config(int argc, const char** argv);
    void launch_training(int argc, const char** argv);
    void run_training(int argc, const char** argv, NCCLCommunicator& comm);
};

void TrainingRunner::load_training_config(int argc, const char** argv) {
    BeginStartup = std::chrono::steady_clock::now();
    CLI::App app;

    std::string matmul_dtype = "";
    std::string gradient_dtype = "";

    Options.KeepAllActivations = false;
    Options.RecomputeSwiGLu = false;
    Options.RecomputeRMSNorm = false;
    Options.RecomputeFFN = false;
    Options.UseCudaGraphs = false;
    int ZeroLevel = 1;

    app.add_option("--model", ModelRootPath, "Path to the huggingface model directory or name of a HF model that is cache locally.");
    auto from_scratch = app.add_flag("--from-scratch", FromScratch, "Train the model from a random initialization");
    app.add_flag("--init-proj-to-zero", Options.InitProjectionsToZero, "Init (ffn.down and att.out) projections to zero, as in modded-nanogpt")->needs(from_scratch);
    app.add_option("--matmul-dtype", matmul_dtype, "Which dtype to use for matmuls. Defaults to model-dtype.");
    app.add_option("--gradient-dtype", gradient_dtype, "Which dtype to use for (activation) gradients. Defaults to matmul-dtype.");
    app.add_option("--model-dtype", ModelDType, "Which dtype to use for model");
    app.add_option("--batch,--batch-size", B, "micro-batch size")->check(CLI::PositiveNumber);
    app.add_option("--seq-len,--seq-length", T, "sequence length")->check(CLI::PositiveNumber);
    app.add_option("--lmhead-chunks", Options.LMHeadChunks, "Run the LM-Head in chunks to avoid materializing the large logit tensor.")->check(CLI::PositiveNumber);
    app.add_option("--attn-bwd-chunks", Options.AttBwdChunks, "Run the attention backward pass in chunks, to avoid having to materialize a large workspace tensor.")->check(CLI::PositiveNumber);

    // debug
    app.add_option("--name", RunName, "Associate a name with this run. This will not influence any computations. You can use %n as part of specifying log, output, and checkpoint file names.");
    app.add_option("--debug-log-allocations", LogAllocations, "Log all memory allocations larger than the given number (in MiB)");
    app.add_flag("--debug-time-breakdown", Options.TriggerTimingEvents, "Log additional timing information");
    app.add_flag("--debug-log-abs-max", DebugLogAbsMaxes, "Log abs-maxes every n steps");

    // optimizer
    app.add_option("--lr,--learning-rate", LearningRate, "Base learning rate")->check(CLI::NonNegativeNumber);
    app.add_option("--warmup", WarmupSteps, "Number of warmup steps.")->check(CLI::NonNegativeNumber);;
    app.add_option("--cooldown", CoolDownSteps, "Number of cool-down steps, using 1-sqrt() to anneal learning rate to zero");
    app.add_option("--final-lr-fraction", FinalLrFraction, "Fraction of base lr to use for the final steps.")->check(CLI::NonNegativeNumber);
    app.add_option("--lr-schedule", LrScheduleType, "Learning rate schedule function: Cosine or Linear");
    app.add_option("--beta-1", Beta1, "Beta 1 for Adam")->check(CLI::NonNegativeNumber);
    app.add_option("--beta-2", Beta2, "Beta 2 for Adam")->check(CLI::NonNegativeNumber);
    app.add_option("--opt-m-dtype", Options.OptMomentumType, "DType for first-order momentum. FP32 or BF16");
    app.add_option("--opt-v-dtype", Options.OptVarianceType, "DType for second-order momentum. FP32 or BF16");
    app.add_option("--grad-accumulation", GradAccSteps, "number of micro-batches per optimizer step")->check(CLI::PositiveNumber);
    app.add_option("--grad-clip", GradClip, "Gradient clipping");
    app.add_option("--weight-decay", WeightDecay, "Weight decay for matrix parameters")->check(CLI::NonNegativeNumber);
    app.add_option("--adam-epsilon", Epsilon, "Epsilon to use for AdamW")->check(CLI::NonNegativeNumber);

    auto steps_opt = app.add_option("--steps", MaxSteps, "Number of training steps");
    app.add_option("--epochs", NumEpochs, "Number of training steps")->excludes(steps_opt)->check(CLI::PositiveNumber);
    app.add_option("--log-gpu-util", LogGPUEvery, "Log the gpu utilization every n steps. Set to 0 to disable.");
    app.add_option("--eval-every-n-steps", EvalEvery, "How many optimizer steps between evaluations");
    app.add_option("--eval-num-steps", EvalNumSteps, "How many batches of eval data to use");

    app.add_option("--train-file", TrainFile, "Tokens for training");
    app.add_option("--train-seed", TrainLoaderSeed, "Seed for the training dataloader");
    app.add_option("--eval-file", EvalFile, "Tokens for validation");
    app.add_option("--eval-seed", EvalLoaderSeed, "Seed for the eval loader");
    app.add_option("--out-dir", OutDir, "Where to save the trained model");
    app.add_option("--checkpoint-dir", CkptDir, "Directory in which to save checkpoints.");
    app.add_option("--ckpt-interval", CkptEvery, "How many optimizer steps between checkpoints");
    app.add_option("--ckpt-keep-n", CkptToKeep, "Clean up old checkpoints, only preserving the latest n.");
    app.add_option("--ckpt-major", MajorCkptEvery, "Make every nth checkpoint a major checkpoint, which does not get cleaned up.");
    auto continue_from_checkpoint = app.add_option("--continue", ContinueFromCheckpoint,
        "Continue from checkpoint. If no number is given, uses the latest checkpoint")->expected(0, 1)->default_str("-1");

    app.add_option("--log-file", LogFile, "Where to save the training log");
    app.add_option("--gpus", NGPUs, "How many GPUs to use for training.");

    //  options
    app.add_flag("--recompute-swiglu", Options.RecomputeSwiGLu, "Recompute swiglu during the backward pass to save activation memory");
    app.add_flag("--recompute-norm", Options.RecomputeRMSNorm, "Recompute rms-norms during the backward pass to save activation memory");
    app.add_flag("--recompute-ffn", Options.RecomputeFFN, "Recompute the feed-forward block during the backward pass to save activation memory");
    app.add_flag("--recompute-qkv", Options.RecomputeQKV, "Recompute the qkv projections during the backward pass");
    app.add_flag("--recompute-att", Options.RecomputeAtt, "Recompute the attention block during the backward pass");
    app.add_flag("--recompute-block", Options.RecomputeBlock, "Recompute the entire transformer block");
    app.add_flag("--offload-residual", Options.OffloadResidual, "Offload the residual of the feed-forward block to host memory");
    app.add_flag("--use-cuda-graphs,!--no-use-cuda-graphs", Options.UseCudaGraphs, "Enable/disable use of cuda graphs");
    auto zero = app.add_option("--zero-level", ZeroLevel, "Zero redundancy level: 1 - sharded optimizer; 2 - sharded gradients; 3 - sharded weights");
    app.add_flag("--offload-master", Options.OffloadMaster, "Store master weights in pinned host memory.");
    auto shard_w = app.add_flag("--shard-weights", Options.ShardWeights, "Shard weights across GPUs")->excludes(zero);
    auto persist = app.add_flag("--persistent-quants", Options.PersistentQuants, "Keep quantized weights around, instead of re-quantizing")->needs(shard_w);
    app.add_flag("--offload-quants", Options.OffloadQuants, "Store quantized weights in pinned host memory.")->needs(persist);
    app.add_flag("--offload-opt-m", Options.OffloadOptM, "Store first-order momentum in pinned host memory.");
    app.add_flag("--offload-opt-v", Options.OffloadOptV, "Store second-order momentum in pinned host memory.");
    app.add_flag("--offload-gradients", Options.OffloadGrads, "Offload gradients to pinned host memory.");
    app.add_flag("--use-zero-copy", Options.UseZeroCopy, "Use ZeroCopy memory access, instead of double-buffered cudaMemcpy, for offloaded optimizer states. On consumer cards, DMA appears to be much slower, whereas on professional cards it is faster.");
    app.add_flag("--shard-gradients", Options.ShardGradients, "Shard gradients across GPUs")->excludes(zero);
    app.add_flag("--memcpy-all-gather", MemcpyAllGather, "Use memcpy to perform all-gathers. Currently only supported by the threads backend.");
    app.add_flag("--memcpy-send-recv", MemcpySendRecv, "Use memcpy to perform send/receive (all-to-all). Currently only supported by the threads backend.");
    app.add_flag("--all-to-all-reduce", Options.UseAllToAllReduce, "Uses an all-to-all-based reduce algorithm. Combine with --memcpy-send-recv.");
    app.add_flag("--write-combined", Options.UseWriteCombined, "Uses write-combined memory for offloaded tensors.");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (!std::filesystem::exists(ModelRootPath)) {
        if (ModelRootPath.find('/') != std::string::npos) {
            std::string hf_path = get_hf_model_files(ModelRootPath);
            if (hf_path.empty()) {
                throw std::runtime_error("Could not find model files for " + ModelRootPath);
            }
            ModelRootPath = hf_path;
        }
    }

    Options.MatmulType = matmul_dtype.empty() ? std::optional<ETensorDType>{} : dtype_from_str(matmul_dtype);
    Options.GradientType = gradient_dtype.empty() ? std::optional<ETensorDType>{} : dtype_from_str(gradient_dtype);

    switch (ZeroLevel) {
    case 0:
        std::cerr << "Warning: ZeRO-level 0 not supported, defaulting to 1" << std::endl;
        break;
    case 1:
        break;
    case 3:
        Options.ShardWeights = true;
        [[fallthrough]];
    case 2:
        Options.ShardGradients = true;
        break;
    default:
        std::cerr << "Warning: Invalid ZeRO-level " << ZeroLevel << std::endl;
    }

    if (Options.RecomputeBlock) {
        Options.RecomputeAtt = true;
        Options.RecomputeFFN = true;
        Options.RecomputeRMSNorm = true;
    }

    if (Options.RecomputeAtt) {
        Options.RecomputeQKV = true;
    }
    if (Options.RecomputeFFN) {
        Options.RecomputeSwiGLu = true;
    }

    LogAllocations *= 1024 * 1024;

    LogFile = replace(LogFile, "%n", RunName);
    OutDir = replace(OutDir, "%n", RunName);
    CkptDir = replace(CkptDir, "%n", RunName);
}

void TrainingRunner::launch_training(int argc, const char** argv) {
    if (const char* mpi_world_size = getenv("OMPI_COMM_WORLD_SIZE"); mpi_world_size) {
        // MPI code path -- this region is entered by all processes, so no additional
        // launching necessary
        NGPUs = std::stoi(mpi_world_size);
        if (NGPUs != 0 && NGPUs != std::stoi(getenv("OMPI_COMM_WORLD_SIZE"))) {
            throw std::runtime_error("Number of GPUs does not match OMPI_COMM_WORLD_SIZE");
        }
        auto comm = NCCLCommunicator::make_mpi_communicator();
        run_training(argc, argv, *comm);
    } else {
        // Threads code path -- launch one thread per GPU
        int gpus_available = 0;
        CUDA_CHECK(cudaGetDeviceCount(&gpus_available));
        if (NGPUs == 0) {
            NGPUs = gpus_available;
        }

        if (NGPUs > gpus_available) {
            std::cerr << "Error: requested " << NGPUs << " GPUs, but only " << gpus_available << " found" << std::endl;
            std::exit(1);
        }
        NCCLCommunicator::launch_threads_communicators(
            NGPUs, MemcpyAllGather, MemcpySendRecv,
            [&](NCCLCommunicator& comm) {
                run_training(argc, argv, comm);
            });
    }
}

TransformerConfig create_config(const std::string& root, bool from_scratch, ETensorDType dtype) {
    std::string config_path = root + "/config.json";
    if(std::filesystem::exists(config_path)) {
        return load_transformer_config(config_path.c_str(), dtype);
    } else if(from_scratch) {
        auto cfg = create_config_from_name(root, dtype);
        return cfg;
    } else {
        std::cerr << "Could not find model config at " << config_path << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void TrainingRunner::run_training(int argc, const char** argv, NCCLCommunicator& comm) {
    // TODO this is a quick hack, implement something better
    // local printf that prints only on rank 0
    auto printf = [enabled = comm.rank() == 0](const char* fmt, const auto& ... args) {
        if (enabled)
            ::printf(fmt, args...);
    };

    std::unique_ptr<IGPUUtilTracker> gpu_util = IGPUUtilTracker::create();
    int total_batch_size = B * T * comm.world_size() * GradAccSteps;

    std::string model_path = ModelRootPath + "/model.safetensors";
    if (!std::filesystem::exists(model_path)) {
        model_path = ModelRootPath + "/model.safetensors.index.json";
    }
    TransformerConfig config = create_config(ModelRootPath, FromScratch, ModelDType);
    Options.ModelType = config.DType;

    TrainingRunLogger logger(LogFile, comm.rank(), TrainingRunLogger::DEFAULT);
    logger.log_cmd(argc, argv);

    DataLoader train_loader({TrainFile}, T, comm.rank(), comm.world_size(), TrainLoaderSeed);
    DataLoader test_loader({EvalFile}, T, comm.rank(), comm.world_size(), EvalLoaderSeed);

    int steps_per_epoch = train_loader.num_tokens() / total_batch_size;
    if (MaxSteps == -1) {
        MaxSteps = steps_per_epoch * NumEpochs;
    }

    logger.log_options({
        {"name",               RunName},
        {"recompute-swiglu",   Options.RecomputeSwiGLu},
        {"recompute-norm",     Options.RecomputeRMSNorm},
        {"recompute-ffn",      Options.RecomputeFFN},
        {"recompute-qkv",      Options.RecomputeQKV},
        {"recompute-att",      Options.RecomputeAtt},
        {"recompute-block",    Options.RecomputeBlock},
        {"lm-head-chunks",     Options.LMHeadChunks},
        {"attn-bwd-chunks",    Options.AttBwdChunks},
        {"offload-master",     Options.OffloadMaster},
        {"offload-quants",     Options.OffloadQuants},
        {"offload-opt-m",      Options.OffloadOptM},
        {"offload-opt-v",      Options.OffloadOptV},
        {"use-zero-copy",      Options.UseZeroCopy},
        {"shard-weights",      Options.ShardWeights},
        {"shard-gradients",    Options.ShardGradients},
        {"persistent-quants",  Options.PersistentQuants},
        {"cuda-graphs",        Options.UseCudaGraphs},
        {"all-to-all-reduce",  Options.UseAllToAllReduce},
        {"memcpy-all-gather",  MemcpyAllGather},
        {"memcpy-send-recv",   MemcpySendRecv},
        {"use-write-combined", Options.UseWriteCombined},
        {"matmul-dtype",       dtype_to_str(Options.matmul_dtype())},
        {"gradient-dtype",     dtype_to_str(Options.grad_dtype())},
        {"model-dtype",        dtype_to_str(ModelDType)},
        {"opt-m-dtype",        dtype_to_str(Options.OptMomentumType)},
        {"opt-v-dtype",        dtype_to_str(Options.OptVarianceType)},

        {"learning-rate",      LearningRate},
        {"warmup",             WarmupSteps},
        {"cooldown",           CoolDownSteps},
        {"final-lr",           FinalLrFraction},
        {"lr-schedule",        LrScheduleType},
        {"beta-1",             Beta1},
        {"beta-2",             Beta2},
        {"grad-clip",          GradClip},
        {"weight-decay",       WeightDecay},
        {"grad-accumulation",  GradAccSteps},

        {"micro-batch",        B},
        {"seq-len",            T},
        {"total-batch",        total_batch_size},
        {"steps-per-epoch",    steps_per_epoch},
        {"steps",              MaxSteps},
        {"eval-every-n-steps", EvalEvery},
        {"eval-num-steps",     EvalNumSteps},
        {"ckpt-every-n-steps", CkptEvery},
        {"ckpt-keep",          CkptToKeep},
        {"ckpt-major",         MajorCkptEvery},

        {"train-loader-seed",  (std::int64_t)TrainLoaderSeed},
        {"eval-loader-seed",   (std::int64_t)EvalLoaderSeed},

        {"arch",               std::string{config.model_name()}},
        {"vocab-size",         config.VocabSize},
        {"hidden-size",        config.HiddenSize},
        {"ffn-size",           config.IntermediateSize},
        {"num-layers",         config.NumLayers},
        {"tied-word-emb",      config.TiedWordEmbeddings},
        {"num-heads",          config.NumQueryHeads},
        {"num-kv-heads",       config.NumKeyValHeads},

        {"train-file",         TrainFile},
        {"eval-file",          EvalFile},
        {"checkpoint-dir",     CkptDir},
        {"out-dir",            OutDir},
        {"log-file",           LogFile},
    });
    logger.log_gpu_model(comm);

    auto allocator = std::make_shared<TensorAllocator>();
    if (LogAllocations >= 0 && comm.rank() == 0) {
        allocator->set_callback([this](const std::string& ctx, const std::string& name, EAllocationType kind, std::size_t amount){
            if(amount >= LogAllocations) {
                const char* kind_str;
                switch(kind) {
                    case EAllocationType::ON_DEVICE: kind_str = "GPU "; break;
                    case EAllocationType::MANAGED: kind_str = "USM "; break;
                    case EAllocationType::WRITE_CMB:
                    case EAllocationType::PINNED:
                    case EAllocationType::ON_HOST: kind_str = "HOST"; break;
                }
                ::printf("%s  %15s - %15s: %4zu MiB\n", kind_str, ctx.c_str(), name.c_str(), amount / 1024 / 1024);
            }
        });
    }

    logger.log_sol_estimate(get_transformer_ops(
                                config.NumLayers * ((long)config.HiddenSize * (config.IntermediateSize * 3 + config.HiddenSize * 1 + config.qkv_channels())),
                                Options.matmul_dtype(), (long)config.VocabSize * config.HiddenSize, config.DType,
                                config.NumQueryHeads * config.head_size(), config.NumLayers, T),
                            comm.world_size());

    LLamaModel model{config, Options, comm.rank(), comm.world_size(), allocator};

    // Note: cannot check for exact equality, because vocab_size differs in tokenizer vs model (implicit padding)
    if (train_loader.vocab_size() > 0 && train_loader.vocab_size() > config.VocabSize) {
        std::cerr << "\033[1;31mError: model vocab size " << config.VocabSize
                  << " does not match training data vocab size " << train_loader.vocab_size() << "\033[0m" << std::endl;
        std::exit(1);
    }
    if (test_loader.vocab_size() > 0 && test_loader.vocab_size() > config.VocabSize) {
        std::cerr << "\033[1;31mError: model vocab size " << config.VocabSize
                  << " does not match validation data vocab size " << test_loader.vocab_size() << "\033[0m"
                  << std::endl;
        std::exit(1);
    }


    int latest_step = -1;
    if (ContinueFromCheckpoint.has_value()) {
        if (ContinueFromCheckpoint.value() < 0) {
            latest_step = find_latest_checkpoint(CkptDir);
        } else {
            latest_step = ContinueFromCheckpoint.value();
        }
        if (latest_step < 0) {
            std::cerr << "No checkpoint found in " << CkptDir << std::endl;
            std::cerr << " starting from scratch" << std::endl;
        }
    }

    model.allocate_run_state(Options, comm, B, T);

    if (latest_step >= 0) {
        auto log = logger.log_section_start(0, fmt::format("Loading checkpoint {} from `{}`", latest_step, CkptDir.c_str()));
        load_checkpoint(CkptDir, latest_step, model, &train_loader, comm);
    } else if (FromScratch) {
        auto log = logger.log_section_start(0, "Initializing model from scratch");
        model.init_weights(comm);
        latest_step = 0;
    } else {
        auto log = logger.log_section_start(0, fmt::format("Loading model from `{}`", model_path.c_str()));
        model.import_weights(model_path, true, comm);
        latest_step = 0;
    }

    logger.log_dataset(train_loader, test_loader);

    logger.log_allocator(model.get_allocator().get_allocation_segments(), model.run_state().Stack.get_allocation_stats());

    Tensor inputs = model.get_input_buffer();
    Tensor targets = model.get_target_buffer();

    std::unique_ptr<ISchedule> lr_schedule;
    int end_steps = MaxSteps - CoolDownSteps;
    if (iequals(LrScheduleType, "cosine")) {
        lr_schedule = std::make_unique<CosineSchedule>(LearningRate, end_steps, WarmupSteps, LearningRate * FinalLrFraction);
    } else if (iequals(LrScheduleType, "linear")) {
        lr_schedule = std::make_unique<LinearSchedule>(LearningRate, LearningRate * FinalLrFraction, end_steps, WarmupSteps);
    } else if (iequals(LrScheduleType, "wsd")) {
        lr_schedule = std::make_unique<LinearSchedule>(LearningRate, LearningRate, end_steps, WarmupSteps);
    } else {
        throw std::invalid_argument("Unknown learning rate schedule: " + LrScheduleType);
    }

    logger.log_message(0, fmt::format("Starting training for {} steps ({:.2f} epochs in total)",
        MaxSteps, float(MaxSteps) / steps_per_epoch));
    logger.log_message(0, fmt::format("Setup took {} seconds",
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - BeginStartup).count()));


    for (int step = latest_step; step < MaxSteps; ++step) {
        bool run_eval = false;
        if (!train_loader.has_next(GradAccSteps * B)) {
            train_loader.advance_epoch();
            run_eval = true;
        }
        if (EvalEvery > 0 && step % EvalEvery == 0 && step > 0) {
            run_eval = true;
        }

        if (CkptEvery > 0 && step % CkptEvery == 0 && step > latest_step) {
            auto log = logger.log_section_start(step, fmt::format("saving checkpoint to `{}`", CkptDir.c_str()));
            std::string save_path = save_checkpoint(CkptDir, step, model, &train_loader, comm);
            if(CkptToKeep > 0) {
                auto cleaned = clean_old_checkpoints(CkptDir, CkptToKeep, MajorCkptEvery);
                logger.log_message(0, fmt::format("Cleaned {} checkpoints", cleaned.size()));
            }
        }

        if (run_eval) {
            run_evaluation(test_loader, model, logger, train_loader.epoch() + 0.01f * train_loader.progress(), step, comm, EvalNumSteps,
                           inputs, targets);
        }

        NvtxRange range("step", step);
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < GradAccSteps; ++j) {
            train_loader.load_batch(inputs, targets);
            model.forward(inputs, comm, j);
            model.backward(inputs, targets, comm, GradAccSteps, j);
        }

        if (LogGPUEvery > 0 && step % LogGPUEvery == 0) {
            logger.log_gpu_state(step, 0, gpu_util->update());
        }

        float lr = lr_schedule->eval(step);
        // learning-rate cooldown
        if (step > end_steps) {
            int cds = step - end_steps;
            float f = static_cast<float>(cds) / static_cast<float>(CoolDownSteps);
            // 1 - sqrt recommended by https://arxiv.org/pdf/2405.18392
            lr = lr_schedule->eval(end_steps) * (1.f - sqrtf(f));
        }
        model.update(comm, lr, Beta1, Beta2, step + 1, Epsilon, WeightDecay, GradClip);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float step_loss = model.get_loss();
        float step_norm = model.get_norm();
        logger.log_step(step, train_loader.epoch() + 0.01f*train_loader.progress(), B*T*GradAccSteps*comm.world_size(), narrow<int>(ms), step_norm, step_loss / (B*T*GradAccSteps), lr);

        if (DebugLogAbsMaxes > 0 && step % DebugLogAbsMaxes == 0) {
            auto& rs = model.run_state();
            std::vector<std::pair<std::string, float>> abs_maxes;
            rs.debug_iterate_abs_maxes([&abs_maxes](const std::string& name, float value) {
                abs_maxes.emplace_back(std::make_pair(name, value));
            });
            logger.log_abs_maxes(step, abs_maxes);
        }

        if(Options.TriggerTimingEvents) {
            // timing breakdown
            printf("%s", "\nTiming breakdown:\n");
            auto& rs = model.run_state();
            for(int i = 0; i < GradAccSteps; ++i) {
                float fwd, bwd, head;
                CUDA_CHECK(cudaEventElapsedTime(&fwd, rs.TimingForwardStart[i], rs.TimingForwardEnd[i]));
                CUDA_CHECK(cudaEventElapsedTime(&head, rs.TimingHeadStart[i], rs.TimingHeadEnd[i]));
                CUDA_CHECK(cudaEventElapsedTime(&bwd, rs.TimingBackwardStart[i], rs.TimingBackwardEnd[i]));
                // not: head events are nested in bwd, so need to subtract times
                printf("  fwd %7.2fms, head %7.2fms, bwd %7.2fms\n", fwd, head, bwd - head);
            }
            float opt;
            CUDA_CHECK(cudaEventElapsedTime(&opt, rs.TimingOptimizerStart, rs.TimingOptimizerEnd));
            printf("  opt %7.2fms\n", opt);
            printf("%s", "\n");
        }
    }

    float loss = run_evaluation(test_loader, model, logger, train_loader.epoch() + 0.01f*train_loader.progress(), MaxSteps, comm, test_loader.num_chunks(), inputs, targets);
    logger.log_message(0, fmt::format("Done. validation loss {:10f}", loss));

    auto log = logger.log_section_start(MaxSteps, fmt::format("Saving model to `{}`", OutDir.c_str()));
    std::filesystem::path p(OutDir);
    std::filesystem::create_directories(p);
    save_transformer_config(config, (p / "config.json").c_str());
    model.export_weights((p / "model.safetensors").c_str(), comm);

    // copy config files from source model, if we have them and they don't exist already
    if (std::filesystem::exists(ModelRootPath)) {
        auto maybe_copy = [root=std::filesystem::path(ModelRootPath),dest=std::filesystem::path(p)](const char* file_name){
            if (std::filesystem::exists(root / file_name) && !std::filesystem::exists(dest / file_name)) {
                std::filesystem::copy_file(root / file_name, dest / file_name);
            }
        };

        maybe_copy("tokenizer_config.json");
        maybe_copy("generation_config.json");
        maybe_copy("merges.txt");
        maybe_copy("vocab.json");
        maybe_copy("tokenizer.json");
    }
}

float run_evaluation(DataLoader& test_loader, LLamaModel& model, TrainingRunLogger& logger, float epoch, int step,
                     NCCLCommunicator& comm, int max_steps, Tensor& inputs, Tensor& targets) {
    NvtxRange range("validate", step);
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    test_loader.set_state(test_loader.seed(), 0, 0, 0);
    float loss = 0.f;
    int batches = 0;
    while (test_loader.has_next(div_exact((int)inputs.nelem(), test_loader.seq_len())) && batches < max_steps) {
        test_loader.load_batch(inputs, targets);
        loss += model.validate(inputs, targets, comm, batches);
        batches++;
    }
    static bool warning = true;
    if (warning && batches == 0) {
        std::cerr << "WARNING: insufficient validation data: " << test_loader.num_tokens() << " need at least "
                  << comm.world_size() * test_loader.seq_len() << std::endl;
        warning = false;
        return 0.f;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    logger.log_eval(step, epoch, batches * targets.nelem(), narrow<int>(ms), loss / batches);
    return loss / batches;
}

int main(int argc, const char** argv) {
    try {
        TrainingRunner runner;
        runner.load_training_config(argc, argv);
        runner.launch_training(argc, argv);
        return 0;
    } catch (const std::exception& e) {
        ::fprintf(stderr, "ERROR: %s\n", e.what());
        fflush(stderr);
        return EXIT_FAILURE;
    }
}
