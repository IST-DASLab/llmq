// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// All rights reserved.
//


#include "utilities/safetensors.h"
#include "utilities/comm.h"
#include "training/checkpoint.h"
#include "models/llama_model.h"

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

int main(int argc, const char** argv) {
    ETensorDType ModelDType = ETensorDType::BF16;
    std::string ModelRootPath = ".";
    std::string OutDir;
    std::string CkptDir = "ckpt";
    LLamaOptions Options;

    CLI::App app;

    app.add_option("--model-dtype", ModelDType, "Which dtype to use for model");
    app.add_option("--out-dir", OutDir, "Where to save the trained model")->required();
    app.add_option("--checkpoint-dir", CkptDir, "Directory in which to save checkpoints.");

    // TODO these should be inferred automatically
    app.add_option("--model", ModelRootPath, "Path to the huggingface model directory or name of a HF model that is cache locally.");
    app.add_option("--opt-m-dtype", Options.OptMomentumType, "DType for first-order momentum. FP32 or BF16");
    app.add_option("--opt-v-dtype", Options.OptVarianceType, "DType for second-order momentum. FP32 or BF16");

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


    std::string config_path = ModelRootPath + "/config.json";
    LLamaConfig config = load_llama_config(config_path.c_str(), ModelDType);

    int latest_step = find_latest_checkpoint(CkptDir);
    if (latest_step < 0) {
        std::cerr << "No checkpoint found in " << CkptDir << std::endl;
        exit(EXIT_FAILURE);
    }

    int world_size = get_checkpoint_world_size(CkptDir, latest_step);

    NCCLCommunicator::launch_threads_communicators(
            world_size, true, true,
            [&](NCCLCommunicator& comm) {

                LLamaModel model{config, Options, comm.rank(), comm.world_size()};
                model.allocate_run_state(Options, comm, 0, 0);
                load_checkpoint(CkptDir, latest_step, model, nullptr, comm);

                std::filesystem::path p(OutDir);
                std::filesystem::create_directories(p);
                save_llama_config(config, (p / "config.json").c_str());
                model.export_weights((p / "model.safetensors").c_str(), comm);
            });
}
