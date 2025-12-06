// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "llama_config.h"
#include "utilities/utils.h"

#include <fstream>

#include <nlohmann/json.hpp>
#include <fmt/core.h>

LLamaConfig load_llama_config(const char* file_name, ETensorDType dtype) {
    std::ifstream file(file_name);
    if(!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open config file {}", file_name));
    }

    auto config_json = nlohmann::json::parse(file);

    auto archs = config_json["architectures"].get<std::vector<std::string>>();
    if(archs.size() != 1) {
        throw std::runtime_error("got multiple values for architecture");
    }
    LLamaConfig::LLamaBasedModels arch_id;
    if(archs.front() == "LlamaForCausalLM") {
        arch_id = LLamaConfig::LLAMA;
    } else if(archs.front() == "Qwen2ForCausalLM") {
        arch_id = LLamaConfig::QWEN2;
    } else {
        throw std::runtime_error(fmt::format("unknown architecture {}", archs.front()));
    }
    LLamaConfig result;
    result.Architecture = arch_id;
    result.DType = dtype;

    result.BosTokenId = config_json["bos_token_id"].get<int>();
    result.EosTokenId = config_json["eos_token_id"].get<int>();

    result.HiddenSize = config_json["hidden_size"].get<int>();
    result.IntermediateSize = config_json["intermediate_size"].get<int>();
    result.VocabSize = config_json["vocab_size"].get<int>();
    result.NumQueryHeads = config_json["num_attention_heads"].get<int>();
    result.NumKeyValHeads = config_json["num_key_value_heads"].get<int>();
    result.NumLayers = config_json["num_hidden_layers"].get<int>();
    result.MaxPositionEmbeddings = config_json["max_position_embeddings"].get<int>();
    result.RopeTheta = config_json["rope_theta"].get<float>();
    result.TiedWordEmbeddings = config_json["tie_word_embeddings"].get<bool>();
    if(config_json.contains("rms_norm_eps")) {
        result.RmsNormEps = config_json["rms_norm_eps"].get<float>();
    } else {
        result.RmsNormEps = result.Architecture == LLamaConfig::LLAMA ? 1e-5 : 1e-6;
    }

    result.UseQKVBias = arch_id == LLamaConfig::QWEN2;

    return result;
}

[[nodiscard]] std::string_view LLamaConfig::model_name() const {
    switch(Architecture) {
        case LLamaConfig::QWEN2:
            return "Qwen2";
        case LLamaConfig::LLAMA:
            return "LLaMA";
        default:
            throw std::logic_error("Unknown architecture");
    }
}

void save_llama_config(const LLamaConfig& config, const char* file_name) {
    std::ofstream file(file_name);
    if(!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open file for writing {}", file_name));
    }

    std::vector<std::string> archs;
    if(config.Architecture == LLamaConfig::QWEN2) {
        archs = {"Qwen2ForCausalLM"};
    } else if (config.Architecture == LLamaConfig::LLAMA) {
        archs = {"LlamaForCausalLM"};
    }

    nlohmann::json config_json;
    config_json["architectures"] = std::move(archs);
    config_json["bos_token_id"] = config.BosTokenId;
    config_json["eos_token_id"] = config.EosTokenId;
    config_json["hidden_size"] = config.HiddenSize;
    config_json["intermediate_size"] = config.IntermediateSize;
    config_json["vocab_size"] = config.VocabSize;
    config_json["num_attention_heads"] = config.NumQueryHeads;
    config_json["num_key_value_heads"] = config.NumKeyValHeads;
    config_json["num_hidden_layers"] = config.NumLayers;
    config_json["max_position_embeddings"] = config.MaxPositionEmbeddings;
    config_json["rope_theta"] = config.RopeTheta;
    config_json["rms_norm_eps"] = config.RmsNormEps;
    config_json["tie_word_embeddings"] = config.TiedWordEmbeddings;
    config_json["torch_dtype"] = dtype_to_torch_str(config.DType);

    config_json["attention_dropout"] = 0.f;
    config_json["initializer_range"] = 0.02f;
    config_json["hidden_act"] = "silu";
    config_json["use_cache"] = true;
    if(config.Architecture == LLamaConfig::QWEN2) {
        config_json["model_type"] = "qwen2";
        config_json["max_window_layers"] = config.NumLayers;
        config_json["sliding_window"] = config.MaxPositionEmbeddings;
        config_json["use_sliding_window"] = false;
        config_json["use_mrope"] = false;
    } else if (config.Architecture == LLamaConfig::LLAMA) {
        config_json["model_type"] = "llama";
        config_json["attention_bias"] = false;
        config_json["mlp_bias"] = false;
    }

    file << config_json.dump(4);
}

static LLamaConfig create_qwen25_config(int hidden_size, int intermediate_size, int q_heads, int kv_heads, int depth, float rms, bool tied, ETensorDType dtype) {
    return {
        .Architecture = LLamaConfig::QWEN2,
        .BosTokenId = 151643,
        .EosTokenId = 151643,
        .HiddenSize = hidden_size,
        .IntermediateSize = intermediate_size,
        .VocabSize = 151936,
        .NumQueryHeads = q_heads,
        .NumKeyValHeads = kv_heads,
        .NumLayers = depth,
        .MaxPositionEmbeddings = 32768,
        .RopeTheta = 1'000'000.0f,
        .RmsNormEps = rms,
        .TiedWordEmbeddings = tied,
        .UseQKVBias = true,
        .DType = dtype
    };
}

static LLamaConfig create_llama2_config(int hidden_size, int intermediate_size, int heads, int depth, ETensorDType dtype) {
    return {
        .Architecture = LLamaConfig::LLAMA,
        .BosTokenId = 1,
        .EosTokenId = 2,
        .PadTokenId = 0,
        .HiddenSize = hidden_size,
        .IntermediateSize = intermediate_size,
        .VocabSize = 32000,
        .NumQueryHeads = heads,
        .NumKeyValHeads = heads,
        .NumLayers = depth,
        .MaxPositionEmbeddings = 4096,
        .RopeTheta = 10000.f,
        .RmsNormEps = 1e-05f,
        .TiedWordEmbeddings = false,
        .UseQKVBias = false,
        .DType = dtype
    };
}

static LLamaConfig create_llama3_config(int hidden_size, int intermediate_size, int q_heads, int kv_heads, int depth, ETensorDType dtype) {
    return {
        .Architecture = LLamaConfig::LLAMA,
        .BosTokenId = 128000,
        .EosTokenId = 128001,
        .PadTokenId = 128255,
        .HiddenSize = hidden_size,
        .IntermediateSize = intermediate_size,
        .VocabSize = 128256,
        .NumQueryHeads = q_heads,
        .NumKeyValHeads = kv_heads,
        .NumLayers = depth,
        .MaxPositionEmbeddings = 4096,
        .RopeTheta = 500000.f,
        .RmsNormEps = 1e-05f,
        .TiedWordEmbeddings = false,
        .UseQKVBias = false,
        .DType = dtype
    };
}

LLamaConfig create_config_from_name(std::string_view name, ETensorDType dtype) {
    if(iequals(name, "Qwen2.5-0.5B")) {
        return create_qwen25_config(896, 4864, 14, 2, 24, 1e-06f, true, dtype);
    } else if(iequals(name, "Qwen2.5-1.5B")) {
        return create_qwen25_config(1536, 8960, 12, 2, 28, 1e-06f, true, dtype);
    } else if(iequals(name, "Qwen2.5-3B")) {
        return create_qwen25_config(2048, 11008, 16, 2, 36, 1e-06f, true, dtype);
    } else if(iequals(name, "Qwen2.5-7B")) {
        return create_qwen25_config(3584, 18944, 28, 4, 28, 1e-06f, false, dtype);
    } else if(iequals(name, "Qwen2.5-14B")) {
        return create_qwen25_config(5120, 13824, 40, 8, 48, 1e-05f, false, dtype);
    } else if(iequals(name, "Qwen2.5-32B")) {
        return create_qwen25_config(5120, 27648, 40, 8, 64, 1e-05f, false, dtype);
    } else if(iequals(name, "Qwen2.5-72B")) {
        return create_qwen25_config(8192, 29568, 64, 8, 80, 1e-05f, false, dtype);
    } else if (iequals(name, "llama-2-7b")) {
        return create_llama2_config(4096, 11008, 32, 32, dtype);
    } else if (iequals(name, "llama-2-13b")) {
        return create_llama2_config(5120, 13824, 40, 40, dtype);
    } else if (iequals(name, "llama-3-8b")) {
        return create_llama3_config(4096, 14336, 32, 8, 32, dtype);
    }
    throw std::runtime_error(fmt::format("unknown model name {}", name));
}
