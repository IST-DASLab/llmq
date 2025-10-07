// Copyright (c) 2025, Aalto University, developed by Erik Schultheis
//

#include "llama_config.h"

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
