// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer_config.h"
#include "utilities/utils.h"

#include <fstream>

#include <nlohmann/json.hpp>
#include <fmt/core.h>

TransformerConfig load_transformer_config(const char* file_name, ETensorDType dtype) {
    std::ifstream file(file_name);
    if(!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open config file {}", file_name));
    }

    auto config_json = nlohmann::json::parse(file);

    auto archs = config_json["architectures"].get<std::vector<std::string>>();
    if(archs.size() != 1) {
        throw std::runtime_error("got multiple values for architecture");
    }
    TransformerConfig::EArchitecture arch_id;
    if(archs.front() == "LlamaForCausalLM") {
        arch_id = TransformerConfig::LLAMA;
    } else if(archs.front() == "MistralForCausalLM") {
        arch_id = TransformerConfig::MISTRAL;
    } else if(archs.front() == "Qwen2ForCausalLM") {
        arch_id = TransformerConfig::QWEN2;
    }  else if(archs.front() == "Qwen3ForCausalLM") {
        arch_id = TransformerConfig::QWEN3;
    } else {
        throw std::runtime_error(fmt::format("unknown architecture {}", archs.front()));
    }
    TransformerConfig result;
    result.Architecture = arch_id;
    result.DType = dtype;

    result.BosTokenId = config_json["bos_token_id"].get<int>();
    result.EosTokenId = config_json["eos_token_id"].get<int>();
    if (config_json.contains("pad_token_id")) {
        result.PadTokenId = config_json["pad_token_id"].get<int>();
    }

    result.HiddenSize = config_json["hidden_size"].get<int>();
    result.IntermediateSize = config_json["intermediate_size"].get<int>();
    result.VocabSize = config_json["vocab_size"].get<int>();
    result.NumQueryHeads = config_json["num_attention_heads"].get<int>();
    result.NumKeyValHeads = config_json["num_key_value_heads"].get<int>();
    result.NumLayers = config_json["num_hidden_layers"].get<int>();
    if (config_json.contains("head_dim")) {
        result.HeadDim = config_json["head_dim"].get<int>();
    }
    result.MaxPositionEmbeddings = config_json["max_position_embeddings"].get<int>();
    result.RopeTheta = config_json["rope_theta"].get<float>();
    result.TiedWordEmbeddings = config_json["tie_word_embeddings"].get<bool>();
    if(config_json.contains("rms_norm_eps")) {
        result.RmsNormEps = config_json["rms_norm_eps"].get<float>();
    } else {
        result.RmsNormEps = result.Architecture == TransformerConfig::LLAMA ? 1e-5 : 1e-6;
    }

    // value() throws on a present-but-null key, and Mistral writes `attention_bias: null`.
    auto get_or = [&](const char* key, auto fallback) {
        auto it = config_json.find(key);
        if(it == config_json.end() || it->is_null()) {
            return fallback;
        }
        return it->template get<decltype(fallback)>();
    };

    result.UseQKNorm = arch_id == TransformerConfig::QWEN3;
    // Qwen2 biases q/k/v only and carries no flag for it; that is not HF's
    // attention_bias, which also biases o_proj, so the flag does not feed in here.
    result.UseQKVBias = arch_id == TransformerConfig::QWEN2;

    // Anything we cannot represent exactly has to fail here: silently training a model
    // that differs from the checkpoint only shows up as degraded quality later.
    auto reject = [&](std::string_view key, std::string_view value) {
        throw std::runtime_error(fmt::format("config {}: cannot represent '{}' = {}", file_name, key, value));
    };

    if(auto it = config_json.find("rope_scaling"); it != config_json.end() && !it->is_null()) {
        reject("rope_scaling", it->dump());
    }
    if(get_or("mlp_bias", false)) {
        reject("mlp_bias", "true");
    }
    if(get_or("attention_bias", false)) {
        reject("attention_bias", "true, which implies an o_proj bias we have no tensor for");
    }
    if(auto act = get_or("hidden_act", std::string{"silu"}); act != "silu") {
        reject("hidden_act", act);
    }
    if(float dropout = get_or("attention_dropout", 0.f); dropout != 0.f) {
        reject("attention_dropout", fmt::format("{}", dropout));
    }
    // Qwen's sliding_window stays inactive unless use_sliding_window is set; Mistral has
    // no such flag, so a non-null window there is always active.
    if(auto it = config_json.find("sliding_window"); it != config_json.end() && !it->is_null()
       && get_or("use_sliding_window", true)) {
        reject("sliding_window", it->dump());
    }

    return result;
}

[[nodiscard]] std::string_view TransformerConfig::model_name() const {
    switch(Architecture) {
        case TransformerConfig::QWEN3:
            return "Qwen3";
        case TransformerConfig::QWEN2:
            return "Qwen2";
        case TransformerConfig::LLAMA:
            return "LLaMA";
        case TransformerConfig::MISTRAL:
            return "Mistral";
        default:
            throw std::logic_error("Unknown architecture");
    }
}

void save_transformer_config(const TransformerConfig& config, const char* file_name) {
    std::ofstream file(file_name);
    if(!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open file for writing {}", file_name));
    }

    std::vector<std::string> archs;
    if (config.Architecture == TransformerConfig::QWEN2) {
        archs = {"Qwen2ForCausalLM"};
    } else if(config.Architecture == TransformerConfig::QWEN3) {
        archs = {"Qwen3ForCausalLM"};
    } else if (config.Architecture == TransformerConfig::LLAMA) {
        archs = {"LlamaForCausalLM"};
    } else if (config.Architecture == TransformerConfig::MISTRAL) {
        archs = {"MistralForCausalLM"};
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
    if (config.HeadDim != 0) {
        config_json["head_dim"] = config.HeadDim;
    }
    config_json["max_position_embeddings"] = config.MaxPositionEmbeddings;
    config_json["rope_theta"] = config.RopeTheta;
    config_json["rms_norm_eps"] = config.RmsNormEps;
    config_json["tie_word_embeddings"] = config.TiedWordEmbeddings;
    config_json["torch_dtype"] = dtype_to_torch_str(config.DType);

    config_json["attention_dropout"] = 0.f;
    config_json["initializer_range"] = 0.02f;
    config_json["hidden_act"] = "silu";
    config_json["use_cache"] = true;
    if (config.Architecture == TransformerConfig::QWEN2 || config.Architecture == TransformerConfig::QWEN3) {
        config_json["model_type"] = config.Architecture == TransformerConfig::QWEN2 ? "qwen2" : "qwen3";
        config_json["max_window_layers"] = config.NumLayers;
        config_json["sliding_window"] = config.MaxPositionEmbeddings;
        config_json["use_sliding_window"] = false;
        config_json["use_mrope"] = false;
    } else if (config.Architecture == TransformerConfig::LLAMA || config.Architecture == TransformerConfig::MISTRAL) {
        bool is_llama = config.Architecture == TransformerConfig::LLAMA;
        config_json["model_type"] = is_llama ? "llama" : "mistral";
        // A q/k/v-only bias has no faithful spelling here: true would promise an o_proj
        // bias we lack, false would disclaim biases we have.
        if(config.UseQKVBias) {
            throw std::runtime_error(fmt::format(
                "cannot save a {} config with QKV biases: HF's attention_bias also implies an o_proj bias",
                config.model_name()));
        }
        config_json["attention_bias"] = false;
        config_json["mlp_bias"] = false;
        if(!is_llama) {
            // we only support Mistral variants that keep the window disabled
            config_json["sliding_window"] = nullptr;
        }
    }

    file << config_json.dump(4);
}

static TransformerConfig create_qwen2_config(int hidden_size, int intermediate_size, int q_heads, int kv_heads, int depth, float rms, bool tied, ETensorDType dtype) {
    return {
        .Architecture = TransformerConfig::QWEN2,
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

static TransformerConfig create_qwen3_config(int hidden_size, int intermediate_size, int q_heads, int kv_heads, int depth, float rms, bool tied, ETensorDType dtype) {
    return {
        .Architecture = TransformerConfig::QWEN3,
        .BosTokenId = 151643,
        .EosTokenId = 151645,
        .HiddenSize = hidden_size,
        .IntermediateSize = intermediate_size,
        .VocabSize = 151936,
        .NumQueryHeads = q_heads,
        .NumKeyValHeads = kv_heads,
        .NumLayers = depth,
        .HeadDim = 128,
        .MaxPositionEmbeddings = 40960,
        .RopeTheta = 1'000'000.f,
        .RmsNormEps = rms,
        .TiedWordEmbeddings = tied,
        .UseQKVBias = false,
        .UseQKNorm = true,
        .DType = dtype
    };
}

static TransformerConfig create_llama2_config(int hidden_size, int intermediate_size, int heads, int depth, ETensorDType dtype) {
    return {
        .Architecture = TransformerConfig::LLAMA,
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

static TransformerConfig create_llama3_config(int hidden_size, int intermediate_size, int q_heads, int kv_heads, int depth, bool tied, ETensorDType dtype) {
    return {
        .Architecture = TransformerConfig::LLAMA,
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
        .TiedWordEmbeddings = tied,
        .UseQKVBias = false,
        .DType = dtype
    };
}

TransformerConfig create_config_from_name(std::string_view name, ETensorDType dtype) {
    if(iequals(name, "Qwen2.5-0.5B") || iequals(name, "Qwen2-0.5B")) {
        return create_qwen2_config(896, 4864, 14, 2, 24, 1e-06f, true, dtype);
    } else if(iequals(name, "Qwen2.5-1.5B") || iequals(name, "Qwen2-1.5B")) {
        return create_qwen2_config(1536, 8960, 12, 2, 28, 1e-06f, true, dtype);
    } else if(iequals(name, "Qwen2.5-3B") || iequals(name, "Qwen2-3B")) {
        return create_qwen2_config(2048, 11008, 16, 2, 36, 1e-06f, true, dtype);
    } else if(iequals(name, "Qwen2.5-7B") || iequals(name, "Qwen2-7B")) {
        return create_qwen2_config(3584, 18944, 28, 4, 28, 1e-06f, false, dtype);
    } else if(iequals(name, "Qwen2.5-14B") || iequals(name, "Qwen2-14B")) {
        return create_qwen2_config(5120, 13824, 40, 8, 48, 1e-05f, false, dtype);
    } else if(iequals(name, "Qwen2.5-32B") || iequals(name, "Qwen2-32B")) {
        return create_qwen2_config(5120, 27648, 40, 8, 64, 1e-05f, false, dtype);
    } else if(iequals(name, "Qwen2.5-72B") || iequals(name, "Qwen2-72B")) {
        return create_qwen2_config(8192, 29568, 64, 8, 80, 1e-05f, false, dtype);
    } else if (iequals(name, "Qwen3-0.6B")) {
        return create_qwen3_config(1024, 3072, 16, 8, 28, 1e-6f, true, dtype);
    } else if (iequals(name, "Qwen3-1.7B")) {
        return create_qwen3_config(2048, 6144, 16, 8, 28, 1e-6f, true, dtype);
    } else if (iequals(name, "Qwen3-4B")) {
        return create_qwen3_config(2560, 9728, 32, 8, 36, 1e-6f, true, dtype);
    } else if (iequals(name, "Qwen3-8B")) {
        return create_qwen3_config(4096, 12288, 32, 8, 36, 1e-6f, false, dtype);
    } else if (iequals(name, "Qwen3-14B")) {
        return create_qwen3_config(5120, 17408, 40, 8, 40, 1e-6f, false, dtype);
    }  else if (iequals(name, "Qwen3-32B")) {
        return create_qwen3_config(5120, 25600, 64, 8, 64, 1e-6f, false, dtype);
    } else if (iequals(name, "llama-2-7b")) {
        return create_llama2_config(4096, 11008, 32, 32, dtype);
    } else if (iequals(name, "llama-2-13b")) {
        return create_llama2_config(5120, 13824, 40, 40, dtype);
    } else if (iequals(name, "llama-3-1b") || iequals(name, "llama-3.2-1b")) {
        return create_llama3_config(2048, 8192, 32, 8, 16, true, dtype);
    } else if (iequals(name, "llama-3-3b") || iequals(name, "llama-3.2-3b")) {
        return create_llama3_config(3072, 8192, 24, 8, 28, true, dtype);
    } else if (iequals(name, "llama-3-8b") || iequals(name, "llama-3.1-8b")) {
        return create_llama3_config(4096, 14336, 32, 8, 32, false, dtype);
    }
    throw std::runtime_error(fmt::format("unknown model name {}", name));
}
