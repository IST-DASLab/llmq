// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "training/transformer_config.h"

#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

namespace {

//! a config.json in the temp directory that removes itself again
class ScratchFile {
public:
    ScratchFile() {
        static int counter = 0;
        mPath = std::filesystem::temp_directory_path() /
                ("llmq-test-config-" + std::to_string(counter++) + ".json");
        std::filesystem::remove(mPath);
    }
    ~ScratchFile() { std::filesystem::remove(mPath); }

    [[nodiscard]] const char* c_str() const { return mPath.c_str(); }

private:
    std::filesystem::path mPath;
};

TransformerConfig make_config(TransformerConfig::EArchitecture arch) {
    TransformerConfig config{};
    config.Architecture = arch;
    config.BosTokenId = 1;
    config.EosTokenId = 2;
    config.HiddenSize = 256;
    config.IntermediateSize = 512;
    config.VocabSize = 32000;
    config.NumQueryHeads = 4;
    config.NumKeyValHeads = 2;
    config.NumLayers = 4;
    config.MaxPositionEmbeddings = 2048;
    config.RopeTheta = 10000.f;
    config.RmsNormEps = 1e-5f;
    config.TiedWordEmbeddings = false;
    config.UseQKVBias = arch == TransformerConfig::QWEN2;
    config.UseQKNorm = arch == TransformerConfig::QWEN3;
    return config;
}

//! save `config`, then patch `overrides` on top, to get a config.json that
//! save_transformer_config would never emit by itself.
void save_with(const TransformerConfig& config, const ScratchFile& file, const nlohmann::json& overrides) {
    save_transformer_config(config, file.c_str());

    std::ifstream in(file.c_str());
    auto json = nlohmann::json::parse(in);
    in.close();
    for(const auto& [key, value] : overrides.items()) {
        json[key] = value;
    }
    std::ofstream out(file.c_str());
    out << json.dump(4);
}

}

TEST_CASE("transformer config survives a save/load round trip") {
    for(auto arch : {TransformerConfig::LLAMA, TransformerConfig::MISTRAL,
                     TransformerConfig::QWEN2, TransformerConfig::QWEN3}) {
        auto original = make_config(arch);
        ScratchFile file;
        save_transformer_config(original, file.c_str());
        auto loaded = load_transformer_config(file.c_str(), original.DType);

        INFO("architecture " << original.model_name());
        CHECK(loaded.Architecture == original.Architecture);
        CHECK(loaded.HiddenSize == original.HiddenSize);
        CHECK(loaded.IntermediateSize == original.IntermediateSize);
        CHECK(loaded.VocabSize == original.VocabSize);
        CHECK(loaded.NumQueryHeads == original.NumQueryHeads);
        CHECK(loaded.NumKeyValHeads == original.NumKeyValHeads);
        CHECK(loaded.NumLayers == original.NumLayers);
        CHECK(loaded.head_size() == original.head_size());
        CHECK(loaded.RopeTheta == original.RopeTheta);
        CHECK(loaded.TiedWordEmbeddings == original.TiedWordEmbeddings);
        CHECK(loaded.UseQKVBias == original.UseQKVBias);
        CHECK(loaded.UseQKNorm == original.UseQKNorm);
    }
}

TEST_CASE("a decoupled head_dim round trips") {
    auto original = make_config(TransformerConfig::QWEN3);
    original.HeadDim = 128;     // HiddenSize / NumQueryHeads would be 256 / 4 == 64
    REQUIRE(original.head_size() == 128);

    ScratchFile file;
    save_transformer_config(original, file.c_str());
    auto loaded = load_transformer_config(file.c_str(), original.DType);
    CHECK(loaded.HeadDim == 128);
    CHECK(loaded.head_size() == 128);
}

TEST_CASE("saving a llama-family config with QKV biases is refused") {
    // neither spelling of attention_bias is correct for a q/k/v-only bias, so we must
    // not write a config.json that we would refuse to load again
    for(auto arch : {TransformerConfig::LLAMA, TransformerConfig::MISTRAL}) {
        auto config = make_config(arch);
        config.UseQKVBias = true;
        ScratchFile file;
        CHECK_THROWS(save_transformer_config(config, file.c_str()));
    }
}

TEST_CASE("configs we cannot represent are rejected on load") {
    auto base = make_config(TransformerConfig::LLAMA);

    auto rejects = [&](const nlohmann::json& overrides) {
        ScratchFile file;
        save_with(base, file, overrides);
        CHECK_THROWS(load_transformer_config(file.c_str(), base.DType));
    };

    rejects({{"attention_bias", true}});
    rejects({{"mlp_bias", true}});
    rejects({{"hidden_act", "gelu"}});
    rejects({{"attention_dropout", 0.1f}});
    rejects({{"rope_scaling", {{"rope_type", "llama3"}, {"factor", 8.0}}}});
    rejects({{"sliding_window", 1024}});
    rejects({{"architectures", nlohmann::json::array({"GemmaForCausalLM"})}});
}

TEST_CASE("a sliding window is only rejected when it is active") {
    auto base = make_config(TransformerConfig::QWEN2);

    ScratchFile inactive;
    save_with(base, inactive, {{"sliding_window", 32768}, {"use_sliding_window", false}});
    CHECK_NOTHROW(load_transformer_config(inactive.c_str(), base.DType));

    ScratchFile active;
    save_with(base, active, {{"sliding_window", 32768}, {"use_sliding_window", true}});
    CHECK_THROWS(load_transformer_config(active.c_str(), base.DType));
}

TEST_CASE("a null-valued key is treated as absent") {
    // Mistral writes `attention_bias: null`, which value() would throw on
    auto base = make_config(TransformerConfig::MISTRAL);
    ScratchFile file;
    save_with(base, file, {{"attention_bias", nullptr}, {"rope_scaling", nullptr}});
    CHECK_NOTHROW(load_transformer_config(file.c_str(), base.DType));
}
