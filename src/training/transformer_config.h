// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODELS_LLAMA_CONFIG_H
#define LLMQ_SRC_MODELS_LLAMA_CONFIG_H

#include <string_view>

#include "utilities/dtype.h"

//! Struct that contains the basic configuration for the model.
//! This includes both architecture and some run configurations.
//! Maps to `config.json` in huggingface.
struct TransformerConfig {
    enum EArchitecture {
        LLAMA,
        QWEN2,
        QWEN3,
    } Architecture;
    int BosTokenId;
    int EosTokenId;
    int PadTokenId = -100;

    int HiddenSize;
    int IntermediateSize;
    int VocabSize;
    int NumQueryHeads;
    int NumKeyValHeads;
    int NumLayers;
    int HeadDim = 0;        // 0: derived as HiddenSize / NumQueryHeads

    int MaxPositionEmbeddings;
    float RopeTheta;
    float RmsNormEps;
    bool TiedWordEmbeddings;
    bool UseQKVBias;
    bool UseQKNorm = false;

    ETensorDType DType = ETensorDType::BF16;

    [[nodiscard]] int head_size() const { return HeadDim != 0 ? HeadDim : HiddenSize / NumQueryHeads; }
    //! number of channels consumed by the attention operation
    [[nodiscard]] int qkv_channels() const { return head_size() * (NumQueryHeads + 2 * NumKeyValHeads); }
    //! number of channels produced by the attention operation
    [[nodiscard]] int attn_channels() const { return head_size() * NumQueryHeads; }
    [[nodiscard]] std::string_view model_name() const;
};

TransformerConfig load_transformer_config(const char* file_name, ETensorDType dtype);
void save_transformer_config(const TransformerConfig& config, const char* file_name);
TransformerConfig create_config_from_name(std::string_view name, ETensorDType dtype);

#endif //LLMQ_SRC_MODELS_LLAMA_CONFIG_H
