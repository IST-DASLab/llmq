// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODELS_LLAMA_CONFIG_H
#define LLMQ_SRC_MODELS_LLAMA_CONFIG_H

#include <string_view>

#include "utilities/dtype.h"

// Struct that contains the basic configuration for the model.
// This includes both architecture and run configurations
struct TransformerConfig {
    enum EArchitecture {
        LLAMA,
        QWEN2,
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

    int MaxPositionEmbeddings;
    float RopeTheta;
    float RmsNormEps;
    bool TiedWordEmbeddings;
    bool UseQKVBias;

    ETensorDType DType = ETensorDType::BF16;

    [[nodiscard]] int head_size() const { return HiddenSize / NumQueryHeads; }
    [[nodiscard]] int qkv_channels() const { return head_size() * (NumQueryHeads + 2 * NumKeyValHeads); }
    [[nodiscard]] std::string_view model_name() const;
};

TransformerConfig load_transformer_config(const char* file_name, ETensorDType dtype);
void save_transformer_config(const TransformerConfig& config, const char* file_name);
TransformerConfig create_config_from_name(std::string_view name, ETensorDType dtype);

#endif //LLMQ_SRC_MODELS_LLAMA_CONFIG_H
