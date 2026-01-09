// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#include "llama_optimizer.h"

#include <fmt/format.h>

#include "training/transformer_config.h"
#include "llama_model.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"
#include "utilities/lazy_allocator.h"
#include "utilities/safetensors.h"

struct OptStateWrapper : ITensorContainer {
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;
    std::vector<GenericTensorContainer>* Blocks;
    GenericTensorContainer* NonBlock;
    OptStateWrapper() = default;
    OptStateWrapper(std::vector<GenericTensorContainer>* b, GenericTensorContainer* nb) : Blocks(b), NonBlock(nb) {};
};

void OptStateWrapper::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    callback("model.embed_tokens.weight", NonBlock->get_tensor(LLamaWeightID::EMBEDDING));
    if(NonBlock->get_tensor(LLamaWeightID::LM_HEAD)) {
        callback("lm_head.weight", NonBlock->get_tensor(LLamaWeightID::LM_HEAD));
    }
    callback("model.norm.weight", NonBlock->get_tensor(LLamaWeightID::LNF_W));

    for(int i = 0; i < Blocks->size(); i++) {
        auto& layer = Blocks->at(i);
        const Tensor& qkv_w = layer.get_tensor(LLamaWeightID::QKV_W);
        const Tensor& up_proj = layer.get_tensor(LLamaWeightID::UP_W);
        std::string prefix = "model.layers." + std::to_string(i);
        callback(prefix + ".self_attn.qkv.weight", qkv_w);
        if (layer.get_tensor(LLamaWeightID::QKV_B)) {
            callback(prefix + ".self_attn.qkv.bias", layer.get_tensor(LLamaWeightID::QKV_B));
        }

        callback(prefix + ".self_attn.o_proj.weight", layer.get_tensor(LLamaWeightID::ATTO_W));
        callback(prefix + ".mlp.up.weight", up_proj);
        callback(prefix + ".mlp.down_proj.weight", layer.get_tensor(LLamaWeightID::DOWN_W));
        callback(prefix + ".input_layernorm.weight", layer.get_tensor(LLamaWeightID::LN1_W));
        callback(prefix + ".post_attention_layernorm.weight", layer.get_tensor(LLamaWeightID::LN2_W));
    }
}

LLamaOptimizerStateManager::LLamaOptimizerStateManager(TransformerConfig cfg, IModel& model, LLamaOptions options, NCCLCommunicator& comm):
        AdamWStateManager(cfg, model, options.OffloadOptM, options.OffloadOptV, options.OptMomentumType, options.OptVarianceType, options.UseZeroCopy, comm.rank(), comm.world_size())
{
}

void LLamaOptimizerStateManager::safe_to_checkpoint(const std::string& checkpoint_dir) {
    OptStateWrapper m_state{&mBlocksM, &mNonBlockM};
    OptStateWrapper v_state{&mBlocksV, &mNonBlockV};

    write_safetensors(checkpoint_dir + fmt::format("/adam.m.shard_{:03}_of_{:03}.safetensors", mRank, mWorld), m_state);
    write_safetensors(checkpoint_dir + fmt::format("/adam.v.shard_{:03}_of_{:03}.safetensors", mRank, mWorld), v_state);
    if (mMType == ETensorDType::FP8_E4M3) {
        OptStateWrapper m_scales{&mBlocksMScales, &mNonBlockMScales};
        write_safetensors(checkpoint_dir + fmt::format("/adam.m.scales.shard_{:03}_of_{:03}.safetensors", mRank, mWorld), m_scales);
    }
}

void LLamaOptimizerStateManager::load_from_checkpoint(const std::string& checkpoint_dir) {
    OptStateWrapper m_state{&mBlocksM, &mNonBlockM};
    OptStateWrapper v_state{&mBlocksV, &mNonBlockV};

    // load optimizer shards
    load_safetensors(checkpoint_dir + fmt::format("/adam.m.shard_{:03}_of_{:03}.safetensors", mRank, mWorld), m_state, false);
    load_safetensors(checkpoint_dir + fmt::format("/adam.v.shard_{:03}_of_{:03}.safetensors", mRank, mWorld), v_state, false);

    if (mMType == ETensorDType::FP8_E4M3) {
        OptStateWrapper m_scales{&mBlocksMScales, &mNonBlockMScales};
        load_safetensors(checkpoint_dir + fmt::format("/adam.m.scales.shard_{:03}_of_{:03}.safetensors", mRank, mWorld), m_scales, false);
    }
}
