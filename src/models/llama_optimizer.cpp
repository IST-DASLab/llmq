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
    using namespace LLamaWeightID;
    for(unsigned id = 0; id < NonBlock->num_tensors(); ++id) {
        if(const Tensor& tensor = NonBlock->get_tensor(id)) {
            callback(non_block_weight_name(id), tensor);
        }
    }

    for(int i = 0; i < Blocks->size(); i++) {
        auto& layer = Blocks->at(i);
        for(unsigned id = 0; id < layer.num_tensors(); ++id) {
            if(const Tensor& tensor = layer.get_tensor(id)) {
                callback(block_weight_name(i, id), tensor);
            }
        }
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
