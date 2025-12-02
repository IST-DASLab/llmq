// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODELS_QWEN2_H
#define LLMQ_SRC_MODELS_QWEN2_H

#include <memory>
#include <optional>
#include <random>

#include "llama_config.h"
#include "training/model.h"
#include "utilities/allocator.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"

// ---------------------------------------------------------------------------------------------------------------------

struct LLamaOptions {
    bool KeepAllActivations = false;
    bool RecomputeSwiGLu = false;
    bool RecomputeRMSNorm = false;
    bool RecomputeFFN = false;
    bool RecomputeQKV = false;
    bool RecomputeAtt = false;
    bool RecomputeBlock = false;
    bool OffloadResidual = false;
    int LMHeadChunks = 1;
    int AttBwdChunks = 1;
    bool UseCudaGraphs = false;
    bool TriggerTimingEvents = false;

    bool OffloadMaster = false;
    bool OffloadQuants = false;
    bool OffloadOptM   = false;
    bool OffloadOptV   = false;
    bool OffloadGrads  = false;
    bool UseZeroCopy   = false;
    bool UseWriteCombined = false;
    bool ShardWeights = false;
    bool PersistentQuants = false;

    bool ShardGradients = false;
    bool UseAllToAllReduce = false;

    bool InitProjectionsToZero = false;

    // ModelType is just a copy of the dtype set in config
    std::optional<ETensorDType> ModelType = std::nullopt;
    std::optional<ETensorDType> MatmulType = std::nullopt;
    std::optional<ETensorDType> GradientType = std::nullopt;
    std::optional<ETensorDType> MasterDType = std::nullopt;
    ETensorDType OptMomentumType = ETensorDType::FP32;
    ETensorDType OptVarianceType = ETensorDType::FP32;

    ETensorDType matmul_dtype() const {
        return MatmulType.value_or(ModelType.value());
    }

    ETensorDType grad_dtype() const {
        return GradientType.value_or(matmul_dtype());
    }

    EAllocationType offload_alloc() const {
        return UseWriteCombined ? EAllocationType::WRITE_CMB : EAllocationType::PINNED;
    }
};

template<class T>
struct sLLamaBlockWeights;
struct sLLamaLayerActivations;
struct sLLamaLayerGradients;
class LLamaWeightsManager;
class LLamaGradsManager;
class LLamaOptimizerStateManager;
struct LLamaRunState;
struct sLLamaWeights;
class NCCLCommunicator;
using sLLamaGradBlock = sLLamaBlockWeights<Tensor>;

//! \brief LLama (1,2,3) and related models (e.g., Qwen)
//! \details Implements a transformer model with a gated linear unit in the feed-forward layer and RoPE attention.
class LLamaModel : public IModel {
public:
    LLamaModel(LLamaConfig config, const LLamaOptions& options, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc = nullptr);
    ~LLamaModel();

    void init_weights(NCCLCommunicator& comm) override;
    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override;
    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override;
    void on_restore_checkpoint(NCCLCommunicator& comm) override;

    // main training loop
    void forward(Tensor inputs, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;
    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon, float weight_decay, float grad_clip) override;

    void allocate_run_state(const LLamaOptions& options, NCCLCommunicator& comm, int B, int T);

    //! \brief Calculates the global norm of the gradient buffers, and gradient clipping scale factor
    //! \details Runs asynchronously, signalling completion through the NormDone event.
    void calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip);

    float get_loss() const override;
    float get_norm() const override;
    Tensor& get_input_buffer() override;
    Tensor& get_target_buffer() override;

    ITensorContainer& weights() override;
    ITensorContainer& opt_momentum() override;
    ITensorContainer& opt_momentum_scales() override;
    ITensorContainer& opt_variance() override;
    std::vector<std::byte> rng_state() const override;
    void set_rng_state(const std::vector<std::byte>& state) override;
    std::string_view model_type() const override;

    const TensorAllocator& get_allocator() const { return *Allocator; }

    const LLamaConfig& config() { return Config; }
    LLamaGradsManager& grads() { return *Grads; }
    LLamaRunState& run_state() { return *RunState; }

protected:
    void _calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip);
    void _reduce_loss(LLamaRunState& acts, NCCLCommunicator& comm, int B, int T);

    void _forward_block(sLLamaBlockWeights<Tensor>& weights, sLLamaLayerActivations& activations, Tensor& residual);
    void _backward_lmhead(long B, long T, int micro_step, int grad_accum_steps, NCCLCommunicator& comm);
    void _recompute_block(sLLamaBlockWeights<Tensor>& weights, sLLamaLayerActivations& activations, Tensor& residual);
    void _backward_block(bool accumulate, sLLamaBlockWeights<Tensor>& weights, sLLamaGradBlock& grads,
                         sLLamaLayerActivations& activations, sLLamaLayerGradients& d_activations);
private:
    LLamaConfig Config;
    LLamaOptions Options;
    std::shared_ptr<TensorAllocator> Allocator;
    std::unique_ptr<LLamaWeightsManager> Parameters;
    std::unique_ptr<LLamaOptimizerStateManager> OptimizerState;
    std::unique_ptr<LLamaGradsManager> Grads;
    std::unique_ptr<LLamaRunState> RunState;

    std::minstd_rand OptimizerRNG;  //!< Seed generator for stochastic rounding in the optimizer
};

#endif //LLMQ_SRC_MODELS_QWEN2_H
