// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_TRAINING_MODEL_H
#define LLMQ_SRC_TRAINING_MODEL_H

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

class ITensorContainer;
class Tensor;
class NCCLCommunicator;

//! \brief Abstract model base class.
//! \details Provides access to the different underlying tensor containers.
class IModel {
public:
    //! \brief Runs the forward pass until just before the logit calculation
    //! \details This function is asynchronous. You need to wait on `run_state.ForwardDone`
    //! before accessing any of the results (or run subsequent work on `run_state.MainStream`).
    //! However, it is guaranteed that `inputs` have been copied to the GPU-side buffer
    //! before this function returns.
    //! Note: We do not calculate the logits here, so that we have more freedom to optimize
    //! this large matmul, e.g., calculating it in chunks, by including it in the backward pass.
    virtual void forward(Tensor inputs, NCCLCommunicator& comm, int micro_step) = 0;

    //! \brief Runs the forward pass and calculates the loss w.r.t. `targets`.
    virtual float validate(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int micro_step) = 0;

    //! \brief Runs the backward pass
    //! \details This function is asynchronous. You need to wait on `run_state.BackwardDone`
    //! before accessing any of the results (or run subsequent work on `run_state.MainStream`).
    //! However, it is guaranteed that `inputs` and `targets` have been copied to the GPU-side buffer
    //! before this function returns.
    virtual void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) = 0;

    //! \brief Runs the AdamW update step.
    //! \details Runs asynchronously, signalling completion through the OptimizerDone event.
    virtual void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon, float weight_decay, float grad_clip) = 0;

    //! Gets the loss of the preceding validate or backward call (forward does _not_ calculate the loss)
    virtual float get_loss() const = 0;

    //! Gets the gradient norm of the preceding update call.
    virtual float get_norm() const = 0;

    //! Gets the tensor into which model inputs are to be placed.
    virtual Tensor& get_input_buffer() = 0;

    //! Gets the tensor into which model targets are to be placed.
    virtual Tensor& get_target_buffer() = 0;

    //! Model (master) weights. Sharded.
    virtual ITensorContainer& weights() = 0;

    //! (First order) momentum. Sharded.
    virtual ITensorContainer& opt_momentum() = 0;

    //! (First order) momentum. Sharded.
    virtual ITensorContainer& opt_momentum_scales() = 0;

    //! Second order moments. Sharded.
    virtual ITensorContainer& opt_variance() = 0;

    //! Get the current RNG state
    virtual std::vector<std::byte> rng_state() const = 0;

    //! Set the RNG state from checkpoint data
    virtual void set_rng_state(const std::vector<std::byte>& state) = 0;

    //! Randomly initialize the model weights.
    virtual void init_weights(NCCLCommunicator& comm) = 0;

    //! Import the model weights from a file. This may be different than just reading into `weights()`,
    //! because it may involve dtype conversion (`allow_cast=true`), and even rearrange some data
    //! (e.g., fused vs unfused QKV)
    virtual void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) = 0;

    //! This function needs to be called after the model has been restored from a checkpoint.
    virtual void on_restore_checkpoint(NCCLCommunicator& comm) = 0;

    //! Export the model weights to a safetensors file.
    virtual void export_weights(const std::string& file_name, NCCLCommunicator& comm) = 0;

    //! Get the model type identifier
    virtual std::string_view model_type() const = 0;

protected:
    ~IModel() = default;
};

#endif //LLMQ_SRC_TRAINING_MODEL_H
