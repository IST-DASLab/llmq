// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "model.h"

#include "transformer_config.h"
#include "utilities/allocator.h"

cudnnHandle_t create_cudnn_handle();
cublasLtHandle_t create_cublaslt_handle();

float IModel::get_loss() const {
    return get_run_state().get_loss();
}
float IModel::get_norm() const {
    return get_run_state().get_norm();
}

Tensor& IModel::get_input_buffer() {
    return get_run_state().Inputs_CPU;
}

Tensor& IModel::get_target_buffer() {
    return get_run_state().Targets_CPU;
}


IRunState::IRunState(TransformerConfig config, long batch_size, long seq_len, std::shared_ptr<TensorAllocator> alloc) : Config(config), B(batch_size), T(seq_len), Allocator(std::move(alloc)) {
    int did;
    CUDA_CHECK(cudaGetDevice(&did));
    CUDA_CHECK(cudaGetDeviceProperties(&DeviceProp, did));

    Inputs = Allocator->allocate(ETensorDType::INT32, "inputs", {B, T});
    Targets = Allocator->allocate(ETensorDType::INT32, "targets", {B, T});
    Inputs_CPU = Allocator->allocate(ETensorDType::INT32, "inputs_cpu", EAllocationType::PINNED, {B, T});
    Targets_CPU = Allocator->allocate(ETensorDType::INT32, "targets_cpu", EAllocationType::PINNED, {B, T});
    Losses = Allocator->allocate(ETensorDType::FP32, "losses", {B, T});

    CudnnHandle = create_cudnn_handle();
    CublasLtHandle = create_cublaslt_handle();

    // https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace
    // recommended workspace size 32MB for sm_90+
    CuBlasWorkspace = Allocator->allocate(ETensorDType::BYTE, "cublas_ws", {32*1024*1024});

    MainStream = create_named_stream("main stream");

    ForwardDone = create_named_event("forward_done");
    BackwardDone = create_named_event("backward_done");
    TransferDone = create_named_event("transfer_done");
    NormDone = create_named_event("norm_done");
    OptimizerDone = create_named_event("optimizer_done");

    Tensor host_buffer = Allocator->allocate(ETensorDType::FP32, "host_buffer", EAllocationType::PINNED, {2});
    NormHost = host_buffer.get<float>();
    LossHost = host_buffer.get<float>() + 1;
}

void IRunState::setup_timing_events(int micro_steps) {
    TimingOptimizerStart = create_named_event("timing_opt_start", true);
    TimingOptimizerEnd = create_named_event("timing_opt_done", true);
    for(int i = TimingForwardStart.size(); i < micro_steps + 1; ++i) {
        TimingForwardStart.push_back(create_named_event(("timing_fwd_" + std::to_string(i) + "_start").c_str(), true));
        TimingForwardEnd.push_back(create_named_event(("timing_fwd_" + std::to_string(i) + "_end").c_str(), true));
        TimingHeadStart.push_back(create_named_event(("timing_head_" + std::to_string(i) + "_start").c_str(), true));
        TimingHeadEnd.push_back(create_named_event(("timing_head_" + std::to_string(i) + "_end").c_str(), true));
        TimingBackwardStart.push_back(create_named_event(("timing_bwd_" + std::to_string(i) + "_start").c_str(), true));
        TimingBackwardEnd.push_back(create_named_event(("timing_bwd_" + std::to_string(i) + "_end").c_str(), true));
    }
}

float IRunState::get_loss() const {
    CUDA_CHECK(cudaEventSynchronize(BackwardDone));
    return LossHost[0];
}

float IRunState::get_norm() const {
    CUDA_CHECK(cudaEventSynchronize(NormDone));
    return NormHost[0];
}

Tensor IRunState::temp_alloc(ETensorDType dtype, const std::vector<long>& shape) {
    return Stack.allocate(dtype, shape);
}

void IRunState::temp_acquire(Tensor& target) {
    if(target.Device != Stack.device_id()) {
        throw std::logic_error("device mismatch");
    }

    target.Data = Stack.allocate(target.bytes());
}

void IRunState::temp_free(Tensor& tensor) {
    Stack.free(tensor);
}
