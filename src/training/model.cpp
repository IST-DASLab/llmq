// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "model.h"
#include "utilities/allocator.h"

cudnnHandle_t create_cudnn_handle();
cublasLtHandle_t create_cublaslt_handle();

RunState::RunState(long batch_size, long seq_len, std::shared_ptr<TensorAllocator> alloc) : B(batch_size), T(seq_len), Allocator(std::move(alloc)) {
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

    MainStream = create_named_stream("main stream");

    ForwardDone = create_named_event("forward done");
    BackwardDone = create_named_event("backward done");
    TransferDone = create_named_event("transfer done");
}

void RunState::setup_timing_events(int micro_steps) {
    for(int i = TimingForwardStart.size(); i < micro_steps + 1; ++i) {
        TimingForwardStart.push_back(create_named_event(("timing_fwd_" + std::to_string(i) + "_start").c_str(), true));
        TimingForwardEnd.push_back(create_named_event(("timing_fwd_" + std::to_string(i) + "_end").c_str(), true));
        TimingHeadStart.push_back(create_named_event(("timing_head_" + std::to_string(i) + "_start").c_str(), true));
        TimingHeadEnd.push_back(create_named_event(("timing_head_" + std::to_string(i) + "_end").c_str(), true));
        TimingBackwardStart.push_back(create_named_event(("timing_bwd_" + std::to_string(i) + "_start").c_str(), true));
        TimingBackwardEnd.push_back(create_named_event(("timing_bwd_" + std::to_string(i) + "_end").c_str(), true));
    }
}
