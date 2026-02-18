// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "model.h"

#include <cmath>

#include "transformer_config.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"

cudnnHandle_t create_cudnn_handle();
cublasLtHandle_t create_cublaslt_handle();

float IModel::get_loss(int max_pos) const {
    return get_run_state().get_loss(max_pos);
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

GenericTensorContainer IModel::create_block_container(const TransformerConfig& config, ETensorDType matrix_dtype,
    ETensorDType other_dtype) const
{
    std::vector<Tensor> tensors(num_block_tensors());
    GenericTensorContainer container(std::move(tensors));
    fill_block_shapes(container, config, matrix_dtype, other_dtype);
    return container;
}

GenericTensorContainer IModel::create_non_block_container(const TransformerConfig& config, ETensorDType matrix_dtype,
    ETensorDType other_dtype) const
{
    std::vector<Tensor> tensors(num_non_block_tensors());
    GenericTensorContainer container(std::move(tensors));
    fill_non_block_shapes(container, config, matrix_dtype, other_dtype);
    return container;
}

IRunState::IRunState(TransformerConfig config, long batch_size, long seq_len, std::shared_ptr<TensorAllocator> alloc) : Config(config), B(batch_size), T(seq_len), Allocator(std::move(alloc)) {
    int did;
    CUDA_CHECK(cudaGetDevice(&did));
    CUDA_CHECK(cudaGetDeviceProperties(&DeviceProp, did));

    long n_loss_groups = div_ceil(T, 512l);

    Inputs = Allocator->allocate(ETensorDType::INT32, "inputs", {B, T});
    Targets = Allocator->allocate(ETensorDType::INT32, "targets", {B, T});
    Inputs_CPU = Allocator->allocate(ETensorDType::INT32, "inputs_cpu", EAllocationType::PINNED, {B, T});
    Targets_CPU = Allocator->allocate(ETensorDType::INT32, "targets_cpu", EAllocationType::PINNED, {B, T});
    Losses = Allocator->allocate(ETensorDType::FP32, "losses", {B, T});
    GroupedLosses = Allocator->allocate(ETensorDType::FP32, "grouped_losses", {n_loss_groups});
    LSE = Allocator->allocate(ETensorDType::FP32, "lse", {B, T});

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
    LSEDone = create_named_event("lse_done");
    OptimizerDone = create_named_event("optimizer_done");

    Tensor host_buffer = Allocator->allocate(ETensorDType::FP32, "host_buffer", EAllocationType::PINNED, {3 + n_loss_groups});
    NormHost = host_buffer.get<float>();
    LSEHost = host_buffer.get<float>() + 1;
    LossHost = host_buffer.get<float>() + 3;
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

float IRunState::get_loss(int max_pos) const {
    CUDA_CHECK(cudaEventSynchronize(BackwardDone));
    if (max_pos != -1 && max_pos % 512 != 0) {
        throw std::logic_error("max_pos must be divisible by 512");
    }
    int avail_groups = GroupedLosses.nelem();
    int max_group = max_pos < 0 ? avail_groups: max_pos / 512;
    max_group = std::min(max_group, avail_groups);
    float loss = 0;
    for (int i = 0; i < max_group; ++i) {
        loss += LossHost[i];
    }
    return loss;
}

float IRunState::get_norm() const {
    CUDA_CHECK(cudaEventSynchronize(NormDone));
    return NormHost[0];
}

float IRunState::get_lse_max() const {
    CUDA_CHECK(cudaEventSynchronize(LSEDone));
    return LSEHost[0];
}

float IRunState::get_lse_sum() const {
    CUDA_CHECK(cudaEventSynchronize(LSEDone));
    return LSEHost[1];
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


std::pair<float, float> IRunState::record_step(float loss, float norm) {
    LossOutliers.record(loss);
    NormOutliers.record(norm);

    return {LossOutliers.eval(loss), NormOutliers.eval(norm)};
}


IRunState::OutlierDetector::OutlierDetector(int window_size) : mWindowSize(window_size){
    mValues.reserve(window_size);
}

void IRunState::OutlierDetector::record(float value) {
    double v_n = value;
    if (mValues.size() < mWindowSize) {
        // simply add the value and accumulate buffers
        mValues.push_back(value);
        mSum += v_n;
        mSumSq += v_n * v_n;
    } else {
        // we need to subtract the old value, and add the new one.
        double v_o = mValues[mIndex];
        mSum += v_n - v_o;
        mSumSq += v_n * v_n - v_o * v_o;

        mValues[mIndex] = value;
        mIndex = (mIndex + 1) % mWindowSize;
    }

    // periodically recompute to prevent accumulation of
    // rounding errors
    if (mIndex == 0 && mValues.size() == mWindowSize) {
        re_evaluate();
    }
}

float IRunState::OutlierDetector::eval(float value) const {
    if (mValues.size() < mWindowSize) {
        return 0.0;
    } else {
        double mean = mSum / mWindowSize;
        double variance = mSumSq / mWindowSize - mean * mean;
        double std_dev = std::sqrt(variance);
        if (std_dev == 0.0) {
            return 0.0;
        }
        return (static_cast<double>(value) - mean) / std_dev;
    }
}

void IRunState::OutlierDetector::re_evaluate() {
    mSum = 0.0;
    mSumSq = 0.0;
    for (float val : mValues) {
        mSum += val;
        mSumSq += val * val;
    }
}

void IRunState::OutlierDetector::reset(int window_size, int index, std::vector<float> values) {
    mWindowSize = window_size;
    mIndex = index;
    mValues = std::move(values);
    re_evaluate();
}
