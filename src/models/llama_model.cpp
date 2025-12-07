// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "llama_model.h"

#include <cmath>
#include <sstream>

#include "kernels/kernels.h"
#include "llama_gradients.h"
#include "llama_optimizer.h"
#include "llama_run_state.h"
#include "llama_weights.h"
#include "utilities/comm.h"

LLamaModel::LLamaModel(LLamaConfig config, const LLamaOptions& options, int rank, int world, const std::shared_ptr<TensorAllocator>& alloc) :
        Config(config), Options(options), Allocator(alloc ? alloc : std::make_shared<TensorAllocator>())
{
    Parameters = LLamaWeightsManager::create(Config, options, rank, world, *Allocator);
}

LLamaModel::~LLamaModel() = default;

void forward_qmm(Tensor& out, QuantizableTensor& inp, Tensor& weight, std::optional<Tensor> bias,
                 cublasLtHandle_t handle, Tensor workspace,
                 int B, int T, int C, int OC,
                 const cudaDeviceProp& dp, bool reuse_inp_quant,
                 cudaStream_t stream) {
    if (weight.DType == inp.Value.DType) {
        matmul(out, weight, inp.Value, bias, nullptr, nullptr, handle, workspace, OC, B*T, C, EMMTranspose::TN, false, stream);
        return;
   }

    if (!reuse_inp_quant) {
        quantize_with_abs_max(inp.Quant.value(), inp.Quant->scale(), inp.Value, inp.Quant->abs_max(), B*T*C, dp, stream);
    }

    if (weight.DType == ETensorDType::BF16) {
        matmul(out, weight, inp.Quant.value(), bias, nullptr, nullptr, handle, workspace, OC, B*T, C, EMMTranspose::TN, false, stream);
    } else {
        matmul(out, weight, inp.Quant.value(), bias, weight.scale(), inp.Quant->scale(), handle, workspace, OC, B*T, C, EMMTranspose::TN, false, stream);
    }
}

template<typename Function>
void trace_or_execute_cuda_graph(Function&& function, cudaStream_t stream, cudaGraphExec_t& instance, bool enabled) {
    if (enabled) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        function();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

        if (instance == nullptr) {
            CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
        }
        cudaGraphExecUpdateResultInfo result;
        if(auto status = cudaGraphExecUpdate(instance, graph, &result); status != cudaSuccess)
        {
            fprintf(stderr, "Graph update failed: %d\n", result.result);
            CUDA_CHECK(status);
        }
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
    } else {
        function();
    }
}

/// If `tensor` has quants, return their scales; otherwise, return nullptr
float* quant_abs_max_ptr(QuantizableTensor& tensor) {
    return tensor.Quant.has_value() ? tensor.Quant->abs_max() : nullptr;
}

void LLamaModel::forward(Tensor inputs, NCCLCommunicator& comm, int micro_step) {
    NVTX_RANGE_FN();

    if(Options.TriggerTimingEvents) {
        RunState->setup_timing_events(micro_step);
        CUDA_CHECK(cudaEventRecord(RunState->TimingForwardStart[micro_step], RunState->MainStream));
    }

    assert(inputs.DType == ETensorDType::INT32);
    auto& rs = RunState;
    cudaStream_t main_stream = rs->MainStream;
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    long V = Config.VocabSize;
    long C = Config.HiddenSize;

    // If this is the first micro-step, the parameters have just changed, and we can not
    // re-use any cached values
    if(micro_step == 0) {
        Parameters->invalidate();
    }

    assert(rs->Inputs.Sizes[0] >= B);
    assert(rs->Inputs.Sizes[1] >= T);
    assert(inputs.Device == -1);
    {
        NvtxRange r{"copy-input"};
        // no point running this copy on side stream: input is needed by embedding gradients, which is
        // the last op in backward.
        CUDA_CHECK(cudaMemcpyAsync(rs->Inputs.Data, inputs.Data, inputs.bytes(), cudaMemcpyHostToDevice, main_stream));
        CUDA_CHECK(cudaEventRecord(rs->TransferDone, main_stream));
    }

    {
        NvtxRange emb_range("embedding");
        Parameters->gather_embeddings(comm);
        encoder_forward(
            rs->Encoded,
            rs->Inputs,
            Parameters->get_embeddings(main_stream),
            std::nullopt, B, T, C, V, main_stream);
        Parameters->release_embeddings(main_stream);
    }

    if(rs->AbsMaxes.has_value())
        fill_zero(rs->AbsMaxes.value(), main_stream);

    Parameters->gather_block(0, comm, *rs);
    for (int l = 0; l < Config.NumLayers; l++) {
        NvtxRange layer_range("Layer", l);

        // prefetch
        if (l != Config.NumLayers - 1) {
            Parameters->gather_block(l + 1, comm, *rs);
        }

        auto& wgt = Parameters->get_block(l, main_stream);
        Tensor residual = l == 0 ? rs->Encoded : rs->get_res_ffn(l-1, main_stream);

        // fuse RMSNorm with residual, except in the first layer when no residual exists yet.
        // mark_res_ffn_ready records an event, and we need to wait for that event outside the
        // graph, so this block has to be separate.
        if (l == 0) {
            rmsnorm_forward(rs->Acts[0].LN1.Value, rs->Acts[0].LN1_Rstd, residual, wgt.LN1_w,
                            quant_abs_max_ptr(rs->Acts[0].LN1), Config.RmsNormEps, B, T, C, main_stream);
        } else {
            auto& prev = rs->Acts[l-1];
            fused_residual_rmsnorm_forward(residual, rs->Acts[l].LN1.Value, rs->Acts[l].LN1_Rstd,
                                           prev.ResidualAtt, prev.MlpDown, wgt.LN1_w,
                                           quant_abs_max_ptr(rs->Acts[l].LN1),
                                           Config.RmsNormEps, B * T, C, main_stream);
            rs->mark_res_ffn_ready(l-1, main_stream);
        }

        rs->Acts[l].MlpUp = rs->acquire_mlp_up(l);
        trace_or_execute_cuda_graph([&](){_forward_block(wgt, rs->Acts[l], residual);},
            main_stream, rs->ForwardBlockGraph, rs->Options.UseCudaGraphs);
        Parameters->release_block(l, main_stream);
        rs->release_mlp_up(rs->Acts[l].MlpUp);
        if(l > 0) {
            rs->put_res_ffn(l-1, rs->SideStream);
        }
    }

    {
        NvtxRange r{"LNF"};
        auto& acts = rs->Acts[Config.NumLayers-1];
        Parameters->gather_lnf(comm);
        fused_residual_rmsnorm_forward(rs->get_res_ffn(Config.NumLayers - 1, main_stream), rs->LNF, rs->LNF_Rstd, acts.ResidualAtt,
                                       acts.MlpDown, Parameters->get_lnf(main_stream), nullptr, Config.RmsNormEps, B * T, C, main_stream);
        Parameters->release_lnf(main_stream);
        rs->mark_res_ffn_ready(Config.NumLayers-1, main_stream);
        rs->put_res_ffn(Config.NumLayers-1, rs->SideStream);
    }

    // do not return before inputs can be accessed again.
    CUDA_CHECK(cudaEventSynchronize(rs->TransferDone));
    CUDA_CHECK(cudaEventRecord(rs->ForwardDone, main_stream));

    if(Options.TriggerTimingEvents) {
        CUDA_CHECK(cudaEventRecord(RunState->TimingForwardEnd[micro_step], RunState->MainStream));
    }

}

void LLamaModel::_forward_block(sLLamaBlockWeights<Tensor>& weights, sLLamaLayerActivations& acts, Tensor& residual)
{
    auto& rs = RunState;
    long B = rs->Inputs.Sizes[0];
    long T = rs->Inputs.Sizes[1];
    long C = Config.HiddenSize;
    long D = Config.IntermediateSize;
    long Hq = Config.NumQueryHeads;
    long Hkv = Config.NumKeyValHeads;
    long Hs = Config.head_size();
    cudaStream_t main_stream = rs->MainStream;

    // 1) projection to QKV vectors (note k,v may be fewer heads than q)
    forward_qmm(acts.QKV, acts.LN1, weights.Attn_QKV_w, weights.Attn_QKV_b,
                rs->CublasLtHandle, rs->CuBlasWorkspace,
                B, T, C, Config.qkv_channels(),
                rs->DeviceProp, false, main_stream);
    // 2) apply RoPE to q,k (potentially in place)
    rope_forward(acts.QKV, acts.QKV, rs->FreqCis, nullptr, B, T, Hq, Hkv, Hs, main_stream);
    // 3) attention: att <- softmax(qk^T)v
    attention_forward_cudnn(acts.Att.Value, acts.LSE, acts.QKV, rs->CuBlasWorkspace, rs->CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);
    // quantize attention if necessary
    if(acts.Att.Quant.has_value()) {
        abs_max(acts.Att.Quant->abs_max(), acts.Att.Value, acts.Att.Value.nelem(), rs->DeviceProp, main_stream);
    }

    forward_qmm(acts.AttO, acts.Att, weights.Attn_Out_w, std::nullopt,
                rs->CublasLtHandle, rs->CuBlasWorkspace,
                B, T, C, C,
                rs->DeviceProp, false, main_stream);

    fused_residual_rmsnorm_forward(acts.ResidualAtt, acts.LN2.Value, acts.LN2_Rstd, residual, acts.AttO, weights.LN2_w,
                                   quant_abs_max_ptr(acts.LN2), Config.RmsNormEps, B * T, C, main_stream);

    forward_qmm(acts.MlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
                rs->CublasLtHandle, rs->CuBlasWorkspace,
                B, T, C, 2 * D,
                rs->DeviceProp, false, main_stream);
    swiglu_forward(acts.SwiGLu.Value, acts.MlpUp, quant_abs_max_ptr(acts.SwiGLu), B, T, D, main_stream);

    forward_qmm(acts.MlpDown, acts.SwiGLu, weights.MLP_Down_w, std::nullopt,
                rs->CublasLtHandle, rs->CuBlasWorkspace,
                B, T, D, C,
                rs->DeviceProp, false, main_stream);
}

float LLamaModel::validate(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    NVTX_RANGE_FN();
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    auto& rs = RunState;
    const size_t V = Config.VocabSize;
    const size_t Vp = Config.VocabSize;
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    long C = Config.HiddenSize;

    cudaStream_t main_stream = rs->MainStream;

    forward(inputs, comm, micro_step);

    NvtxRange classifier_and_loss_range("classifier_and_loss");
    // fused classifier: does the forward pass and first part of the backward pass
    const float d_loss = 1.0f / float(B * T); // results in the uniform average loss over all elements
    // note: we don't need to generate dlogits here
    fill_zero(rs->Losses, main_stream);
    if(targets.Device == -1) {
        CUDA_CHECK(cudaMemcpy(rs->Targets.Data, targets.Data, targets.bytes(), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(rs->Targets.Data, targets.Data, targets.bytes(), cudaMemcpyDeviceToDevice));
    }


    long nano_batches = Options.LMHeadChunks;
    int nano_batch_size = div_exact(B * T, nano_batches);
    Parameters->gather_head(comm);
    rs->temp_acquire(rs->Output);
    for(int nano_step = 0; nano_step < nano_batches; nano_step++) {
        Tensor lnf_slice = rs->LNF;
        lnf_slice.Data += nano_step * nano_batch_size * C * get_dtype_size(lnf_slice.DType);
        Tensor tgt = rs->Targets;
        tgt.Data += nano_step *  nano_batch_size * get_dtype_size(tgt.DType);
        Tensor losses = rs->Losses;
        losses.Data += nano_step * nano_batch_size * get_dtype_size(losses.DType);

        matmul(rs->Output, Parameters->get_head(main_stream), lnf_slice,
               std::nullopt, nullptr, nullptr, rs->CublasLtHandle, rs->CuBlasWorkspace, V, nano_batch_size, C, EMMTranspose::TN, false, main_stream);

        // accumulate the losses inside rs->losses, and kick off the backward pass inside the fused classifier
        fused_classifier(rs->Output, losses, d_loss, tgt, nano_batch_size, V, Vp, true, main_stream);
    }
    rs->temp_free(rs->Output);
    Parameters->release_head(main_stream);
    _reduce_loss(*rs, comm, B, T);

    CUDA_CHECK(cudaDeviceSynchronize());
    *rs->LossHost /= B * T;
    return *rs->LossHost;
}

void backward_qmm(Tensor& dinp, Tensor& dweight, std::optional<Tensor> dbias,
                  QuantizableTensor& dout, QuantizableTensor& inp, Tensor& weight, std::optional<Tensor> bias_buffer,
                  bool accumulate_gradient,
                  LLamaRunState& rs,
                  int B, int T, int C, int OC,
                  bool reuse_inp, cudaStream_t stream) {
    if (weight.DType == inp.Value.DType) {
        matmul(dinp, weight, dout.Value, std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace, C, B*T, OC, EMMTranspose::NN, false, stream);
        matmul(dweight, inp.Value, dout.Value, std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace, C, OC, B*T, EMMTranspose::NT, accumulate_gradient, stream);

        if (dbias.has_value()) {
            backward_bias(dbias.value(), dout.Value, nullptr, nullptr, bias_buffer.value(), B, T, OC, rs.DeviceProp, stream);
        }
    } else if (weight.DType == ETensorDType::BF16) {
        quantize_with_abs_max(dout.Quant.value(), dout.Quant->scale(), dout.Value, nullptr, B*T*OC, rs.DeviceProp, stream);
        if(!reuse_inp) {
            quantize_with_abs_max(inp.Quant.value(), dout.Quant->scale(), inp.Value, nullptr, B*T*C, rs.DeviceProp, stream);
        }

        matmul(dinp, weight, dout.Quant.value(), std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace, C, B*T, OC, EMMTranspose::NN, false, stream);
        matmul(dweight, inp.Quant.value(), dout.Quant.value(), std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace, C, OC, B*T, EMMTranspose::NT, accumulate_gradient, stream);

        if (dbias.has_value()) {
            backward_bias(dbias.value(), dout.Value, nullptr, nullptr, bias_buffer.value(), B, T, OC, rs.DeviceProp, stream);
        }
    } else {
        quantize_with_abs_max(dout.Quant.value(), dout.Quant->scale(), dout.Value, dout.Quant->abs_max(), B*T*OC, rs.DeviceProp, stream);

        auto& inp_q = inp.Quant.value();
        auto weight_tp = rs.temp_alloc(inp_q.DType, {C, OC});
        transpose(weight_tp, weight, OC, C, stream);

        matmul(dinp, weight_tp, dout.Quant.value(), std::nullopt, weight.scale(), dout.Quant->scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace, C, B*T, OC, EMMTranspose::TN, false, stream);
        rs.temp_free(weight_tp);

        auto activation_tp = rs.temp_alloc(inp_q.DType, {C, B*T});
        auto grad_tp = rs.temp_alloc(rs.Options.grad_dtype(), {OC, B*T});
        if(reuse_inp) {
            // inp is already quantized from the forward pass, so just transpose here
            transpose(activation_tp, inp_q, B*T, C, stream);
        } else {
            // even though we're re-using (and overwriting) the main tensor, each tensor still has its own version
            // of the absmax-scale, so we can reuse the existing scale from the forward pass
            quantize_and_transpose_with_abs_max(activation_tp, activation_tp.scale(), inp.Value, inp.Quant->abs_max(), B*T, C, rs.DeviceProp, stream);
        }
        transpose(grad_tp, dout.Quant.value(), B*T, OC, stream);

        matmul(dweight, activation_tp, grad_tp, std::nullopt, inp_q.scale(), dout.Quant->scale(), rs.CublasLtHandle, rs.CuBlasWorkspace, C, OC, B*T, EMMTranspose::TN, accumulate_gradient, stream);
        if (dbias.has_value()) {
            backward_bias(dbias.value(), dout.Quant.value(), inp_q.scale(), dout.Quant->scale(), bias_buffer.value(), B, T, OC, rs.DeviceProp, stream);
        }
        rs.temp_free(grad_tp);
        rs.temp_free(activation_tp);
    }
}


void LLamaModel::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    auto& rs = RunState;
    cudaStream_t main_stream = rs->MainStream;

    NVTX_RANGE_FN();

    if(Options.TriggerTimingEvents) {
        CUDA_CHECK(cudaEventRecord(rs->TimingBackwardStart[micro_step], main_stream));
    }

    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    const size_t C = Config.HiddenSize;
    const size_t L = Config.NumLayers;
    {
        NvtxRange r{"copy-targets"};
        // make sure rs->Targets is no longer needed by the previous step.
        CUDA_CHECK(cudaStreamWaitEvent(rs->SideStream, rs->BackwardDone, 0));
        CUDA_CHECK(cudaMemcpyAsync(rs->Targets.Data, targets.Data, targets.bytes(), cudaMemcpyHostToDevice, rs->SideStream));
        CUDA_CHECK(cudaEventRecord(rs->TransferDone, rs->SideStream));
        // we will wait in _backward_lmhead for this transfer to be done.
    }

    bool last_step = micro_step == grad_accum_steps - 1;
    // on the first micro-step zero the gradients, as we're about to += accumulate into them
    if (micro_step == 0) {
        NvtxRange classifier_and_loss_range("zero gradients");
        // there are currently two state vars during the gradient accumulation inner loop:
        // 1) the losses accumulate += into rs->losses, reset here
        // 2) the gradients accumulate += into grads_memory, reset here
        fill_zero(rs->Losses, main_stream);
        Grads->start_micro_step(rs->SideStream, micro_step, grad_accum_steps);
        CUDA_CHECK(cudaEventRecord(rs->SideStreamEvent, rs->SideStream));
    } else {
        Grads->start_micro_step(main_stream, micro_step, grad_accum_steps);
    }

    // reset residual stream gradients (put here to work with gradient accumulation)
    fill_zero(rs->DLNF, main_stream);
    fill_zero(rs->DActs[L-1].DResFFN.Value, main_stream);
    _backward_lmhead(B, T, micro_step, grad_accum_steps, comm);

    // ok, now reduce the loss across all ranks
    if (last_step) {
        _reduce_loss(*rs, comm, B, T);
    }

    bool accumulate;
    auto& d_lnf_w = Grads->get_lnf_w_full(main_stream, comm, accumulate);
    Parameters->gather_lnf(comm);
    // backward the final layernorm
    rmsnorm_backward(rs->DActs[L-1].DResFFN.Value, d_lnf_w, rs->RMSNormScratch, rs->DActs[L - 1].DResFFN.Value, rs->DLNF,
                     rs->get_res_ffn(L-1, main_stream), Parameters->get_lnf(main_stream), rs->LNF_Rstd,
                     quant_abs_max_ptr(rs->DActs[L-1].DResFFN), B, T, C, rs->DeviceProp, main_stream);
    rs->release_res_ffn(L-1, main_stream);

    Parameters->release_lnf(main_stream);
    Grads->notify_lnf_w(main_stream, comm);
    rs->fetch_res_ffn(L-2, comm.stream());
    Parameters->gather_block(L - 1, comm, *rs);
    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);
        auto& dw = Grads->get_block_full(l, main_stream, comm, accumulate);

        // prefetch previous layer
        if(l > 1) {
            rs->fetch_res_ffn(l-2, comm.stream());
        }
        if(l > 0) {
            Parameters->gather_block(l - 1, comm, *rs);
        } else if (!last_step) {
            Parameters->gather_embeddings(comm);
        }

        auto& weights = Parameters->get_block(l, main_stream);
        auto& d_acts = rs->DActs.at(l);
        Tensor residual = l == 0 ? rs->Encoded : rs->get_res_ffn(l - 1, main_stream);
        rs->Acts[l].MlpUp = rs->acquire_mlp_up(l);
        rs->DActs[l].DMlpUp.Value = rs->Acts[l].MlpUp;
        trace_or_execute_cuda_graph([&]() {
            _recompute_block(weights, rs->Acts[l], residual);
            _backward_block(accumulate, weights, dw, rs->Acts[l], rs->DActs[l]);
            }, main_stream, rs->BackwardBlockGraph, rs->Options.UseCudaGraphs);
        rs->release_mlp_up(rs->Acts[l].MlpUp);

        if(l > 0) {
            auto& prev_dacts = rs->DActs.at(l - 1);
            rmsnorm_backward(prev_dacts.DResFFN.Value, dw.LN1_w, rs->RMSNormScratch, prev_dacts.DResAtt.Value, d_acts.DLN1,
                             rs->get_res_ffn(l-1, main_stream), weights.LN1_w, rs->Acts[l].LN1_Rstd, quant_abs_max_ptr(prev_dacts.DResFFN),
                             B, T, C, rs->DeviceProp, main_stream);
            rs->release_res_ffn(l - 1, main_stream);
        } else {
            rmsnorm_backward(rs->DEmb, dw.LN1_w, rs->RMSNormScratch, d_acts.DResAtt.Value, d_acts.DLN1,
                             rs->Encoded, weights.LN1_w, rs->Acts[l].LN1_Rstd, nullptr, B, T, C, rs->DeviceProp, main_stream);
        }
        Parameters->release_block(l, main_stream);
        Grads->notify_block(l, main_stream, comm);
    }

    auto& d_emb = Grads->get_embeddings_full(main_stream, comm, accumulate);
    encoder_backward(d_emb, rs->EncoderBwdScratch, rs->EncoderBwdIndices, rs->EncoderBwdInfo,
                     rs->DEmb, rs->Inputs, inputs, B, T, C, OptimizerRNG(), main_stream, rs->SideStreamEvent, rs->SideStream);
    Grads->notify_embeddings(main_stream, comm);

    // make sure all gradients are communicated before we go to the update step.
    Grads->end_micro_step(main_stream, comm);
    CUDA_CHECK(cudaEventRecord(rs->BackwardDone, main_stream));

    // do not return before inputs can be accessed again.
    CUDA_CHECK(cudaEventSynchronize(rs->TransferDone));

    if(Options.TriggerTimingEvents) {
        CUDA_CHECK(cudaEventRecord(rs->TimingBackwardEnd[micro_step], main_stream));
    }
}

void LLamaModel::_backward_lmhead(long B, long T, int micro_step, int grad_accum_steps, NCCLCommunicator& comm) {
    auto& rs = RunState;
    const size_t C = Config.HiddenSize;
    const size_t V = Config.VocabSize;
    const size_t Vp = Config.VocabSize;
    cudaStream_t main_stream = rs->MainStream;

    if(Options.TriggerTimingEvents) {
        CUDA_CHECK(cudaEventRecord(rs->TimingHeadStart[micro_step], main_stream));
    }

    long nano_batches = Options.LMHeadChunks;
    int nano_batch_size = div_exact(B * T, nano_batches);

    const float d_loss =
        1.0f / (float) (B * T * grad_accum_steps); // results in the uniform average loss over all elements

    NvtxRange classifier_and_loss_range("lm-head");
    Parameters->gather_head(comm);
    rs->temp_acquire(rs->Output);
    for (int nano_step = 0; nano_step < nano_batches; nano_step++) {
        Tensor lnf_slice = rs->LNF;
        lnf_slice.Data += nano_step * nano_batch_size * C * get_dtype_size(lnf_slice.DType);
        Tensor tgt = rs->Targets;
        tgt.Data += nano_step * nano_batch_size * get_dtype_size(tgt.DType);
        Tensor losses = rs->Losses;
        losses.Data += nano_step * nano_batch_size * get_dtype_size(losses.DType);
        Tensor dlnf_slice = rs->DLNF;
        dlnf_slice.Data += nano_step * nano_batch_size * C * get_dtype_size(dlnf_slice.DType);

        matmul(rs->Output, Parameters->get_head(main_stream), lnf_slice, std::nullopt,
               nullptr, nullptr, rs->CublasLtHandle, rs->CuBlasWorkspace, V, nano_batch_size, C, EMMTranspose::TN,
               false, main_stream);

        if(nano_step == 0) {
            // make sure Targets have been copied
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs->TransferDone, 0));
        }

        // accumulate the losses inside rs->losses, and kick off the backward pass inside the fused classifier
        fused_classifier(rs->Output, losses, d_loss, tgt, nano_batch_size, V, Vp, true, main_stream);

        // if we reset model grads to zero, now is the time we need to wait
        if (micro_step == 0 && nano_step == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs->SideStreamEvent, 0));
        }

        // handle the LM-head. We run the d_lmhead matmul first, so that the gradient reduction can overlap with the DLNF matmul.
        bool accumulate;
        auto& d_lmhead = Grads->get_lmhead_full(main_stream, comm, accumulate);
        accumulate |= nano_step != 0;
        matmul(d_lmhead, lnf_slice, rs->Output, std::nullopt, nullptr, nullptr,
               rs->CublasLtHandle, rs->CuBlasWorkspace, C, V, nano_batch_size, EMMTranspose::NT, accumulate, main_stream);
        if (nano_step == nano_batches - 1) {
            Grads->notify_lmhead(main_stream, comm);
        }

        matmul(dlnf_slice, Parameters->get_head(main_stream), rs->Output, std::nullopt, nullptr, nullptr,
               rs->CublasLtHandle, rs->CuBlasWorkspace, C, nano_batch_size, V, EMMTranspose::NN, false, main_stream);

    }
    rs->temp_free(rs->Output);
    Parameters->release_head(main_stream);


    if(Options.TriggerTimingEvents) {
        CUDA_CHECK(cudaEventRecord(rs->TimingHeadEnd[micro_step], main_stream));
    }
}

void LLamaModel::_recompute_block(sLLamaBlockWeights<Tensor>& weights, sLLamaLayerActivations& acts, Tensor& residual) {
    NvtxRange classifier_and_loss_range("recompute");
    auto& rs = RunState;
    cudaStream_t main_stream = rs->MainStream;
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    long B = rs->Inputs.Sizes[0];
    long T = rs->Inputs.Sizes[1];
    const size_t C = Config.HiddenSize;
    long D = Config.IntermediateSize;
    long Hq = Config.NumQueryHeads;
    long Hkv = Config.NumKeyValHeads;
    long Hs = Config.head_size();

    auto& opt = rs->Options;

    // Figure out which parts we need to recompute
    bool recompute_ln1 = opt.RecomputeRMSNorm || opt.RecomputeAtt || opt.RecomputeBlock;
    bool recompute_ln2 = opt.RecomputeRMSNorm || opt.RecomputeFFN || opt.RecomputeBlock;
    bool recompute_qkv = opt.RecomputeQKV || opt.RecomputeAtt || opt.RecomputeBlock;
    bool recompute_swiglu = opt.RecomputeSwiGLu || opt.RecomputeFFN || opt.RecomputeBlock;
    bool recompute_att = opt.RecomputeAtt || opt.RecomputeBlock;

    // Attention block
    if(recompute_ln1) {
        rmsnorm_forward(acts.LN1.Value, acts.LN1_Rstd, residual, weights.LN1_w, nullptr, Config.RmsNormEps, B, T, C, main_stream);
    }

    if (recompute_qkv) {
        // two scenarios: 1) we do not recompute the RMSnorm; then, we _will_ overwrite the full-precision copy of acts.LN1,
        //                   but _can_ reuse the quantized version
        //                2) we recompute RMSNorm; then, acts.LN1 will be correct, but its quantized version will not, so
        //                   we have to re-quantize
        forward_qmm(acts.QKV, acts.LN1, weights.Attn_QKV_w, weights.Attn_QKV_b,
                     rs->CublasLtHandle, rs->CuBlasWorkspace,
                     B, T, C, Config.qkv_channels(),
                     rs->DeviceProp, !recompute_ln1, main_stream);
        rope_forward(acts.QKV, acts.QKV, rs->FreqCis, nullptr, B, T, Hq, Hkv, Hs, main_stream);
    }

    if (recompute_att) {
        attention_forward_cudnn(acts.Att.Value, acts.LSE, acts.QKV, rs->CuBlasWorkspace, rs->CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);
        // AttO not needed in backward pass; but if we want to recompute the entire transformer block, we need its output
        // to recompute the FFN part
        if (opt.RecomputeBlock) {
            forward_qmm(acts.AttO, acts.Att, weights.Attn_Out_w, std::nullopt,
                         rs->CublasLtHandle, rs->CuBlasWorkspace,
                         B, T, C, C,
                         rs->DeviceProp, false, main_stream);
        }
    }

    // Feed-forward block
    if(recompute_ln2) {
        if (opt.RecomputeBlock) {
            fused_residual_rmsnorm_forward(acts.ResidualAtt, acts.LN2.Value, acts.LN2_Rstd, residual, acts.AttO, weights.LN2_w,
                                 nullptr, Config.RmsNormEps, B * T, C, main_stream);
        } else {
            rmsnorm_forward(acts.LN2.Value, acts.LN2_Rstd, acts.ResidualAtt, weights.LN2_w,
                  nullptr, Config.RmsNormEps, B, T, C, main_stream);
        }
    }

    if(opt.RecomputeFFN) {
        forward_qmm(acts.MlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
                         rs->CublasLtHandle, rs->CuBlasWorkspace,
                         B, T, C, 2 * D,
                         rs->DeviceProp, false, main_stream);
    }

    if(recompute_swiglu) {
        if (acts.SwiGLu.Quant.has_value()) {
            swiglu_forward_quant(acts.SwiGLu.Quant.value(), acts.SwiGLu.Quant->scale(), acts.MlpUp, acts.SwiGLu.Quant->abs_max(), B, T, D, main_stream);
        } else {
            swiglu_forward(acts.SwiGLu.Value, acts.MlpUp, nullptr, B, T, D, main_stream);
        }
    }
}

void LLamaModel::_backward_block(bool accumulate, sLLamaBlockWeights<Tensor>& weights, sLLamaGradBlock& d_weights,
                                 sLLamaLayerActivations& acts, sLLamaLayerGradients& d_acts) {
    auto& rs = RunState;
    cudaStream_t main_stream = rs->MainStream;
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    long B = rs->Inputs.Sizes[0];
    long T = rs->Inputs.Sizes[1];
    const size_t C = Config.HiddenSize;
    long D = Config.IntermediateSize;
    long Hq = Config.NumQueryHeads;
    long Hkv = Config.NumKeyValHeads;
    long Hs = Config.head_size();

    // backward the 2nd matmul of MLP
    // note that _recompute_block guarantees that if SwiGLu is already quantized (if necessary)
    rs->temp_acquire(d_acts.DSwiGLU);
    backward_qmm(d_acts.DSwiGLU, d_weights.MLP_Down_w, std::nullopt, d_acts.DResFFN, acts.SwiGLu, weights.MLP_Down_w, std::nullopt,
                 accumulate, *rs, B, T, D, C, true, main_stream);

    swiglu_backward(d_acts.DMlpUp.Value, d_acts.DSwiGLU, acts.MlpUp, quant_abs_max_ptr(d_acts.DMlpUp), B, T, D, main_stream);
    rs->temp_free(d_acts.DSwiGLU);

    if(d_acts.DMlpUp.Quant.has_value()) {
        rs->temp_acquire(d_acts.DMlpUp.Quant.value());
    }
    backward_qmm(d_acts.DLN2, d_weights.MLP_Up_w, std::nullopt, d_acts.DMlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
                 accumulate, *rs, B, T, C, 2 * D, !rs->Options.RecomputeRMSNorm, main_stream);
    if(d_acts.DMlpUp.Quant.has_value()) {
        rs->temp_free(d_acts.DMlpUp.Quant.value());
    }

    // rmsnorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
    rmsnorm_backward(d_acts.DResAtt.Value, d_weights.LN2_w, rs->RMSNormScratch, d_acts.DResFFN.Value, d_acts.DLN2,
                     acts.ResidualAtt, weights.LN2_w, acts.LN2_Rstd, quant_abs_max_ptr(d_acts.DResAtt), B, T, C, rs->DeviceProp, main_stream);

    bool recompute_ln1 = rs->Options.RecomputeRMSNorm || rs->Options.RecomputeAtt;
    backward_qmm(d_acts.DAttY, d_weights.Attn_Out_w, std::nullopt, d_acts.DResAtt, acts.Att, weights.Attn_Out_w, std::nullopt,
                 accumulate, *rs, B, T, C, C, false, main_stream);

    rs->temp_acquire(d_acts.DQKV.Value);
    rs->temp_acquire(rs->CuDNNWorkspace);
    for (int i=0; i < Options.AttBwdChunks; ++i) {
        long chunk_batch_size = div_exact(B, (long)Options.AttBwdChunks);
        Tensor d_qkv = shard_view(d_acts.DQKV.Value, i, Options.AttBwdChunks);
        Tensor lse = shard_view(acts.LSE, i, Options.AttBwdChunks);
        Tensor att = shard_view(acts.Att.Value, i, Options.AttBwdChunks);
        Tensor d_atty = shard_view(d_acts.DAttY, i, Options.AttBwdChunks);
        Tensor qkv = shard_view(acts.QKV, i, Options.AttBwdChunks);
        attention_backward_cudnn(d_qkv, lse, att, d_atty, qkv, rs->CuDNNWorkspace, rs->CudnnHandle,
            chunk_batch_size, T, Hq, Hkv, Hs, main_stream);
    }
    rs->temp_free(rs->CuDNNWorkspace);
    rope_backward(d_acts.DQKV.Value, d_acts.DQKV.Value, rs->FreqCis, quant_abs_max_ptr(d_acts.DQKV), B, T, Hq, Hkv, Hs, main_stream);

    backward_qmm(d_acts.DLN1, d_weights.Attn_QKV_w, d_weights.Attn_QKV_b, d_acts.DQKV, acts.LN1, weights.Attn_QKV_w, rs->MatmulBiasScratch,
                 accumulate, *rs, B, T, C, Config.qkv_channels(), !recompute_ln1, main_stream);
    rs->temp_free(d_acts.DQKV.Value);
}

void LLamaModel::_reduce_loss(LLamaRunState& acts, NCCLCommunicator& comm, int B, int T) {
    NVTX_RANGE_FN();
    // reduce all the losses within the current GPU (across all microsteps)
    deterministic_sum(acts.Losses.get<float>(), acts.Losses.get<float>(), B*T, acts.MainStream);
    // reduce loss across GPUs to a single, final float across all GPUs
    comm.reduce_loss(acts.Losses.get<float>(), acts.MainStream);
    CUDA_CHECK(cudaMemcpyAsync(acts.LossHost, acts.Losses.get<float>(), sizeof(float), cudaMemcpyDeviceToHost, acts.MainStream));
}

void LLamaModel::calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip) {
    NVTX_RANGE_FN();
    auto& rs = RunState;

    cudaStream_t main_stream = rs->MainStream;
    CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs->BackwardDone));

    if(rs->Options.UseCudaGraphs) {
        if(!rs->GlobalNormGraph) {
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeThreadLocal));
            _calculate_gradient_norm(comm, grad_clip);
            CUDA_CHECK(cudaStreamEndCapture(main_stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&rs->GlobalNormGraph, graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        CUDA_CHECK(cudaGraphLaunch(rs->GlobalNormGraph, main_stream));
    } else {
        _calculate_gradient_norm(comm, grad_clip);
    }

    CUDA_CHECK(cudaEventRecord(rs->NormDone, main_stream));
}

void LLamaModel::_calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip) {
    auto& rs = RunState;
    cudaStream_t main_stream = rs->MainStream;

    fill_zero(rs->NormBuffer, main_stream);
    auto norm_squared = [&](const TensorShard& grad){
        global_norm_squared(rs->NormBuffer, grad, grad.nelem(), rs->DeviceProp, main_stream);
    };

    norm_squared(Grads->get_embeddings_shard(main_stream));

    if(!Config.TiedWordEmbeddings) {
        norm_squared(Grads->get_lmhead_shard(main_stream));
    }
    norm_squared(Grads->get_lnf_w_shard(main_stream));

    for(int i = 0; i < Config.NumLayers; i++) {
        auto& block = Grads->get_block_shard(i, main_stream);
        norm_squared(block.LN1_w);
        norm_squared(block.LN2_w);
        norm_squared(block.Attn_QKV_w);
        if(block.Attn_QKV_b.has_value()) {
            norm_squared(block.Attn_QKV_b.value());
        }
        norm_squared(block.Attn_Out_w);
        norm_squared(block.MLP_Up_w);
        norm_squared(block.MLP_Down_w);
    }

    // final reduction to a single norm-squared element
    deterministic_sum(rs->NormBuffer.get<float>(), rs->NormBuffer.get<float>(), rs->NormBuffer.nelem(), main_stream);

    // potential cross-gpu reduction
    comm.reduce_norm(rs->NormBuffer.get<float>(), main_stream);

    // tiny kernel (1 thread) that calculates norm, scale factor, and puts the result on the host for later display
    global_norm_sqrt(rs->NormBuffer.get<float>(), rs->NormHost, grad_clip, rs->DeviceProp, main_stream);
}

void LLamaModel::update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon, float weight_decay, float grad_clip) {
    NVTX_RANGE_FN();
    auto& rs = RunState;
    cudaStream_t main_stream = rs->MainStream;

    if(!OptimizerState) {
        throw std::logic_error("LLamaModel::update() but no optimizer available");
    }

    auto& rng = OptimizerRNG;

    if(Options.TriggerTimingEvents) {
        CUDA_CHECK(cudaEventRecord(rs->TimingOptimizerStart, main_stream));
    }

    Parameters->begin_optimizer(rs->Stack, rs->MainStream);
    OptimizerState->begin_optimizer(rs->Stack);

    // grad_scale gets deposited into NormBuffer[1] and can be used on main_stream after this.
    calculate_gradient_norm(comm, grad_clip);
    float* grad_scale = rs->NormBuffer.get<float>() + 1;

    auto run_update = [&](Tensor& val, Tensor& grad, Tensor& m, Tensor& v, Tensor& scales, float wd) {
        adamw_update(val, grad, m, v, grad.nelem(),
                     learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale, scales, val.abs_max(), rng(), main_stream);
    };

    auto& m_scales = OptimizerState->scales_m();

    run_update(Parameters->get_master_embeddings(), Grads->get_embeddings_shard(main_stream),
               OptimizerState->non_block_m().Embeddings, OptimizerState->non_block_v().Embeddings, m_scales.NonBlocks.Embeddings, weight_decay);
    comm.reduce_max(Parameters->get_master_embeddings().abs_max());
    run_update(Parameters->get_master_lnf_w(), Grads->get_lnf_w_shard(main_stream),
               OptimizerState->non_block_m().LNF_w, OptimizerState->non_block_v().LNF_w, m_scales.NonBlocks.LNF_w, 0.f);
    comm.reduce_max(Parameters->get_master_lnf_w().abs_max());
    CUDA_CHECK(cudaEventRecord(rs->OptEmbeddingsDone, main_stream));

    for(int i = 0; i < Config.NumLayers; i++) {
        NvtxRange layer_range("Layer", i);
        Parameters->fetch_master_block(i, comm.stream());
        OptimizerState->fetch_block(i, comm.stream());
        auto& bw = Parameters->get_master_block(i, main_stream);
        auto& bg = Grads->get_block_shard(i, main_stream);
        auto& bm = OptimizerState->get_block_m(i, main_stream);
        auto& bv = OptimizerState->get_block_v(i, main_stream);
        auto& sm = m_scales.Blocks[i];
        run_update(bw.LN1_w, bg.LN1_w, bm.LN1_w, bv.LN1_w, sm.LN1_w, 0.f);
        run_update(bw.LN2_w, bg.LN2_w, bm.LN2_w, bv.LN2_w, sm.LN2_w, 0.f);

        run_update(bw.Attn_QKV_w, bg.Attn_QKV_w, bm.Attn_QKV_w, bv.Attn_QKV_w,
                   sm.Attn_QKV_w, weight_decay);
        if(bm.Attn_QKV_b.has_value()) {
            Tensor qkv_b_scales = sm.Attn_QKV_b.value_or(Tensor{});
            run_update(bw.Attn_QKV_b.value(), bg.Attn_QKV_b.value(), bm.Attn_QKV_b.value(),
                         bv.Attn_QKV_b.value(), qkv_b_scales, 0.f);
        }
        run_update(bw.Attn_Out_w, bg.Attn_Out_w, bm.Attn_Out_w, bv.Attn_Out_w, sm.Attn_Out_w, weight_decay);

        run_update(bw.MLP_Up_w, bg.MLP_Up_w, bm.MLP_Up_w, bv.MLP_Up_w, sm.MLP_Up_w, weight_decay);
        run_update(bw.MLP_Down_w, bg.MLP_Down_w, bm.MLP_Down_w, bv.MLP_Down_w, sm.MLP_Down_w, weight_decay);
        auto scales = Parameters->get_scales_for_block(i);
        // yes, we run this on main stream. Yes, this isn't nice because it prevents kernels from running in parallel.
        // the communication is tiny, though, so it doesn't matter, and this setup guarantees that the abs-maxes are
        // ready once we try to quantize on the main stream (which happens in `release_master_block`), so in that case
        // we'd have to wait anyway.
        // TODO there's probably a way to schedule this so that we can avoid this idle time. If it turns out to actually
        //      matter (e.g., for small models), we can investigate more.
        comm.reduce_max(scales.first, scales.second - scales.first, main_stream);
        Parameters->release_master_block(i, main_stream, rs->SideStream, *rs);
        OptimizerState->store_block(i, main_stream, rs->SideStream);

        CUDA_CHECK(cudaEventRecord(rs->LayerUpdateDone[i], main_stream));
    }

    if(!Config.TiedWordEmbeddings) {
        run_update(Parameters->get_master_lmhead(), Grads->get_lmhead_shard(main_stream),
                   OptimizerState->non_block_m().LMHead, OptimizerState->non_block_v().LMHead, m_scales.NonBlocks.LMHead, weight_decay);
        comm.reduce_max(Parameters->get_master_lmhead().abs_max());
    }
    comm.wait_on_comms(main_stream);
    OptimizerState->end_optimizer(rs->Stack);
    Parameters->end_optimizer(rs->Stack);
    CUDA_CHECK(cudaEventRecord(rs->OptimizerDone, main_stream));
    if(Options.TriggerTimingEvents) {
        CUDA_CHECK(cudaEventRecord(rs->TimingOptimizerEnd, main_stream));
    }
}

void LLamaModel::allocate_run_state(const LLamaOptions& options, NCCLCommunicator& comm, int B, int T) {
    NVTX_RANGE_FN();

    std::vector<std::pair<const char*, std::size_t>> stack_watermark;

    // create a dummy stack and simulate the way we're going to use temporaries later, to determine how much we need to allocate
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    DeviceMemoryStack stack(nullptr, 1024 * 1024 * 1024 * 1024ll, dev);

    LLamaRunState acts;
    {
        auto ctx = Allocator->with_context("Activations");
        acts = ::allocate_run_state(Config, options, B, T, stack, Allocator);
    }

    OptimizerState = std::make_unique<LLamaOptimizerStateManager>(Config, options, acts.MainStream, comm, *Allocator);

    Parameters->begin_optimizer(stack, comm.stream());
    OptimizerState->begin_optimizer(stack);
    OptimizerState->end_optimizer(stack);
    Parameters->end_optimizer(stack);

    {
        auto ctx = Allocator->with_context("Stack");
        long required_size = stack.max_utilization();
        acts.Stack = DeviceMemoryStack{Allocator->allocate(ETensorDType::BYTE, "stack", {required_size}).Data, (std::size_t)required_size, dev};
        acts.Stack.set_high_mark(stack.get_high_mark());
    }

    {
        auto ctx = Allocator->with_context("Gradients");
        Grads = LLamaGradsManager::create(42, 0, Config, options, comm.rank(), comm.world_size(), Allocator);
    }

    OptimizerRNG = std::minstd_rand{42};
    RunState = std::make_unique<LLamaRunState>(std::move(acts));
    comm.barrier();     // make sure *all* GPUs have allocated the model before returning
}

ITensorContainer& LLamaModel::weights() {
    return *Parameters;
}

ITensorContainer& LLamaModel::opt_momentum() {
    return OptimizerState->full_m();
}

ITensorContainer& LLamaModel::opt_momentum_scales() {
    return OptimizerState->scales_m();
}

ITensorContainer& LLamaModel::opt_variance() {
    return OptimizerState->full_v();
}

std::vector<std::byte> LLamaModel::rng_state() const {
    std::stringstream tmp;
    tmp << OptimizerRNG;
    auto view = tmp.rdbuf()->view();
    std::vector<std::byte> state;
    state.reserve(view.size());
    std::transform(view.begin(), view.end(), std::back_inserter(state), [](char c) { return static_cast<std::byte>(c); });
    return state;
}

void LLamaModel::set_rng_state(const std::vector<std::byte>& state) {
    std::stringstream tmp;
    tmp.write(reinterpret_cast<const char*>(state.data()), state.size());
    tmp >> OptimizerRNG;
}

void LLamaModel::import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    Parameters->import_from_file(file_name, allow_cast, comm);
}

void LLamaModel::init_weights(NCCLCommunicator& comm) {
    Parameters->random_init(42, Options, comm);
}

void LLamaModel::export_weights(const std::string& file_name, NCCLCommunicator& comm) {
    Parameters->export_to_file(file_name, comm);
}

void LLamaModel::on_restore_checkpoint(NCCLCommunicator& comm) {
    Parameters->synchronize_absmax(comm);
}

std::string_view LLamaModel::model_type() const {
    return Config.model_name();
}

float LLamaModel::get_loss() const {
    return ::get_loss(*RunState);
}
float LLamaModel::get_norm() const {
    return ::get_norm(*RunState);
}
Tensor& LLamaModel::get_input_buffer() {
    return ::get_input_buffer(*RunState);
}
Tensor& LLamaModel::get_target_buffer() {
    return ::get_target_buffer(*RunState);
}
