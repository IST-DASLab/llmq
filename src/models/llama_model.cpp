// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//

#include "llama_model.h"

#include <cmath>
#include <sstream>

#include "kernels/kernels.h"
#include "llama_gradients.h"
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
                      const cudaDeviceProp& dp, bool reuse_inp_quant, bool inp_has_abs_max,
                      cudaStream_t stream) {
    if (weight.DType == inp.Value.DType) {
        matmul(out, weight, inp.Value, bias, nullptr, handle, workspace, OC, B*T, C, EMMTranspose::TN, false, stream);
    } else {
        float* act_scale = inp.Quant->Scales;
        float* wgt_scale = weight.Scales;
        float* out_scale = out.Scales;

        if (!reuse_inp_quant) {
            if(!inp_has_abs_max) {
                abs_max(act_scale, inp.Value, B * T * C, dp, stream);
            }
            quantize_with_abs_max(inp.Quant.value(), inp.Value, act_scale, B*T*C, dp, stream);
        }

        float scale = weight.DType == ETensorDType::FP8_E4M3 ? 448.f : std::numeric_limits<std::int8_t>::max();
        matmul_out_scale(out_scale, act_scale, wgt_scale, 1.f / scale / scale, stream);
        matmul(out, weight, inp.Quant.value(), bias, out_scale, handle, workspace, OC, B*T, C, EMMTranspose::TN, false, stream);
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
    return tensor.Quant.has_value() ? tensor.Quant->Scales : nullptr;
}

void LLamaModel::forward(Tensor inputs, NCCLCommunicator& comm, int micro_step) {
    NVTX_RANGE_FN();

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
    CUDA_CHECK(cudaMemcpyAsync(rs->Inputs.Data, inputs.Data, inputs.bytes(), cudaMemcpyHostToDevice, main_stream));
    CUDA_CHECK(cudaEventRecord(rs->TransferDone, main_stream));

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
        // fuse RMSNorm with residual; except, of course, in the first block, where we don't have a residual
        // because this is a change in topology, this isn't part of the cuda graph
        if (l == 0) {
            rmsnorm_forward(rs->Acts[0].LN1.Value, rs->Acts[0].LN1_Rstd, rs->Encoded, wgt.LN1_w,
                            quant_abs_max_ptr(rs->Acts[0].LN1), Config.RmsNormEps, B, T, C, main_stream);
        } else if(l != Config.NumLayers) {
            auto& acts = rs->Acts[l-1];
            fused_residual_rmsnorm_forward(acts.ResidualFFN, rs->Acts[l].LN1.Value, rs->Acts[l].LN1_Rstd,
                                           acts.ResidualAtt, acts.MlpDown, wgt.LN1_w,
                                           quant_abs_max_ptr(rs->Acts[l].LN1),
                                           Config.RmsNormEps, B * T, C, main_stream);
        }

        trace_or_execute_cuda_graph([&](){_forward_block(l, wgt);},
            main_stream, rs->ForwardBlockGraph, rs->Options.UseCudaGraphs);
        Parameters->release_block(l, main_stream);
    }

    {
        auto& acts = rs->Acts[Config.NumLayers-1];
        Parameters->gather_lnf(comm);
        fused_residual_rmsnorm_forward(acts.ResidualFFN, rs->LNF, rs->LNF_Rstd, acts.ResidualAtt,
                                       acts.MlpDown, Parameters->get_lnf(main_stream), nullptr, Config.RmsNormEps, B * T, C, main_stream);
        Parameters->release_lnf(main_stream);
    }

    Parameters->gather_head(comm);
    matmul(rs->Output, Parameters->get_head(main_stream), rs->LNF,
           std::nullopt, nullptr, rs->CublasLtHandle, rs->Workspace, V, B*T, C, EMMTranspose::TN, false, main_stream);
    Parameters->release_head(main_stream);

    // do not return before inputs can be accessed again.
    CUDA_CHECK(cudaEventSynchronize(rs->TransferDone));
    CUDA_CHECK(cudaEventRecord(rs->ForwardDone, main_stream));
}

void LLamaModel::_forward_block(int layer, sLLamaBlockWeights<Tensor>& weights)
{
    auto& rs = RunState;
    int l = layer;
    long B = rs->Inputs.Sizes[0];
    long T = rs->Inputs.Sizes[1];
    long C = Config.HiddenSize;
    long D = Config.IntermediateSize;
    long Hq = Config.NumQueryHeads;
    long Hkv = Config.NumKeyValHeads;
    long Hs = Config.head_size();
    cudaStream_t main_stream = rs->MainStream;

    Tensor residual = l == 0 ? rs->Encoded : rs->Acts[l - 1].ResidualFFN;

    auto& acts = rs->Acts[l];

    // 1) projection to QKV vectors (note k,v may be fewer heads than q)
    forward_qmm(acts.QKV, acts.LN1, weights.Attn_QKV_w, weights.Attn_QKV_b,
                rs->CublasLtHandle, rs->Workspace,
                B, T, C, Config.qkv_channels(),
                rs->DeviceProp, false, true, main_stream);
    // 2) apply RoPE to q,k (potentially in place)
    rope_forward(acts.Rope, acts.QKV, rs->FreqCis, B, T, Hq, Hkv, Hs, main_stream);
    // 3) attention: att <- softmax(qk^T)v
    attention_forward_cudnn(acts.Att.Value, acts.LSE, acts.Rope, rs->Workspace, rs->CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);

    forward_qmm(acts.AttO, acts.Att, weights.Attn_Out_w, std::nullopt,
                rs->CublasLtHandle, rs->Workspace,
                B, T, C, C,
                rs->DeviceProp, false, false, main_stream);

    fused_residual_rmsnorm_forward(acts.ResidualAtt, acts.LN2.Value, acts.LN2_Rstd, residual, acts.AttO, weights.LN2_w,
                                   quant_abs_max_ptr(acts.LN2), Config.RmsNormEps, B * T, C, main_stream);

    forward_qmm(acts.MlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
                rs->CublasLtHandle, rs->Workspace,
                B, T, C, 2 * D,
                rs->DeviceProp, false, true, main_stream);
    swiglu_forward(acts.SwiGLu.Value, acts.MlpUp, quant_abs_max_ptr(acts.SwiGLu), B, T, D, main_stream);

    forward_qmm(acts.MlpDown, acts.SwiGLu, weights.MLP_Down_w, std::nullopt,
                rs->CublasLtHandle, rs->Workspace,
                B, T, D, C,
                rs->DeviceProp, false, true, main_stream);
}

float LLamaModel::validate(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    NVTX_RANGE_FN();
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    auto& rs = RunState;
    const size_t V = Config.VocabSize;
    const size_t Vp = Config.VocabSize;
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];

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
    fused_classifier(rs->Output, rs->Losses, d_loss, rs->Targets, B, T, V, Vp, false, main_stream);
    // reduce all the losses within the current GPU (across all microsteps)
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
                  bool reuse_inp, bool dout_has_absmax, cudaStream_t stream) {
    if (weight.DType == inp.Value.DType) {
        matmul_backward(dinp, dweight, dbias, dout.Value, inp.Value, weight, bias_buffer, nullptr, nullptr, accumulate_gradient,
                        rs.CublasLtHandle, rs.Workspace, B, T, C, OC, rs.DeviceProp, stream);
    } else {
        float* act_scale = inp.Quant->Scales;
        float* wgt_scale = weight.Scales;
        float* dout_scale = dout.Quant->Scales;
        float* dinp_scale = rs.MatmulScales.get<float>();
        float* dwgt_scale = dinp_scale + 1;

        if (!dout_has_absmax) {
            abs_max(dout_scale, dout.Value, B*T*OC, rs.DeviceProp, stream);
        }
        quantize_with_abs_max(dout.Quant.value(), dout.Value, dout_scale, B*T*OC, rs.DeviceProp, stream);
        transpose(rs.GradientTranspose, dout.Quant.value(), B*T, OC, stream);

        if(reuse_inp) {
            // inp is already quantized from the forward pass, so just transpose here
            transpose(rs.ActivationTranspose, inp.Quant.value(), B*T, C, stream);
        } else {
            // even though we're re-using (and overwriting) the main tensor, each tensor still has its own version
            // of the absmax-scale, so we can reuse the existing scale from the forward pass
            quantize_and_transpose_with_abs_max(rs.ActivationTranspose, inp.Value, act_scale, B*T, C, rs.DeviceProp, stream);
        }
        transpose(rs.WeightTranspose, weight, OC, C, stream);

        float scale = weight.DType == ETensorDType::FP8_E4M3 ? 448.f : std::numeric_limits<std::int8_t>::max();
        matmul_out_scale(dinp_scale, dout_scale, wgt_scale, 1.f / scale / scale, stream);
        matmul_out_scale(dwgt_scale, dout_scale, act_scale, 1.f / scale / scale, stream);

        matmul_backward_fp8(dinp, dweight, dbias, dout.Quant.value(), rs.GradientTranspose, rs.ActivationTranspose, rs.WeightTranspose,
                            bias_buffer, dinp_scale, dwgt_scale, dout_scale, accumulate_gradient, rs.CublasLtHandle, rs.Workspace, B, T, C, OC, rs.DeviceProp, stream);
    }
}


void LLamaModel::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    auto& rs = RunState;
    cudaStream_t main_stream = rs->MainStream;
    auto& d_acts = rs->DActs;

    NVTX_RANGE_FN();

    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    const size_t V = Config.VocabSize;
    const size_t Vp = Config.VocabSize;
    const size_t C = Config.HiddenSize;
    const size_t L = Config.NumLayers;

    const float d_loss =
        1.0f / (float) (B * T * grad_accum_steps); // results in the uniform average loss over all elements
    CUDA_CHECK(cudaMemcpyAsync(rs->Targets.Data, targets.Data, targets.bytes(), cudaMemcpyHostToDevice, main_stream));
    CUDA_CHECK(cudaEventRecord(rs->TransferDone, main_stream));

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

    // accumulate the losses inside rs->losses, and kick off the backward pass inside the fused classifier
    fused_classifier(rs->Output, rs->Losses, d_loss, rs->Targets, B, T, V, Vp, true, main_stream);
    if (last_step) {
        _reduce_loss(*rs, comm, B, T);
    }
    // if we reset model grads to zero, now is the time we need to wait
    if (micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs->SideStreamEvent, 0));
    }

    // ------------------------------------------------------------------------
    // backward pass: go in the reverse order of the forward pass, and call *_backward() functions

    // reset residual stream gradients (put here to work with gradient accumulation)
    fill_zero(rs->DLNF, main_stream);
    fill_zero(d_acts[L-1].DResFFN.Value, main_stream);

    bool accumulate;

    // ZeRO-3 note: We just finished the forward pass, so LMHead and LNF_w are still available locally, no further gathering needed.
    //              Same for layer L-1, so the first thing we need to prefetch is L-2 down in the loop below.

    // BackwarDone ensures that zero-2 gradient accumulation of the previous step has finished, so we can safely write to d_lmhead again.
    CUDA_CHECK(cudaEventSynchronize(rs->BackwardDone));
    auto& d_lmhead = Grads->get_lmhead_full(main_stream, comm, accumulate);
    Parameters->gather_head(comm);
    matmul_backward(rs->DLNF, d_lmhead, std::nullopt, rs->Output, rs->LNF, Parameters->get_head(main_stream), std::nullopt,
                    nullptr, nullptr, accumulate, rs->CublasLtHandle, rs->Workspace, B, T, C, Vp, rs->DeviceProp, main_stream);
    Parameters->release_head(main_stream);
    Grads->notify_lmhead(main_stream, comm);

    auto& d_lnf_w = Grads->get_lnf_w_full(main_stream, comm, accumulate);
    Parameters->gather_lnf(comm);
    // backward the final layernorm
    rmsnorm_backward(d_acts[L-1].DResFFN.Value, d_lnf_w, rs->RMSNormScratch, d_acts[L - 1].DResFFN.Value, rs->DLNF,
                     rs->Acts[L - 1].ResidualFFN, Parameters->get_lnf(main_stream), rs->LNF_Rstd, quant_abs_max_ptr(d_acts[L-1].DResFFN), B, T, C, rs->DeviceProp, main_stream);
    Parameters->release_lnf(main_stream);
    Grads->notify_lnf_w(main_stream, comm);

    Parameters->gather_block(L - 1, comm, *rs);
    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);
        auto& dw = Grads->get_block_full(l, main_stream, comm, accumulate);

        // prefetch previous layer
        if(l > 0) {
            Parameters->gather_block(l - 1, comm, *rs);
        }
        auto& weights = Parameters->get_block(l, main_stream);
        // last layer changes topology, so for now just don't use graph
        trace_or_execute_cuda_graph([&](){
            _recompute_block(l, weights);
            _backward_block(l, accumulate, weights, dw);
            }, main_stream, rs->BackwardBlockGraph, rs->Options.UseCudaGraphs && l != 0);
        Parameters->release_block(l, main_stream);
        Grads->notify_block(l, main_stream, comm);
    }

    auto& d_emb = Grads->get_embeddings_full(main_stream, comm, accumulate);
    encoder_backward(d_emb, rs->EncoderBwdScratch, rs->EncoderBwdIndices, rs->EncoderBwdInfo,
                     rs->DEmb, rs->Inputs, inputs, B, T, C, OptimizerRNG(), main_stream);
    Grads->notify_embeddings(main_stream, comm);

    // make sure all gradients are communicated before we go to the update step.
    Grads->end_micro_step(main_stream, comm);
    CUDA_CHECK(cudaEventRecord(rs->BackwardDone, main_stream));

    // do not return before inputs can be accessed again.
    CUDA_CHECK(cudaEventSynchronize(rs->TransferDone));
}

void LLamaModel::_recompute_block(int layer, sLLamaBlockWeights<Tensor>& weights) {
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

    auto& acts = rs->Acts[layer];
    auto& opt = rs->Options;

    // Figure out which parts we need to recompute
    bool recompute_ln1 = opt.RecomputeRMSNorm || opt.RecomputeAtt || opt.RecomputeBlock;
    bool recompute_ln2 = opt.RecomputeRMSNorm || opt.RecomputeFFN || opt.RecomputeBlock;
    bool recompute_qkv = opt.RecomputeQKV || opt.RecomputeAtt || opt.RecomputeBlock;
    bool recompute_swiglu = opt.RecomputeSwiGLu || opt.RecomputeFFN || opt.RecomputeBlock;

    // Attention block
    if(recompute_ln1) {
        Tensor& residual = layer == 0 ? rs->Encoded : rs->Acts[layer - 1].ResidualFFN;
        rmsnorm_forward(acts.LN1.Value, acts.LN1_Rstd, residual, weights.LN1_w, nullptr, Config.RmsNormEps, B, T, C, main_stream);
    }

    if (recompute_qkv) {
        // two scenarios: 1) we do not recompute the RMSnorm; then, we _will_ overwrite the full-precision copy of acts.LN1,
        //                   but _can_ reuse the quantized version
        //                2) we recompute RMSNorm; then, acts.LN1 will be correct, but its quantized version will not, so
        //                   we have to re-quantize
        forward_qmm(acts.QKV, acts.LN1, weights.Attn_QKV_w, weights.Attn_QKV_b,
                     rs->CublasLtHandle, rs->Workspace,
                     B, T, C, Config.qkv_channels(),
                     rs->DeviceProp, !recompute_ln1, true, main_stream);
        rope_forward(acts.Rope, acts.QKV, rs->FreqCis, B, T, Hq, Hkv, Hs, main_stream);
    }

    if (opt.RecomputeAtt) {
        attention_forward_cudnn(acts.Att.Value, acts.LSE, acts.Rope, rs->Workspace, rs->CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);
        // AttO not needed in backward pass; but if we want to recompute the entire transformer block, we need its output
        // to recompute the FFN part
        if (opt.RecomputeBlock) {
            forward_qmm(acts.AttO, acts.Att, weights.Attn_Out_w, std::nullopt,
                         rs->CublasLtHandle, rs->Workspace,
                         B, T, C, C,
                         rs->DeviceProp, false, true, main_stream);
        }
    }

    // Feed-forward block
    if(recompute_ln2) {
        if (opt.RecomputeBlock) {
            Tensor residual = layer == 0 ? rs->Encoded : rs->Acts[layer - 1].ResidualFFN;
            fused_residual_rmsnorm_forward(acts.ResidualAtt, acts.LN2.Value, acts.LN2_Rstd, residual, acts.AttO, weights.LN2_w,
                                 nullptr, Config.RmsNormEps, B * T, C, main_stream);
        } else {
            rmsnorm_forward(acts.LN2.Value, acts.LN2_Rstd, acts.ResidualAtt, weights.LN2_w,
                  nullptr, Config.RmsNormEps, B, T, C, main_stream);
        }
    }

    if(opt.RecomputeFFN) {
        forward_qmm(acts.MlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
                         rs->CublasLtHandle, rs->Workspace,
                         B, T, C, 2 * D,
                         rs->DeviceProp, false, true, main_stream);
    }

    if(recompute_swiglu) {
        if (acts.SwiGLu.Quant.has_value()) {
            swiglu_forward_quant(acts.SwiGLu.Quant.value(), acts.MlpUp, acts.SwiGLu.Quant->Scales, B, T, D, main_stream);
        } else {
            swiglu_forward(acts.SwiGLu.Value, acts.MlpUp, nullptr, B, T, D, main_stream);
        }
    }
}

void LLamaModel::_backward_block(int layer, bool accumulate, sLLamaBlockWeights<Tensor>& weights, sLLamaGradBlock& d_weights) {
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

    auto& acts = rs->Acts[layer];
    auto& d_acts = rs->DActs.at(layer);

    // backward the 2nd matmul of MLP
    // note that _recompute_block guarantees that if SwiGLu is already quantized (if necessary)
    backward_qmm(d_acts.DSwiGLU, d_weights.MLP_Down_w, std::nullopt, d_acts.DResFFN, acts.SwiGLu, weights.MLP_Down_w, std::nullopt,
                 accumulate, *rs, B, T, D, C, true, true, main_stream);

    swiglu_backward(d_acts.DMlpUp.Value, d_acts.DSwiGLU, acts.MlpUp, quant_abs_max_ptr(d_acts.DMlpUp), B, T, D, main_stream);

    backward_qmm(d_acts.DLN2, d_weights.MLP_Up_w, std::nullopt, d_acts.DMlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
                 accumulate, *rs, B, T, C, 2 * D, !rs->Options.RecomputeRMSNorm, true, main_stream);

    // rmsnorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
    rmsnorm_backward(d_acts.DResAtt.Value, d_weights.LN2_w, rs->RMSNormScratch, d_acts.DResFFN.Value, d_acts.DLN2,
                     acts.ResidualAtt, weights.LN2_w, acts.LN2_Rstd, quant_abs_max_ptr(d_acts.DResAtt), B, T, C, rs->DeviceProp, main_stream);

    bool recompute_ln1 = rs->Options.RecomputeRMSNorm || rs->Options.RecomputeAtt;
    backward_qmm(d_acts.DAttY, d_weights.Attn_Out_w, std::nullopt, d_acts.DResAtt, acts.Att, weights.Attn_Out_w, std::nullopt,
                 accumulate, *rs, B, T, C, C, false, true, main_stream);

    attention_backward_cudnn(d_acts.DRope, acts.LSE, acts.Att.Value, d_acts.DAttY, acts.Rope, rs->Workspace, rs->CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);
    rope_backward(d_acts.DQKV.Value, d_acts.DRope, rs->FreqCis, B, T, Hq, Hkv, Hs, main_stream);

    backward_qmm(d_acts.DLN1, d_weights.Attn_QKV_w, d_weights.Attn_QKV_b, d_acts.DQKV, acts.LN1, weights.Attn_QKV_w, rs->MatmulBiasScratch,
                       accumulate, *rs, B, T, C, Config.qkv_channels(), !recompute_ln1, false, main_stream);

    if(layer > 0) {
        auto& prev_dacts = rs->DActs.at(layer - 1);
        rmsnorm_backward(prev_dacts.DResFFN.Value, d_weights.LN1_w, rs->RMSNormScratch, prev_dacts.DResAtt.Value, d_acts.DLN1,
                         rs->Acts[layer - 1].ResidualFFN, weights.LN1_w, acts.LN1_Rstd, quant_abs_max_ptr(prev_dacts.DResFFN), B, T, C, rs->DeviceProp, main_stream);
    } else {
        rmsnorm_backward(rs->DEmb, d_weights.LN1_w, rs->RMSNormScratch, d_acts.DResAtt.Value, d_acts.DLN1,
                         rs->Encoded, weights.LN1_w, acts.LN1_Rstd, nullptr, B, T, C, rs->DeviceProp, main_stream);
    }
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

    if(!OptM || !OptV) {
        throw std::logic_error("LLamaModel::update() but no optimizer available");
    }

    auto& opt_m = *OptM;
    auto& opt_v = *OptV;

    auto& rng = OptimizerRNG;

    // make sure we can run abs_max again
    Parameters->reset_scales(main_stream);

    // grad_scale gets deposited into NormBuffer[1] and can be used on main_stream after this.
    calculate_gradient_norm(comm, grad_clip);
    float* grad_scale = rs->NormBuffer.get<float>() + 1;

    auto run_update = [&](Tensor& val, Tensor& grad, Tensor& m, Tensor& v, float wd){
        adamw_update(val, grad, m, v, grad.nelem(),
                     learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale, val.Scales, rng(), main_stream);
        comm.reduce_abs_max(val.Scales);
    };

    run_update(Parameters->get_master_embeddings(), Grads->get_embeddings_shard(main_stream), opt_m.NonBlocks.Embeddings, opt_v.NonBlocks.Embeddings, weight_decay);
    run_update(Parameters->get_master_lnf_w(), Grads->get_lnf_w_shard(main_stream), opt_m.NonBlocks.LNF_w, opt_v.NonBlocks.LNF_w, 0.f);
    CUDA_CHECK(cudaEventRecord(rs->OptEmbeddingsDone, main_stream));

    for(int i = 0; i < Config.NumLayers; i++) {
        NvtxRange layer_range("Layer", i);
        auto& bw = Parameters->get_master_block(i);
        auto& bg = Grads->get_block_shard(i, main_stream);
        auto& bm = opt_m.Blocks[i];
        auto& bv = opt_v.Blocks[i];
        run_update(bw.LN1_w, bg.LN1_w, bm.LN1_w, bv.LN1_w, 0.f);
        run_update(bw.LN2_w, bg.LN2_w, bm.LN2_w, bv.LN2_w, 0.f);

        run_update(bw.Attn_QKV_w, bg.Attn_QKV_w, bm.Attn_QKV_w, bv.Attn_QKV_w,
                   weight_decay);
        if(bm.Attn_QKV_b.has_value()) {
            run_update(bw.Attn_QKV_b.value(), bg.Attn_QKV_b.value(), bm.Attn_QKV_b.value(),
                         bv.Attn_QKV_b.value(), 0.f);
        }
        run_update(bw.Attn_Out_w, bg.Attn_Out_w, bm.Attn_Out_w, bv.Attn_Out_w, weight_decay);

        run_update(bw.MLP_Up_w, bg.MLP_Up_w, bm.MLP_Up_w, bv.MLP_Up_w, weight_decay);
        run_update(bw.MLP_Down_w, bg.MLP_Down_w, bm.MLP_Down_w, bv.MLP_Down_w, weight_decay);

        CUDA_CHECK(cudaEventRecord(rs->LayerUpdateDone[i], main_stream));
    }

    if(!Config.TiedWordEmbeddings) {
        run_update(Parameters->get_master_lmhead(), Grads->get_lmhead_shard(main_stream), opt_m.NonBlocks.LMHead, opt_v.NonBlocks.LMHead, weight_decay);
    }
    comm.wait_on_comms(main_stream);
    CUDA_CHECK(cudaEventRecord(rs->OptimizerDone, main_stream));
}

void zero_opt_buffer(sLLamaWeights& weights, cudaStream_t stream) {
    // here's the first disadvantage of having individual buffers: We need to make a ton of memset calls
    fill_zero(weights.NonBlocks.Embeddings, stream);
    fill_zero(weights.NonBlocks.LNF_w, stream);
    if(weights.NonBlocks.LMHead.Data != weights.NonBlocks.Embeddings.Data) {
        fill_zero(weights.NonBlocks.LMHead, stream);
    }
    for(auto& layer: weights.Blocks) {
        fill_zero(layer.LN1_w, stream);
        fill_zero(layer.LN2_w, stream);
        fill_zero(layer.Attn_QKV_w, stream);
        if(auto& qkv_b = layer.Attn_QKV_b; qkv_b.has_value()) {
            fill_zero(qkv_b.value(), stream);
        }
        fill_zero(layer.Attn_Out_w, stream);
        fill_zero(layer.MLP_Up_w, stream);
        fill_zero(layer.MLP_Down_w, stream);
    }
}

void LLamaModel::allocate_run_state(const LLamaOptions& options, NCCLCommunicator& comm, int B, int T) {
    NVTX_RANGE_FN();
    LLamaRunState acts;
    {
        auto ctx = Allocator->with_context("Activations");
        acts = ::allocate_run_state(Config, options, B, T, Allocator);
    }

    {
        auto ctx = Allocator->with_context("Gradients");
        Grads = LLamaGradsManager::create(42, 0, Config, options, comm.rank(), comm.world_size(), Allocator);
    }

    {
        auto ctx = Allocator->with_context("Adam M");
        EAllocationType alloc_type = options.OffloadOptM ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        LLamaConfig c = Config;
        c.DType = acts.Options.OptMomentumType;
        OptM = std::make_unique<sLLamaWeights>(allocate_weights(c, alloc_type, comm.rank(), comm.world_size(), *Allocator));
        zero_opt_buffer(*OptM, acts.MainStream);
    }

    {
        auto ctx = Allocator->with_context("Adam V");
        EAllocationType alloc_type = options.OffloadOptV ? options.offload_alloc() : EAllocationType::ON_DEVICE;
        LLamaConfig c = Config;
        c.DType = acts.Options.OptVarianceType;
        OptV = std::make_unique<sLLamaWeights>(allocate_weights(c, alloc_type, comm.rank(), comm.world_size(), *Allocator));
        zero_opt_buffer(*OptV, acts.MainStream);
    }

    OptimizerRNG = std::minstd_rand{42};
    RunState = std::make_unique<LLamaRunState>(std::move(acts));
    comm.barrier();     // make sure *all* GPUs have allocated the model before returning
}

ITensorContainer& LLamaModel::weights() {
    return *Parameters;
}

ITensorContainer& LLamaModel::opt_momentum() {
    return *OptM;
}

ITensorContainer& LLamaModel::opt_variance() {
    return *OptV;
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
