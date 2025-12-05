// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "llama_run_state.h"

#include <cuda_runtime.h>

#include "kernels/kernels.h"

cudnnHandle_t create_cudnn_handle();
cublasLtHandle_t create_cublaslt_handle();

// FIXME
constexpr const int QWEN2_NUM_LINEAR_OPS = 4;

class RunStateBuilder {
public:
    RunStateBuilder(LLamaConfig config, LLamaOptions options, int B, int T, std::shared_ptr<TensorAllocator> alloc)
        : Config(config), Options(options), B(B), T(T), C(config.HiddenSize), H(config.IntermediateSize), Alloc(alloc)
    {
    }

    Tensor generate_frequencies();

    LLamaRunState::LayerActivations allocate_basic_fwd_tensors(Tensor lnf);
    void allocate_fwd_quant_tensors(LLamaRunState::LayerActivations& act);
    void keep_fwd_quant_tensors(LLamaRunState::LayerActivations& act, LLamaRunState::LayerActivations& src);
    std::vector<LLamaRunState::LayerActivations> allocate_forward_buffers(Tensor lnf);

    LLamaRunState::LayerGradients allocate_basic_bwd_tensors(Tensor d_lnf);

    std::vector<LLamaRunState::LayerGradients> allocate_backward_buffers(Tensor d_lnf);

private:
    template<typename... Args>
    Tensor allocate(ETensorDType type, const char* name, Args&&... args) {
        return Alloc->allocate(type, name, {std::forward<Args>(args)...});
    }

    template<typename... Args>
    Tensor allocate_or_reuse(bool reuse, Tensor& buffer, ETensorDType type, Args&&... args) {
        if(reuse && !Options.KeepAllActivations) {
            if(buffer.Data == nullptr) {
                buffer = allocate(type, std::forward<Args>(args)...);
            }
            return buffer;
        } else {
            return allocate(type, std::forward<Args>(args)...);
        }
    }

    LLamaConfig Config;
    LLamaOptions Options;
    long B;
    long T;
    long C;     // Config.HiddenSize;
    long H;     // Config.IntermediateSize;
    std::shared_ptr<TensorAllocator> Alloc;

    Tensor tSwiGluBuffer;
    Tensor tMlpUpBuffer;
    Tensor tQKVBuffer;
    Tensor tAttBuffer;
    Tensor tLN1Buffer;
    Tensor tResAttBuffer;
};

Tensor RunStateBuilder::generate_frequencies() {
    Tensor freq = allocate(Config.DType, "freqs", Config.MaxPositionEmbeddings, 2 * Config.head_size());
    // Generate frequencies
    if(Config.DType == ETensorDType::BF16) {
        std::vector<nv_bfloat16> freq_cpu(Config.MaxPositionEmbeddings * 2 * Config.head_size());
        precompute_freqs_cis(freq_cpu.data(), Config.head_size(), Config.MaxPositionEmbeddings, Config.RopeTheta);
        CUDA_CHECK(cudaMemcpy(freq.Data, freq_cpu.data(), freq_cpu.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    } else if (Config.DType == ETensorDType::FP32) {
        std::vector<float> freq_cpu(Config.MaxPositionEmbeddings * 2 * Config.head_size());
        precompute_freqs_cis(freq_cpu.data(), Config.head_size(), Config.MaxPositionEmbeddings, Config.RopeTheta);
        CUDA_CHECK(cudaMemcpy(freq.Data, freq_cpu.data(), freq_cpu.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    return freq;
}

LLamaRunState::LayerActivations RunStateBuilder::allocate_basic_fwd_tensors(Tensor lnf) {
    Tensor ln1_rstd = allocate(ETensorDType::FP32, "ln1_rstd", B, T);
    Tensor ln2_rstd = allocate(ETensorDType::FP32, "ln2_rstd", B, T);
    bool quant = (Options.matmul_dtype() != Config.DType) && !Options.KeepAllActivations;
    bool reuse_ln_buffer = Options.RecomputeRMSNorm || quant;
    Tensor ln1_v = allocate_or_reuse(reuse_ln_buffer || Options.RecomputeAtt, tLN1Buffer, Config.DType, "ln1", B, T, C);
    Tensor ln2_v = allocate_or_reuse(reuse_ln_buffer || Options.RecomputeFFN, lnf, Config.DType, "ln2", B, T, C);

    Tensor qkv = allocate_or_reuse(Options.RecomputeQKV, tQKVBuffer, Config.DType, "qkv", B, T, Config.qkv_channels());
    Tensor res_att = allocate_or_reuse(Options.RecomputeBlock, tResAttBuffer, Config.DType, "res_att", B, T, C);
    Tensor lse = allocate(ETensorDType::FP32, "lse", B, T, Config.NumQueryHeads);
    Tensor att_v = allocate_or_reuse(Options.RecomputeAtt, tAttBuffer, Config.DType, "att_v", B, T, C);
    // not needed for backward, so can reuse an existing buffer
    // we can use the same buffer as for the rms norms, because those support
    // inplace transforms.
    Tensor atto = allocate_or_reuse(true, lnf, Config.DType, "att_o", B, T, C);
    bool reuse_swiglu = Options.RecomputeSwiGLu || quant || Options.RecomputeFFN;
    Tensor swiglu_v = allocate_or_reuse(reuse_swiglu, tSwiGluBuffer, Config.DType, "swiglu", B, T, H);
    Tensor mlp_up;
    if(!Options.RecomputeFFN)
        mlp_up = allocate_or_reuse(Options.RecomputeFFN, tMlpUpBuffer, Config.DType, "mlp_up", B, T, 2 * H);

    QuantizableTensor ln1 = {ln1_v, std::nullopt};
    QuantizableTensor ln2 = {ln2_v, std::nullopt};
    QuantizableTensor att = {att_v, std::nullopt};
    QuantizableTensor swiglu = {swiglu_v, std::nullopt};

    Tensor mlp_down = allocate_or_reuse(true, lnf, Config.DType, "mlp_down", B, T, C);

    return LLamaRunState::LayerActivations{ln1_rstd, ln1, ln2_rstd, ln2, qkv, lse, att, atto,
                                           res_att, mlp_up, mlp_down, swiglu};
}

void RunStateBuilder::allocate_fwd_quant_tensors(LLamaRunState::LayerActivations& act) {
    ETensorDType matmul_dtype = Options.matmul_dtype();
    // allocate a new buffer for every forward quantization
    act.LN1.Quant = allocate(matmul_dtype, "ln1.q", B, T, C);
    act.LN2.Quant = allocate(matmul_dtype, "ln2.q", B, T, C);
    act.Att.Quant = allocate(matmul_dtype, "att.q", B, T, C);
    act.SwiGLu.Quant = allocate(matmul_dtype, "swiglu.q", B, T, H);
}

void RunStateBuilder::keep_fwd_quant_tensors(LLamaRunState::LayerActivations& act, LLamaRunState::LayerActivations& src) {
    // allocate new buffers for activation quants (so we can drop the unquantized ones), but reuse
    // the weight quant buffers.
    act.LN1.Quant = Options.RecomputeRMSNorm ? src.LN1.Quant : allocate(Options.matmul_dtype(), "ln1.q", B, T, C);
    act.LN2.Quant = Options.RecomputeRMSNorm ? src.LN2.Quant : allocate(Options.matmul_dtype(), "ln2.q", B, T, C);
    // note: Att is needed unquantized for attention-backward
    act.Att.Quant = src.Att.Quant;
    act.SwiGLu.Quant = (Options.RecomputeSwiGLu || Options.RecomputeFFN) ? src.SwiGLu.Quant : allocate(Options.matmul_dtype(), "swiglu.q", B, T, H);
}

std::vector<LLamaRunState::LayerActivations> RunStateBuilder::allocate_forward_buffers(Tensor lnf)
{
    std::vector<LLamaRunState::LayerActivations> layers;
    layers.reserve(Config.NumLayers);
    for(int l = 0; l < Config.NumLayers; ++l) {
        LLamaRunState::LayerActivations act = allocate_basic_fwd_tensors(lnf);

        if(Options.matmul_dtype() != Config.DType) {
            if(l == 0) {
                allocate_fwd_quant_tensors(act);
            } else {
                keep_fwd_quant_tensors(act, layers.front());
            }
        }

        layers.push_back(act);
    }

    return layers;
}

LLamaRunState::LayerGradients RunStateBuilder::allocate_basic_bwd_tensors(Tensor d_lnf) {
    QuantizableTensor d_res_ffn{allocate(Config.DType, "d_res_ffn", B, T, C)};
    Tensor d_swiglu = Tensor{Config.DType, {B, T, H}, nullptr, nullptr, 3, d_lnf.Device};
    QuantizableTensor d_mlp_up{};   // this will be handled in-place
    Tensor d_ln2 = Options.KeepAllActivations ? allocate(Config.DType, "d_ln2", B, T, C) : d_lnf;
    Tensor d_att_y = Options.KeepAllActivations ? allocate(Config.DType, "d_att_y", B, T, C) : d_lnf;
    QuantizableTensor d_qkv{Tensor{Config.DType, {B, T, Config.qkv_channels()}, nullptr, nullptr, 3, d_lnf.Device}};
    Tensor d_ln1 = Options.KeepAllActivations ? allocate(Config.DType, "d_ln1", B, T, C) : d_lnf;
    QuantizableTensor d_res_att = Options.KeepAllActivations ? QuantizableTensor{allocate(Config.DType, "d_res_att", B, T, C)} : d_res_ffn;

    return LLamaRunState::LayerGradients{d_res_ffn, d_swiglu, d_mlp_up, d_ln2, d_res_att, d_att_y, d_qkv, d_ln1};
}

std::vector<LLamaRunState::LayerGradients> RunStateBuilder::allocate_backward_buffers(Tensor d_lnf)
{
    std::vector<LLamaRunState::LayerGradients> LGrads;
    LGrads.reserve(Config.NumLayers);
    for (int l = 0; l < Config.NumLayers; ++l) {
        if (Options.KeepAllActivations || l == 0) {
            LLamaRunState::LayerGradients grads = allocate_basic_bwd_tensors(d_lnf);
            ETensorDType matmul_dtype = Options.matmul_dtype();
            ETensorDType grad_dtype = Options.grad_dtype();
            if(grad_dtype != Config.DType) {
                grads.DResFFN.Quant = allocate(grad_dtype, "d_res_ffn.q", B, T, C);
                grads.DResAtt.Quant = Options.KeepAllActivations ? allocate(grad_dtype, "d_res_att.q", B, T, C) : grads.DResFFN.Quant;
                grads.DMlpUp.Quant = {grad_dtype, {B, T, 2 * Config.IntermediateSize}, nullptr, nullptr, 3, d_lnf.Device};
                grads.DQKV.Quant = allocate(grad_dtype, "d_qkv.q", Config.qkv_channels(), B, T);
            }
            LGrads.push_back(grads);
        } else {
            LGrads.push_back(LGrads.front());   // just duplicate the pointers
        }
    }
    return LGrads;
}

void LLamaRunState::fetch_res_ffn(int layer_idx, cudaStream_t fetch_stream) {
    if(!Options.OffloadResidual) {
        return;
    }

    int l2 = layer_idx % 2;
    auto& status = OffloadedResidualState.at(l2);

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.Event, 0));
    status.Layer = layer_idx;
    status.Ready = false;
    CUDA_CHECK(cudaMemcpyAsync(DeviceResiduals.at(l2).Data, OffloadedResiduals.at(layer_idx).Data,
                               DeviceResiduals.at(l2).bytes(), cudaMemcpyHostToDevice, fetch_stream));
    CUDA_CHECK(cudaEventRecord(status.Event, fetch_stream));
}

void LLamaRunState::mark_res_ffn_ready(int layer_idx, cudaStream_t main_stream) {
    if(!Options.OffloadResidual) {
        return;
    }
    auto& status = OffloadedResidualState.at(layer_idx % 2);
    status.Layer = layer_idx;
    CUDA_CHECK(cudaEventRecord(ResidualsAreReady, main_stream));
}

void LLamaRunState::put_res_ffn(int layer_idx, cudaStream_t put_stream) {
    if(!Options.OffloadResidual) {
        return;
    }

    int l2 = layer_idx % 2;
    auto& status = OffloadedResidualState.at(l2);
    status.Ready = false;
    if(status.Layer != layer_idx) {
        throw std::logic_error("Mismatched layer index in put_res_ffn");
    }

    CUDA_CHECK(cudaStreamWaitEvent(put_stream, ResidualsAreReady, 0));
    CUDA_CHECK(cudaMemcpyAsync(OffloadedResiduals.at(layer_idx).Data, DeviceResiduals.at(l2).Data,
                               DeviceResiduals.at(l2).bytes(), cudaMemcpyDeviceToHost, put_stream));
    CUDA_CHECK(cudaEventRecord(status.Event, put_stream));
}

Tensor& LLamaRunState::get_res_ffn(int layer_idx, cudaStream_t main_stream) {
    if(!Options.OffloadResidual) {
        return DeviceResiduals.at(layer_idx);
    }

    int l2 = layer_idx % 2;
    auto& status = OffloadedResidualState.at(l2);
    if(!status.Ready) {
        CUDA_CHECK(cudaStreamWaitEvent(main_stream, status.Event, 0));
        status.Ready = true;
    }
    return DeviceResiduals.at(l2);
}

void LLamaRunState::release_res_ffn(int layer_idx, cudaStream_t main_stream) {
    if(!Options.OffloadResidual) {
        return;
    }
    int l2 = layer_idx % 2;
    auto& status = OffloadedResidualState.at(l2);
    CUDA_CHECK(cudaEventRecord(status.Event, main_stream));
}

void LLamaRunState::init(LLamaConfig config, long B, long T, DeviceMemoryStack& stack) {
    long V = config.VocabSize;
    long C = config.HiddenSize;
    long H = config.IntermediateSize;
    auto& alloc = Allocator;

    RunStateBuilder builder(config, Options, B, T, alloc);

    int did;
    CUDA_CHECK(cudaGetDevice(&did));
    CUDA_CHECK(cudaGetDeviceProperties(&DeviceProp, did));

    Inputs = alloc->allocate(ETensorDType::INT32, "inputs", {B, T});
    Targets = alloc->allocate(ETensorDType::INT32, "targets", {B, T});
    Inputs_CPU = alloc->allocate(ETensorDType::INT32, "inputs_cpu", EAllocationType::PINNED, {B, T});
    Targets_CPU = alloc->allocate(ETensorDType::INT32, "targets_cpu", EAllocationType::PINNED, {B, T});
    Losses = alloc->allocate(ETensorDType::FP32, "losses", {B, T});
    Encoded = alloc->allocate(config.DType, "encoded", {B, T, C});
    FreqCis = builder.generate_frequencies();
    // We're chunking the logit computation, so we can allocate a much smaller tensor.
    long out_size = div_exact(B*T, (long)Options.LMHeadChunks);
    Output = Tensor{config.DType,{out_size, V}, nullptr, nullptr, 2, Inputs.Device};
    LNF = alloc->allocate(config.DType, "lnf", {B, T, C});
    LNF_Rstd = alloc->allocate(ETensorDType::FP32, "lnf_rstd", {B, T});
    DLNF = alloc->allocate(config.DType, "d_lnf", {B, T, C});
    long rms_scratch_size = get_rmsnorm_backward_scratch_size(C, DeviceProp);
    long bias_scratch_size = get_bias_backward_scratch_size(config.DType, config.qkv_channels(), DeviceProp);
    CudnnHandle = create_cudnn_handle();
    CublasLtHandle = create_cublaslt_handle();
    RMSNormScratch = alloc->allocate(ETensorDType::BYTE, "rms_scratch", {rms_scratch_size});
    MatmulBiasScratch = alloc->allocate(ETensorDType::FP32, "bias_scratch", {bias_scratch_size / (long)sizeof(float)});
    CuBlasWorkspace = alloc->allocate(ETensorDType::BYTE, "cublas_ws", {32*1024*1024});
    // batch size for chunked attention backward
    long chunk_batch_size = div_exact(B, (long)Options.AttBwdChunks);
    long cudnn_ws_size = cudnn_get_workspace_size(chunk_batch_size, T, config.NumQueryHeads, config.NumKeyValHeads, config.head_size(), CudnnHandle);
    CuDNNWorkspace = Tensor{ETensorDType::BYTE, {cudnn_ws_size}, nullptr, nullptr, 1, Inputs.Device};
    DEmb = alloc->allocate(config.DType, "d_emb", {B, T, C});

    Acts = builder.allocate_forward_buffers(LNF);
    ResidualsAreReady = create_named_event("residual_ready");
    if(Options.OffloadResidual) {
        DeviceResiduals.reserve(2);
        OffloadedResiduals.reserve(config.NumLayers);
        for(int i = 0; i < 2; ++i) {
            DeviceResiduals.emplace_back(alloc->allocate(config.DType, "res_ffn", {B, T, C}));
            OffloadedResidualState.emplace_back(
                create_named_event((std::string("offload_res_ffn_") + std::to_string(i)).c_str()),
                -1, false
            );
        }
        for(int i = 0; i < config.NumLayers; ++i) {
            OffloadedResiduals.push_back(
                alloc->allocate(config.DType, "ffn-res-off", EAllocationType::PINNED, {B, T, C}));
        }
    } else {
        DeviceResiduals.reserve(config.NumLayers);
        for(int i = 0; i < config.NumLayers; ++i) {
            DeviceResiduals.emplace_back(alloc->allocate(config.DType, "res_ffn", {B, T, C}));
        }
    }

    long num_c_groups = div_ceil(C, (long)(16 / get_dtype_size(config.DType) * 32));
    EncoderBwdScratch = alloc->allocate(ETensorDType::INT32, "enc_bw_scratch", {B, T, num_c_groups * 5});
    EncoderBwdIndices = alloc->allocate(ETensorDType::INT32, "enc_bw_idx", EAllocationType::PINNED, {B, T, num_c_groups});
    EncoderBwdInfo = alloc->allocate(ETensorDType::INT32, "env_bw_info", EAllocationType::PINNED, {B, T, 4 * num_c_groups});

    NormBuffer = alloc->allocate(ETensorDType::FP32, "norm_buffer", {get_max_num_block_sums(DeviceProp)});;
    Tensor host_buffer = alloc->allocate(ETensorDType::FP32, "host_buffer", EAllocationType::PINNED, {2});
    NormHost = host_buffer.get<float>();
    LossHost = host_buffer.get<float>() + 1;

    MainStream = create_named_stream("main stream");
    SideStream = create_named_stream("side stream");

    SideStreamEvent = create_named_event("side stream event");

    ForwardDone = create_named_event("forward done");
    BackwardDone = create_named_event("backward done");
    NormDone = create_named_event("norm done");
    OptEmbeddingsDone = create_named_event("optimizer lmhead done");
    OptimizerDone = create_named_event("optimizer done");
    TransferDone = create_named_event("transfer done");

    for(int i = 0; i < config.NumLayers + 1; ++i) {
        LayerUpdateDone.push_back(create_named_event(("opt " + std::to_string(i) + " done").c_str()));
    }

    DActs = builder.allocate_backward_buffers(DLNF);
    for (int i = 0; i < config.NumLayers; ++i) {
        // DMlpUp is handled in-place
        DActs[i].DMlpUp.Value = Acts[i].MlpUp;
    }

    if(Options.matmul_dtype() != config.DType) {
        AbsMaxes = alloc->allocate(ETensorDType::FP32, "abs_max", {config.NumLayers, 8l*QWEN2_NUM_LINEAR_OPS});
        float* abs_max_ptr = AbsMaxes->get<float>();
        for(int i = 0; i < config.NumLayers; ++i) {
            float* layer_abs_maxes = abs_max_ptr + 8 * QWEN2_NUM_LINEAR_OPS * i;

            Acts[i].LN1.Quant->Stats = layer_abs_maxes + 0;
            Acts[i].QKV.Stats = layer_abs_maxes + 2;
            DActs.at(i).DQKV.Quant->Stats = layer_abs_maxes + 4;
            DActs.at(i).DLN1.Stats = layer_abs_maxes + 6;

            Acts[i].Att.Quant->Stats = layer_abs_maxes + 8;
            Acts[i].AttO.Stats = layer_abs_maxes + 10;
            DActs.at(i).DResAtt.Quant->Stats = layer_abs_maxes + 12;
            DActs.at(i).DAttY.Stats = layer_abs_maxes + 14;

            Acts[i].LN2.Quant->Stats = layer_abs_maxes + 16;
            Acts[i].MlpUp.Stats = layer_abs_maxes + 18;
            DActs.at(i).DMlpUp.Quant->Stats = layer_abs_maxes + 20;
            DActs.at(i).DLN2.Stats = layer_abs_maxes + 22;

            Acts[i].SwiGLu.Quant->Stats = layer_abs_maxes + 24;
            Acts[i].MlpDown.Stats = layer_abs_maxes + 26;
            DActs.at(i).DResFFN.Quant->Stats = layer_abs_maxes + 28;
            DActs.at(i).DSwiGLU.Stats = layer_abs_maxes + 30;
        }
    }

    bool use_fp8 = Options.grad_dtype() == ETensorDType::FP8_E4M3 || Options.grad_dtype() == ETensorDType::FP8_E5M2;
    auto bw_qmm = [&](int B, int T, int C, int OC) {
        if(use_fp8) {
            auto wgt_tp = stack.allocate(ETensorDType::FP8_E4M3, {C, OC}, "wgt_tp");
            stack.free(wgt_tp.Data);
            auto act_tp = stack.allocate(ETensorDType::FP8_E4M3, {C, B * T}, "act_tp");
            auto grd_tp = stack.allocate(Options.grad_dtype(), {OC, B * T}, "grd_tp");
            stack.free(grd_tp.Data);
            stack.free(act_tp.Data);
        }
    };

    // simulate to determine required stack size
    auto mlp_up = stack.allocate(Config.DType, {B, T, 2 * Config.IntermediateSize}, "mlp_up");
    auto ws = stack.allocate(CuDNNWorkspace.bytes(), "workspace");
    stack.free(stack.allocate(DActs[0].DQKV.Value.bytes(), "dqkv"));   // attention
    stack.free(ws);   // attention

    auto dswi = stack.allocate(DActs[0].DSwiGLU.bytes(), "dswiglu");
    bw_qmm(B, T, H, C);         // backward qmm swiglu
    stack.free(dswi);

    if(use_fp8) {
        auto dupq = stack.allocate(DActs[0].DMlpUp.Quant->bytes(), "dup.q");
        bw_qmm(B, T, C, 2 * H);     // backward qmm up
        stack.free(dupq);
    }
    stack.free(mlp_up);
    stack.free(stack.allocate(Output.bytes(), "output"));  // lm-head

    MatmulScales = alloc->allocate(ETensorDType::FP32, "mm_scales", {2});

    // debug timings
    if(Options.TriggerTimingEvents) {
        TimingOptimizerStart = create_named_event("timing_opt_start", true);
        TimingOptimizerEnd = create_named_event("timing_opt_done", true);
    }
}

LLamaRunState allocate_run_state(LLamaConfig config, LLamaOptions options, long B, long T, DeviceMemoryStack& stack, std::shared_ptr<TensorAllocator> alloc) {
    LLamaRunState state{config, B, T, options, std::move(alloc)};
    state.init(config, B, T, stack);
    return state;
}

Tensor LLamaRunState::temp_alloc(ETensorDType dtype, const std::vector<long>& shape) {
    return  Stack.allocate(dtype, shape);
}

void LLamaRunState::temp_acquire(Tensor& target) {
    if(target.Device != Stack.device_id()) {
        throw std::logic_error("device mismatch");
    }

    target.Data = Stack.allocate(target.bytes());
}

void LLamaRunState::temp_free(Tensor& tensor) {
    Stack.free(tensor);
}

void LLamaRunState::setup_timing_events(int micro_steps) {
    for(int i = TimingForwardStart.size(); i < micro_steps + 1; ++i) {
        TimingForwardStart.push_back(create_named_event(("timing_fwd_" + std::to_string(i) + "_start").c_str(), true));
        TimingForwardEnd.push_back(create_named_event(("timing_fwd_" + std::to_string(i) + "_end").c_str(), true));
        TimingHeadStart.push_back(create_named_event(("timing_head_" + std::to_string(i) + "_start").c_str(), true));
        TimingHeadEnd.push_back(create_named_event(("timing_head_" + std::to_string(i) + "_end").c_str(), true));
        TimingBackwardStart.push_back(create_named_event(("timing_bwd_" + std::to_string(i) + "_start").c_str(), true));
        TimingBackwardEnd.push_back(create_named_event(("timing_bwd_" + std::to_string(i) + "_end").c_str(), true));
    }
}

Tensor LLamaRunState::acquire_mlp_up(int layer) {
    if(Options.RecomputeFFN) {
        return Stack.allocate(Options.ModelType.value(), {B, T, 2 * Config.IntermediateSize});
    } else {
        return Acts[layer].MlpUp;
    }
}

void LLamaRunState::release_mlp_up(Tensor& mlp_up) {
    if(Options.RecomputeFFN) {
        Stack.free(mlp_up);
    }
}

float get_loss(LLamaRunState& acts) {
    CUDA_CHECK(cudaEventSynchronize(acts.BackwardDone));
    return acts.LossHost[0];
}

float get_norm(LLamaRunState& acts) {
    CUDA_CHECK(cudaEventSynchronize(acts.NormDone));
    return acts.NormHost[0];
}

Tensor& get_input_buffer(LLamaRunState& acts) {
    return acts.Inputs_CPU;
}

Tensor& get_target_buffer(LLamaRunState& acts) {
    return acts.Targets_CPU;
}
