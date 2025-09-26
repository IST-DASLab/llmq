// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
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
    bool quant = (Options.MatmulType.value_or(Config.DType) != Config.DType) && !Options.KeepAllActivations;
    bool reuse_ln_buffer = Options.RecomputeRMSNorm || quant;
    Tensor ln1_v = allocate_or_reuse(reuse_ln_buffer || Options.RecomputeAtt, tLN1Buffer, Config.DType, "ln1", B, T, C);
    Tensor ln2_v = allocate_or_reuse(reuse_ln_buffer || Options.RecomputeFFN, lnf, Config.DType, "ln2", B, T, C);

    Tensor qkv = allocate_or_reuse(Options.RecomputeQKV, tQKVBuffer, Config.DType, "qkv", B, T, Config.qkv_channels());
    Tensor res_att = allocate(Config.DType, "res_att", B, T, C);
    Tensor res_ffn = allocate(Config.DType, "res_ffn", B, T, C);
    Tensor lse = allocate(ETensorDType::FP32, "lse", B, T, Config.NumQueryHeads);
    Tensor att_v = allocate_or_reuse(Options.RecomputeAtt, tAttBuffer, Config.DType, "att_v", B, T, C);
    // not needed for backward, so can reuse an existing buffer
    // we can use the same buffer as for the rms norms, because those support
    // inplace transforms.
    Tensor atto = allocate_or_reuse(true, lnf, Config.DType, "att_o", B, T, C);
    bool reuse_swiglu = Options.RecomputeSwiGLu || quant || Options.RecomputeFFN;
    Tensor swiglu_v = allocate_or_reuse(reuse_swiglu, tSwiGluBuffer, Config.DType, "swiglu", B, T, H);
    Tensor mlp_up = allocate_or_reuse(Options.RecomputeFFN, tMlpUpBuffer, Config.DType, "mlp_up", B, T, 2 * H);

    QuantizableTensor ln1 = {ln1_v, std::nullopt};
    QuantizableTensor ln2 = {ln2_v, std::nullopt};
    QuantizableTensor att = {att_v, std::nullopt};
    QuantizableTensor swiglu = {swiglu_v, std::nullopt};

    Tensor mlp_down = allocate_or_reuse(true, lnf, Config.DType, "mlp_down", B, T, C);
    Tensor rope = allocate_or_reuse(true, qkv, Config.DType, "rope", B, T, Config.qkv_channels());

    return LLamaRunState::LayerActivations{ln1_rstd, ln1, ln2_rstd, ln2, qkv, lse, rope, att, atto,
                                           res_att, mlp_up, mlp_down, swiglu, res_ffn};
}

void RunStateBuilder::allocate_fwd_quant_tensors(LLamaRunState::LayerActivations& act) {
    ETensorDType matmul_dtype = Options.MatmulType.value();
    // allocate a new buffer for every forward quantization
    act.LN1.Quant = allocate(matmul_dtype, "ln1.q", B, T, C);
    act.LN2.Quant = allocate(matmul_dtype, "ln2.q", B, T, C);
    act.Att.Quant = allocate(matmul_dtype, "att.q", B, T, C);
    act.SwiGLu.Quant = allocate(matmul_dtype, "swiglu.q", B, T, H);
}

void RunStateBuilder::keep_fwd_quant_tensors(LLamaRunState::LayerActivations& act, LLamaRunState::LayerActivations& src) {
    // allocate new buffers for activation quants (so we can drop the unquantized ones), but reuse
    // the weight quant buffers.
    act.LN1.Quant = Options.RecomputeRMSNorm ? src.LN1.Quant : allocate(Options.MatmulType.value(), "ln1.q", B, T, C);
    act.LN2.Quant = Options.RecomputeRMSNorm ? src.LN2.Quant : allocate(Options.MatmulType.value(), "ln2.q", B, T, C);
    // note: Att is needed unquantized for attention-backward
    act.Att.Quant = src.Att.Quant;
    act.SwiGLu.Quant = (Options.RecomputeSwiGLu || Options.RecomputeFFN) ? src.SwiGLu.Quant : allocate(Options.MatmulType.value(), "swiglu.q", B, T, H);
}

std::vector<LLamaRunState::LayerActivations> RunStateBuilder::allocate_forward_buffers(Tensor lnf)
{
    std::vector<LLamaRunState::LayerActivations> layers;
    layers.reserve(Config.NumLayers);
    for(int l = 0; l < Config.NumLayers; ++l) {
        LLamaRunState::LayerActivations act = allocate_basic_fwd_tensors(lnf);

        if(Options.MatmulType.value_or(Config.DType) != Config.DType) {
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
    Tensor d_swiglu = allocate(Config.DType, "d_swiglu", B, T, H);
    QuantizableTensor d_mlp_up{allocate(Config.DType, "d_mlp_up", B, T, 2 * H)};
    Tensor d_ln2 = Options.KeepAllActivations ? allocate(Config.DType, "d_ln2", B, T, C) : d_lnf;
    Tensor d_att_y = Options.KeepAllActivations ? allocate(Config.DType, "d_att_y", B, T, C) : d_lnf;
    QuantizableTensor d_qkv{allocate(Config.DType, "d_qkv", B, T, Config.qkv_channels())};
    Tensor d_ln1 = Options.KeepAllActivations ? allocate(Config.DType, "d_ln1", B, T, C) : d_lnf;
    QuantizableTensor d_res_att = Options.KeepAllActivations ? QuantizableTensor{allocate(Config.DType, "d_res_att", B, T, C)} : d_res_ffn;
    Tensor d_rope = Options.KeepAllActivations ? allocate(Config.DType, "d_rope", B, T, Config.qkv_channels()) : d_qkv.Value;

    return LLamaRunState::LayerGradients{d_res_ffn, d_swiglu, d_mlp_up, d_ln2, d_res_att, d_att_y, d_rope, d_qkv,
                                         d_ln1};
}

std::vector<LLamaRunState::LayerGradients> RunStateBuilder::allocate_backward_buffers(Tensor d_lnf)
{
    std::vector<LLamaRunState::LayerGradients> LGrads;
    LGrads.reserve(Config.NumLayers);
    for (int l = 0; l < Config.NumLayers; ++l) {
        if (Options.KeepAllActivations || l == 0) {
            LLamaRunState::LayerGradients grads = allocate_basic_bwd_tensors(d_lnf);
            ETensorDType matmul_dtype = Options.MatmulType.value_or(Config.DType);
            if(matmul_dtype != Config.DType) {
                grads.DResFFN.Quant = allocate(matmul_dtype, "d_res_ffn.q", B, T, C);
                grads.DResAtt.Quant = Options.KeepAllActivations ? allocate(matmul_dtype, "d_res_att.q", B, T, C) : grads.DResFFN.Quant;
                grads.DMlpUp.Quant = allocate(matmul_dtype, "d_mlp_up.q", B, T, 2 * Config.IntermediateSize);
                grads.DQKV.Quant = allocate(matmul_dtype, "d_qkv.q", Config.qkv_channels(), B, T);
            }
            LGrads.push_back(grads);
        } else {
            LGrads.push_back(LGrads.front());   // just duplicate the pointers
        }
    }
    return LGrads;
}

LLamaRunState allocate_run_state(LLamaConfig config, LLamaOptions options, int B, int T, std::shared_ptr<TensorAllocator> alloc) {
    // we are *not* handling this as a single giant allocation
    // by using independent allocations, we give sanitizers a better chance of catching errors,
    // and we never have to worry about tensor alignment
    long V = config.VocabSize;
    long C = config.HiddenSize;
    long H = config.IntermediateSize;

    RunStateBuilder builder(config, options, B, T, alloc);

    int did;
    CUDA_CHECK(cudaGetDevice(&did));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, did));

    Tensor inputs = alloc->allocate(ETensorDType::INT32, "inputs", {B, T});
    Tensor targets = alloc->allocate(ETensorDType::INT32, "targets", {B, T});
    Tensor inputs_cpu = alloc->allocate(ETensorDType::INT32, "inputs_cpu", EAllocationType::PINNED, {B, T});
    Tensor targets_cpu = alloc->allocate(ETensorDType::INT32, "targets_cpu", EAllocationType::PINNED, {B, T});
    Tensor losses = alloc->allocate(ETensorDType::FP32, "losses", {B, T});
    Tensor encoded = alloc->allocate(config.DType, "encoded", {B, T, C});
    Tensor freq_cis = builder.generate_frequencies();
    Tensor output = alloc->allocate(config.DType, "output", {B, T, V});
    Tensor lnf = alloc->allocate(config.DType, "lnf", {B, T, C});
    Tensor lnf_rstd = alloc->allocate(ETensorDType::FP32, "lnf_rstd", {B, T});
    Tensor d_lnf = alloc->allocate(config.DType, "d_lnf", {B, T, C});
    long rms_scratch_size = get_rmsnorm_backward_scratch_size(C, deviceProp);
    long bias_scratch_size = get_bias_backward_scratch_size(config.DType, config.qkv_channels(), deviceProp);
    cudnnHandle_t cudnn_handle = create_cudnn_handle();
    long ws_size = cudnn_get_workspace_size(B, T, config.NumQueryHeads, config.NumKeyValHeads, config.head_size(), cudnn_handle);
    ws_size = std::max(ws_size, 32l * 1024 * 1024); // Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
    cublasLtHandle_t cublas_handle = create_cublaslt_handle();
    Tensor rms_scratch = alloc->allocate(ETensorDType::BYTE, "rms_scratch", {rms_scratch_size});
    Tensor bias_scratch = alloc->allocate(ETensorDType::FP32, "bias_scratch", {bias_scratch_size / (long)sizeof(float)});
    Tensor cudnn_ws = alloc->allocate(ETensorDType::BYTE, "workspace", {ws_size});
    Tensor d_emb = alloc->allocate(config.DType, "d_emb", {B, T, C});

    auto layers = builder.allocate_forward_buffers(lnf);


    long num_c_groups = div_ceil(C, (long)(16 / get_dtype_size(config.DType) * 32));
    Tensor enc_bw_scratch = alloc->allocate(ETensorDType::INT32, "enc_bw_scratch", {B, T, num_c_groups * 5});
    Tensor enc_bw_idx = alloc->allocate(ETensorDType::INT32, "enc_bw_idx", EAllocationType::PINNED, {B, T, num_c_groups});
    Tensor env_bw_info = alloc->allocate(ETensorDType::INT32, "env_bw_info", EAllocationType::PINNED, {B, T, 4 * num_c_groups});
    Tensor wgt_tp_buffer = alloc->allocate(ETensorDType::FP8_E4M3, "wgt_tp_buffer", {config.HiddenSize, 2 * config.IntermediateSize});
    Tensor act_tp_buffer = alloc->allocate(ETensorDType::FP8_E4M3, "wgt_tp_buffer", {H, B, T});
    Tensor grd_tp_buffer = alloc->allocate(ETensorDType::FP8_E4M3, "grd_tp_buffer", {2*H, B, T});

    Tensor norm_buffer = alloc->allocate(ETensorDType::FP32, "norm_buffer", {get_max_num_block_sums(deviceProp)});
    Tensor host_buffer = alloc->allocate(ETensorDType::FP32, "host_buffer", EAllocationType::PINNED, {2});

    cudaStream_t main_stream = create_named_stream("main stream");
    cudaStream_t side_stream = create_named_stream("side stream");

    cudaEvent_t side_stream_event = create_named_event("side stream event");

    cudaEvent_t forward_done_event = create_named_event("forward done");
    cudaEvent_t backward_done_event = create_named_event("backward done");
    cudaEvent_t norm_done_event = create_named_event("norm done");
    cudaEvent_t lmhead_done = create_named_event("optimizer lmhead done");
    cudaEvent_t optimizer_done_event = create_named_event("optimizer done");
    cudaEvent_t transfer_done_event = create_named_event("transfer done");

    std::vector<cudaEvent_t> optimizer_events;
    for(int i = 0; i < config.NumLayers + 1; ++i) {
        optimizer_events.push_back(create_named_event(("opt " + std::to_string(i) + " done").c_str()));
    }

    std::vector<LLamaRunState::LayerGradients> grads = builder.allocate_backward_buffers(d_lnf);

    std::optional<Tensor> abs_maxes;
    if(options.MatmulType.value_or(config.DType) != config.DType) {
        abs_maxes = alloc->allocate(ETensorDType::FP32, "abs_max", {config.NumLayers, 6l*QWEN2_NUM_LINEAR_OPS});
        float* abs_max_ptr = abs_maxes->get<float>();
        for(int i = 0; i < config.NumLayers; ++i) {
            float* layer_abs_maxes = abs_max_ptr + 6 * QWEN2_NUM_LINEAR_OPS * i;

            layers[i].LN1.Quant->Scales = layer_abs_maxes + 0;
            layers[i].QKV.Scales = layer_abs_maxes + 2;
            grads.at(i).DQKV.Quant->Scales = layer_abs_maxes + 3;
            grads.at(i).DLN1.Scales = layer_abs_maxes + 4;

            layers[i].Att.Quant->Scales = layer_abs_maxes + 6;
            layers[i].AttO.Scales = layer_abs_maxes + 8;
            grads.at(i).DResAtt.Quant->Scales = layer_abs_maxes + 9;
            grads.at(i).DAttY.Scales = layer_abs_maxes + 10;

            layers[i].LN2.Quant->Scales = layer_abs_maxes + 12;
            layers[i].MlpUp.Scales = layer_abs_maxes + 14;
            grads.at(i).DMlpUp.Quant->Scales = layer_abs_maxes + 15;
            grads.at(i).DLN2.Scales = layer_abs_maxes + 16;

            layers[i].SwiGLu.Quant->Scales = layer_abs_maxes + 18;
            layers[i].MlpDown.Scales = layer_abs_maxes + 20;
            grads.at(i).DResFFN.Quant->Scales = layer_abs_maxes + 21;
            grads.at(i).DSwiGLU.Scales = layer_abs_maxes + 22;
        }
    }

    Tensor mm_scales = alloc->allocate(ETensorDType::FP32, "mm_scales", {2});

    return LLamaRunState{inputs, targets, inputs_cpu, targets_cpu, losses,
                         encoded, freq_cis, output, lnf, lnf_rstd, std::move(layers),
                         std::move(grads),
                         d_lnf, d_emb, rms_scratch, bias_scratch, cudnn_ws, enc_bw_scratch, enc_bw_idx, env_bw_info,
            wgt_tp_buffer, act_tp_buffer, grd_tp_buffer,
                         abs_maxes, mm_scales,
                         norm_buffer, host_buffer.get<float>(), host_buffer.get<float>() + 1,
                         options, std::move(alloc), deviceProp, main_stream, side_stream, side_stream_event,
                         forward_done_event, backward_done_event, transfer_done_event, norm_done_event, lmhead_done, std::move(optimizer_events), optimizer_done_event,
                         nullptr, nullptr, nullptr, cudnn_handle, cublas_handle};
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
