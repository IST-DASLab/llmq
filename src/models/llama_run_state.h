// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_LLAMA_RUN_STATE_H
#define LLMQ_LLAMA_RUN_STATE_H

#include "llama_model.h"
#include "llama_weights.h"
#include "utilities/tensor.h"
#include "utilities/stack.h"

class TensorAllocator;
using sLLamaGradients = sLLamaWeightsSet<TensorShard>;
typedef struct cudnnContext* cudnnHandle_t;
typedef struct cublasLtContext* cublasLtHandle_t;
class LLamaGradsManager;

struct QuantizableTensor {
    /// original, high-precision value
    Tensor Value;
    /// Quantized value
    std::optional<Tensor> Quant = std::nullopt;
};


struct sLLamaLayerActivations {
    using QTensor = QuantizableTensor;
    Tensor LN1_Rstd;    // (B, T)
    QTensor LN1;        // (B, T, C)
    Tensor LN2_Rstd;    // (B, T)
    QTensor LN2;        // (B, T, C)
    Tensor QKV;         // (B, T, QKV_C)
    Tensor LSE;         // (B, T)
    QTensor Att;        // (B, T, C)
    Tensor AttO;        // (B, T, C)
    Tensor ResidualAtt; // (B, T, C)
    Tensor MlpUp;       // (B, T, 2*Ch)
    Tensor MlpDown;     // (B, T, C)
    QTensor SwiGLu;     // (B, T, Ch)
};

struct sLLamaLayerGradients {
    using QTensor = QuantizableTensor;

    QTensor DResFFN;                   // (B, T, C)
    Tensor DSwiGLU;                    // (B, T, Ch)
    QTensor DMlpUp;                    // (B, T, 2*Ch)
    Tensor DLN2;                       // (B, T, C)
    QTensor DResAtt;                   // (B, T, C)
    Tensor DAttY;                      // (B, T, C)
    QTensor DQKV;                      // (B, T, QKV_C)
    Tensor DLN1;                       // (B, T, C)
};

struct LLamaRunState {
    using LayerActivations = ::sLLamaLayerActivations;
    using LayerGradients = ::sLLamaLayerGradients;

    LLamaConfig Config;
    long B;
    long T;
    LLamaOptions Options;
    std::shared_ptr<TensorAllocator> Allocator;

    Tensor Inputs;          // (B, T) Int32
    Tensor Targets;         // (B, T) Int32
    Tensor Inputs_CPU;      // (B, T) Int32
    Tensor Targets_CPU;     // (B, T) Int32
    Tensor Losses;          // (B, T) FP32

    // Activations
    Tensor Encoded;         // (B, T, C)
    Tensor FreqCis;         // (mT, 2*HS)
    Tensor Output;          // (B, T, V)
    Tensor LNF;             // (B, T, C)
    Tensor LNF_Rstd;        // (B, T)
    std::vector<LayerActivations> Acts;

    std::vector<Tensor> DeviceResiduals;        // (B, T, C)
    std::vector<Tensor> OffloadedResiduals;

    struct sOffloadedResidualState {
        cudaEvent_t Event;
        int Layer;
        int Ready;
    };

    std::vector<sOffloadedResidualState> OffloadedResidualState;

    void fetch_res_ffn(int layer_idx, cudaStream_t fetch_stream);
    void put_res_ffn(int layer_idx, cudaStream_t put_stream);
    Tensor& get_res_ffn(int layer_idx, cudaStream_t main_stream);
    void mark_res_ffn_ready(int layer_idx, cudaStream_t main_stream);
    void release_res_ffn(int layer_idx, cudaStream_t main_stream);

    // Gradients
    std::vector<LayerGradients> DActs;

    Tensor DLNF;                // (B, T, C)
    Tensor DEmb;                // (B, T, C)

    // scratch buffers
    Tensor RMSNormScratch;      // (#Blocks*C+128)
    Tensor MatmulBiasScratch;   // TODO
    Tensor CuBlasWorkspace;
    Tensor CuDNNWorkspace;
    Tensor EncoderBwdScratch;   // (B, T, 5 * C / (x128::size * 32))
    Tensor EncoderBwdIndices;   // (B, T, 1 * C / (x128::size * 32)) [on CPU!]
    Tensor EncoderBwdInfo;      // (B, T, 4 * C / (x128::size * 32)) [on CPU!]

    std::optional<Tensor> AbsMaxes; // (L, ...)
    Tensor MatmulScales;        // 2 floats

    // Optimizer scratch
    Tensor NormBuffer;
    float* NormHost;
    float* LossHost;

    // temporary buffers
    Tensor temp_alloc(ETensorDType dtype, const std::vector<long>& shape);
    void temp_acquire(Tensor& target);
    void temp_free(Tensor& tensor);

    Tensor acquire_mlp_up(int layer);
    void release_mlp_up(Tensor& mlp_up);

    DeviceMemoryStack Stack;

    // cached GPU info
    cudaDeviceProp DeviceProp;

    cudaStream_t MainStream;
    cudaStream_t SideStream;
    cudaEvent_t SideStreamEvent;
    cudaEvent_t ForwardDone;        //!< recorded at the end of the forward pass
    cudaEvent_t BackwardDone;       //!< recorded at the end of the backward pass
    cudaEvent_t TransferDone;       //!< recorded once CPU-side buffers have been copied to GPU
    cudaEvent_t NormDone;           //!< recorded after norm calculation completes
    cudaEvent_t OptEmbeddingsDone;      //!< recorded after the optimizer has done an update to the LMHead
    std::vector<cudaEvent_t> LayerUpdateDone;   //!< Recorded after the optimizer has done an update to the specified layer.
    cudaEvent_t OptimizerDone;      //!< recorded after the optimizer completes
    cudaEvent_t ResidualsAreReady;  //!< recorded after the residuals have been calculated in forward, indicating that they may be offloaded

    cudaGraphExec_t GlobalNormGraph = nullptr;
    cudaGraphExec_t ForwardBlockGraph = nullptr;
    cudaGraphExec_t BackwardBlockGraph = nullptr;

    // events for debugging timings
    cudaEvent_t TimingOptimizerStart = nullptr;
    cudaEvent_t TimingOptimizerEnd   = nullptr;

    void setup_timing_events(int micro_steps);

    std::vector<cudaEvent_t> TimingForwardStart;
    std::vector<cudaEvent_t> TimingForwardEnd;
    std::vector<cudaEvent_t> TimingHeadStart;
    std::vector<cudaEvent_t> TimingHeadEnd;
    std::vector<cudaEvent_t> TimingBackwardStart;
    std::vector<cudaEvent_t> TimingBackwardEnd;

    cudnnHandle_t CudnnHandle;
    cublasLtHandle_t CublasLtHandle;

    void init(LLamaConfig config, long B, long T, DeviceMemoryStack& stack);
};

LLamaRunState allocate_run_state(LLamaConfig config, LLamaOptions options, long B, long T, DeviceMemoryStack& stack, std::shared_ptr<TensorAllocator> alloc);

float get_loss(LLamaRunState& acts);
float get_norm(LLamaRunState& acts);
Tensor& get_input_buffer(LLamaRunState& acts);
Tensor& get_target_buffer(LLamaRunState& acts);

#endif //LLMQ_LLAMA_RUN_STATE_H
