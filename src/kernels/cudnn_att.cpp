// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cudnn_frontend.h>

#include "utilities/utils.h"

namespace fe = cudnn_frontend;

static void cuDNNCheck_(cudnnStatus_t error, const char *file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck_(err, __FILE__, __LINE__))

static void checkCudnnFE(const fe::error_object& e, const char *file, int line) {
    if(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

enum UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    Stats_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID
};

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::map<std::tuple<int,int,int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<int,int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
auto lookup_cache_or_build_graph_fwd(int B, int Hq, int Hkv, int T, int HS, int is_inference_only, cudnnHandle_t cudnn_handle) {

    thread_local cache_type_fwd user_maintained_cache_fwd;

    auto key = std::make_tuple(B, Hq, Hkv, T, HS, is_inference_only);

    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    // for (B, N, (NH + 2*(NH/replicate_factor)) * HS)
    // (B, T, Hq + 2Hkv, HS)
    int H = Hq + 2 * Hkv;
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, Hq, T, HS})
                               .set_uid(Q_UID)
                               .set_stride({H * HS * T,  HS, H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, Hkv, T, HS})
                               .set_uid(K_UID)
                               .set_stride({H * HS * T, HS, H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, Hkv, T, HS})
                               .set_uid(V_UID)
                               .set_stride({H * HS * T, HS, H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_uid(Attn_scale_UID)
                               .set_is_pass_by_value(true)
                               .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, Hq, HS) BF16/FP16 and stats for backward pass is (B, Hq, T) FP32
    O->set_output(true).set_dim({B, Hq, T, HS}).set_stride({Hq * HS * T, HS, Hq * HS, 1}).set_uid(O_UID);

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, Hq, T, 1})
                               .set_stride({Hq * T, T, 1, 1})
                               .set_uid(Stats_UID);
    }

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    user_maintained_cache_fwd.insert({key, graph});

    return graph;
}

auto lookup_cache_or_build_graph_bwd(int B, int Hq, int Hkv, int T, int HS, cudnnHandle_t cudnn_handle) {
    thread_local cache_type_bwd user_maintained_cache_bwd;

    auto key = std::make_tuple(B, Hq, Hkv, T, HS);

    auto it = user_maintained_cache_bwd.find(key);
    if (it != user_maintained_cache_bwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    int H = Hq + 2*Hkv;
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, Hq, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({H * HS * T, HS, H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, Hkv, T, HS})
                            .set_uid(K_UID)
                            .set_stride({H * HS * T, HS, H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, Hkv, T, HS})
                            .set_uid(V_UID)
                            .set_stride({H * HS * T, HS, H * HS, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                            .set_dim({B, Hq, T, HS})
                            .set_uid(O_UID)
                            .set_stride({Hq * HS * T, HS, Hq * HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                            .set_dim({B, Hq, T, HS})
                            .set_uid(dO_UID)
                            .set_stride({Hq * HS * T, HS, Hq * HS, 1}));

    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                            .set_dim({B, Hq, T, 1})
                            .set_uid(Stats_UID)
                            .set_stride({Hq * T, T, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_uid(Attn_scale_UID)
                            .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(true) // 1.5+ needs this for determinism
#endif
                            .set_causal_mask(true)
                            .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, Hq, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, Hkv, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, Hkv, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    user_maintained_cache_bwd.insert({key, graph});
    return graph;
}

void attention_forward_cudnn(nv_bfloat16* out,  // output: (B, T, Hq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const nv_bfloat16* inp,  // input: (B, T, Hq + Hk + Hv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    NVTX_RANGE_FN();
    bool is_inference_only = (stats == nullptr);

    cuDNNCheck(cudnnSetStream(handle, stream));

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_fwd(B, Hq, Hkv, T, HS, is_inference_only, handle);

    // Prepare all the tensor pointers for executing the graph
    const void* devPtrQ = inp;
    const void* devPtrK = (inp + Hq * HS);
    const void* devPtrV = (inp + (Hq + Hkv) * HS);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , void*> variant_pack = {
        {Q_UID, (void*)devPtrQ}, {K_UID, (void*)devPtrK}, {V_UID, (void*)devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(graph->execute(handle, variant_pack, workspace));
    CUDA_CHECK(cudaGetLastError());
}

void attention_backward_cudnn(nv_bfloat16* dqkvr,                                       // output
                              const float* stats,
                              const nv_bfloat16* dout, const nv_bfloat16* qkvr, const nv_bfloat16* o, // inputs
                              std::byte* workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    NVTX_RANGE_FN();

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_bwd(B, Hq, Hkv, T, HS, handle);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = (void*)qkvr;
    void* devPtrK = (void*)(qkvr + Hq * HS);
    void* devPtrV = (void*)(qkvr + (Hq + Hkv) * HS);
    void* devPtrO = (void*)o;
    void* devPtrdO = (void*)dout;
    void* devPtrStats = (void*)stats;
    float attn_scale_cpu = 1.f / sqrtf(HS);

    void* devPtrdQ = dqkvr;
    void* devPtrdK = (dqkvr + Hq * HS);
    void* devPtrdV = (dqkvr + (Hq + Hkv) * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnnSetStream(handle, stream));
    checkCudnnFE(graph->execute(handle, variant_pack, workspace));
    CUDA_CHECK(cudaGetLastError());
}

std::size_t cudnn_get_workspace_size(int B, int T, int Hq, int Hkv, int HS, cudnnHandle_t handle)
{
    auto graph = lookup_cache_or_build_graph_bwd(B, Hq, Hkv, T, HS, handle);
    return graph->get_workspace_size();
}

cudnnHandle_t create_cudnn_handle() {
    cudnnHandle_t handle;
    cuDNNCheck(cudnnCreate(&handle));
    return handle;
}

// TODO cudnn destroy