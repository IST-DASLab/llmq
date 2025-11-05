This post provides a deep dive into the implementation of llm.q.
It is organized in a top-down fashion: We will start by looking at the
full training loop, and then gradually zoom in to the forward and backward passes,
and the individual components that enable them.


## Training Loop
Implemented in [train.cpp](train.cpp), the training loop corresponds roughly to the following:
```cpp
for (int step = latest_step; step < MaxSteps; ++step) {
    for (int j = 0; j < GradAccSteps; ++j) {
        train_loader.load_batch(inputs, targets);
        model.forward(inputs, comm, j);
        model.backward(inputs, targets, comm, GradAccSteps, j);
    }

    float lr = schedule.get_lr(step);
    model.update(comm, lr, Beta1, Beta2, step + 1, 1e-8f, WeightDecay, GradClip);
    cudaDeviceSynchronize();
}
```
Each step comprises `GradAccSteps` forward and backward passes, followed by a single model update.
The `forward`, `backward`, and `update` functions (mostly) are asynchronous; only after the update do we enforce synchronization between CPU and GPU. However, `forward` and `backward` ensure that
they only return _after_ `inputs` and `targets` have been transferred to the GPU, so that these buffers can be safely reused for the next batch.


## Forward Pass
The forward pass is implemented in [llama_model.cpp](src/models/llama_model.cpp). The forward pass
can be broken down into three main parts, plus some additional orchestration. Let's look at the overall flow first:
```cpp
void LLamaModel::forward(Tensor inputs, NCCLCommunicator& comm, int micro_step) {
    // [define convenience variables]
    
    // If this is the first micro-step, the parameters have just changed, and we can not
    // re-use any cached values
    if(micro_step == 0) {
        Parameters->invalidate();
    }
    
    // copy inputs to GPU
    cudaMemcpyAsync(rs->Inputs.Data, inputs.Data, inputs.bytes(), cudaMemcpyHostToDevice, main_stream);
    cudaEventRecord(rs->TransferDone, main_stream);

    // [handle embedding layer]

    if(rs->AbsMaxes.has_value())
        fill_zero(rs->AbsMaxes.value(), main_stream);

    Parameters->gather_block(0, comm, *rs);
    for (int l = 0; l < Config.NumLayers; l++) {
        // [handle transformer block]
    }

    // [handle lm-head]

    // do not return before inputs can be accessed again.
    cudaEventSynchronize(rs->TransferDone);
}
```
After defining some convenience variables (e.g., B, T for the batch size and sequence length), we
copy the input data to the GPU and record a cuda event (the `invalidate` function will be explained later).
Then, we handle the embedding layer and reset abs-maxes that are needed for activation quantization.
Then, we loop over all transformer blocks, before handling the LM-head.
Finally, we wait for the transfer-done event (potentially stalling the CPU), to ensure that the function does not return before it is safe to access the input memory.

> For training, this is actually overly cautious, because each forward call will be followed by a backward call, only after which new inputs will be fetched. But during validation, there is no backward so this synchronization is necessary. As the bulk of GPU work is still scheduled asynchronously, though, an implementation like this is perfectly fine.

Now, let's look at the individual parts. 

### Embeddings
```cpp
Parameters->gather_embeddings(comm);
encoder_forward(
    rs->Encoded,
    rs->Inputs,
    Parameters->get_embeddings(main_stream),
    std::nullopt, B, T, C, V, main_stream); 
Parameters->release_embeddings(main_stream);
```
In addition to the actual kernel call `encoder_forward`, there are three more functions involved:
`gather_embeddings`, `get_embeddings`, and `release_embeddings`. These are responsible for FSDP/DDP/offloading. The `gather` function checks whether an up-to-date copy of the embeddings is already available on the GPU, and if not, it initiates the transfer that gathers the embeddings either from peer GPUs or from the CPU. 
The `get_embeddings` function returns a pointer to the embeddings on the GPU and inserts an event synchronization into the main stream that ensures the `gather` operation has finished. Finally, `release_embeddings` signals that we are finished using the embeddings, so their memory _may_ be reused.

This now explains the need for the `invalidate` function: In the first micro-step, the parameters have just been changed by the optimizer, and we need to ensure that `gather_*` does not try to  re-use cached values.

The importance of the gather-get-release cycle, and why this is implemented in three different functions, becomes more obvious for the transformer blocks.

### Transformer Blocks
The transformer blocks are processed as follows:
```cpp
Parameters->gather_block(0, comm, *rs);
for (int l = 0; l < Config.NumLayers; l++) {
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
    
    trace_or_execute_cuda_graph([&](){_forward_block(l, weights, *rs);},
        main_stream, rs->ForwardBlockGraph, rs->Options.UseCudaGraphs);
    Parameters->release_block(l, main_stream);
}
```
There is a lot to unpack here. First, note that gather-get-release calls are now _interleaved_ between different layers. In particular, the order of calls is
```txt
gather(0)
gather(1)
get(0)
release(0)
gather(2)
get(1)
release(1)
...
```
This enables efficient overlap of communication and computation: While the call to gather(l+1) is
gathering the parameters for the next layer, the current layer can be processed. The three functions take care of correctly synchronizing the `main_stream` with the communication stream.
The `release` function is necessary to implement double-buffered memory management. In case the communication is faster than the computation, gather(l+2) cannot be allowed to start until release(l) has been called to signal that the memory buffer shared between these two layers is no longer needed.

Furthermore, you can see that the actual computation is split into two parts: First, the initial RMS-norm is handled, either by directly calling an `rmsnorm_forward` kernel (for the first block), or by calling a `fused_residual_rmsnorm_forward` kernel (for all other blocks).
The fused kernel takes the output of the previous block, adds it to the residual stream, and writes the rms-normed result to a second output, thus avoiding the need to read the residual from memory. Because this means that different kernels are called depending on the layer number, this part of the computation is separated from the rest of the forward pass, which has a static computation graph.
Therefore, we can run the rest of the forward pass, defined in the `_forward_block` function, in a cuda graph.

The forward block itself is straightforward, now that we've ensured that all the parameters are available:
```cpp
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
// quantize attention if necessary
if(acts.Att.Quant.has_value()) {
    abs_max(acts.Att.Quant->Scales, acts.Att.Value, acts.Att.Value.nelem(), rs->DeviceProp, main_stream);
}
    
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
```
This is just a sequence of kernel calls, which can be this simple because the `acts` variable takes care of selecting the right pointers to load/store activations.
You might think that there is still a problem with the first layer being handled differently, but changing `residual` only affects which _pointer_ is passed to the cuda kernel, not the computation graph.

> There is a deliberate decision here in handling communication/weight gathering at the transformer-block level. While it would be possible to have individual gather calls for each parameter, this would result in a large number of additional kernel calls and event synchronizations. While there might be a latency advantage for the first layer (we can start computing as soon as the first parameter is available, instead of the whole layer), in subsequent layers there is no benefit: Either communication is faster than computation, and the next block can start immediately, or it is slower, in which case double-buffering still ensures that data is transferred at maximum bandwidth.
> 
> Double-buffering in itself is also much more difficult to implement at smaller granularity: If we had one buffer for the attention part, and one for the MLP part, for example, then we would  get significant imbalances, with the MLP part requiring more communication and (for moderate sequence lengths) more computation. This would make it more difficult to ensure perfect communication/computation overlap.
> 
> In all, the block-level gather provides a simple mental model, keeps the forward_block function straightforward, and does not have significant performance drawbacks compared to more complicated implementations. 

### LM-Head
The LM-head follows the now-familiar pattern:
```cpp
auto& acts = rs->Acts[Config.NumLayers-1];
Parameters->gather_lnf(comm);
fused_residual_rmsnorm_forward(
    acts.ResidualFFN, rs->LNF, rs->LNF_Rstd, acts.ResidualAtt,
    acts.MlpDown, Parameters->get_lnf(main_stream), nullptr, Config.RmsNormEps, 
    B * T, C, main_stream);
Parameters->release_lnf(main_stream);

Parameters->gather_head(comm);
matmul_forward(rs->Output, rs->LNF, Parameters->get_head(main_stream),
               std::nullopt, nullptr, rs->CublasLtHandle, rs->Workspace, 
               B, T, C, V, main_stream);
Parameters->release_head(main_stream);
```
Note that in case of the LM-head being tied with the embedding weights, the `gather_head` call would automatically detect if the embeddings are still present locally, and avoid doing needless communication.

### Quantized Matmul
Let's take a look inside the `forward_qmm` function that we've used above.
```cpp
void forward_qmm(Tensor& out, QuantizableTensor& inp, Tensor& weight, std::optional<Tensor> bias,
                 cublasLtHandle_t handle, Tensor workspace,
                 int B, int T, int C, int OC,
                 const cudaDeviceProp& dp, bool reuse_inp_quant,
                 cudaStream_t stream) {
    if (weight.DType == inp.Value.DType) {
        matmul(out, weight, inp.Value, bias, nullptr, handle, workspace, OC, B*T, C, EMMTranspose::TN, false, stream);
    } else {
        float* act_scale = inp.Quant->Scales;
        float* wgt_scale = weight.Scales;
        float* out_scale = out.Scales;

        if (!reuse_inp_quant) {
            quantize_with_abs_max(inp.Quant.value(), inp.Value, act_scale, B*T*C, dp, stream);
        }

        float scale = weight.DType == ETensorDType::FP8_E4M3 ? 448.f : std::numeric_limits<std::int8_t>::max();
        matmul_out_scale(out_scale, act_scale, wgt_scale, 1.f / scale / scale, stream);
        matmul(out, weight, inp.Quant.value(), bias, out_scale, handle, workspace, OC, B*T, C, EMMTranspose::TN, false, stream);
    }
}
```
There is the trivial code-path, when the matmul is not quantized (in the sense of not having any scale factors), identified by weights having the same dtype as unquantized inputs.
If the weights are quantized, we need to ensure the input dtype matches. In the ideal case,
a quantized version of the input is already available (`reuse_inp_quant`), otherwise, we need to quantize the input.
We assume, however, that at least the abs-max has already been computed (usually, we fuse this into the previous op anyway).

After both operands to the matmul are quantized, there is a slightly peculiar operation: `matmul_out_scale` launches a cuda kernel with exactly one thread. This kernel computes the scale factor for the output of the subsequent matmul. To avoid a CPU-GPU sync, this single calculation is done inside a cuda kernel.
Finally, the actual matmul is launched.

### Implementation of weight management
Finally, let's break open the gather/get/release black-boxes and see how these functions are implemented.

The first decision the `gather` function needs to make is to check whether the weights are already present locally. For that purpose, we maintain the following data structure for each 
block buffer:
```cpp
struct sGatherData {
    int LayerIdx = -1;                  // which layer currently stored in this buffer
    cudaEvent_t DoneEvent = nullptr;    // cuda event to synchronize actions
    bool Fetch = false;                 // indicates whether a gather op has been scheduled
    bool Done = true;                   // indicates whether the param is in use
    int Version = -1;                   // last step at which we gathered this param
};
```
Then we can check whether data needs to be gathered as follows:
```cpp
bool is_in_cache(sGatherData& data, int expected) const {
    if(!data.Done) {
        throw std::logic_error("still in use");
    }

    if(data.LayerIdx == expected && data.Version == mVersion) {
        data.Fetch = false;
        return true;
    }

    data.LayerIdx = expected;
    data.Fetch = true;
    return false;
}
```
This first checks that the buffer is not still in use, logically (i.e., we forgot to call `release` -- this is a programming error, the branch will never be taken with correct usage).
As we tag each buffer with the layer it currently stores, and a step number (version), we can check if the data currently present if from the same layer _and_ same step. This can happen at the transition from forward to backward pass, when the last transformer block is still available locally. In such cases, we record that we do not need to fetch any data.
Otherwise, we update `LayerIdx` to indicate that this buffer now stores the new layer, and mark that data will be fetched.

The gather function itself first checks if cached data is present. If not, it checks if an already quantized version of the weights exists. If not, dtype conversions are started.
If there happens to be any conversion, we need to update the `DoneEvent` to ensure that communication only starts after the dtype conversion is complete. The `DoneEvent` would have been previously reported at the release of the old weights within this buffer, which happens on the main stream, too, so updating the event is a safe operation that pushes the event completion to a later point in time.
Finally, we can schedule the gather operations themselves, execute them, and record a new version of `DoneEvent`, this time triggered on the communication stream.
```cpp
void gather_block(int layer_idx, NCCLCommunicator& comm, LLamaRunState& run_state) {
    auto& src = get_master_block(layer_idx);     auto& qnt = lookup_block_quants(layer_idx);
    auto& dst = lookup_block_weights(layer_idx); auto& gather_data = lookup_block_status(layer_idx);

    // Check if data is still in cache
    if(is_in_cache(gather_data, layer_idx))
        return;

    bool convert_any = false;
    if (qnt.Version != mVersion || qnt.LayerIdx != layer_idx) {
        convert_dtype_for_gather(src.LN1_w, qnt.Block.LN1_w, convert_any, run_state);
        // [other weights]
        qnt.Version = mVersion;
        qnt.LayerIdx = layer_idx;
    }

    // make sure the target scales are set up correctly
    dst.LN1_w.Scales = qnt.Block.LN1_w.Scales;
    // [other weights]
    if (convert_any)
        cudaEventRecord(gather_data.DoneEvent, run_state.MainStream);

    comm.begin_transaction(gather_data.DoneEvent);
    comm.schedule_all_gather(qnt.Block.LN1_w, dst.LN1_w);
    // [other weights]
    comm.execute_transaction(gather_data.DoneEvent);
}
```

After the gather call, to ensure that the weights are available, the `get_block` function needs to be called. 
```cpp
void update_get_status(sGatherData& data, int expected, cudaStream_t stream) const {
    data.Done = false;

    cudaEvent_t done_event = data.DoneEvent;
    if(data.LayerIdx != expected)
        throw std::logic_error("Gather data is not for the requested layer");

    // if we needed to fetch, we need to wait
    if(data.Fetch) 
        cudaStreamWaitEvent(stream, done_event, 0);

    data.Version = mVersion;
}

sLLamaBlockWeights<Tensor>& get_block(int layer_idx, cudaStream_t stream) {
    auto& gather_data = lookup_block_status(layer_idx);
    update_get_status(gather_data, layer_idx, stream);
    return lookup_block_weights(layer_idx);
}
```
The main implementation happens in the `update_get_status` helper, which does the following:
1) It marks the block as in use, so subsequent gather calls will fail without an intermediate release.
2) It ensures consistency between the gathered layer idx and the requested one.
3) If data was fetched, the (main) stream is blocked until the transfer is complete.
4) Set the version number to the current step.

Then the actual computation with the block can happen. Finally, to mark the buffer as available again, we need to call the `release` function. This again checks consistency, records the `DoneEvent`, and marks the buffer as done.
```cpp
void release_status(sGatherData& data, int expected, cudaStream_t stream) {
    if(data.LayerIdx != expected) {
        throw std::logic_error("Gather data is not for the requested layer");
    }
    CUDA_CHECK(cudaEventRecord(data.DoneEvent, stream));
    data.Done = true;
}

void release_block(int layer_idx, cudaStream_t stream) {
    auto& gather_data = lookup_block_status(layer_idx);
    release_status(gather_data, layer_idx, stream);
}
```
