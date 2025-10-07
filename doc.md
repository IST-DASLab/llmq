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

    // LM-head will be handled in chunks during backward

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

    trace_or_execute_cuda_graph([&](){_forward_block(l, wgt, residual);},
        main_stream, rs->ForwardBlockGraph, rs->Options.UseCudaGraphs);
    Parameters->release_block(l, main_stream);
    if(l > 0) {
        rs->put_res_ffn(l-1, rs->SideStream);
    }
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

In addition to fetching parameters, if residuals are offloaded, we also need to write and send them with double-buffering. The `get_res_ffn` gets a buffer that is ready, `mark_res_ffn_ready` triggers and event that indicates the computation is complete, and `put_res_ffn` handles the actual transfer to host memory. If residuals are not offloaded, all these functions are no-ops.

Furthermore, you can see that the actual computation is split into two parts: First, the initial RMS-norm is handled, either by directly calling an `rmsnorm_forward` kernel (for the first block), or by calling a `fused_residual_rmsnorm_forward` kernel (for all other blocks).
The fused kernel takes the output of the previous block, adds it to the residual stream, and writes the rms-normed result to a second output, thus avoiding the need to read the residual from memory. Because this means that different kernels are called depending on the layer number, this part of the computation is separated from the rest of the forward pass, which has a static computation graph. 
Additional difficulties arise due to residual offloading; even if the graphs were the same, we would trigger an event inside the graph that would be waited on outside the graph, which results in an error.
Therefore, we can run the rest of the forward pass, defined in the `_forward_block` function, in a cuda graph.

The forward block itself is straightforward, now that we've ensured that all the parameters are available:
```cpp
auto& acts = rs->Acts[l];

// 1) projection to QKV vectors (note k,v may be fewer heads than q)
forward_qmm(acts.QKV, acts.LN1, weights.Attn_QKV_w, weights.Attn_QKV_b,
            rs->CublasLtHandle, rs->Workspace,
            B, T, C, Config.qkv_channels(),
            rs->DeviceProp, false, main_stream);
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
            rs->DeviceProp, false, main_stream);

fused_residual_rmsnorm_forward(acts.ResidualAtt, acts.LN2.Value, acts.LN2_Rstd, residual, acts.AttO, weights.LN2_w,
                               quant_abs_max_ptr(acts.LN2), Config.RmsNormEps, B * T, C, main_stream);

forward_qmm(acts.MlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
            rs->CublasLtHandle, rs->Workspace,
            B, T, C, 2 * D,
            rs->DeviceProp, false, main_stream);
swiglu_forward(acts.SwiGLu.Value, acts.MlpUp, quant_abs_max_ptr(acts.SwiGLu), B, T, D, main_stream);

forward_qmm(acts.MlpDown, acts.SwiGLu, weights.MLP_Down_w, std::nullopt,
            rs->CublasLtHandle, rs->Workspace,
            B, T, D, C,
            rs->DeviceProp, false, main_stream);
```
This is just a sequence of kernel calls, which can be this simple because the `acts` variable takes care of selecting the right pointers to load/store activations.

> There is a deliberate decision here in handling communication/weight gathering at the transformer-block level. While it would be possible to have individual gather calls for each parameter, this would result in a large number of additional kernel calls and event synchronizations. While there might be a latency advantage for the first layer (we can start computing as soon as the first parameter is available, instead of the whole layer), in subsequent layers there is no benefit: Either communication is faster than computation, and the next block can start immediately, or it is slower, in which case double-buffering still ensures that data is transferred at maximum bandwidth.
> 
> Double-buffering in itself is also much more difficult to implement at smaller granularity: If we had one buffer for the attention part, and one for the MLP part, for example, then we would get significant imbalances, with the MLP part requiring more communication and (for moderate sequence lengths) more computation. This would make it more difficult to ensure perfect communication/computation overlap.
> 
> In all, the block-level gather provides a simple mental model, keeps the forward_block function straightforward, and does not have significant performance drawbacks compared to more complicated implementations.

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
## Backward Pass
The backward pass is implemented in [llama_model.cpp](src/models/llama_model.cpp). The backward pass can similarly broken
down, though it has more parts.
```cpp
void LLamaModel::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int micro_step, int step) {
    // [define convenience variables]

    // results in the uniform average loss over all elements
    const float d_loss = 1.0f / (float) (B * T * grad_accum_steps);
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

    // reset residual stream gradients (put here to work with gradient accumulation)
    fill_zero(rs->DLNF, main_stream);
    fill_zero(d_acts[L-1].DResFFN.Value, main_stream);
    
    // [handle chunked LM-head]

    // ok, now reduce the loss across all ranks
    if (last_step) {
        _reduce_loss(*rs, comm, B, T);
    }

    auto& d_lnf_w = Grads->get_lnf_w_full(main_stream, comm, accumulate);
    Parameters->gather_lnf(comm);
    // backward the final layernorm
    rmsnorm_backward(d_acts[L-1].DResFFN.Value, d_lnf_w, rs->RMSNormScratch, d_acts[L - 1].DResFFN.Value, rs->DLNF,
                     rs->get_res_ffn(L-1, main_stream), Parameters->get_lnf(main_stream), rs->LNF_Rstd, quant_abs_max_ptr(d_acts[L-1].DResFFN), B, T, C, rs->DeviceProp, main_stream);
    rs->release_res_ffn(L-1, main_stream);

    Parameters->release_lnf(main_stream);
    Grads->notify_lnf_w(main_stream, comm);
    rs->fetch_res_ffn(L-2, comm.stream());
    Parameters->gather_block(L - 1, comm, *rs);
    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        // [handle transformer block]
    }

    // [handle embedding layer]

    // make sure all gradients are communicated before we go to the update step.
    Grads->end_micro_step(main_stream, comm);
    CUDA_CHECK(cudaEventRecord(rs->BackwardDone, main_stream));

    // do not return before inputs can be accessed again.
    CUDA_CHECK(cudaEventSynchronize(rs->TransferDone));
}
```
Let's look at the main flow first:
In the beginning, the targets are transferred from CPU to GPU. Then, the micro-step set-up is handled.
If it is the first micro-step in the batch, we need to zero out the loss accumulator and gradients. Gradient zeroing
is handled by the `Grads->start_micro_step` function and happens on the side stream. In any case, we need to inform
the `Grads` object that a new step is starting, so it can do the necessary bookkeeping.

Next, the classifier head and gradients are calculated in a fused kernel, i.e., we never write an unreduduced loss tensor to memory,
and update the logits in-place to their gradients. If this is the last step, we also need to reduce the loss across all GPUs.
At this point, we need to ensure that all gradient zeroing has finished.

With distributed training, it is possible for gradient accumulation to run on a different stream than main_stream, so we
ensure that the previous micro-step has finished before we allow any access to gradient buffers again.

Then we handle the LM-head, followed by a loop over all transformer blocks, and finally the embedding layer.
At the end, we infor the `Grads` object that the current micro-step is finished, which may trigger any outstanding gradient reductions,
record that the step has finished, and ensure that the target tensor has been transferred before we allow the function to return.

### LM-Head
The LM-head is handled in chunks, so that it is not necessary to allocate a very large logit tensor.
As with the other layers, the main computation is surrounded by gather/release calls.
Note that in case of the LM-head being tied with the embedding weights, the `gather_head` call would automatically detect if the embeddings are still present locally, and avoid doing needless communication.

```cpp
bool accumulate;
int nano_batch_size = div_exact(B * T, nano_batches);
Parameters->gather_head(comm);
for(int nano_step = 0; nano_step < nano_batches; nano_step++) {
    // [setup tensors to current chunk]

    matmul(rs->Output, Parameters->get_head(main_stream), lnf_slice,
           std::nullopt, nullptr, rs->CublasLtHandle, rs->Workspace, V, nano_batch_size, C, EMMTranspose::TN, false, main_stream);

    // accumulate the losses inside rs->losses, and kick off the backward pass inside the fused classifier
    fused_classifier(rs->Output, losses, d_loss, tgt, nano_batch_size, V, Vp, true, main_stream);

    // [wait until we're ready for gradient computation]

    // handle the LM-head. We run the d_lmhead matmul first, so that the gradient reduction can overlap with the DLNF matmul.
    auto& d_lmhead = Grads->get_lmhead_full(main_stream, comm, accumulate);
    accumulate |= nano_step != 0;
    matmul(d_lmhead, lnf_slice, rs->Output, std::nullopt, get_device_one(),
           rs->CublasLtHandle, rs->Workspace, C, V, nano_batch_size, EMMTranspose::NT, accumulate, main_stream);
    if (nano_step == nano_batches - 1) {
        Grads->notify_lmhead(main_stream, comm);
    }
    
    matmul(dlnf_slice, Parameters->get_head(main_stream), rs->Output, std::nullopt, get_device_one(),
           rs->CublasLtHandle, rs->Workspace, C, nano_batch_size, V, EMMTranspose::NN, false, main_stream);
}
Parameters->release_head(main_stream);
```
The main computation flow is as follows: First, generate tensors that point to the current chunk of tokens in the LNF, Target, Lss, and dlNF tensors. Then, run the forward matmul to calculate the logits.
Next, the classifier head and gradients are calculated in a fused kernel, i.e., we never write an unreduced loss tensor to memory and update the logits in-place to their gradients instead.
Before we can actually start calculating gradients, we may need to wait for previous calculations to finish.

For the gradient computation, we also need to have gradient buffers. Requesting with `Grads->get_lmhead_full` gets us a full-sized (i.e., unsharded) buffer for the gradient that can be written to.
Only after the last chunk has been processed, we can notify the `Grads` object that the LM-head gradients are ready for reduction across GPUs. 
Note that, as backward consists of two matmuls, we can actually start communicating the gradients after the last iteration of the _first_ matmul is done, so there is some overlap between the two. This is useful because the LM-head is generally much more expensive than a transformer block, so hiding its latency is difficult.

Finally, what are the events that we need to wait for before starting gradient calculations?
First, we want to ensure that zeroing of gradients has finished before we start generating new ones.
Second, we also need to ensure that the previous step's backward has finished.
In both cases, it is only necessary to wait on the first chunk.
```cpp
 // if we reset model grads to zero, now is the time we need to wait
if (micro_step == 0 && nano_step == 0) {
    CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs->SideStreamEvent, 0));
}

if(nano_step == 0) {
    // BackwardDone ensures that zero-2 gradient accumulation of the previous step has finished, so we can safely write to d_lmhead again.
    CUDA_CHECK(cudaEventSynchronize(rs->BackwardDone));
}
```


### Transformer Blocks
The transformer blocks are processed in the same way as in the forward pass.
```cpp
auto& dw = Grads->get_block_full(l, main_stream, comm, accumulate);
// prefetch previous layer
if(l > 0) {
    Parameters->gather_block(l - 1, comm, *rs);
}
auto& weights = Parameters->get_block(l, main_stream);
Tensor residual = l == 0 ? rs->Encoded : rs->get_res_ffn(l - 1, main_stream);
trace_or_execute_cuda_graph([&]() {
    _recompute_block(l, weights, residual);
    _backward_block(l, accumulate, weights, dw);
    }, main_stream, rs->BackwardBlockGraph, rs->Options.UseCudaGraphs && l != 0);
if(l > 0) {
    rs->release_res_ffn(l - 1, main_stream);
}
```
Again, we have a topology change for the last layer, so we don't cuda-graph it.

There is one more caveat here: To handle gradients, three steps are necessary:
First, calculate the full gradient for the local batch.
Second, scatter gradient shards to their respective GPUs.
Third, reduce the shard contributions from all GPUs.
In `nccl`, the `scatter_reduce` call combines steps 2 and 3 and runs them on the communication stream. However, in scenarios
where we handle the communication ourselves, we use `cudaMemcpyAsync` to handle transfers, and run the reduction manually on the main stream,
which means that no SMs need to be set aside. However, it also means that we need to manually schedule when the reduction happens.
In a double-buffered setup, that would be at the point in time when the gradient buffer is requested by the main stream again.
That still leaves a gap, because the last gradients to be transferred do not get their buffers reused in this step. For that reason,
the `Grads->end_micro_step(main_stream, comm);` is needed later in the backward pass.

As we may not keep all necessary activations in memory to make larger models fit, it may be necessary to recompute the missing subset. Therefore, inside the cuda graph context, we need to call two functions. First, `_recompute_block` recomputes the missing activations for the current block, and second, `_backward_block` calculates the gradients for the current block.

Finally, in the first micro-batch, we may directly set the gradients to the calculated values  (potentially saving some memory reads), but in subsequent micro-batches, we need to accumulate the gradients.

### Embeddings
The embeddings are surprisingly simple to handle:
```cpp
auto& d_emb = Grads->get_embeddings_full(main_stream, comm, accumulate);
encoder_backward(d_emb, rs->EncoderBwdScratch, rs->EncoderBwdIndices, rs->EncoderBwdInfo,
                 rs->DEmb, rs->Inputs, inputs, B, T, C, OptimizerRNG(), main_stream);
Grads->notify_embeddings(main_stream, comm);
```
In particular, embedding gradients do not require that the embedding weights are present locally.
Gradients are accumulated with stochastic rounding, which is seeded from OptimizerRNG.
> Random states are not yet handled systematically; TODO

### Recomputation
Now, let's look at the `_recompute_block` function.
```cpp
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
                 rs->CublasLtHandle, rs->Workspace,
                 B, T, C, Config.qkv_channels(),
                 rs->DeviceProp, !recompute_ln1, main_stream);
    rope_forward(acts.Rope, acts.QKV, rs->FreqCis, B, T, Hq, Hkv, Hs, main_stream);
}

if (recompute_att) {
    attention_forward_cudnn(acts.Att.Value, acts.LSE, acts.Rope, rs->Workspace, rs->CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);
    // AttO not needed in backward pass; but if we want to recompute the entire transformer block, we need its output
    // to recompute the FFN part
    if (opt.RecomputeBlock) {
        forward_qmm(acts.AttO, acts.Att, weights.Attn_Out_w, std::nullopt,
                     rs->CublasLtHandle, rs->Workspace,
                     B, T, C, C,
                     rs->DeviceProp, false, main_stream);
    }
}

// Feed-forward block
if(recompute_ln2) {
    if (opt.RecomputeBlock) {
        Tensor residual = layer == 0 ? rs->Encoded : rs->get_res_ffn(layer - 1, main_stream);
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
                     rs->DeviceProp, false, main_stream);
}

if(recompute_swiglu) {
    if (acts.SwiGLu.Quant.has_value()) {
        swiglu_forward_quant(acts.SwiGLu.Quant.value(), acts.MlpUp, acts.SwiGLu.Quant->Scales, B, T, D, main_stream);
    } else {
        swiglu_forward(acts.SwiGLu.Value, acts.MlpUp, nullptr, B, T, D, main_stream);
    }
}
```
This is mostly straightforward, doing subsets of the normal forward pass depending on which parts of the network are to be recomputed.
Two noteworthy observations:
1) As not all activations are necessary for the backward pass, some forward pass computations never have to be re-done. In particular, the gradient for the addition operation into the residual does not need to know the addends, so we need not rerun the matmul at the end of the feed-forward block. The output matmul of the attention block is only needed if we recalculate the entire transformer block; not for the backward pass, but so that we have the inputs required for the recalculated rms-norm of the ffn block.
2) We already know abs-maxes from the initial forward pass. Therefore, we can skip the absmax kernel calls in the recomputation phase. Even better, if the absmax is know, quantization can be fused into the nonlinearity kernel. This is currently only implemented for swiglu, which calls the `swiglu_forward_quant` function in case of fp8 activations.

> The way activation memory is handled right now is rather ad-hoc. We have a `Tensor` object for each activation, but if the activation is to be recomputed, the corresponding `Tensor`s in all blocks point to the same memory. In contrast, though, the `Scales` pointers still point to distinct locations, so that abs-maxes don't get overwritten.

### Backward Block
Finally, the actual computation of the backward pass. Now that all data movement and recomputation
has been taken care of, the remaining work is simply a sequence of function calls for the individual operations.
```cpp
// backward the 2nd matmul of MLP
// note that _recompute_block guarantees that if SwiGLu is already quantized (if necessary)
backward_qmm(d_acts.DSwiGLU, d_weights.MLP_Down_w, std::nullopt, d_acts.DResFFN, acts.SwiGLu, weights.MLP_Down_w, std::nullopt,
             accumulate, *rs, B, T, D, C, true, main_stream);

swiglu_backward(d_acts.DMlpUp.Value, d_acts.DSwiGLU, acts.MlpUp, quant_abs_max_ptr(d_acts.DMlpUp), B, T, D, main_stream);

backward_qmm(d_acts.DLN2, d_weights.MLP_Up_w, std::nullopt, d_acts.DMlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
             accumulate, *rs, B, T, C, 2 * D, !rs->Options.RecomputeRMSNorm, main_stream);

// rmsnorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
rmsnorm_backward(d_acts.DResAtt.Value, d_weights.LN2_w, rs->RMSNormScratch, d_acts.DResFFN.Value, d_acts.DLN2,
                 acts.ResidualAtt, weights.LN2_w, acts.LN2_Rstd, quant_abs_max_ptr(d_acts.DResAtt), B, T, C, rs->DeviceProp, main_stream);

bool recompute_ln1 = rs->Options.RecomputeRMSNorm || rs->Options.RecomputeAtt;
backward_qmm(d_acts.DAttY, d_weights.Attn_Out_w, std::nullopt, d_acts.DResAtt, acts.Att, weights.Attn_Out_w, std::nullopt,
             accumulate, *rs, B, T, C, C, false, main_stream);

attention_backward_cudnn(d_acts.DRope, acts.LSE, acts.Att.Value, d_acts.DAttY, acts.Rope, rs->Workspace, rs->CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);
rope_backward(d_acts.DQKV.Value, d_acts.DRope, rs->FreqCis, B, T, Hq, Hkv, Hs, main_stream);

if (d_acts.DQKV.Quant.has_value()) {
    abs_max(d_acts.DQKV.Quant->Scales, d_acts.DQKV.Value, d_acts.DQKV.Value.nelem(), rs->DeviceProp, main_stream);
}

backward_qmm(d_acts.DLN1, d_weights.Attn_QKV_w, d_weights.Attn_QKV_b, d_acts.DQKV, acts.LN1, weights.Attn_QKV_w, rs->MatmulBiasScratch,
             accumulate, *rs, B, T, C, Config.qkv_channels(), !recompute_ln1, main_stream);

if(layer > 0) {
    auto& prev_dacts = rs->DActs.at(layer - 1);
    rmsnorm_backward(prev_dacts.DResFFN.Value, d_weights.LN1_w, rs->RMSNormScratch, prev_dacts.DResAtt.Value, d_acts.DLN1,
                     rs->get_res_ffn(layer-1, main_stream), weights.LN1_w, acts.LN1_Rstd, quant_abs_max_ptr(prev_dacts.DResFFN),
                     B, T, C, rs->DeviceProp, main_stream);
} else {
    rmsnorm_backward(rs->DEmb, d_weights.LN1_w, rs->RMSNormScratch, d_acts.DResAtt.Value, d_acts.DLN1,
                     rs->Encoded, weights.LN1_w, acts.LN1_Rstd, nullptr, B, T, C, rs->DeviceProp, main_stream);
}
```
