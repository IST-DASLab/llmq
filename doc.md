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
                        rs->Acts[0].LN1.Quant.has_value() ? rs->Acts[0].LN1.Quant->Scales : nullptr,
                        Config.RmsNormEps, B, T, C, main_stream);
    } else if(l != Config.NumLayers) {
        auto& acts = rs->Acts[l-1];
        fused_residual_rmsnorm_forward(acts.ResidualFFN, rs->Acts[l].LN1.Value, rs->Acts[l].LN1_Rstd,
                                       acts.ResidualAtt, acts.MlpDown, wgt.LN1_w,
                                       rs->Acts[l].LN1.Quant.has_value() ? rs->Acts[l].LN1.Quant->Scales : nullptr,
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

Furthermore, you can see that the actual computation is split into two parts: First, the initial RMS-norm is handled, either by directly calling an `rmsnorm_forward` kernel  (for the first block), or by calling a `fused_residual_rmsnorm_forward` kernel (for all other blocks).
The fused kernel takes the output of the previous block, adds it to the residual stream, and writes the rms-normed result to a second output, thus avoiding the need to read the residual from memory. Because this means that different kernels are called depending on the layer number, this part of the computation is separated from the rest of the forward pass, which has a static computation graph.
Therefore, we can run the rest of the forward pass, defined in the `_forward_block` function, in a cuda graph.

The forward block itself is straightfoward, now that we've ensured that all the parameters are available:
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

forward_qmm(acts.AttO, acts.Att, weights.Attn_Out_w, std::nullopt,
            rs->CublasLtHandle, rs->Workspace,
            B, T, C, C,
            rs->DeviceProp, false, false, main_stream);

fused_residual_rmsnorm_forward(acts.ResidualAtt, acts.LN2.Value, acts.LN2_Rstd, residual, acts.AttO, weights.LN2_w,
                               acts.LN2.Quant.has_value() ? acts.LN2.Quant->Scales : nullptr,
                               Config.RmsNormEps, B * T, C, main_stream);

forward_qmm(acts.MlpUp, acts.LN2, weights.MLP_Up_w, std::nullopt,
            rs->CublasLtHandle, rs->Workspace,
            B, T, C, 2 * D,
            rs->DeviceProp, false, true, main_stream);
swiglu_forward(acts.SwiGLu.Value, acts.MlpUp, acts.SwiGLu.Quant.has_value() ? acts.SwiGLu.Quant->Scales: nullptr, B, T, D, main_stream);

forward_qmm(acts.MlpDown, acts.SwiGLu, weights.MLP_Down_w, std::nullopt,
            rs->CublasLtHandle, rs->Workspace,
            B, T, D, C,
            rs->DeviceProp, false, true, main_stream);
```
This is just a sequence of kernel calls, which can be this simple because the `acts` variable takes care of selecting the right pointers to load/store activations.
You might think that there is still a problem with the first layer being handled differently, but  changing `residual` only affects which _pointer_ is passed to the cuda kernel, not the computation graph.

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


