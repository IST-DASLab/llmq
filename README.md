# LLM.Q
[![Build Wheels](https://github.com/IST-DASLab/llmq/actions/workflows/wheel.yml/badge.svg)](https://github.com/IST-DASLab/llmq/actions/workflows/wheel.yml)

Quantized LLM training in pure CUDA/C++.

## Overview
`llm.q` is an implementation of (quantized) large language model training in CUDA, inspired by [llm.c](https://github.com/karpathy/llm.c). It is particularly aimed at medium-sized training setups, i.e., a single node with multiple GPUs.

## Build instructions
The code is written in C++20 and requires CUDA 12 or later.  It depends on [nccl](https://developer.nvidia.com/nccl) for communication, and [cudnn](https://developer.nvidia.com/cudnn) for fast attention. Multi-GPU training can either be run in multi-process mode (requires OpenMPI) or in multi-thread mode. On a recent ubuntu system, this should provide
the required dependencies (adapt cuda version as needed):
```´shell
# build tools
apt install cmake ninja-build git gcc-13 g++-13
# libs
apt install cuda-12-8 cudnn9-cuda-12-8 libnccl2 libnccl-dev libopenmpi-dev
```

Additional header-only dependencies are automatically downloaded by cmake during the build process.
These are:
* [json](https://github.com/nlohmann/json)
* [cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend)
* [CLI11](https://github.com/CLIUtils/CLI11)
* [fmt](https://github.com/fmtlib/fmt)
* [nanobind](https://github.com/wjakob/nanobind) (for optional python bindings)

To build the training executable, run
```shell
mkdir build
cmake -S . -B build
cmake --build build --parallel --target train
```

## How to train your (quantized) llm
### Data preparation
In order to train/fine-tune a model, you first need some data. The [tokenize_data](scripts/tokenize_data.py) script provides a utility to prepare token files for training.
It only supports a limited number of datasets, but hacking it for your own dataset should be straightforward.
```shell
uv run scripts/tokenize_data.py --dataset tiny-shakespeare --model qwen
```
This will create `tiny-shakespeare-qwen-train.bin` and `tiny-shakespeare-qwen-eval.bin`.

### Training run
Let's fine-tune the smallest Qwen model on this data:
```shell
./build/train --model=Qwen/Qwen2.5-0.5B \
  --train-file=data/tiny-shakespeare-qwen/train.bin \
  --eval-file=data/tiny-shakespeare-qwen/eval.bin \
  --model-dtype=bf16 --opt-m-dtype=bf16 --opt-v-dtype=bf16 \
  --matmul-dtype=e4m3 \
  --recompute-block \
  --grad-accumulation=8 --steps=30 \
  --learning-rate=1e-5 --gpus=1 --batch-size=8
```
The program will print some logging information, such as the following:
```text
[Options]
  recompute-swiglu  : true
  recompute-norm    : true
  [...]

[System 0]
  Device 0: NVIDIA GeForce RTX 4090
  CUDA version: driver 13000, runtime 13000
  Memory: 906 MiB / 24080 MiB

Loading model from `/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors`
 done

[Dataset]
 train:  329k tokens
   data/tiny-shakespeare-qwen/train.bin :     329663
 eval:   33k tokens
   data/tiny-shakespeare-qwen/eval.bin :      33698

[Allocator State]
        Adam V:   942 MiB
        Adam M:   942 MiB
     Gradients:   942 MiB
   Activations:  4505 MiB
        Master:   682 MiB
       Weights:   601 MiB
          Free: 14078 MiB
      Reserved:   483 MiB
         Other:   903 MiB
```

If `[Allocator State]` shows a lot of `Free` memory (as it does here), you can try increasing the batch size (and adjust `--grad-accumulation` accordingly), or reduce the amount of activation checkpointing.
For example, on a 16GiB 4060Ti, `--recompute-block` can be replaced by `--recompute-swiglu`, which increases the activation memory from `4.5 GiB` to `9 GiB`, and
the speed from ~11k tps to ~13k tps.

Then, the actual training will begin:
````text
[T] step     0 [ 19.9%] | time:  1869 ms | norm   4.315545 | loss   3.282568 | tps 35064 | sol 42.9%
[T] step     1 [ 39.8%] | time:  1709 ms | norm   8.423664 | loss   3.310652 | tps 38347 | sol 46.9%
[T] step     2 [ 59.6%] | time:  1708 ms | norm   4.818971 | loss   3.330125 | tps 38370 | sol 47.0%
[T] step     3 [ 79.5%] | time:  1715 ms | norm   5.247286 | loss   3.259991 | tps 38213 | sol 46.8%
[V] step     4 [  0.0%] | time:   165 ms | eval   2.945187 | train  3.295834 | tps  148k
````
Each `[T]` line is a training step (in contrast to `[V]` validation). It shows the step number, the progress within the epoch,
the elapsed time, as well as the current loss and gradient norm. It also calculates the current throuput in tokens per second,
and sets this in relation to the GPU's speed of light (SOL), i.e., the fastest possible speed if the GPU was only running strictly necessary matmuls at peak flop/s.

### Inspecting the logs
After 50 steps, the training will finish, and save the final model to `model.safetensors`. In addition, a log file will be created,
which contains the training log in JSON format. We can visualize the log using the [plot_training_run.py](scripts/plot_training_run.py) utility script:`
```shell
uv run scripts/plot_training_run.py log.json
```
This shows the training and evaluation losses over time, for quick inspection.
For a more detailed and interactive workflow, you can export the log to weights&biases:
```shell
uv run scripts/export_wandb.py --log-file log.json --project <ProjectName>
```

### Evaluations
Finally, we want to evaluate the produced model, which by default is placed in `output/model.safetensors`. This
file is [transformers](https://huggingface.co/docs/transformers) compatible, in contrast to checkpoints that get generated
at intermediate steps of longer training runs. However, as `llmq` does not include any tokenization, in order to run this
with transformers we still need the tokenizers. A simple way to get them is to just copy them over from the original HF
checkpoint, in this example:
```shell
cp /huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer* output/
```
Then we can run [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) evaluations:
```shell
uv run lm_eval --model hf --model_args pretrained=./output --tasks hellaswag --batch_size auto
```
which yields

| Tasks     | Version | Filter | n-shot | Metric   |   |  Value |   | Stderr |
|-----------|--------:|--------|-------:|----------|---|-------:|---|-------:|
| hellaswag |       1 | none   |      0 | acc      | ↑ | 0.4043 | ± | 0.0049 |
|           |         | none   |      0 | acc_norm | ↑ | 0.5270 | ± | 0.0050 |

Of course, we wouldn't expect tiny-shakespeare to improve these evaluation metrics, but for a real training run,
running evaluations with lm-eval provides independent verification of the resulting model.

## A larger example
For a real training run, we aim for a 1.5B model in Qwen architecture, trained on 10B tokens of [Climb](https://research.nvidia.com/labs/lpr/climb/)
using 4 x RTX 4090 GPUs:
```shell
uv run scripts/tokenize_data.py --dataset climb-10b --model qwen
./build/train --model=Qwen/Qwen2.5-1.5B \
     --from-scratch \
     "--train-file=data/climb-10b-qwen/train-*.bin"  --eval-file=data/climb-10b-qwen/eval.bin \
     --ckpt-interval=10000  --steps=33900 \
     --eval-num-steps=40 \
     --learning-rate=0.0006 --final-lr-fraction=0.1 --warmup=150 \
     --seq-len=2048 --batch-size=9 \
     --model-dtype=bf16 --matmul-dtype=e4m3 --opt-m-dtype=bf16 --opt-v-dtype=bf16 \
     --gpus=4 \
     --recompute-ffn --recompute-norm \
     --shard-weights --persistent-quants --offload-quants --write-combined --memcpy-all-gather
```
In this case, we tokenize the data into multiple files, so we need to specify a glob expression for `--train-file`.
At the moment, the glob gets resolved _inside_ the program, so we need to escape it to prevent the shell from providing multiple values to the option.
We configure the total number of training steps, the learning rate schedule, dtypes, recomputations and weight sharding.
Note that despite the `--from-scratch` flag, we need to specify the HF model name; this allows the training script to look up the model
dimensions in the `config.json` file.`

Then we can watch the loss go down.
![Evaluation Loss](assets/1.5b-climb-eval.png)

After about 40 hours, the training finishes, and we can run evaluations as above:

| Tasks         | Version | Filter | n-shot | Metric   |   | |   Ours | Qwen2.5-1.5B | TinyLLama |
|---------------|--------:|--------|-------:|----------|---|-|-------:|--------------|-----------|
| arc_challenge |       1 | none   |      0 | acc      | ↑ | | 0.3490 | 0.4104       | 0.3097    |
|               |         | none   |      0 | acc_norm | ↑ | | 0.3729 | 0.4539       | 0.3268    |
| arc_easy      |       1 | none   |      0 | acc      | ↑ | | 0.6911 | 0.7529       | 0.6166    |
|               |         | none   |      0 | acc_norm | ↑ | | 0.6292 | 0.7184       | 0.5547    |
| boolq         |       2 | none   |      0 | acc      | ↑ | | 0.5936 | 0.7251       | 0.5599    |
| copa          |       1 | none   |      0 | acc      | ↑ | | 0.7100 | 0.8300       | 0.7500    |
| hellaswag     |       1 | none   |      0 | acc      | ↑ | | 0.4094 | 0.5020       | 0.4670    |
|               |         | none   |      0 | acc_norm | ↑ | | 0.5301 | 0.6781       | 0.6147    |
| openbookqa    |       1 | none   |      0 | acc      | ↑ | | 0.2680 | 0.3240       | 0.2520    |
|               |         | none   |      0 | acc_norm | ↑ | | 0.3560 | 0.4040       | 0.3680    |
| piqa          |       1 | none   |      0 | acc      | ↑ | | 0.7329 | 0.7584       | 0.7263    |
|               |         | none   |      0 | acc_norm | ↑ | | 0.7269 | 0.7617       | 0.7356    |
| winogrande    |       1 | none   |      0 | acc      | ↑ | | 0.5485 | 0.6377       | 0.5943    |

Unsurprisingly, the model performs worse than the official Qwen2.5-1.5B model,
which has been trained on 1000x as much data. Pitted against TinyLLama, the model does manage to
hold its own, in particular the higher-quality climb data seems to have helped with tasks like arc.
Not bad form something that can be done on a single workstation in a reasonable amount of time.
If you were to rent the corresponding compute on [vast.ai](https://vast.ai), at \$0.31 per GPU-hour (Sept 26, 2025),
this training run would have cost less than \$50.



## Detailed Usage
The training script accepts numerous command-line arguments to configure model training, optimization, memory management, and data handling. Below is a detailed description of each argument category:

### Model Configuration
- `--model <path>` - Path to the Hugging Face model directory or name of a HF model that is cached locally. The script will automatically locate model files if given a model name, but it will _not_ download new models.
- `--from-scratch` - Train the model from random initialization instead of loading pre-trained weights.
- `--model-dtype <dtype>` - Data type for model parameters. Supported types are `fp32`, `bf16`. This is the data type used for model master weights, and also for all activations and gradients. Matrix multiplications may be performed in lower precision according to the `--matmul-dtype` argument.

### Data Configuration
- `--train-file <path>` - Path to the binary file containing training tokens.
- `--eval-file <path>` - Path to the binary file containing validation tokens.
- `--batch-size, --batch <int>` - Micro-batch size (default: 4). This is the batch size per GPU.
- `--seq-len <int>` - Sequence length for training (default: 1024).

### Optimization Parameters and Training Schedule
- `--learning-rate, --lr <float>` - Base learning rate (default: 1e-5).
- `--warmup <int>` - Number of warmup steps. If not specified, defaults to a fraction of total steps.
- `--final-lr-fraction <float>` - Fraction of base learning rate to use for the final steps (default: 1.0).
- `--beta-1 <float>` - Beta1 parameter for Adam optimizer (default: 0.9).
- `--beta-2 <float>` - Beta2 parameter for Adam optimizer (default: 0.95).
- `--grad-accumulation <int>` - Number of micro-batches per optimizer step (default: 4). Effective batch size = batch-size × grad-accumulation × num-gpus.
- `--grad-clip <float>` - Gradient clipping threshold (default: 1.0).
- `--weight-decay <float>` - Weight decay for matrix parameters (default: 1.0).
- `--steps <int>` - Total number of training steps (default: 1000).
- `--eval-every-n-steps <int>` - Run evaluation every N optimizer steps (default: 100).
- `--eval-num-steps <int>` - Number of evaluation batches to process (default: 100). The evaluation at the end of training is always performed on the full dataset.

### Checkpointing and Output
- `--out-file <path>` - Path where to save the final trained model (default: "model.safetensors").
- `--checkpoint-dir <path>` - Directory for saving training checkpoints (default: "ckpt").
- `--ckpt-interval <int>` - Save checkpoint every N optimizer steps (default: 100).
- `--continue` - Continue training from the latest checkpoint in checkpoint-dir.
- `--log-file <path>` - Path for training log in JSON format (default: "log.json").
- `--log-gpu-util <int>` - Log GPU utilization every N steps (default: 25). Set to 0 to disable.

### Low-bit settings
- `--matmul-dtype <dtype>` - Data type for matrix multiplications. If not specified, uses model-dtype.
- `--opt-m-dtype <dtype>` - Data type for first-order momentum (Adam's m). Can be `fp32` or `bf16`.
- `--opt-v-dtype <dtype>` - Data type for second-order momentum (Adam's v). Can be `fp32` or `bf16`.

### Activation Checkpointing/Recomputation
The following switches enable fine-grained control over which parts of the activations
will be recomputed during the backward pass. Recomputing `swiglu` and `norm` is computationally cheap -- these operations are memory bound and run at the speed of memcpy. Recomputing matrix multiplications, as required by the other options, is more expensive.
Additional memory savings, especially for larger models, can be achieved by the offloading options described later.
- `--recompute-swiglu` - Recompute SwiGLU activation during backward pass to save activation memory. As swiglu is at the widest part of the model, this will result in substantial memory savings at only moderate compute increases (especially for large models).
- `--recompute-norm` - Recompute RMS normalizations during backward pass.
- `--recompute-ffn` - Recompute the entire feed-forward block during backward pass (implies --recompute-swiglu).
- `--recompute-qkv` - Recompute QKV projections during backward pass.
- `--recompute-att` - Recompute the entire attention block during backward pass (implies --recompute-qkv).
- `--recompute-block` - Recompute the entire transformer block during backward pass (subsumes the other recompute flags).
- `--offload-residuals` - Offload the residuals (of the ffn block; the only remaining part of the block that is not recomputed) to pinned host memory. Combined with `--recompute-block`, the total activation memory consumption becomes independent of the network depth.
- `--lmhead-chunks=N` - Split LM-head computation into `N` chunks, so that the required size of the logit tensor is reduced by a factor of `N`.

### Multi-GPU
- `--gpus <int>` - Number of GPUs to use. If 0, uses all available GPUs.
- `--zero-level <int>` - ZeRO redundancy optimization level:
    - 1: Sharded optimizer states (default)
    - 2: Sharded gradients + optimizer states
    - 3: Sharded weights + gradients + optimizer states

  You can also configure weights and gradients individually, using the `--shard-weights` and `--shard-gradients` flags. When training in fp8, for example, it makes sense to enable weight sharding before gradient sharding, as weights need only half the amount of bandwidth.

### Offloading
These options enable offloading parts of the model and optimizer state to pinned host memory.
For the optimizer state, this will slow down the optimizer step drastically (memory-bound operation), but if enough gradient accumulation steps are performed, the overall contribution of the optimizer step will be negligible. When training with lower precision, the `--persist-quants` option allows avoiding re-quantization of weights; this increases memory, however, when combined with `--offload-quants`, the additional memory is placed on the host. In a PCIe setting where any GPU-to-GPU communication has to pass through host memory anway, this can actually lead to significant speed-ups, especially if combined with the `--memcpy-all-gather` option.

- `--offload-master` - Store master weights in pinned host memory.
- `--offload-opt-m` - Store first-order momentum in pinned host memory.
- `--offload-opt-v` - Store second-order momentum in pinned host memory.
- `--persistent-quants` - Keep quantized weights in memory instead of re-quantizing (requires --shard-weights).
- `--offload-quants` - Store quantized weights in pinned host memory (requires --persistent-quants).
- `--use-write-combined` - Use write-combined memory for offloaded tensors. In some situations, this may improve PCie throughput.

### Algorithm selection
- `--memcpy-all-gather` - Use memcpy for all-gather operations (threads backend only). Memcpy generally gets better bandwidth utilization on PCIe, and does not consume SM resources.
- `--memcpy-send-recv` - Use memcpy for send/receive operations (threads backend only).
- `--all-to-all-reduce` - Use all-to-all-based reduce algorithm (combine with --memcpy-send-recv).
- `--use-cuda-graphs` / `--no-use-cuda-graphs` - Enable or disable CUDA graphs for performance.

## Python bindings
While it is nice to demonstrate training in pure C++/Cuda, there are scenarios where it is desirable to use Python for training, e.g., when using an alternative learning-rate schedule.

The Python bindings are provided in the `src/bindings` directory, and can be built using the
`_pyllmq` target. The library can be built manually (`-DPYTHON_BINDING=ON`), or directly into a wheel file
using `uv build --wheel`.
The [demo.py](scripts/demo.py) script provides an example of how to use the bindings. Running it with `uv run pyllmq-demo` will trigger the wheel build automatically.

Pre-built wheels are available from [GitHub Releases](https://github.com/IST-DASLab/llmq/releases) for convenience.
Download the latest `.whl` file and install it with `uv pip install 'pyllmq-0.2.0-cp312-abi3-linux_x86_64.whl[scripts]'`,
or run example scripts directly: `uv run --with 'pyllmq-0.2.0+cu128-cp312-abi3-linux_x86_64.whl[scripts]' pyllmq-demo`, replacing the file name as appropriate. The `[scripts]` extra installs additional packages that aren't strictly required for pyllmq, but are used in the utility scripts, such as `datasets` and `matplotlib`.
The wheels are built against CUDA 12.8 and 13.0 and support compute capabilities 89, 90, 100f, and 120f.

By design, the bindings expose only coarse-grained operations; that is, the minimum unit
of work is a full forward+backward pass across all GPUs. While this may be a bit inflexible,
it allows benefiting from the full optimization of the C++ backend, and there are no GPU-CPU
synchronizations until the `update` call.

You can also use the fully-fledged training script in [scripts/train.py](scripts/train.py), which
has mostly feature-parity with the C++ version. It is a bit less debuggable,
but does support directly logging to weights and biases, which can be very convenient.
The python version supports only multi-threaded mode, but not multi-process.

Note that the pre-built wheel files declare dependencies on specific cuda version packages, but the
`pyproject.toml` does no such declaration; during development, it is expected that you manage
the cuda version yourself/use the system installation of cude, not a pip-provided one.
The [wheel build-script](.github/workflows/wheel.yml) [modifies](.github/scripts/add_cuda_deps.py) the
toml file just before building the wheel.

## Code organization
The [src](src) directory contains four major subdirectories:
[kernels](src/kernels), [models](src/models), [training](src/training), and [utilities](src/utilities).
* [Kernels](src/kernels): The kernels directory contains the CUDA kernels, and each file is mostly self-contained, except for using a small subset of [utilities](utilities). Each kernel should be declared in [kernels.h](src/kernels/kernels.h) with all its dtype combinations. The actual cuda kernels should only tyke primitive parameters (i.e., pointers, integers, floats, etc.); but `kernels.h` should also provide a `Tensor` version of the kernel, which is implemented in [kernels.cpp](src/kernels/kernels.cpp) and extracts the data pointers from the `Tensor` object before dispatching to the right dtype combination.
* [Utilities](src/utilities): Contains code for memory management, safetensors, error handling, gpu monitoring, etc., as well as basic [type declarations](src/utilities/dtype.h) and the [Tensor](src/utilities/tensor.h) class.
* [Training](src/training): Contains training utilities, such as checkpointing, data loading, and logging. It also defines an abstract [Model](src/training/model.h) interface, which is the only way in which training utilities should interact with the model.
* [Models](src/models): Contains the model implementations. This should not include anything from `training`, except for the [Model](src/models/model.h) interface.

Additionally, the [scripts](scripts) directory contains utility python scripts, and
[binding](src/binding) the glue code for python bindings.

For more details on how the model's forward and backward passes are organized, see [doc.md](doc.md).

## Speed
TPS: tokens per second.
SOL: Speed of light.
TTB: Time to a billion tokens.

Note that the numbers below are snapshots on a single machine. There can be significant variability
depending on cooling, power supply, etc, especially in non-datacenter deployments.

### RTX PRO 6000 (courtesy of [Datacrunch](https://datacrunch.io/))
Using `--model-dtype=bf16 --opt-m-dtype=bf16 --opt-v-dtype=bf16 --use-cuda-graphs`, and a total batch size of 524288
for <= 3B (single-GPU 7B uses half the total batch size).

| Model        | nGPU | DType | Batch | TPS    | SOL | TTB   |
|--------------|------|-------|-------|--------|-----|-------|
| Qwen2.5-0.5B | 1    | fp8   | 32    | 90k    | 36% | 3:05h |
| Qwen2.5-0.5B | 1    | bf16  | 32    | 85k    | 52% | 3:16h |
| Qwen2.5-1.5B | 1    | fp8   | 32    | 36k    | 41% | 7:43h |
| Qwen2.5-1.5B | 1    | bf16  | 32    | 32k    | 61% | 8:41h |
| Qwen2.5-3B   | 1    | fp8   | 16    | 22k    | 46% | 13h   |
| Qwen2.5-3B   | 1    | bf16  | 16    | 17k    | 63% | 16h   |
| Qwen2.5-7B   | 1    | fp8   | 4     | 11k    | 53% | 25h   |
| Qwen2.5-7B   | 1    | bf16  | 4     | 8k     | 67% | 35h   |
| Qwen2.5-1.5B | 4    | fp8   | 32    | 145k   | 40% | 1:55h |
| Qwen2.5-1.5B | 4    | bf16  | 32    | 129k   | 61% | 2:09h |
| Qwen2.5-3B   | 4    | fp8   | 16    | 87k    | 46% | 3:11h |
| Qwen2.5-3B   | 4    | bf16  | 16    | 68k    | 64% | 4:05h |
| Qwen2.5-7B   | 4    | fp8   | 8     | 45k    | 53% | 6:10h |
| Qwen2.5-7B   | 4    | bf16  | 4     | 23k    | 62% | 12h   |
| Qwen2.5-14B  | 4    | fp8   | 4     | 23k    | 52% | 12h   |
| Qwen2.5-7B*  | 4    | fp8   | 16    | 46k    | 54% | 6:02h |

The run marked with * uses additional arguments `--shard-weights --memcpy-all-gather --persistent-quants --offload-quants` to offload quantized weights, which enables the increase in batch size.

### H100
| Model        | nGPU | DType | Batch | TPS  | SOL | TTB   |
|--------------|------|-------|-------|------|-----|-------|
| Qwen2.5-0.5B | 1    | fp8   | 32    | 155k | 32% | 1:47h |
| Qwen2.5-0.5B | 1    | bf16  | 32    | 144k | 45% | 1:55h |
| Qwen2.5-1.5B | 1    | fp8   | 16    | 68k  | 39% | 4:05h |
| Qwen2.5-1.5B | 1    | bf16  | 16    | 58k  | 55% | 4.47h |
| Qwen2.5-3B   | 1    | fp8   | 16    | 39k  | 42% | 7:07h |
| Qwen2.5-3B   | 1    | bf16  | 8     | 30k  | 57% | 9:15h |
| Qwen2.5-7B¹  | 1    | fp8   | 4     | 18k  | 42% | 15h   |
| Qwen2.5-7B   | 1    | bf16  | 4     | 14k  | 60% | 20h   |

^1: `--recompute-swiglu --recompute-norm` to fit batch size 4. Smaller batch sizes drastically reduce efficiency.

### RTX 4090

All settings use `--use-cuda-graphs --memcpy-all-gather --lmhead-chunks=${BATCH_SIZE}`

| Model         | nGPU | DType | Batch | TPS  | SOL | TTB   |
|---------------|------|-------|-------|------|-----|-------|
| Qwen2.5-0.5B  | 1    | fp8   | 16    | 48k  | 59% | 5.47h |
| Qwen2.5-0.5B  | 1    | bf16  | 16    | 41k  | 76% | 6:48h |
| Qwen2.5-1.5B  | 1    | fp8   | 4     | 20k  | 70% | 14h   |
| Qwen2.5-1.5B  | 1    | bf16  | 4     | 14k  | 83% | 19h   |
| Qwen2.5-3B¹   | 1    | fp8   | 4     | 10k  | 66% | 28h   |
| Qwen2.5-3B²   | 1    | bf16  | 4     | 7.0k | 80% | 40h   |
| Qwen2.5-7B³   | 1    | fp8   | 4     | 3.9k | 56% | 71h   |
| Qwen2.5-7B⁴   | 1    | bf16  | 4     | 2.4k | 64% | 116h  |
| Qwen2.5-0.5B  | 4    | fp8   | 16    | 175k | 53% | 1:35h |
| Qwen2.5-0.5B  | 4    | bf16  | 16    | 148k | 69% | 1:52h |
| Qwen2.5-1.5B  | 4    | fp8   | 8     | 71k  | 60% | 3:55h |
| Qwen2.5-1.5B⁵ | 4    | bf16  | 8     | 50k  | 73% | 5:30h |
| Qwen2.5-3B¹⁰  | 4    | fp8   | 4     | 36k  | 58% | 7:40h |
| Qwen2.5-3B¹⁰  | 4    | bf16  | 4     | 25k  | 71% | 11h   |
| Qwen2.5-7B⁶   | 4    | fp8   | 16    | 16k  | 57% | 17h   |
| Qwen2.5-7B⁷   | 4    | bf16  | 16    | 9.3k | 61% | 30h   |
| Qwen2.5-14B⁸  | 4    | fp8   | 16    | 7.8k | 54% | 36h   |
| Qwen2.5-14B⁹  | 4    | bf16  | 16    | 5.1k | 66% | 54h   |


^1: `--offload-opt-m --offload-opt-v --recompute-swiglu --offload-master`
^2: `--offload-opt-m --offload-opt-v --recompute-swiglu --recompute-norm`
^3: `--offload-opt-m --offload-opt-v --recompute-ffn --recompute-norm --recompute-att --offload-master --shard-weights --persistent-quants --offload-quants`
^4: `--offload-opt-m --offload-opt-v --recompute-ffn --recompute-norm --recompute-att --offload-master --shard-weights`
^5: `--recompute-norm --recompute-swiglu`
^6: `--offload-opt-m --offload-opt-v --recompute-ffn --recompute-norm --recompute-att --offload-master --shard-weights --shard-gradients --memcpy-send-recv --all-to-all-reduce --persistent-quants --offload-quants`
^7: `--offload-opt-m --offload-opt-v --recompute-ffn --recompute-norm --recompute-att --offload-master --shard-weights --shard-gradients --memcpy-send-recv --all-to-all-reduce`
^8: `--offload-opt-m --offload-opt-v --recompute-block --offload-master --offload-residual --shard-weights --memcpy-all-gather --shard-gradients --memcpy-send-recv --all-to-all-reduce --persistent-quants --offload-quants`
^9: `--offload-opt-m --offload-opt-v --recompute-block --offload-master --offload-residual --shard-weights --memcpy-all-gather --shard-gradients --memcpy-send-recv --all-to-all-reduce`
^10: `--offload-opt-m --recompute-swiglu --recompute-norm`

### L40S
On the L40S system, we found zero-copy access to offloaded optimizer states (`--use-zero-copy`) to be much more performant than cudaMemcpy-based double buffering.

| Model        | nGPU | DType | Batch | TPS  | SOL | TTB   |
|--------------|------|-------|-------|------|-----|-------|
| Qwen2.5-0.5B | 1    | fp8   | 32    | 47k  | 26% | 5:55h |
| Qwen2.5-0.5B | 1    | bf16  | 16    | 44k  | 37% | 6:20h |
| Qwen2.5-1.5B | 1    | fp8   | 8     | 20k  | 31% | 14h   |
| Qwen2.5-1.5B | 1    | bf16  | 8     | 17k  | 43% | 16h   |
| Qwen2.5-3B   | 1    | fp8   | 4     | 11k  | 33% | 25h   |
| Qwen2.5-3B   | 1    | bf16  | 4     | 8.9k | 56% | 31h   |
| Qwen2.5-7B¹  | 1    | fp8   | 4     | 5.2k | 33% | 54h   |
| Qwen2.5-7B²  | 1    | bf16  | 4     | 3.9k | 47% | 70h   |
| Qwen2.5-14B³ | 1    | fp8   | 16    | 1.7k | 22% | 159h  |
| Qwen2.5-14B⁴ | 1    | bf16  | 16    | 1.5k | 35% | 185h  |
| Qwen2.5-0.5B | 4    | fp8   | 32    | 185k | 25% | 1:30h |
| Qwen2.5-0.5B | 4    | bf16  | 16    | 169k | 35% | 1:38h |
| Qwen2.5-1.5B | 4    | fp8   | 16    | 79k  | 30% | 3:30h |
| Qwen2.5-1.5B | 4    | bf16  | 8     | 65k  | 42% | 4:16h |
| Qwen2.5-3B   | 4    | fp8   | 8     | 44k  | 31% | 6:20h |
| Qwen2.5-3B   | 4    | bf16  | 8     | 35k  | 45% | 7:56h |
| Qwen2.5-7B¹  | 4    | fp8   | 4     | 18k  | 28% | 16h   |
| Qwen2.5-7B²  | 4    | bf16  | 4     | 15k  | 43% | 18h   |
| Qwen2.5-14B⁵ | 4    | fp8   | 64    | 8.7k | 27% | 32h   |
| Qwen2.5-14B⁶ | 4    | bf16  | 64    | 6.3k | 36% | 44h   |

^1: `--offload-opt-m --offload-opt-v --offload-master`
^2: `--offload-opt-m --offload-opt-v --recompute-swiglu --recompute-norm`
^3: `--offload-opt-m --offload-opt-v --offload-master --recompute-block --use-zero-copy --lmhead-chunks=16 --shard-weights --offload-residual --persistent-quants --offload-quants`
^4: `--offload-opt-m --offload-opt-v --offload-master --recompute-block --use-zero-copy --lmhead-chunks=16 --shard-weights --offload-residual`
^5: `--offload-opt-m --offload-opt-v --offload-master --recompute-block --use-zero-copy --lmhead-chunks=16 --shard-weights --offload-residual --shard-gradients --persistent-quants --offload-quants`
^6: `--offload-opt-m --offload-opt-v --offload-master --recompute-block --use-zero-copy --lmhead-chunks=16 --shard-weights --offload-residual --shard-gradients`

### RTX 5090 Performance Benchmarks

| Model         | nGPU | DType | Batch | TPS   | SOL | TTB    |
|---------------|------|-------|-------|-------|-----|--------|
| Qwen2.5-0.5B  | 1    | fp8   | 8     | 70k   | 68% | 4:00h  |
| Qwen2.5-0.5B  | 1    | fp8   | 16    | 69k   | 67% | 4:02h  |
| Qwen2.5-0.5B¹ | 1    | fp8   | 32    | 54k   | 52% | 5:08h  |
| Qwen2.5-0.5B  | 1    | bf16  | 8     | 55k   | 81% | 5:03h  |
| Qwen2.5-0.5B  | 1    | bf16  | 16    | 55k   | 82% | 5:03h  |
| Qwen2.5-0.5B¹ | 1    | bf16  | 32    | 48k   | 71% | 5:47h  |
| Qwen2.5-1.5B  | 1    | fp8   | 8     | 27k   | 73% | 10:16h |
| Qwen2.5-1.5B¹ | 1    | bf16  | 8     | 17k   | 77% | 16:22h |
| Qwen2.5-3B¹   | 1    | fp8   | 4     | 13k   | 64% | 21:24h |
| Qwen2.5-3B¹   | 1    | fp8   | 8     | 13k   | 64% | 21:24h |
| Qwen2.5-3B¹   | 1    | bf16  | 4     | 8.2k  | 75% | 33:53h |
| Qwen2.5-3B¹   | 1    | bf16  | 8     | 8.6k  | 78% | 32:18h |
| Qwen2.5-7B³   | 1    | fp8   | 4     | 3.2k  | 36% | 87h    |
| Qwen2.5-7B³   | 1    | fp8   | 8     | 4.4k  | 50% | 63h    |
| Qwen2.5-7B³   | 1    | bf16  | 4     | 2.5k  | 52% | 111h   |
| Qwen2.5-7B³   | 1    | bf16  | 8     | 3.1k  | 66% | 90h    |
| Qwen2.5-1.5B  | 4    | fp8   | 8     | 107k  | 72% | 2:36h  |
| Qwen2.5-1.5B⁴ | 4    | bf16  | 4     | 71.2k | 81% | 3:41h  |
| Qwen2.5-3B¹   | 4    | fp8   | 4     | 48k   | 61% | 5:47h  |
| Qwen2.5-3B²   | 4    | fp8   | 8     | 49k   | 62% | 5:40h  |
| Qwen2.5-3B¹   | 4    | bf16  | 4     | 32k   | 72% | 8:41h  |
| Qwen2.5-7B³   | 4    | fp8   | 8     | 18.9k | 53% | 14:42h |
| Qwen2.5-7B³   | 4    | bf16  | 8     | 13.9k | 71% | 20:00h |
| Qwen2.5-14B³  | 4    | fp8   | 4     | 8.1k  | 23% | 34:20h |
| Qwen2.5-14B³  | 4    | bf16  | 4     | 3.9k  | 20% | 71:20h |

¹: `--recompute-ffn --recompute-att`
²: `--recompute-norm --recompute-swiglu --recompute-ffn --recompute-att`
³: `--recompute-norm --recompute-swiglu --recompute-ffn --recompute-att --offload-opt-m --offload-opt-v --shard-weights --offload-master --shard-gradients`
⁴: `--recompute-swiglu`

Command used: `./build/train --model=./Qwen2.5-0.5B --train-file=tiny-shakespeare-qwen-train.bin --eval-file=tiny-shakespeare-qwen-eval.bin --model-dtype=bf16 --opt-m-dtype=bf16 --opt-v-dtype=bf16 --matmul-dtype=e4m3 --grad-accumulation=8 --steps=1000 --learning-rate=1e-5 --gpus=1 --batch-size=8`

## Testing
Testing is handled through the python bindings.
At this point, we have three types of tests:
- recomputation
- fixed reference
- python reference


### Recomputation tests
Recomputation tests are responsible for verifying that the various `--recompute-*` flags produce the same results as the
reference implementation without recomputation. To that end, the same
training script is run twice with different recomputation settings,
and the resulting losses and gradient norms are compared.

There are two ways to run the recomputation tests. First, you can run them
remotely on [Modal](https://modal.com/). In order for this to be successful,
modal needs to be able to install the library as a wheel file.
```bash
uv build --wheel
uv run --with auditwheel --with patchelf auditwheel repair dist/*.whl -w wheelhouse/ --exclude libcuda.so.1 --exclude libcudart.so.12  --exclude libcudart.so.13 --exclude libcudnn.so.9 --exclude libcufile.so.0 --exclude libnccl.so.2 --exclude libcublasLt.so.12 --exclude libcublasLt.so.13 --exclude libnvidia-ml.so.1
```
(The second command is there to shrink the size of the wheel, and make it so it does not use the libraries from your local development system.)

Then you can run
```bash
modal run scripts/modal_test_app.py::test_recompute --recompute-swiglu
```

If you have a GPU locally, you can instead run the test script directly:
```bash
uv run python src/bindings/python/tests/recompute.py --recompute-swiglu
```
In this case, `uv run` will take care of building the wheel.

### Fixed reference tests
Fixed reference tests are designed to catch unintended changes in the model's
computations. As outputs inevitably vary depending on GPU and CUDA version,
these tests are only available to run on modal, where they will always pick
an L4 GPU.
After following the same setup as above, you can run them as
```bash
modal run scripts/modal_test_app.py::test_fixed --dtype bf16
modal run scripts/modal_test_app.py::test_fixed --dtype e4m3
```
Note that at this point, our FP32 attention backward pass is non-deterministic, so we cannot
expect it to succeed in bitwise reproduction tests.

### Python reference tests
These tests compare, with some tolerance based on cosine similarity and norms of the gradients, that our implementation produces results that are close to torch/transformers.

You can run these tests locally
```bash
uv run python src/binding/python/tests/torch_reference.py --seq-len 512 --grad-accum 4 --model-dtype bf16 --matmul-dtype e4m3
```
or using modal
```bash
modal run scripts/modal_test_app.py::test_torch_step --grad-accum 2 --model-dtype fp32 --matmul-dtype fp32
```

### CI tests:
To facilitate CI testing, there is an additional script [modal_test_ci](scripts/modal_test_ci.py). This script assumes that the modal app has already been deployed, and provides a convenient interface to run the tests in that case.

## Acknowledgements
This implementation wouldn't have been possible without Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c).
