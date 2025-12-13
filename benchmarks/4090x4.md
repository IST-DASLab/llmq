# Benchmarks on 4xRTX4090

All commands expect the following environment variable for common arguments:
````shell
export ARGS="--train-file=data/tiny-stories-qwen/train-*.bin --eval-file=data/tiny-stories-qwen/eval.bin \
 --ckpt-interval=10000 --from-scratch --seq-length=1024 --model-dtype=bf16 --opt-m-dtype=bf16 \
 --opt-v-dtype=bf16 --gpus=4 --use-cuda-graphs"
````

## Model size: 0.5B
### FP8
```shell
./train ${ARGS} --model=Qwen2.5-0.5B --matmul-dtype=e4m3 --batch-size=16 --grad-accumulation=8 \
 --lmhead-chunks=2 --attn-bwd-chunks=4 --memcpy-all-gather

# [T] step     0 [  0.5%] | time:  3092 ms | norm  13.797879 | loss  12.097197 | tps  169k | sol 51.9%
# [T] step     1 [  1.0%] | time:  2888 ms | norm  17.203356 | loss  11.545008 | tps  181k | sol 55.5%
# [T] step     2 [  1.6%] | time:  2892 ms | norm  13.326680 | loss  10.987377 | tps  181k | sol 55.5%
# [T] step     3 [  2.1%] | time:  2886 ms | norm   8.308699 | loss  10.633677 | tps  181k | sol 55.6%
# [T] step     4 [  2.6%] | time:  2885 ms | norm   5.889706 | loss  10.434742 | tps  181k | sol 55.6%
# [T] step     5 [  3.1%] | time:  2884 ms | norm   4.652005 | loss  10.297182 | tps  181k | sol 55.6%
```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-0.5B --matmul-dtype=bf16 --batch-size=16 --grad-accumulation=8 \
 --lmhead-chunks=2 --attn-bwd-chunks=4 --memcpy-all-gather

# [T] step     0 [  0.5%] | time:  3537 ms | norm  13.756177 | loss  12.098649 | tps  148k | sol 69.4%
# [T] step     1 [  1.0%] | time:  3391 ms | norm  17.134048 | loss  11.545815 | tps  154k | sol 72.4%
# [T] step     2 [  1.6%] | time:  3387 ms | norm  13.369566 | loss  10.989109 | tps  154k | sol 72.5%
# [T] step     3 [  2.1%] | time:  3386 ms | norm   8.338972 | loss  10.634527 | tps  154k | sol 72.5%
# [T] step     4 [  2.6%] | time:  3387 ms | norm   5.888421 | loss  10.435131 | tps  154k | sol 72.5%
# [T] step     5 [  3.1%] | time:  3384 ms | norm   4.660016 | loss  10.297523 | tps  154k | sol 72.5%
```

### LLama-Factory
Uses batch size 128.
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/4x4090/qwen-0.5b.yaml
# 1/11 [00:05<00:58,  5.89s/it]
# 2/11 [00:10<00:45,  5.08s/it]
```
This implies `(512*1024)/(2*5.08-5.89) = 122784` tps.

## Model size: 1.5B

### FP8
```shell
./train ${ARGS} --model=Qwen2.5-1.5B --matmul-dtype=e4m3 --batch-size=8 --grad-accumulation=16 \
  --lmhead-chunks=2 --attn-bwd-chunks=4 --memcpy-all-gather

# [T] step     0 [  0.5%] | time:  7467 ms | norm  15.715532 | loss  12.245522 | tps 70214 | sol 59.4%
# [T] step     1 [  1.0%] | time:  7282 ms | norm  20.995054 | loss  11.077904 | tps 71997 | sol 61.0%
# [T] step     2 [  1.6%] | time:  7279 ms | norm  13.600585 | loss   9.926165 | tps 72027 | sol 61.0%
# [T] step     3 [  2.1%] | time:  7287 ms | norm  14.205133 | loss   9.357170 | tps 71948 | sol 60.9%
# [T] step     4 [  2.6%] | time:  7285 ms | norm  14.431196 | loss   9.023902 | tps 71968 | sol 60.9%
# [T] step     5 [  3.1%] | time:  7287 ms | norm  11.897770 | loss   8.827998 | tps 71948 | sol 60.9%
```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-1.5B --matmul-dtype=bf16 --batch-size=8 --grad-accumulation=16 \
  --lmhead-chunks=4 --attn-bwd-chunks=4 --memcpy-all-gather --offload-residual --recompute-norm

# [T] step     0 [  0.5%] | time: 10129 ms | norm  15.911036 | loss  12.246178 | tps 51761 | sol 74.6%
# [T] step     1 [  1.0%] | time:  9991 ms | norm  20.902443 | loss  11.071198 | tps 52476 | sol 75.6%
# [T] step     2 [  1.6%] | time: 10001 ms | norm  13.584165 | loss   9.920774 | tps 52423 | sol 75.6%
# [T] step     3 [  2.1%] | time:  9997 ms | norm  14.167023 | loss   9.349775 | tps 52444 | sol 75.6%
# [T] step     4 [  2.6%] | time: 10006 ms | norm  14.351781 | loss   9.007559 | tps 52397 | sol 75.5%
# [T] step     5 [  3.1%] | time: 10002 ms | norm  11.713161 | loss   8.807705 | tps 52418 | sol 75.6%
```

### LLama-Factory
Uses batch size 32.
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/4x4090/qwen-1.5b.yaml
# 1/11 [00:15<02:30, 15.02s/it]
# 2/11 [00:28<02:05, 13.91s/it]
```
This implies `(512*1024)/(2*13.91-15.02) = 40960` tps.

## Model size: 3B
### FP8
```shell
./train ${ARGS} --model=Qwen2.5-3B --matmul-dtype=e4m3 --batch-size=4 --grad-accumulation=32 \
  --lmhead-chunks=4 --attn-bwd-chunks=4 --memcpy-all-gather --offload-opt-m --recompute-norm

# [T] step     0 [  0.5%] | time: 13875 ms | norm  17.317457 | loss  12.328534 | tps 37786 | sol 60.9%
# [T] step     1 [  1.0%] | time: 13657 ms | norm  29.196331 | loss  10.716395 | tps 38389 | sol 61.8%
# [T] step     2 [  1.6%] | time: 13714 ms | norm 183.937256 | loss  11.608831 | tps 38230 | sol 61.6%
# [T] step     3 [  2.1%] | time: 13670 ms | norm  44.871384 | loss   9.644464 | tps 38353 | sol 61.8%
# [T] step     4 [  2.6%] | time: 13706 ms | norm  38.677887 | loss   9.612577 | tps 38252 | sol 61.6%
# [T] step     5 [  3.1%] | time: 13700 ms | norm  21.954212 | loss   9.558939 | tps 38269 | sol 61.6%
```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-3B --matmul-dtype=bf16 --batch-size=4 --grad-accumulation=32 \
  --lmhead-chunks=4 --attn-bwd-chunks=4 --memcpy-all-gather --offload-opt-m --recompute-norm --recompute-swiglu

# [T] step     0 [  0.5%] | time: 20330 ms | norm  17.463358 | loss  12.329319 | tps 25788 | sol 74.0%
# [T] step     1 [  1.0%] | time: 20183 ms | norm  29.274010 | loss  10.707371 | tps 25976 | sol 74.6%
# [T] step     2 [  1.6%] | time: 20190 ms | norm 183.013153 | loss  11.573620 | tps 25967 | sol 74.5%
# [T] step     3 [  2.1%] | time: 20165 ms | norm  45.117264 | loss   9.630610 | tps 25999 | sol 74.6%
# [T] step     4 [  2.6%] | time: 20162 ms | norm  39.397141 | loss   9.605055 | tps 26003 | sol 74.6%
# [T] step     5 [  3.1%] | time: 20151 ms | norm  22.821035 | loss   9.550476 | tps 26017 | sol 74.7%
```
### LLama-Factory
Uses ZeRO-3 offload and batch size 64. (Faster than ZeRO-2 with batch size 32)
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/4x4090/qwen-3b.yaml
# 1/11 [00:36<06:09, 36.97s/it]
# 2/11 [01:06<04:54, 32.67s/it]
```
This implies `(512*1024)/(2*32.67-36.97) = 18480` tps.

## Model size: 7B

### FP8:
```shell
./train ${ARGS} --model=Qwen2.5-7B --matmul-dtype=e4m3 --batch-size=16 --grad-accumulation=8 \
  --lmhead-chunks=8 --attn-bwd-chunks=4 --memcpy-all-gather --offload-opt-m --offload-opt-v \
  --recompute-ffn --recompute-norm --offload-master --shard-weights --shard-gradients \
  --memcpy-send-recv --all-to-all-reduce --persistent-quants --offload-quants

# [T] step     0 [  0.5%] | time: 32454 ms | norm  12.868343 | loss  12.642083 | tps 16154 | sol 57.4%
# [T] step     1 [  1.0%] | time: 31763 ms | norm  23.637371 | loss  10.683868 | tps 16506 | sol 58.6%
# [T] step     2 [  1.6%] | time: 31798 ms | norm 186.705399 | loss  14.978359 | tps 16488 | sol 58.5%
# [T] step     3 [  2.1%] | time: 31780 ms | norm 178.402359 | loss  16.339745 | tps 16497 | sol 58.6%
# [T] step     4 [  2.6%] | time: 31796 ms | norm 133.099213 | loss  16.463650 | tps 16489 | sol 58.6%
# [T] step     5 [  3.1%] | time: 31799 ms | norm  99.226608 | loss  13.198079 | tps 16487 | sol 58.5%
```

### BF16:
```shell
./train ${ARGS} --model=Qwen2.5-7B --matmul-dtype=bf16 --batch-size=16 --grad-accumulation=8 \
  --lmhead-chunks=8 --attn-bwd-chunks=4 --memcpy-all-gather --offload-opt-m --offload-opt-v \
  --recompute-ffn --recompute-norm --offload-master --shard-weights --shard-gradients \
  --memcpy-send-recv --all-to-all-reduce --offload-residual

# [T] step     0 [  0.5%] | time: 48276 ms | norm  12.933082 | loss  12.642547 | tps 10860 | sol 70.7%
# [T] step     1 [  1.0%] | time: 48119 ms | norm  23.672430 | loss  10.680092 | tps 10895 | sol 71.0%
# [T] step     2 [  1.6%] | time: 48264 ms | norm 189.932938 | loss  15.121484 | tps 10862 | sol 70.7%
# [T] step     3 [  2.1%] | time: 47977 ms | norm 182.855759 | loss  16.397915 | tps 10927 | sol 71.2%
# [T] step     4 [  2.6%] | time: 48093 ms | norm 132.847656 | loss  16.454048 | tps 10901 | sol 71.0%
# [T] step     5 [  3.1%] | time: 48113 ms | norm  98.846687 | loss  13.139793 | tps 10897 | sol 71.0%
```

Uses ZeRO-3 offload and batch size 32.
### LLama-Factory
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/4x4090/qwen-7b.yaml
# 1/11 [01:34<15:47, 94.77s/it]
# 2/11 [02:53<12:46, 85.22s/it]
```
This implies `(512*1024)/(2*85.22-94.77) = 6929` tps.


## Model size: 14B

### FP8
Using batch size 32 fits into memory, but turns out to be slower than batch size 16.
```shell
./train ${ARGS} --model=Qwen2.5-14B --matmul-dtype=e4m3 --batch-size=16 --grad-accumulation=8 \
  --lmhead-chunks=8 --attn-bwd-chunks=4 --memcpy-all-gather --offload-opt-m --offload-opt-v \
  --recompute-block --offload-master --shard-weights --shard-gradients   --memcpy-send-recv \
  --all-to-all-reduce --persistent-quants --offload-quants --offload-residual

# [T] step     0 [  0.5%] | time: 67914 ms | norm  17.418232 | loss  12.959469 | tps  7719 | sol 53.5%
# [T] step     1 [  1.0%] | time: 66801 ms | norm  42.953403 | loss  10.883582 | tps  7848 | sol 54.4%
# [T] step     2 [  1.6%] | time: 66860 ms | norm  41.940796 | loss  12.041933 | tps  7841 | sol 54.4%
# [T] step     3 [  2.1%] | time: 66891 ms | norm 197.702728 | loss  21.225594 | tps  7837 | sol 54.4%
# [T] step     4 [  2.6%] | time: 66893 ms | norm 162.353333 | loss  22.532246 | tps  7837 | sol 54.4%
# [T] step     5 [  3.1%] | time: 66831 ms | norm 174.296371 | loss  22.894928 | tps  7844 | sol 54.4%
```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-14B --matmul-dtype=bf16 --batch-size=32 --grad-accumulation=4 \
  --lmhead-chunks=8 --attn-bwd-chunks=4 --memcpy-all-gather --offload-opt-m --offload-opt-v \
  --recompute-block --offload-master --shard-weights --shard-gradients   --memcpy-send-recv \
  --all-to-all-reduce --offload-residual

# [T] step     0 [  0.5%] | time: 99861 ms | norm  17.456240 | loss  12.959760 | tps  5250 | sol 67.9%
# [T] step     1 [  1.0%] | time: 99651 ms | norm  44.145153 | loss  10.877642 | tps  5261 | sol 68.0%
# [T] step     2 [  1.6%] | time: 99719 ms | norm  42.026581 | loss  12.082806 | tps  5257 | sol 68.0%
# [T] step     3 [  2.1%] | time: 99646 ms | norm 197.631607 | loss  20.850994 | tps  5261 | sol 68.0%
# [T] step     4 [  2.6%] | time: 99583 ms | norm 160.973190 | loss  22.075367 | tps  5264 | sol 68.1%
# [T] step     5 [  3.1%] | time: 99520 ms | norm 169.567917 | loss  22.356665 | tps  5268 | sol 68.1%
```

### LLama-Factory
Uses ZeRO-3 offload and batch size 21.
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/4x4090/qwen-14b.yaml

# [INFO] After initializing ZeRO optimizer
# [INFO]  MA 0.05 GB         Max_MA 2.95 GB         CA 2.95 GB         Max_CA 3 GB
# [INFO]  CPU Virtual Memory:  used = 256.66 GB, percent = 51.0%
# [INFO] DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer_Stage3
# 1/11 [03:41<36:55, 221.55s/it]
# 2/11 [07:01<31:19, 208.87s/it]
```
This implies `(6*4*21*1024)/(2*208.87 - 221.55) = 2630` tps.


## Model size: 32B
We do not use `--memcpy-send-recv --all-to-all-reduce` with `--offload-gradients`, as (with the current implementation)
this leads to one more round-trip of the gradient tensors.
### FP8
```shell
./train ${ARGS} --model=Qwen2.5-32B --matmul-dtype=e4m3 --batch-size=32 --grad-accumulation=4 \
  --lmhead-chunks=8 --attn-bwd-chunks=4 --memcpy-all-gather --offload-opt-m --offload-opt-v \
  --recompute-block --offload-master --shard-weights --shard-gradients --offload-residual \
  --offload-gradients --persistent-quants --offload-quants

# [T] step     0 [  0.5%] | time:   156  s | norm  18.507004 | loss  12.956062 | tps  3340 | sol 50.7%
# [T] step     1 [  1.0%] | time:   154  s | norm  23.586407 | loss  10.638527 | tps  3398 | sol 51.6%
# [T] step     2 [  1.6%] | time:   154  s | norm 205.904739 | loss  24.402885 | tps  3396 | sol 51.6%
# [T] step     3 [  2.1%] | time:   154  s | norm 269.729919 | loss  26.261187 | tps  3396 | sol 51.6%
# [T] step     4 [  2.6%] | time:   154  s | norm 202.609467 | loss  25.441338 | tps  3395 | sol 51.6%
# [T] step     5 [  3.1%] | time:   154  s | norm 160.604309 | loss  22.940529 | tps  3395 | sol 51.6%

```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-32B --matmul-dtype=bf16 --batch-size=32 --grad-accumulation=4 \
  --lmhead-chunks=8 --attn-bwd-chunks=4 --memcpy-all-gather --offload-opt-m --offload-opt-v \
  --recompute-block --offload-master --shard-weights --shard-gradients --offload-residual --offload-gradients

# [T] step     0 [  0.5%] | time:   238  s | norm  18.545193 | loss  12.957092 | tps  2199 | sol 64.6%
# [T] step     1 [  1.0%] | time:   237  s | norm  23.550249 | loss  10.628635 | tps  2205 | sol 64.7%
# [T] step     2 [  1.6%] | time:   237  s | norm 208.712555 | loss  23.791492 | tps  2204 | sol 64.7%
# [T] step     3 [  2.1%] | time:   237  s | norm 260.918335 | loss  25.454247 | tps  2206 | sol 64.7%
# [T] step     4 [  2.6%] | time:   237  s | norm 175.231873 | loss  24.600286 | tps  2206 | sol 64.8%
# [T] step     5 [  3.1%] | time:   237  s | norm 237.979965 | loss  22.747318 | tps  2208 | sol 64.8%
```

### LLama-Factory
OOM even with Zero-3 offload; We're running out of host-side memory (500GB).
```shell
# [2025-12-09 14:53:56,994] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 465.26 GB, percent = 92.4%
# [2025-12-09 14:53:57,062] [INFO] [stage3.py:534:_setup_for_real_optimizer] optimizer state initialized
```
