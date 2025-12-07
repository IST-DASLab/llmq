# Benchmarks on 1xRTX4090

All commands expect the following environment variable for common arguments:
````shell
export ARGS="--train-file=data/tiny-stories-qwen/train-*.bin --eval-file=data/tiny-stories-qwen/eval.bin \
 --ckpt-interval=10000 --from-scratch --seq-length=1024 --model-dtype=bf16 --opt-m-dtype=bf16 \
 --opt-v-dtype=bf16 --gpus=1 --use-cuda-graphs"
````

## Model size: 0.5B
### FP8
```shell
./train ${ARGS} --model=Qwen2.5-0.5B --matmul-dtype=e4m3 --batch-size=16 --grad-accumulation=32 \
  --lmhead-chunks=16

# [T] step     0 [  0.5%] | time: 11063 ms | norm  14.363219 | loss  12.122908 | tps 47391 | sol 58.0%
# [T] step     1 [  1.0%] | time: 11006 ms | norm  15.861144 | loss  11.502910 | tps 47636 | sol 58.3%
# [T] step     2 [  1.6%] | time: 11004 ms | norm  12.688044 | loss  10.917389 | tps 47645 | sol 58.3%
# [T] step     3 [  2.1%] | time: 11003 ms | norm   9.507539 | loss  10.507158 | tps 47649 | sol 58.3%
# [T] step     4 [  2.6%] | time: 11004 ms | norm   7.739807 | loss  10.238192 | tps 47645 | sol 58.3%
# [T] step     5 [  3.1%] | time: 11012 ms | norm   6.635383 | loss  10.009802 | tps 47610 | sol 58.3%
```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-0.5B --matmul-dtype=bf16 --batch-size=16 --grad-accumulation=32 \
  --lmhead-chunks=16

# [T] step     0 [  0.5%] | time: 13188 ms | norm  14.518435 | loss  12.124965 | tps 39754 | sol 74.5%
# [T] step     1 [  1.0%] | time: 13162 ms | norm  16.111782 | loss  11.504132 | tps 39833 | sol 74.6%
# [T] step     2 [  1.6%] | time: 13158 ms | norm  12.694994 | loss  10.918768 | tps 39845 | sol 74.7%
# [T] step     3 [  2.1%] | time: 13153 ms | norm   9.664483 | loss  10.509415 | tps 39860 | sol 74.7%
# [T] step     4 [  2.6%] | time: 13148 ms | norm   7.806265 | loss  10.237169 | tps 39875 | sol 74.7%
# [T] step     5 [  3.1%] | time: 13159 ms | norm   6.712699 | loss  10.007376 | tps 39842 | sol 74.7%
```

### LLama-Factory
Uses batch size 128.
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/1x4090/qwen-0.5b.yaml
# | 1/11 [00:19<03:10, 19.05s/it]
# | 2/11 [00:36<02:43, 18.21s/it]
```
This implies `(512*1024)/(2*18.21-19.05) = 30183` tps.

## Model size: 1.5B

### FP8
```shell
./train ${ARGS} --model=Qwen2.5-1.5B --matmul-dtype=e4m3 --batch-size=4 --grad-accumulation=128 \
  --lmhead-chunks=4

# [T] step     0 [  0.5%] | time: 25806 ms | norm  16.094933 | loss  12.240793 | tps 20316 | sol 68.8%
# [T] step     1 [  1.0%] | time: 25781 ms | norm  20.687757 | loss  11.067363 | tps 20336 | sol 68.9%
# [T] step     2 [  1.6%] | time: 25784 ms | norm  12.294410 | loss   9.908209 | tps 20333 | sol 68.9%
# [T] step     3 [  2.1%] | time: 25779 ms | norm  12.014793 | loss   9.316853 | tps 20337 | sol 68.9%
# [T] step     4 [  2.6%] | time: 25788 ms | norm  13.171033 | loss   9.040373 | tps 20330 | sol 68.9%
# [T] step     5 [  3.1%] | time: 25792 ms | norm  11.691628 | loss   8.760139 | tps 20327 | sol 68.9%
```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-1.5B --matmul-dtype=bf16 --batch-size=4 --grad-accumulation=128 \
  --lmhead-chunks=4

# [T] step     0 [  0.5%] | time: 37266 ms | norm  16.175560 | loss  12.242585 | tps 14068 | sol 81.1%
# [T] step     1 [  1.0%] | time: 37299 ms | norm  20.669195 | loss  11.053426 | tps 14056 | sol 81.0%
# [T] step     2 [  1.6%] | time: 37249 ms | norm  12.252068 | loss   9.897452 | tps 14075 | sol 81.2%
# [T] step     3 [  2.1%] | time: 37246 ms | norm  11.863964 | loss   9.304607 | tps 14076 | sol 81.2%
# [T] step     4 [  2.6%] | time: 37257 ms | norm  12.692739 | loss   9.010376 | tps 14072 | sol 81.1%
# [T] step     5 [  3.1%] | time: 37281 ms | norm  13.741744 | loss   8.745119 | tps 14063 | sol 81.1%
```

### LLama-Factory
Uses batch size 32.
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/1x4090/qwen-1.5b.yaml
# 1/11 [00:57<09:35, 57.54s/it]
# 2/11 [01:53<08:29, 56.60s/it]
```
This implies `(512*1024)/(2*56.60-57.54) = 9419` tps.

## Model size: 3B
### FP8
Recomputing swiglu and norms to enable bs=6 turns out to be slower (and in fact, bs=6 is generally slower than bf=4).
```shell
./train ${ARGS} --model=Qwen2.5-3B --matmul-dtype=e4m3 --batch-size=4 --grad-accumulation=128 \
  --lmhead-chunks=4 --offload-master --offload-opt-v --offload-opt-m

# [T] step     0 [  0.5%] | time: 49607 ms | norm  17.148050 | loss  12.362766 | tps 10568 | sol 68.1%
# [T] step     1 [  1.0%] | time: 49137 ms | norm  30.098303 | loss  10.854187 | tps 10669 | sol 68.8%
# [T] step     2 [  1.6%] | time: 49131 ms | norm 144.022827 | loss  10.563226 | tps 10671 | sol 68.8%
# [T] step     3 [  2.1%] | time: 49154 ms | norm  21.841425 | loss   9.985455 | tps 10666 | sol 68.7%
# [T] step     4 [  2.6%] | time: 49146 ms | norm  19.755840 | loss   9.189474 | tps 10667 | sol 68.7%
# [T] step     5 [  3.1%] | time: 49223 ms | norm  55.030193 | loss   9.246952 | tps 10651 | sol 68.6%
```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-3B --matmul-dtype=bf16 --batch-size=4 --grad-accumulation=128 --lmhead-chunks=4 \
  --offload-opt-v --offload-opt-m --recompute-swiglu

# [T] step     0 [  0.5%] | time: 74800 ms | norm  17.324688 | loss  12.364298 | tps  7009 | sol 80.5%
# [T] step     1 [  1.0%] | time: 74811 ms | norm  30.018398 | loss  10.843289 | tps  7008 | sol 80.5%
# [T] step     2 [  1.6%] | time: 74824 ms | norm 141.218719 | loss  10.519794 | tps  7006 | sol 80.4%
# [T] step     3 [  2.1%] | time: 74848 ms | norm  21.392759 | loss   9.972561 | tps  7004 | sol 80.4%
# [T] step     4 [  2.6%] | time: 74857 ms | norm  19.199148 | loss   9.148648 | tps  7003 | sol 80.4%
# [T] step     5 [  3.1%] | time: 74878 ms | norm  54.493061 | loss   9.253811 | tps  7001 | sol 80.4%
```
### LLama-Factory
Uses ZeRO-2 offload and batch size 48. (Slightly faster than ZeRO-3 with batch size 64)
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/1x4090/qwen-3b.yaml
# 1/10 [01:51<16:43, 111.52s/it]
| 2/10 [03:32<14:03, 105.38s/it]
```
This implies `(48*11*1024)/(2*105.38-111.52) = 5448` tps.

## Model size: 7B

### FP8:
```shell
./train ${ARGS} --model=Qwen2.5-7B --matmul-dtype=e4m3 --batch-size=16 --grad-accumulation=32 --lmhead-chunks=16 \
  --offload-opt-m --offload-opt-v --recompute-block --offload-master --shard-weights --persistent-quants \
  --offload-quants --offload-residual  --attn-bwd-chunks=16

# [T] step     0 [  0.5%] | time:   123  s | norm  12.337703 | loss  12.652873 | tps  4259 | sol 60.5%
# [T] step     1 [  1.0%] | time:   120  s | norm  23.787540 | loss  10.815722 | tps  4336 | sol 61.6%
# [T] step     2 [  1.6%] | time:   121  s | norm 197.903992 | loss  15.796431 | tps  4332 | sol 61.5%
# [T] step     3 [  2.1%] | time:   120  s | norm 148.991211 | loss  15.137796 | tps  4334 | sol 61.6%
# [T] step     4 [  2.6%] | time:   120  s | norm 130.631546 | loss  15.109634 | tps  4333 | sol 61.5%
# [T] step     5 [  3.1%] | time:   121  s | norm 105.668236 | loss  13.091249 | tps  4328 | sol 61.5%
```

### BF16:
```shell
./train ${ARGS} --model=Qwen2.5-7B --matmul-dtype=e4m3 --batch-size=16 --grad-accumulation=32 --lmhead-chunks=16 \
  --offload-opt-m --offload-opt-v --recompute-block --offload-master \
  --shard-weights --offload-residual  --attn-bwd-chunks=16

# [T] step     0 [  0.5%] | time:   189  s | norm  12.375451 | loss  12.651476 | tps  2765 | sol 72.0%
# [T] step     1 [  1.0%] | time:   189  s | norm  23.730993 | loss  10.803814 | tps  2773 | sol 72.3%
# [T] step     2 [  1.6%] | time:   188  s | norm 197.866272 | loss  15.808683 | tps  2776 | sol 72.3%
# [T] step     3 [  2.1%] | time:   188  s | norm 149.391525 | loss  15.174353 | tps  2779 | sol 72.4%
# [T] step     4 [  2.6%] | time:   188  s | norm 131.170563 | loss  15.151785 | tps  2780 | sol 72.4%
# [T] step     5 [  3.1%] | time:   188  s | norm 105.834908 | loss  13.080410 | tps  2777 | sol 72.4%
```

Uses ZeRO-3 offload and batch size 32.
### LLama-Factory
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/1x4090/qwen-7b.yaml
# 1/11 [03:37<36:19, 217.99s/it]
# 2/11 [07:07<31:56, 212.89s/it]
```
This implies `(512*1024)/(2* 212.89-217.99) = 2523` tps.


## Model size: 14B

### FP8
Using batch size 32 fits into memory, but turns out to be slower than batch size 16.
```shell
./train ${ARGS} --model=Qwen2.5-14B --matmul-dtype=e4m3 --batch-size=32 --grad-accumulation=16 \
  --lmhead-chunks=32 --offload-opt-m --offload-opt-v --recompute-block --offload-master \
  --shard-weights --persistent-quants --offload-quants --offload-residual --shard-gradients \
  --offload-gradients --attn-bwd-chunks=32

# [T] step     0 [  0.5%] | time:   265  s | norm  17.939161 | loss  12.974737 | tps  1972 | sol 54.7%
# [T] step     1 [  1.0%] | time:   261  s | norm  87.351166 | loss  11.101963 | tps  2001 | sol 55.5%
# [T] step     2 [  1.6%] | time:   261  s | norm  47.228184 | loss  12.675285 | tps  2001 | sol 55.5%
# [T] step     3 [  2.1%] | time:   261  s | norm 211.750778 | loss  16.330353 | tps  2001 | sol 55.5%
# [T] step     4 [  2.6%] | time:   261  s | norm 158.123169 | loss  16.382170 | tps  2001 | sol 55.5%
# [T] step     5 [  3.1%] | time:   262  s | norm 208.784302 | loss  13.173750 | tps  2000 | sol 55.5%
```

### BF16
```shell
./train ${ARGS} --model=Qwen2.5-14B --matmul-dtype=bf16 --batch-size=32 --grad-accumulation=16 \
  --lmhead-chunks=32 --offload-opt-m --offload-opt-v --recompute-block --offload-master \
  --shard-weights --offload-residual --shard-gradients --offload-gradients --attn-bwd-chunks=32

# [T] step     0 [  0.5%] | time:   389  s | norm  17.926729 | loss  12.973790 | tps  1345 | sol 69.6%
# [T] step     1 [  1.0%] | time:   388  s | norm  86.474739 | loss  11.093394 | tps  1350 | sol 69.9%
# [T] step     2 [  1.6%] | time:   388  s | norm  46.611458 | loss  12.639969 | tps  1349 | sol 69.8%
# [T] step     3 [  2.1%] | time:   388  s | norm 211.994537 | loss  16.385487 | tps  1349 | sol 69.8%
# [T] step     4 [  2.6%] | time:   388  s | norm 161.280350 | loss  16.579582 | tps  1348 | sol 69.8%
# [T] step     5 [  3.1%] | time:   388  s | norm 210.529739 | loss  13.557438 | tps  1348 | sol 69.8%
```

### LLama-Factory
Uses ZeRO-3 offload and batch size 20.
```shell
FORCE_TORCHRUN=1 llamafactory-cli train benchmarks/1x4090/qwen-14b.yaml

# 1/11 [08:02<1:20:25, 482.54s/it]
# 2/11 [15:53<1:11:21, 475.68s/it]
# 
```
This implies `(20*26*1024)/(2*475.68 - 482.54) = 1135` tps.
