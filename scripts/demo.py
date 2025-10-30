#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyllmq", "wandb", "tqdm"]
# ///

# Note: make sure pyllmq.cpython... is on the path, or run with uv
import pyllmq
import numpy as np
import os
import time
from tqdm import tqdm
from pathlib import Path


def main():
    # configure the model architecture
    config = pyllmq.LLamaConfig(
        architecture="qwen2",
        hidden_size=896,
        intermediate_size=4864,
        max_position_embeddings=32768,
        num_attention_heads=14,
        num_hidden_layers=24,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        tie_word_embeddings=True,
        dtype="bfloat16",
        vocab_size=151936
    )

    # configure the training options
    options = pyllmq.LLamaOptions()

    # more options
    ngpu = 1
    batch_size = 2
    seq_len = 1024
    grad_accumulation = 4
    steps = 100
    eval_steps = 10
    grad_clip = 1.0
    weight_decay = 1e-2
    beta_1 = 0.9
    beta_2 = 0.95
    lr = 1e-5

    # ensure training data exists
    if not Path("data/tiny-stories-qwen").exists():
        print("generating training data...")
        from .tokenize_data import generate_tokenized_dataset
        generate_tokenized_dataset( "tiny-stories", "qwen")

    tiny_stories = list(map(str, Path("data/tiny-stories-qwen").glob("train-*.bin")))

    # set up the data loaders. it is _not_ required to use pyllmq.DataLoader, you can
    # fill in_tokens and out_tokens yourself;
    in_tokens = np.empty((ngpu * batch_size, seq_len), dtype=np.int32)
    out_tokens = np.empty((ngpu * batch_size, seq_len), dtype=np.int32)
    train_loader = pyllmq.DataLoader(tiny_stories, ngpu * batch_size * seq_len, 42)
    eval_loader = pyllmq.DataLoader(["data/tiny-stories-qwen/eval.bin"], ngpu * batch_size * seq_len, 42)

    # create the trainer object and initialize the weights
    print("creating trainer...")
    trainer = pyllmq.LLMQTrainer(ngpu=ngpu, config=config, options=options, batch_size=batch_size, seq_len=seq_len, grad_accum=grad_accumulation)
    print("initializing weights...")
    trainer.init_weights()
    # alternative: pyllmq.LLMQTrainer.from_pretrained("Qwen/Qwen2.5-0.5B", ngpu=ngpu, dtype="bf16", options=options, batch_size=2, seq_len=1024, grad_accum=grad_accumulation)

    print("\nmemory consumption:")
    for k, v in trainer.get_allocator_info(0).items():
        print(f" {k:20}: {v // 1024 // 1024:6} MiB")

    train_loader.load_batch(in_tokens, out_tokens)

    print("\nstarting training...\n")
    start = time.perf_counter()
    for step in range(steps):
        for s in range(grad_accumulation):
            trainer.step(in_tokens, out_tokens)
            # overlap next batch loading with step
            train_loader.load_batch(in_tokens, out_tokens)

        # hide latency by doing this before update()
        train_loader.load_batch(in_tokens, out_tokens)
        if step % 10 == 0:
            infos = trainer.get_gpu_info()
            for info in infos:
                print(f"             power: {info.power // 1000:4}W  temp: {info.temperature:3}°C   rx: {info.pcie_rx // 1024 // 1024:4}MiB/s  tx: {info.pcie_tx // 1024 // 1024:4}MiB/s")
                print(f"             clock: {info.clock / 1000:3.1f}GHz fan:  {info.fan:3}%    throttle: {info.throttle_reason}")

        result = trainer.update(lr, beta_1, beta_2, step + 1, weight_decay, grad_clip)
        duration = time.perf_counter() - start
        start = time.perf_counter()
        print(f"step: {step:5}  loss: {result['loss']:6.3f}  norm: {result['norm']:6.3f}  time: {duration:6.3f}s")


    val_loss = 0.0
    print("\neval...")
    for i in tqdm(range(min(eval_steps, eval_loader.num_chunks-1))):
        eval_loader.load_batch(in_tokens, out_tokens)
        val_loss += trainer.validate(in_tokens, out_tokens)

    print(f"eval loss: {val_loss / eval_steps:6.3f}")

    trainer.export_model("demo-model")


if __name__ == "__main__":
    main()