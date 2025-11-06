#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyllmq", "tqdm", "numpy"]
# ///

"""
Test script to verify that recomputation strategies produce identical results.

This script runs training twice:
1. With specified recomputation options
2. Without any recomputation (baseline)

Then compares losses and gradient norms to ensure they match exactly.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

import pyllmq


class RecomputeTestRunner:
    def __init__(self):
        # Training hyperparameters
        self.batch_size = 2
        self.seq_len = 1024
        self.max_steps = 10

        # Optimizer settings
        self.beta_1 = 0.9
        self.beta_2 = 0.95
        self.grad_clip = 1.0
        self.weight_decay = 0.1
        self.grad_accum = 4

        # Model settings
        self.model_dtype = "bf16"
        self.model = "Qwen/Qwen2.5-0.5B"
        self.train_file = "data/tiny-shakespeare-qwen/train.bin"

        # Communication settings
        self.memcpy_all_gather = False
        self.memcpy_send_recv = False

        # Test options
        self.options = pyllmq.LLamaOptions()
        self.options.recompute_swiglu = False
        self.options.recompute_rms_norm = False
        self.options.recompute_ffn = False
        self.options.recompute_qkv = False
        self.options.recompute_att = False
        self.options.recompute_block = False
        self.options.use_cuda_graphs = False

    def parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Test recomputation strategies produce identical results"
        )

        parser.add_argument("--model", default=self.model,
                            help="Path to HuggingFace model directory or cached model name")
        parser.add_argument("--matmul-dtype", default=None,
                            help="Which dtype to use for matmuls (defaults to model-dtype)")
        parser.add_argument("--model-dtype", default=self.model_dtype,
                            help="Which dtype to use for model")
        parser.add_argument("--train-file", default=self.train_file,
                            help="Tokens for training")
        parser.add_argument("--grad-accumulation", type=int, default=self.grad_accum,
                            help="Number of micro-batches per optimizer step")

        # Recomputation options
        parser.add_argument("--recompute-swiglu", action="store_true",
                            help="Recompute swiglu during backward pass")
        parser.add_argument("--recompute-norm", action="store_true",
                            help="Recompute rms-norms during backward pass")
        parser.add_argument("--recompute-ffn", action="store_true",
                            help="Recompute feed-forward block during backward pass")
        parser.add_argument("--recompute-qkv", action="store_true",
                            help="Recompute qkv projections during backward pass")
        parser.add_argument("--recompute-att", action="store_true",
                            help="Recompute attention block during backward pass")
        parser.add_argument("--recompute-block", action="store_true",
                            help="Recompute entire transformer block")
        parser.add_argument("--use-cuda-graphs", action="store_true",
                            help="Enable CUDA graphs")

        # Optional parameters
        parser.add_argument("--batch-size", "--batch", type=int, default=self.batch_size,
                            help="Micro-batch size")
        parser.add_argument("--seq-len", "--seq-length", type=int, default=self.seq_len,
                            help="Sequence length")
        parser.add_argument("--steps", type=int, default=self.max_steps,
                            help="Number of training steps")
        parser.add_argument("--beta-1", type=float, default=self.beta_1,
                            help="Beta 1 for Adam")
        parser.add_argument("--beta-2", type=float, default=self.beta_2,
                            help="Beta 2 for Adam")
        parser.add_argument("--opt-m-dtype", default="fp32",
                            help="DType for first-order momentum")
        parser.add_argument("--opt-v-dtype", default="fp32",
                            help="DType for second-order momentum")
        parser.add_argument("--grad-clip", type=float, default=self.grad_clip,
                            help="Gradient clipping")
        parser.add_argument("--weight-decay", type=float, default=self.weight_decay,
                            help="Weight decay for matrix parameters")

        args = parser.parse_args()

        # Update settings from args
        self.model = args.model
        self.model_dtype = args.model_dtype
        self.train_file = args.train_file
        self.grad_accum = args.grad_accumulation
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.max_steps = args.steps
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.grad_clip = args.grad_clip
        self.weight_decay = args.weight_decay

        # Set recomputation options
        self.options.recompute_swiglu = args.recompute_swiglu
        self.options.recompute_rms_norm = args.recompute_norm
        self.options.recompute_ffn = args.recompute_ffn
        self.options.recompute_qkv = args.recompute_qkv
        self.options.recompute_att = args.recompute_att
        self.options.recompute_block = args.recompute_block
        self.options.use_cuda_graphs = args.use_cuda_graphs

        # Set optimizer dtypes
        self.options.momentum_type = args.opt_m_dtype
        self.options.variance_type = args.opt_v_dtype

        # Set matmul dtype if specified
        if args.matmul_dtype:
            self.options.matmul_type = args.matmul_dtype

        # Apply recomputation dependencies
        if self.options.recompute_att:
            self.options.recompute_qkv = True
        if self.options.recompute_ffn:
            self.options.recompute_swiglu = True

    def run_training(self, options: pyllmq.LLamaOptions) -> Tuple[List[float], List[float]]:
        """Run training with given options and return losses and norms"""

        # Create trainer
        trainer = pyllmq.LLMQTrainer.from_pretrained(
            name=self.model,
            ngpu=1,
            dtype=self.model_dtype,
            options=options,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            grad_accum=self.grad_accum,
            memcpy_all_gather=self.memcpy_all_gather,
            memcpy_send_recv=self.memcpy_send_recv
        )

        # Create data loader
        train_loader = pyllmq.DataLoader(
            [self.train_file],
            self.batch_size * self.seq_len,
            seed=0x83b45442
        )

        # Prepare input/output buffers
        in_tokens = np.empty((self.batch_size, self.seq_len), dtype=np.int32)
        out_tokens = np.empty((self.batch_size, self.seq_len), dtype=np.int32)

        losses = []
        norms = []

        # Training loop
        for step in range(self.max_steps):
            # Gradient accumulation loop
            for j in range(self.grad_accum):
                train_loader.load_batch(in_tokens, out_tokens)
                trainer.step(in_tokens, out_tokens)

            # Optimizer update
            result = trainer.update(
                1e-5,
                self.beta_1,
                self.beta_2,
                step + 1,
                self.weight_decay,
                self.grad_clip
            )

            # Store results
            norms.append(result['norm'])
            losses.append(result['loss'] / (self.batch_size * self.seq_len * self.grad_accum))

        return losses, norms


def main():
    runner = RecomputeTestRunner()
    runner.parse_args()

    print("Running training with test configuration...")
    options = runner.options
    losses, norms = runner.run_training(options)

    print("Running baseline training (no recomputation)...")
    options.recompute_swiglu = False
    options.recompute_rms_norm = False
    options.recompute_ffn = False
    options.recompute_qkv = False
    options.recompute_att = False
    options.recompute_block = False
    options.use_cuda_graphs = False

    ref_losses, ref_norms = runner.run_training(options)

    # Compare results
    all_ok = True

    print("\nlosses:")
    for i, (loss, ref_loss) in enumerate(zip(losses, ref_losses)):
        if ref_loss != loss:
            print(f" \033[1;31m✗\033[0m step {i}: {loss:.10f} ≠ {ref_loss:.10f}")
            all_ok = False
        else:
            print(f" \033[1;32m✓\033[0m step {i}: {loss:.10f} = {ref_loss:.10f}")

    print("\nnorms:")
    for i, (norm, ref_norm) in enumerate(zip(norms, ref_norms)):
        if ref_norm != norm:
            print(f" \033[1;31m✗\033[0m step {i}: {norm:.10f} ≠ {ref_norm:.10f}")
            all_ok = False
        else:
            print(f" \033[1;32m✓\033[0m step {i}: {norm:.10f} = {ref_norm:.10f}")

    if all_ok:
        print("\n\033[1;32mPASS\033[0m")
        sys.exit(0)
    else:
        print("\n\033[1;31mFAIL\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()