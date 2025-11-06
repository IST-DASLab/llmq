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
import copy
import sys
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

import pyllmq


@dataclass
class TestConfig:
    """Configuration for recomputation testing."""
    # Training hyperparameters
    batch_size: int = 2
    seq_len: int = 1024
    max_steps: int = 10

    # Optimizer settings
    beta_1: float = 0.9
    beta_2: float = 0.95
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    grad_accum: int = 4
    learning_rate: float = 1e-5

    # Model settings
    model_dtype: str = "bf16"
    model: str = "Qwen/Qwen2.5-0.5B"
    train_file: str = "data/tiny-shakespeare-qwen/train.bin"
    matmul_dtype: Optional[str] = None

    # Communication settings
    memcpy_all_gather: bool = False
    memcpy_send_recv: bool = False

    # Optimizer dtypes
    opt_m_dtype: str = "fp32"
    opt_v_dtype: str = "fp32"

    # Recomputation options
    recompute_swiglu: bool = False
    recompute_rms_norm: bool = False
    recompute_ffn: bool = False
    recompute_qkv: bool = False
    recompute_att: bool = False
    recompute_block: bool = False
    use_cuda_graphs: bool = False

    # Test settings
    seed: int = 0x83b45442
    verbose: bool = True


@dataclass
class TestResult:
    """Results from a recomputation test run."""
    losses: List[float]
    norms: List[float]
    baseline_losses: List[float]
    baseline_norms: List[float]
    passed: bool

    def print_comparison(self):
        """Print a formatted comparison of results."""
        print("\nlosses:")
        all_match = True
        for i, (loss, ref_loss) in enumerate(zip(self.losses, self.baseline_losses)):
            if ref_loss != loss:
                print(f" \033[1;31m✗\033[0m step {i}: {loss:.10f} ≠ {ref_loss:.10f}")
                all_match = False
            else:
                print(f" \033[1;32m✓\033[0m step {i}: {loss:.10f} = {ref_loss:.10f}")

        print("\nnorms:")
        for i, (norm, ref_norm) in enumerate(zip(self.norms, self.baseline_norms)):
            if ref_norm != norm:
                print(f" \033[1;31m✗\033[0m step {i}: {norm:.10f} ≠ {ref_norm:.10f}")
                all_match = False
            else:
                print(f" \033[1;32m✓\033[0m step {i}: {norm:.10f} = {ref_norm:.10f}")

        if self.passed:
            print("\n\033[1;32mPASS\033[0m")
        else:
            print("\n\033[1;31mFAIL\033[0m")


class RecomputeTestRunner:
    """Runner for recomputation correctness tests."""

    def __init__(self, config: Optional[TestConfig] = None):
        """Initialize test runner with optional configuration."""
        self.config = config or TestConfig()

    def _create_options(self, config: TestConfig) -> pyllmq.LLamaOptions:
        """Create LLamaOptions from config."""
        options = pyllmq.LLamaOptions()
        options.recompute_swiglu = config.recompute_swiglu
        options.recompute_rms_norm = config.recompute_rms_norm
        options.recompute_ffn = config.recompute_ffn
        options.recompute_qkv = config.recompute_qkv
        options.recompute_att = config.recompute_att
        options.recompute_block = config.recompute_block
        options.use_cuda_graphs = config.use_cuda_graphs
        options.momentum_type = config.opt_m_dtype
        options.variance_type = config.opt_v_dtype

        if config.matmul_dtype:
            options.matmul_type = config.matmul_dtype

        # Apply recomputation dependencies
        if options.recompute_att:
            options.recompute_qkv = True
        if options.recompute_ffn:
            options.recompute_swiglu = True

        return options

    def run_training(self, options: pyllmq.LLamaOptions) -> Tuple[List[float], List[float]]:
        """Run training with given options and return losses and norms."""
        config = self.config

        # Create trainer
        trainer = pyllmq.LLMQTrainer.from_pretrained(
            name=config.model,
            ngpu=1,
            dtype=config.model_dtype,
            options=options,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            grad_accum=config.grad_accum,
            memcpy_all_gather=config.memcpy_all_gather,
            memcpy_send_recv=config.memcpy_send_recv
        )

        # Create data loader
        train_loader = pyllmq.DataLoader(
            [config.train_file],
            config.batch_size * config.seq_len,
            seed=config.seed
        )

        # Prepare input/output buffers
        in_tokens = np.empty((config.batch_size, config.seq_len), dtype=np.int32)
        out_tokens = np.empty((config.batch_size, config.seq_len), dtype=np.int32)

        losses = []
        norms = []

        # Training loop
        for step in range(config.max_steps):
            # Gradient accumulation loop
            for j in range(config.grad_accum):
                train_loader.load_batch(in_tokens, out_tokens)
                trainer.step(in_tokens, out_tokens)

            # Optimizer update
            result = trainer.update(
                config.learning_rate,
                config.beta_1,
                config.beta_2,
                step + 1,
                config.weight_decay,
                config.grad_clip
            )

            # Store results
            norms.append(result['norm'])
            losses.append(result['loss'] / (config.batch_size * config.seq_len * config.grad_accum))

        return losses, norms

    def run_test(self) -> TestResult:
        """
        Run the recomputation test and return results.

        Returns:
            TestResult containing losses, norms, and pass/fail status
        """
        config = self.config

        if config.verbose:
            print("Running training with test configuration...")

        # Run with test configuration
        test_options = self._create_options(config)
        losses, norms = self.run_training(test_options)

        if config.verbose:
            print("Running baseline training (no recomputation)...")

        # Run baseline (no recomputation)
        baseline_config = copy.deepcopy(config)
        baseline_config.recompute_swiglu = False
        baseline_config.recompute_rms_norm = False
        baseline_config.recompute_ffn = False
        baseline_config.recompute_qkv = False
        baseline_config.recompute_att = False
        baseline_config.recompute_block = False
        baseline_config.use_cuda_graphs = False

        baseline_options = self._create_options(baseline_config)
        ref_losses, ref_norms = self.run_training(baseline_options)

        # Check if results match
        passed = (losses == ref_losses) and (norms == ref_norms)

        return TestResult(
            losses=losses,
            norms=norms,
            baseline_losses=ref_losses,
            baseline_norms=ref_norms,
            passed=passed
        )


def parse_args(args: list = None) -> TestConfig:
    """Parse command line arguments and return TestConfig."""
    parser = argparse.ArgumentParser(
        description="Test recomputation strategies produce identical results"
    )

    parser.add_argument("--model", default=TestConfig.model,
                        help="Path to HuggingFace model directory or cached model name")
    parser.add_argument("--matmul-dtype", default=None,
                        help="Which dtype to use for matmuls (defaults to model-dtype)")
    parser.add_argument("--model-dtype", default=TestConfig.model_dtype,
                        help="Which dtype to use for model")
    parser.add_argument("--train-file", default=TestConfig.train_file,
                        help="Tokens for training")
    parser.add_argument("--grad-accumulation", type=int, default=TestConfig.grad_accum,
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
    parser.add_argument("--batch-size", "--batch", type=int, default=TestConfig.batch_size,
                        help="Micro-batch size")
    parser.add_argument("--seq-len", "--seq-length", type=int, default=TestConfig.seq_len,
                        help="Sequence length")
    parser.add_argument("--steps", type=int, default=TestConfig.max_steps,
                        help="Number of training steps")
    parser.add_argument("--beta-1", type=float, default=TestConfig.beta_1,
                        help="Beta 1 for Adam")
    parser.add_argument("--beta-2", type=float, default=TestConfig.beta_2,
                        help="Beta 2 for Adam")
    parser.add_argument("--opt-m-dtype", default=TestConfig.opt_m_dtype,
                        help="DType for first-order momentum")
    parser.add_argument("--opt-v-dtype", default=TestConfig.opt_v_dtype,
                        help="DType for second-order momentum")
    parser.add_argument("--grad-clip", type=float, default=TestConfig.grad_clip,
                        help="Gradient clipping")
    parser.add_argument("--weight-decay", type=float, default=TestConfig.weight_decay,
                        help="Weight decay for matrix parameters")
    parser.add_argument("--learning-rate", "--lr", type=float, default=TestConfig.learning_rate,
                        help="Learning rate")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args(args=args)

    return TestConfig(
        model=args.model,
        model_dtype=args.model_dtype,
        train_file=args.train_file,
        grad_accum=args.grad_accumulation,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.steps,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        matmul_dtype=args.matmul_dtype,
        opt_m_dtype=args.opt_m_dtype,
        opt_v_dtype=args.opt_v_dtype,
        recompute_swiglu=args.recompute_swiglu,
        recompute_rms_norm=args.recompute_norm,
        recompute_ffn=args.recompute_ffn,
        recompute_qkv=args.recompute_qkv,
        recompute_att=args.recompute_att,
        recompute_block=args.recompute_block,
        use_cuda_graphs=args.use_cuda_graphs,
        verbose=not args.quiet,
    )


def main():
    """Main entry point for command-line usage."""
    config = parse_args()
    runner = RecomputeTestRunner(config)
    result = runner.run_test()

    if config.verbose:
        result.print_comparison()

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()