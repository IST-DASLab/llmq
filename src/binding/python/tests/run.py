import argparse
import math
from dataclasses import dataclass, asdict
from typing import Optional, List
import pyllmq
import numpy as np
from pyllmq.training import TrainingConfig


@dataclass
class RunResult:
    """Results from a recomputation test run."""
    losses: List[float]
    norms: List[float]


def compare_results(result: RunResult, expected: RunResult, *, file=None, atol=0, rtol=0.0):
    passed = True
    if len(result.losses) != len(expected.losses):
        print("losses have different lengths", file=file)
        passed = False

    if len(result.norms) != len(expected.norms):
        print("norms have different lengths", file=file)
        passed = False

    print("\nlosses:", file=file)
    for i, (loss, ref_loss) in enumerate(zip(result.losses, expected.losses)):
        if loss == ref_loss:
            print(f" \033[1;32m✓\033[0m step {i}: {loss:.10f} = {ref_loss:.10f}", file=file)
        elif math.isclose(ref_loss, loss, abs_tol=atol, rel_tol=rtol):
            print(f" \033[1;32m✓\033[0m step {i}: {loss:.10f} ≈ {ref_loss:.10f}", file=file)
        else:
            passed = False
            print(f" \033[1;31m✗\033[0m step {i}: {loss:.10f} ≠ {ref_loss:.10f}", file=file)

    print("\nnorms:", file=file)
    for i, (norm, ref_norm) in enumerate(zip(result.norms, expected.norms)):
        if norm == ref_norm:
            print(f" \033[1;32m✓\033[0m step {i}: {norm:13.10f} = {ref_norm:13.10f}", file=file)
        elif math.isclose(ref_norm, norm, abs_tol=atol, rel_tol=rtol):
            print(f" \033[1;32m✓\033[0m step {i}: {norm:13.10f} ≈ {ref_norm:13.10f}", file=file)
        else:
            passed = False
            print(f" \033[1;31m✗\033[0m step {i}: {norm:13.10f} ≠ {ref_norm:13.10f}", file=file)

    if passed:
        print("\n\033[1;32mPASS\033[0m", file=file)
    else:
        print("\n\033[1;31mFAIL\033[0m", file=file)

    return passed


def _create_options(config: TrainingConfig) -> pyllmq.LLamaOptions:
    """Create LLamaOptions from config."""
    options = pyllmq.LLamaOptions()
    options.recompute_swiglu = config.recompute_swiglu
    options.recompute_rms_norm = config.recompute_norm
    options.recompute_ffn = config.recompute_ffn
    options.recompute_qkv = config.recompute_qkv
    options.recompute_att = config.recompute_att
    options.recompute_block = config.recompute_block
    options.use_cuda_graphs = config.use_cuda_graphs
    options.momentum_type = config.opt_m_dtype
    options.variance_type = config.opt_v_dtype
    options.lmhead_chunks = config.lmhead_chunks
    options.attn_bwd_chunks = config.attn_bwd_chunks
    options.offload_residual = config.offload_residual

    options.offload_opt_m = config.offload_opt_m
    options.offload_opt_v = config.offload_opt_v
    options.offload_quants = config.offload_quants
    options.offload_master = config.offload_master
    options.persistent_quants = config.persistent_quants

    if config.matmul_dtype:
        options.matmul_type = config.matmul_dtype
    if config.gradient_dtype:
        options.gradient_type = config.gradient_dtype

    # Apply recomputation dependencies
    if options.recompute_att:
        options.recompute_qkv = True
    if options.recompute_ffn:
        options.recompute_swiglu = True

    return options


def run_training(config: TrainingConfig) -> RunResult:
    """Run training with given options and return losses and norms."""

    options = _create_options(config)

    # Create trainer
    trainer = pyllmq.LLMQTrainer.from_pretrained(
        name=config.model,
        ngpu=1,
        dtype=config.model_dtype,
        options=options,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        grad_accum=config.grad_accumulation,
        memcpy_all_gather=config.memcpy_all_gather,
        memcpy_send_recv=config.memcpy_send_recv
    )

    # Create data loader
    train_loader = pyllmq.DataLoader(
        [config.train_file],
        config.batch_size * config.seq_len,
        seed=0x83b45442
    )

    # Prepare input/output buffers
    in_tokens = np.empty((config.batch_size, config.seq_len), dtype=np.int32)
    out_tokens = np.empty((config.batch_size, config.seq_len), dtype=np.int32)

    losses = []
    norms = []

    # Training loop
    for step in range(config.steps):
        # Gradient accumulation loop
        for j in range(config.grad_accumulation):
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
        losses.append(result['loss'])

    # add one eval step
    losses.append(trainer.validate(in_tokens, out_tokens))

    return RunResult(losses=losses, norms=norms)


def parse_args(args: list = None) -> TrainingConfig:
    """Parse command line arguments and return TestConfig."""
    parser = argparse.ArgumentParser(
        description="Test recomputation strategies produce identical results"
    )
    from pyllmq.training import add_training_args
    default = TrainingConfig()
    default.steps = 10
    default.train_file = "data/tiny-shakespeare-qwen/train.bin"
    default.batch_size = 2
    add_training_args(parser, default=default)
    parser.add_argument("--use-cuda-graphs", action="store_true")
    parser.add_argument("--memcpy-all-gather", action="store_true")
    parser.add_argument("--memcpy-send-recv", action="store_true")
    parser.add_argument("--all-to-all-reduce", action="store_true")
    parser.add_argument("--write-combined", action="store_true")
    args = parser.parse_args(args=args)

    cfg = TrainingConfig(**vars(args))
    cfg.eval_file = cfg.train_file
    return cfg
