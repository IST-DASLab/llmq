import argparse
import math
from dataclasses import dataclass, asdict
from typing import Optional, List
import pyllmq
import numpy as np


@dataclass
class RunConfig:
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

    # offloading
    offload_master: bool = False
    offload_quants: bool = False
    offload_opt_m: bool = False
    offload_opt_v: bool = False
    persistent_quants: bool = False

    # Test settings
    seed: int = 0x83b45442


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


def _create_options(config: RunConfig) -> pyllmq.LLamaOptions:
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

    options.offload_opt_m = config.offload_opt_m
    options.offload_opt_v = config.offload_opt_v
    options.offload_quants = config.offload_quants
    options.offload_master = config.offload_master
    options.persistent_quants = config.persistent_quants

    if config.matmul_dtype:
        options.matmul_type = config.matmul_dtype

    # Apply recomputation dependencies
    if options.recompute_att:
        options.recompute_qkv = True
    if options.recompute_ffn:
        options.recompute_swiglu = True

    return options


def run_training(config: RunConfig) -> RunResult:
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
        losses.append(result['loss'])

    return RunResult(losses=losses, norms=norms)


def parse_args(args: list = None) -> RunConfig:
    """Parse command line arguments and return TestConfig."""
    parser = argparse.ArgumentParser(
        description="Test recomputation strategies produce identical results"
    )

    parser.add_argument("--model", default=RunConfig.model,
                        help="Path to HuggingFace model directory or cached model name")
    parser.add_argument("--matmul-dtype", default=None,
                        help="Which dtype to use for matmuls (defaults to model-dtype)")
    parser.add_argument("--model-dtype", default=RunConfig.model_dtype,
                        help="Which dtype to use for model")
    parser.add_argument("--train-file", default=RunConfig.train_file,
                        help="Tokens for training")
    parser.add_argument("--grad-accumulation", type=int, default=RunConfig.grad_accum,
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

    parser.add_argument("--offload-master", action="store_true", help="Offload master weights to CPU")
    parser.add_argument("--offload-quants", action="store_true", help="Offload quantized weights")
    parser.add_argument("--offload-opt-m", action="store_true", help="Offload first-order momentum")
    parser.add_argument("--offload-opt-v", action="store_true", help="Offload second-order momentum")
    parser.add_argument("--persistent-quants", action="store_true", help="Keep quantized weights")

    # Optional parameters
    parser.add_argument("--batch-size", "--batch", type=int, default=RunConfig.batch_size,
                        help="Micro-batch size")
    parser.add_argument("--seq-len", "--seq-length", type=int, default=RunConfig.seq_len,
                        help="Sequence length")
    parser.add_argument("--steps", type=int, default=RunConfig.max_steps,
                        help="Number of training steps")
    parser.add_argument("--beta-1", type=float, default=RunConfig.beta_1,
                        help="Beta 1 for Adam")
    parser.add_argument("--beta-2", type=float, default=RunConfig.beta_2,
                        help="Beta 2 for Adam")
    parser.add_argument("--opt-m-dtype", default=RunConfig.opt_m_dtype,
                        help="DType for first-order momentum")
    parser.add_argument("--opt-v-dtype", default=RunConfig.opt_v_dtype,
                        help="DType for second-order momentum")
    parser.add_argument("--grad-clip", type=float, default=RunConfig.grad_clip,
                        help="Gradient clipping")
    parser.add_argument("--weight-decay", type=float, default=RunConfig.weight_decay,
                        help="Weight decay for matrix parameters")
    parser.add_argument("--learning-rate", "--lr", type=float, default=RunConfig.learning_rate,
                        help="Learning rate")

    args = parser.parse_args(args=args)

    return RunConfig(
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
    )
