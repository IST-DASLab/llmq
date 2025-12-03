import argparse
import contextlib
import sys
from dataclasses import dataclass, asdict
from typing import Optional, TYPE_CHECKING
import math
import json
from . import _pyllmq

if TYPE_CHECKING:
    import wandb


@dataclass
class TrainingConfig:
    # Model configuration
    model: str = "Qwen/Qwen2.5-0.5B"
    from_scratch: bool = False
    init_proj_to_zero: bool = False
    model_dtype: str = "bfloat16"
    matmul_dtype: Optional[str] = None
    gradient_dtype: Optional[str] = None

    # Batch configuration
    batch_size: int = 1
    seq_len: int = 1024
    grad_accumulation: int = 4
    lmhead_chunks: int = 1
    attn_bwd_chunks: int = 1

    # Optimizer configuration
    learning_rate: float = 1e-5
    warmup_steps: int = -1
    final_lr_fraction: float = 1.0
    beta_1: float = 0.9
    beta_2: float = 0.95
    opt_m_dtype: str = "float32"
    opt_v_dtype: str = "float32"
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    # Training steps
    steps: int = -1

    # Evaluation
    eval_every: int = 100
    eval_num_steps: int = 100

    # Data files
    train_file: str = "data/tiny-stories-qwen/train*.bin"
    eval_file: str = "data/tiny-stories-qwen/eval.bin"

    # Output and checkpointing
    out_dir: str = "output"
    checkpoint_dir: str = "ckpt"
    log_file: str = "log.json"
    ckpt_interval: int = 1000
    ckpt_keep_n: int = -1
    ckpt_major: int = -1
    continue_from_checkpoint: bool = False
    log_gpu_util: int = 25

    # Multi-GPU
    gpus: int = 1

    # Memory optimization options
    recompute_swiglu: bool = False
    recompute_norm: bool = False
    recompute_ffn: bool = False
    recompute_qkv: bool = False
    recompute_att: bool = False
    recompute_block: bool = False
    offload_residual: bool = False

    # Distributed training options
    zero_level: int = 1
    shard_weights: bool = False
    shard_gradients: bool = False
    offload_master: bool = False
    offload_quants: bool = False
    offload_opt_m: bool = False
    offload_opt_v: bool = False
    offload_grads: bool = False
    persistent_quants: bool = False

    # Performance options
    use_cuda_graphs: bool = True
    memcpy_all_gather: bool = False
    memcpy_send_recv: bool = False
    all_to_all_reduce: bool = False
    write_combined: bool = False
    use_zero_copy: bool = False

    # Logging verbosity
    verbosity: str = _pyllmq.LogVerbosity.DEFAULT

    # Wandb integration
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_name: str = "llmq"


def add_training_args(parser: argparse.ArgumentParser, default: Optional[TrainingConfig] = None):
    default = TrainingConfig() if default is None else default

    # Model configuration
    parser.add_argument("--model", default=default.model, help="Path to model directory or HuggingFace model name")
    parser.add_argument("--from-scratch", action="store_true", help="Train from random initialization")
    parser.add_argument("--init-proj-to-zero", action="store_true", help="Initialize projections to zero")
    parser.add_argument("--model-dtype", default=default.model_dtype, help="Model dtype")
    parser.add_argument("--matmul-dtype", help="Matmul dtype (defaults to model-dtype)")
    parser.add_argument("--gradient-dtype",
                        help="Gradient dtype (defaults to matmul-dtype, except e4m3 matmul uses m5m2 gradients)")

    # Batch configuration
    parser.add_argument("--batch-size", "--batch", type=int, default=default.batch_size, help="Micro-batch size")
    parser.add_argument("--seq-len", "--seq-length", type=int, default=default.seq_len, help="Sequence length")
    parser.add_argument("--grad-accumulation", type=int, default=default.grad_accumulation,
                        help="Gradient accumulation steps")
    parser.add_argument("--lmhead-chunks", type=int, default=default.lmhead_chunks,
                        help="Run LM-head in smaller chunks")
    parser.add_argument("--attn-bwd-chunks", type=int, default=default.attn_bwd_chunks,
                        help="Run attention backward in smaller chunks")

    # Optimizer
    parser.add_argument("--learning-rate", "--lr", type=float, default=default.learning_rate, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=default.warmup_steps, dest="warmup_steps", help="Warmup steps")
    parser.add_argument("--final-lr-fraction", type=float, default=default.final_lr_fraction, help="Final LR fraction")
    parser.add_argument("--beta-1", type=float, default=default.beta_1, help="Adam beta 1")
    parser.add_argument("--beta-2", type=float, default=default.beta_2, help="Adam beta 2")
    parser.add_argument("--opt-m-dtype", default=default.opt_m_dtype, help="First-order momentum dtype")
    parser.add_argument("--opt-v-dtype", default=default.opt_v_dtype, help="Second-order momentum dtype")
    parser.add_argument("--grad-clip", type=float, default=default.grad_clip, help="Gradient clipping")
    parser.add_argument("--weight-decay", type=float, default=default.weight_decay, help="Weight decay")

    # Training
    parser.add_argument("--steps", type=int, default=default.steps, help="Training steps")
    parser.add_argument("--eval-every-n-steps", type=int, default=default.eval_every,
                        dest="eval_every", help="Evaluation interval")
    parser.add_argument("--eval-num-steps", type=int, default=default.eval_num_steps, help="Number of eval batches")
    parser.add_argument("--log-gpu-util", type=int, default=default.log_gpu_util,
                        help="GPU logging interval (0 to disable)")

    # Data
    parser.add_argument("--train-file", default=default.train_file, help="Training data file")
    parser.add_argument("--eval-file", default=default.eval_file, help="Evaluation data file")

    # Output
    parser.add_argument("--out-dir", default=default.out_dir, help="Output directory")
    parser.add_argument("--checkpoint-dir", default=default.checkpoint_dir, help="Checkpoint directory")
    parser.add_argument("--log-file", default=default.log_file, help="Log file")
    parser.add_argument("--ckpt-interval", type=int, default=default.ckpt_interval, help="Checkpoint interval")
    parser.add_argument("--ckpt-keep-n", type=int, default=default.ckpt_keep_n, help="Number of checkpoints to keep")
    parser.add_argument("--ckpt-major", type=int, default=default.ckpt_major, help="Major checkpoint interval")
    parser.add_argument("--continue", dest="continue_from_checkpoint", action="store_true",
                        help="Continue from checkpoint")

    # Multi-GPU
    parser.add_argument("--gpus", type=int, default=_pyllmq.get_num_gpus(), help="Number of GPUs")

    # Memory optimization
    parser.add_argument("--recompute-swiglu", action="store_true", help="Recompute SwiGLU")
    parser.add_argument("--recompute-norm", action="store_true", help="Recompute RMSNorm")
    parser.add_argument("--recompute-ffn", action="store_true", help="Recompute FFN")
    parser.add_argument("--recompute-qkv", action="store_true", help="Recompute QKV")
    parser.add_argument("--recompute-att", action="store_true", help="Recompute attention")
    parser.add_argument("--recompute-block", action="store_true", help="Recompute entire block")
    parser.add_argument("--offload-residual", action="store_true", help="Offload residual activations")

    # Distributed training
    parser.add_argument("--zero-level", type=int, default=1, help="ZeRO optimization level (1-3)")
    parser.add_argument("--shard-weights", action="store_true", help="Shard weights across GPUs")
    parser.add_argument("--shard-gradients", action="store_true", help="Shard gradients across GPUs")
    parser.add_argument("--offload-master", action="store_true", help="Offload master weights to CPU")
    parser.add_argument("--offload-quants", action="store_true", help="Offload quantized weights")
    parser.add_argument("--offload-opt-m", action="store_true", help="Offload first-order momentum")
    parser.add_argument("--offload-opt-v", action="store_true", help="Offload second-order momentum")
    parser.add_argument("--offload-grads", action="store_true", help="Offload gradients")
    parser.add_argument("--persistent-quants", action="store_true", help="Keep quantized weights")
    parser.add_argument("--use-zero-copy", action="store_true", help="Use zero-copy DMA for offloaded optimizer states")

    # Logging
    parser.add_argument("-qq", "--silent", dest="verbosity", action="store_const", const=_pyllmq.LogVerbosity.SILENT,
                        help="Silent mode (no output)")
    parser.add_argument("-q", "--quiet", dest="verbosity", action="store_const", const=_pyllmq.LogVerbosity.QUIET,
                        help="Quiet mode (minimal output)")
    parser.add_argument("-v", "--verbose", dest="verbosity", action="store_const", const=_pyllmq.LogVerbosity.VERBOSE,
                        help="Verbose mode (detailed output)")
    parser.set_defaults(verbosity=default.verbosity)
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=default.wandb_project, help="W&B project name (defaults to 'LLMQ')")
    parser.add_argument("--wandb-name", default=default.wandb_name, help="W&B run name")


class CosineLRSchedule:
    """Cosine learning rate schedule with linear warmup."""

    def __init__(self, base_lr: float, max_steps: int, warmup_steps: int, final_lr: float):
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps if warmup_steps >= 0 else 0
        self.final_lr = final_lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step + 1) / self.warmup_steps
        elif self.warmup_steps < self.max_steps:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + (self.base_lr - self.final_lr) * cosine_decay
        else:
            return self.final_lr


def log_line_to_wandb(run: "wandb.Run", entry: dict):
    kind = entry["log"]
    del entry["log"]
    step = entry["step"]
    del entry["step"]
    del entry["time"]  # TODO can we associate a datetime with step?
    if kind == "step":
        tps = entry["step_tokens"] / (entry["duration_ms"] / 1000)
        del entry["step_tokens"]
        run.log({f"train/{k}": v for k, v in entry.items()}, step=step)
        run.log({"train/tokens_per_second": tps}, step=step)
    elif kind == "eval":
        tps = entry["eval_tokens"] / (entry["duration_ms"] / 1000)
        del entry["eval_tokens"]
        run.log({f"eval/{k}": v for k, v in entry.items()}, step=step)
        run.log({"eval/tokens_per_second": tps}, step=step)
    elif kind == "gpu":
        del entry["throttle"]  # can't log this nicely?
        del entry["id"]        # not useful?
        if entry["fan"] == 0:  # indicates not recorded
            del entry["fan"]
        entry["dram_free"] /= 1024**2   # MiB
        entry["pcie_rx"] /= 1024**2     # MiB/s
        entry["pcie_tx"] /= 1024**2     # MiB/s
        run.log({f"gpu/{k}": v for k, v in entry.items()}, step=step)
    elif kind == "cmd":
        # TODO figure out if we can actually put this in the _wandb config object
        # where is belongs
        run.config["cmd"] = entry["cmd"]
    elif kind == "gpu-model":
        if entry["rank"] == 0:
            run.config["gpu"] = entry
        else:
            run.config[f"gpu-{entry['rank']}"] = entry
    elif kind == "allocator":
        import plotly.express as px
        names = [alloc["name"] for alloc in entry["stats"]]
        amounts = [round(alloc["device"] / 1024 / 1024, 1) for alloc in entry["stats"]]

        fig = px.pie(
            names=names,
            values=amounts,
            title=f"GPU Allocations",
        )
        run.log({"allocations": fig}, step=step)
    elif kind == "dataset":
        pass
        # run.config["dataset"] = entry
    elif kind in ["option", "info"]:
        pass
    elif kind == "sol":
        if entry["rank"] != 0:
            return
        import plotly.express as px
        names = ["Blocks", "LM-Head", "Attention"]
        amounts = [entry["blocks"], entry["lm_head"], entry["attention"]]

        fig = px.pie(
            names=names,
            values=amounts,
            title=f"FLOPs",
        )
        run.log({"ops": fig}, step=step)
    else:
        raise RuntimeError(f"Unknown kind {kind}")


def make_wandb_log_callback(run):
    def callback(entry: str):
        log_line_to_wandb(run, json.loads(entry))
    return callback


@contextlib.contextmanager
def training_logger_context(config: TrainingConfig):
    import wandb

    wandb_context = contextlib.nullcontext()
    log_callback = None

    if config.use_wandb:
        # Initialize a wandb run with optional parameters
        wandb_context = wandb.init(
            project=config.wandb_project,
            config=config,
        )
        log_callback = make_wandb_log_callback(wandb_context)

    with wandb_context:
        logger = _pyllmq.TrainingRunLogger(
            config.log_file,
            callback=log_callback,
            verbosity=config.verbosity
        )
        logger.log_cmd(sys.argv)
        log_options = asdict(config)
        log_options["matmul_dtype"] = log_options["matmul_dtype"] or config.model_dtype
        log_options["gradient_dtype"] = log_options["gradient_dtype"] or log_options["matmul_dtype"]
        log_options["verbosity"] = str(config.verbosity)
        logger.log_options(log_options)
        yield logger
