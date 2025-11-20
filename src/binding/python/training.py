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

    # Distributed training options
    zero_level: int = 1
    shard_weights: bool = False
    shard_gradients: bool = False
    offload_master: bool = False
    offload_quants: bool = False
    offload_opt_m: bool = False
    offload_opt_v: bool = False
    persistent_quants: bool = False

    # Performance options
    use_cuda_graphs: bool = True
    memcpy_all_gather: bool = False
    memcpy_send_recv: bool = False
    all_to_all_reduce: bool = False
    write_combined: bool = False

    # Logging verbosity
    verbosity: str = _pyllmq.LogVerbosity.DEFAULT

    # Wandb integration
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_name: str = "llmq"


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
        #run.config["dataset"] = entry
    elif kind == "option":
        pass
    elif kind == "checkpoint":
        pass
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
        log_options["verbosity"] = str(config.verbosity)
        logger.log_options(log_options)
        yield logger
