#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyllmq", "tqdm", "numpy"]
# ///

import pyllmq
import numpy as np
import argparse
import json
import time
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    # Model configuration
    model: str = "Qwen/Qwen2.5-0.5B"
    from_scratch: bool = False
    init_proj_to_zero: bool = False
    model_dtype: str = "bfloat16"
    matmul_dtype: Optional[str] = None

    # Batch configuration
    batch_size: int = 4
    seq_len: int = 1024
    grad_accumulation: int = 4

    # Optimizer configuration
    learning_rate: float = 1e-5
    warmup_steps: int = -1
    final_lr_fraction: float = 1.0
    beta_1: float = 0.9
    beta_2: float = 0.95
    opt_m_dtype: str = "float32"
    opt_v_dtype: str = "float32"
    grad_clip: float = 1.0
    weight_decay: float = 1.0

    # Training steps
    steps: int = 1000

    # Evaluation
    eval_every: int = 100
    eval_num_steps: int = 100

    # Data files
    train_file: str = "tiny-shakespeare-train.bin"
    eval_file: str = "tiny-shakespeare-test.bin"

    # Output and checkpointing
    out_dir: str = "output"
    checkpoint_dir: str = "ckpt"
    log_file: str = "log.json"
    ckpt_interval: int = 100
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


class TrainingLogger:
    def __init__(self, log_file: str, rank: int):
        self.log_file = Path(log_file)
        self.rank = rank
        self.logs = []

    def log(self, entry: dict):
        """Add a log entry"""
        if self.rank == 0:
            entry["timestamp"] = time.time()
            self.logs.append(entry)

    def save(self):
        """Save logs to file"""
        if self.rank == 0:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)

    def log_config(self, config: dict):
        """Log training configuration"""
        self.log({"type": "config", "data": config})

    def log_step(self, step: int, epoch: float, tokens: int, time_ms: int,
                 norm: float, loss: float, lr: float):
        """Log a training step"""
        self.log({
            "type": "train_step",
            "step": step,
            "epoch": epoch,
            "tokens": tokens,
            "time_ms": time_ms,
            "norm": norm,
            "loss": loss,
            "lr": lr,
            "tok_per_sec": tokens * 1000.0 / time_ms if time_ms > 0 else 0
        })

    def log_eval(self, step: int, epoch: float, tokens: int, time_ms: int, loss: float):
        """Log evaluation results"""
        self.log({
            "type": "eval",
            "step": step,
            "epoch": epoch,
            "tokens": tokens,
            "time_ms": time_ms,
            "loss": loss
        })

    def log_checkpoint(self, step: int, path: str, time_ms: int):
        """Log checkpoint save"""
        self.log({
            "type": "checkpoint",
            "step": step,
            "path": path,
            "time_ms": time_ms
        })

    def log_gpu_state(self, step: int, gpu_info: dict):
        """Log GPU utilization"""
        self.log({
            "type": "gpu_state",
            "step": step,
            "info": gpu_info
        })


class LRSchedule:
    def __init__(self, base_lr: float, max_steps: int, warmup_steps: int, final_lr: float):
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps if warmup_steps >= 0 else 0
        self.final_lr = final_lr

    def get_lr(self, step: int) -> float:
        """Calculate learning rate for given step"""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step + 1) / self.warmup_steps
        elif self.warmup_steps < self.max_steps:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return self.final_lr + (self.base_lr - self.final_lr) * cosine_decay
        else:
            return self.final_lr


def setup_options(config: TrainingConfig) -> pyllmq.LLamaOptions:
    """Configure LLamaOptions from training config"""
    options = pyllmq.LLamaOptions()

    # Recomputation options
    options.recompute_swiglu = config.recompute_swiglu or config.recompute_ffn or config.recompute_block
    options.recompute_rms_norm = config.recompute_norm or config.recompute_block
    options.recompute_ffn = config.recompute_ffn or config.recompute_block
    options.recompute_qkv = config.recompute_qkv or config.recompute_att or config.recompute_block
    options.recompute_att = config.recompute_att or config.recompute_block

    # Optimizer dtype
    options.momentum_type = config.opt_m_dtype
    options.variance_type = config.opt_v_dtype

    # Distributed training options
    if config.zero_level >= 2:
        config.shard_gradients = True
    if config.zero_level >= 3:
        config.shard_weights = True

    options.shard_weights = config.shard_weights
    options.shard_gradients = config.shard_gradients
    options.offload_master = config.offload_master
    options.offload_quants = config.offload_quants
    options.offload_opt_m = config.offload_opt_m
    options.offload_opt_v = config.offload_opt_v
    options.persistent_quants = config.persistent_quants

    # Performance options
    options.use_cuda_graphs = config.use_cuda_graphs
    options.use_all_to_all_reduce = config.all_to_all_reduce
    options.use_write_combined = config.write_combined

    # Other options
    options.init_projections_to_zero = config.init_proj_to_zero
    if config.matmul_dtype:
        options.matmul_type = config.matmul_dtype

    return options


def run_evaluation(trainer: pyllmq.LLMQTrainer, eval_loader: pyllmq.DataLoader,
                   in_tokens: np.ndarray, out_tokens: np.ndarray, max_steps: int) -> float:
    """Run evaluation on test set"""
    eval_loader.set_state(eval_loader.seed, 0, 0, 0)
    total_loss = 0.0
    batches = 0

    while batches < max_steps and batches < eval_loader.num_chunks:
        eval_loader.load_batch(in_tokens, out_tokens)
        loss = trainer.validate(in_tokens, out_tokens)
        total_loss += loss
        batches += 1

    if batches == 0:
        print("WARNING: insufficient validation data", file=sys.stderr)
        return 0.0

    return total_loss / batches


def main():
    parser = argparse.ArgumentParser(description="Train LLaMa model")

    # Model configuration
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Path to model directory or HuggingFace model name")
    parser.add_argument("--from-scratch", action="store_true", help="Train from random initialization")
    parser.add_argument("--init-proj-to-zero", action="store_true", help="Initialize projections to zero")
    parser.add_argument("--model-dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--matmul-dtype", help="Matmul dtype (defaults to model-dtype)")

    # Batch configuration
    parser.add_argument("--batch-size", "--batch", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--seq-len", "--seq-length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--grad-accumulation", type=int, default=4, help="Gradient accumulation steps")

    # Optimizer
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=-1, dest="warmup_steps", help="Warmup steps")
    parser.add_argument("--final-lr-fraction", type=float, default=1.0, help="Final LR fraction")
    parser.add_argument("--beta-1", type=float, default=0.9, help="Adam beta 1")
    parser.add_argument("--beta-2", type=float, default=0.95, help="Adam beta 2")
    parser.add_argument("--opt-m-dtype", default="float32", help="First-order momentum dtype")
    parser.add_argument("--opt-v-dtype", default="float32", help="Second-order momentum dtype")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--weight-decay", type=float, default=1.0, help="Weight decay")

    # Training
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--eval-every-n-steps", type=int, default=100, dest="eval_every", help="Evaluation interval")
    parser.add_argument("--eval-num-steps", type=int, default=100, help="Number of eval batches")
    parser.add_argument("--log-gpu-util", type=int, default=25, help="GPU logging interval (0 to disable)")

    # Data
    parser.add_argument("--train-file", default="tiny-shakespeare-train.bin", help="Training data file")
    parser.add_argument("--eval-file", default="tiny-shakespeare-test.bin", help="Evaluation data file")

    # Output
    parser.add_argument("--out-dir", default="output", help="Output directory")
    parser.add_argument("--checkpoint-dir", default="ckpt", help="Checkpoint directory")
    parser.add_argument("--log-file", default="log.json", help="Log file")
    parser.add_argument("--ckpt-interval", type=int, default=100, help="Checkpoint interval")
    parser.add_argument("--ckpt-keep-n", type=int, default=-1, help="Number of checkpoints to keep")
    parser.add_argument("--ckpt-major", type=int, default=-1, help="Major checkpoint interval")
    parser.add_argument("--continue", dest="continue_from_checkpoint", action="store_true",
                        help="Continue from checkpoint")

    # Multi-GPU
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")

    # Memory optimization
    parser.add_argument("--recompute-swiglu", action="store_true", help="Recompute SwiGLU")
    parser.add_argument("--recompute-norm", action="store_true", help="Recompute RMSNorm")
    parser.add_argument("--recompute-ffn", action="store_true", help="Recompute FFN")
    parser.add_argument("--recompute-qkv", action="store_true", help="Recompute QKV")
    parser.add_argument("--recompute-att", action="store_true", help="Recompute attention")
    parser.add_argument("--recompute-block", action="store_true", help="Recompute entire block")

    # Distributed training
    parser.add_argument("--zero-level", type=int, default=1, help="ZeRO optimization level (1-3)")
    parser.add_argument("--shard-weights", action="store_true", help="Shard weights across GPUs")
    parser.add_argument("--shard-gradients", action="store_true", help="Shard gradients across GPUs")
    parser.add_argument("--offload-master", action="store_true", help="Offload master weights to CPU")
    parser.add_argument("--offload-quants", action="store_true", help="Offload quantized weights")
    parser.add_argument("--offload-opt-m", action="store_true", help="Offload first-order momentum")
    parser.add_argument("--offload-opt-v", action="store_true", help="Offload second-order momentum")
    parser.add_argument("--persistent-quants", action="store_true", help="Keep quantized weights")

    # Performance
    parser.add_argument("--use-cuda-graphs", action="store_true", default=True, help="Use CUDA graphs")
    parser.add_argument("--no-use-cuda-graphs", dest="use_cuda_graphs", action="store_false")
    parser.add_argument("--memcpy-all-gather", action="store_true", help="Use memcpy for all-gather")
    parser.add_argument("--memcpy-send-recv", action="store_true", help="Use memcpy for send/recv")
    parser.add_argument("--all-to-all-reduce", action="store_true", help="Use all-to-all reduce")
    parser.add_argument("--write-combined", action="store_true", help="Use write-combined memory")

    args = parser.parse_args()

    # Create config from args
    config = TrainingConfig(**vars(args))

    # Setup options
    options = setup_options(config)

    # Setup paths
    ckpt_dir = Path(config.checkpoint_dir)
    out_dir = Path(config.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(config.log_file, rank=0)
    logger.log_config(asdict(config))

    # Load model configuration
    print(f"Loading model from {config.model}...")
    if config.from_scratch:
        # For from-scratch training, you'd need to provide architecture details
        # This is simplified - in practice you'd load from a config file
        raise NotImplementedError("--from-scratch requires specifying model architecture")

    # Setup data loaders
    total_batch_size = config.batch_size * config.seq_len * config.gpus * config.grad_accumulation

    train_files = [config.train_file]
    eval_files = [config.eval_file]

    in_tokens = np.empty((config.gpus * config.batch_size, config.seq_len), dtype=np.int32)
    out_tokens = np.empty((config.gpus * config.batch_size, config.seq_len), dtype=np.int32)

    train_loader = pyllmq.DataLoader(train_files, config.batch_size * config.seq_len * config.gpus, seed=0x83b45442)
    eval_loader = pyllmq.DataLoader(eval_files, config.batch_size * config.seq_len * config.gpus, seed=0x83b45442)

    # Create trainer
    print("Creating trainer...")
    latest_step = -1

    if config.continue_from_checkpoint:
        latest_step = pyllmq.find_latest_checkpoint(ckpt_dir)
        if latest_step >= 0:
            print(f"Loading checkpoint from step {latest_step}...")
            trainer = pyllmq.LLMQTrainer.from_checkpoint(
                str(ckpt_dir / f"step_{latest_step}.ckpt"),
                ngpu=config.gpus,
                options=options,
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                grad_accum=config.grad_accumulation
            )
        else:
            print("No checkpoint found, starting from pretrained model")

    if latest_step < 0:
        trainer = pyllmq.LLMQTrainer.from_pretrained(
            name=config.model,
            ngpu=config.gpus,
            dtype=config.model_dtype,
            options=options,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            grad_accum=config.grad_accumulation,
            memcpy_all_gather=config.memcpy_all_gather,
            memcpy_send_recv=config.memcpy_send_recv
        )
        latest_step = 0

    # Setup learning rate schedule
    lr_schedule = LRSchedule(
        config.learning_rate,
        config.steps,
        config.warmup_steps,
        config.learning_rate * config.final_lr_fraction
    )

    # Load first batch
    train_loader.load_batch(in_tokens, out_tokens)

    print(f"\nStarting training from step {latest_step}...\n")
    print(f"Total batch size: {total_batch_size} tokens")
    print(f"Steps per epoch: {train_loader.num_tokens // total_batch_size}")

    # Training loop
    for step in range(latest_step, config.steps):
        # Check if evaluation is needed
        run_eval = False
        if config.eval_every > 0 and step % config.eval_every == 0 and step > latest_step:
            run_eval = True

        # Checkpointing
        if config.ckpt_interval > 0 and step % config.ckpt_interval == 0 and step > latest_step:
            print(f"Saving checkpoint to {config.checkpoint_dir}...")
            start_time = time.time()
            ckpt_path = ckpt_dir / f"step_{step}.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.log_checkpoint(step, str(ckpt_path), elapsed_ms)
            print("done")

            # Clean old checkpoints
            if config.ckpt_keep_n > 0:
                removed = pyllmq.clean_old_checkpoints(ckpt_dir, config.ckpt_keep_n, config.ckpt_major)
                if removed:
                    print(f"Cleaned {len(removed)} checkpoints")

        # Run evaluation
        if run_eval:
            print("Running evaluation...")
            start_time = time.time()
            val_loss = run_evaluation(trainer, eval_loader, in_tokens, out_tokens, config.eval_num_steps)
            elapsed_ms = int((time.time() - start_time) * 1000)
            epoch = train_loader.epoch() + 0.01 * (train_loader.current_position / train_loader.num_tokens)
            logger.log_eval(step, epoch, config.eval_num_steps * config.batch_size * config.seq_len,
                            elapsed_ms, val_loss)
            print(f"Validation loss: {val_loss:.4f}")

        # Training step
        step_start = time.time()

        for micro_step in range(config.grad_accumulation):
            trainer.step(in_tokens, out_tokens)
            train_loader.load_batch(in_tokens, out_tokens)

        # Log GPU info
        if config.log_gpu_util > 0 and step % config.log_gpu_util == 0:
            infos = trainer.get_gpu_info()
            for i, info in enumerate(infos):
                print(f"GPU {i}:  power: {info.power // 1000:4}W  temp: {info.temperature:3}°C  "
                      f"rx: {info.pcie_rx // 1024 // 1024:4}MiB/s  tx: {info.pcie_tx // 1024 // 1024:4}MiB/s")
                if hasattr(info, 'clock'):
                    print(f"         clock: {info.clock / 1000:.1f}GHz  fan: {info.fan:3}%")
                logger.log_gpu_state(step, {
                    "gpu": i,
                    "power": info.power,
                    "temperature": info.temperature,
                    "pcie_rx": info.pcie_rx,
                    "pcie_tx": info.pcie_tx
                })

        # Optimizer update
        lr = lr_schedule.get_lr(step)
        result = trainer.update(lr, config.beta_1, config.beta_2, step + 1,
                                config.weight_decay, config.grad_clip)

        step_time = time.time() - step_start
        elapsed_ms = int(step_time * 1000)

        # Log step
        tokens_processed = config.batch_size * config.seq_len * config.grad_accumulation * config.gpus
        epoch = train_loader.epoch() + 0.01 * (train_loader.progress() / 100)
        logger.log_step(step, epoch, tokens_processed, elapsed_ms,
                        result['norm'], result['loss'], lr)

        # Print progress
        tok_per_sec = tokens_processed / step_time
        print(f"step: {step:5}  loss: {result['loss']:6.3f}  norm: {result['norm']:6.3f}  "
              f"lr: {lr:.2e}  time: {step_time:6.3f}s  tok/s: {tok_per_sec:8.1f}")

        # Save logs periodically
        if step % 10 == 0:
            logger.save()

    # Final evaluation
    print("\nRunning final evaluation...")
    final_loss = run_evaluation(trainer, eval_loader, in_tokens, out_tokens, eval_loader.num_chunks)
    print(f"Final validation loss: {final_loss:.4f}")

    # Save final model
    print(f"Saving model to {config.out_dir}...")
    trainer.export_model(str(out_dir))
    print("done")

    # Save final logs
    logger.save()
    print(f"\nTraining complete! Logs saved to {config.log_file}")


if __name__ == "__main__":
    main()