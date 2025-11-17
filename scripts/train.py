#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyllmq", "numpy", "wandb"]
# ///

import pyllmq
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple


def setup_options(config: pyllmq.TrainingConfig) -> pyllmq.LLamaOptions:
    """Configure LLamaOptions from training config"""
    options = pyllmq.LLamaOptions()

    # Recomputation options
    options.recompute_swiglu = config.recompute_swiglu or config.recompute_ffn or config.recompute_block
    options.recompute_rms_norm = config.recompute_norm or config.recompute_block
    options.recompute_ffn = config.recompute_ffn or config.recompute_block
    options.recompute_qkv = config.recompute_qkv or config.recompute_att or config.recompute_block
    options.recompute_att = config.recompute_att or config.recompute_block

    options.lmhead_chunks = config.lmhead_chunks

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
    if config.gradient_dtype:
        options.gradient_type = config.gradient_dtype

    return options


def run_evaluation(trainer: pyllmq.LLMQTrainer, eval_loader: pyllmq.DataLoader,
                   in_tokens: np.ndarray, out_tokens: np.ndarray, max_steps: int) -> Tuple[float, int]:
    """Run evaluation on test set"""
    start_time = time.time()
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
        return 0.0, 0

    return total_loss / batches, int((time.time() - start_time) * 1000)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLaMa model")
    default = pyllmq.TrainingConfig()

    # Model configuration
    parser.add_argument("--model", default=default.model, help="Path to model directory or HuggingFace model name")
    parser.add_argument("--from-scratch", action="store_true", help="Train from random initialization")
    parser.add_argument("--init-proj-to-zero", action="store_true", help="Initialize projections to zero")
    parser.add_argument("--model-dtype", default=default.model_dtype, help="Model dtype")
    parser.add_argument("--matmul-dtype", help="Matmul dtype (defaults to model-dtype)")
    parser.add_argument("--gradient-dtype", help="Gradient dtype (defaults to matmul-dtype, except e4m3 matmul uses m5m2 gradients)")

    # Batch configuration
    parser.add_argument("--batch-size", "--batch", type=int, default=default.batch_size, help="Micro-batch size")
    parser.add_argument("--seq-len", "--seq-length", type=int, default=default.seq_len, help="Sequence length")
    parser.add_argument("--grad-accumulation", type=int, default=default.grad_accumulation, help="Gradient accumulation steps")
    parser.add_argument("--lmhead-chunks", type=int, default=default.lmhead_chunks, help="Run LM-head in smaller chunks")

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
    parser.add_argument("--eval-every-n-steps", type=int, default=default.eval_every, dest="eval_every", help="Evaluation interval")
    parser.add_argument("--eval-num-steps", type=int, default=default.eval_num_steps, help="Number of eval batches")
    parser.add_argument("--log-gpu-util", type=int, default=default.log_gpu_util, help="GPU logging interval (0 to disable)")

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
    parser.add_argument("--gpus", type=int, default=pyllmq.get_num_gpus(), help="Number of GPUs")

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

    def add_toggle(arg: str, default: bool, help: str):
        dest = arg.replace("-", "_")
        parser.add_argument(f"--{arg}", dest=dest, action="store_true", default=default, help=help)
        parser.add_argument(f"--no-{arg}", dest=dest, action="store_false")

    # Performance
    add_toggle("use-cuda-graphs", True, "Use CUDA graphs for transformer blocks")
    add_toggle("memcpy-all-gather", True, "Use cudaMemcpyAsync for all-gather (faster on PCIe)")
    add_toggle("memcpy-send-recv", True, "Use cudaMemcpyAsync for send/recv (faster on PCIe). Only meaningful in conjunction with all-to-all-reduce")
    add_toggle("all-to-all-reduce", True, "Use custom all-to-all reduce which can be used with memcpy-send-recv")
    add_toggle("write-combined", False, "Use write-combined memory. May give faster PCIe transfers.")

    # Logging
    parser.add_argument("-qq", "--silent", dest="verbosity", action="store_const", const=pyllmq.LogVerbosity.SILENT,
                        help="Silent mode (no output)")
    parser.add_argument("-q", "--quiet", dest="verbosity", action="store_const", const=pyllmq.LogVerbosity.QUIET,
                        help="Quiet mode (minimal output)")
    parser.add_argument("-v", "--verbose", dest="verbosity", action="store_const", const=pyllmq.LogVerbosity.VERBOSE,
                        help="Verbose mode (detailed output)")
    parser.set_defaults(verbosity=default.verbosity)
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=default.wandb_project, help="W&B project name (defaults to 'LLMQ')")
    parser.add_argument("--wandb-name", default=default.wandb_name, help="W&B run name")


    args = parser.parse_args()
    return pyllmq.TrainingConfig(**vars(args))


def main():
    # Create config from args
    config = parse_args()

    # Setup options
    options = setup_options(config)

    # Setup paths
    ckpt_dir = Path(config.checkpoint_dir)
    out_dir = Path(config.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with pyllmq.training_logger_context(config) as logger:
        # Setup data loaders
        train_files = list(map(str, Path.glob(Path(), config.train_file)))
        eval_files = list(map(str, Path.glob(Path(), config.eval_file)))

        train_loader = pyllmq.DataLoader(train_files, config.batch_size * config.seq_len * config.gpus, seed=0x83b45442)
        eval_loader = pyllmq.DataLoader(eval_files, config.batch_size * config.seq_len * config.gpus, seed=0x83b45442)

        # Log dataset information
        logger.log_dataset(train_loader, eval_loader)

        total_batch_size = config.batch_size * config.seq_len * config.gpus * config.grad_accumulation
        steps_per_epoch = train_loader.num_tokens // total_batch_size

        # Create trainer
        print("Creating trainer...")
        if config.continue_from_checkpoint:
            latest_step = pyllmq.find_latest_checkpoint(ckpt_dir)
            if latest_step >= 0:
                trainer = pyllmq.LLMQTrainer(ngpu=config.gpus, config=pyllmq.LLamaConfig.from_pretrained(config.model, config.model_dtype),
                                             options=options, batch_size=config.batch_size, seq_len=config.seq_len, grad_accum=config.grad_accumulation,
                                             memcpy_all_gather=config.memcpy_all_gather, memcpy_send_recv=config.memcpy_send_recv)
                print(f"Loading checkpoint from step {latest_step}...")
                trainer.load_checkpoint(ckpt_dir, latest_step)
            else:
                print("No checkpoint found")
                exit(1)
        elif config.from_scratch:
            print(f"Creating {config.model} from scratch...")
            trainer = pyllmq.LLMQTrainer(ngpu=config.gpus, config=pyllmq.LLamaConfig.from_name(config.model, config.model_dtype),
                                         options=options, batch_size=config.batch_size, seq_len=config.seq_len, grad_accum=config.grad_accumulation,
                                         memcpy_all_gather=config.memcpy_all_gather, memcpy_send_recv=config.memcpy_send_recv)
            trainer.init_weights()
            latest_step = 0
        else:
            print(f"Loading model from {config.model}...")
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

        if config.steps <= 0:
            config.steps = steps_per_epoch

        # Setup learning rate schedule
        lr_schedule = pyllmq.CosineLRSchedule(
            config.learning_rate,
            config.steps,
            config.warmup_steps,
            config.learning_rate * config.final_lr_fraction
        )

        # Log allocator stats
        for idx in range(config.gpus):
            logger.log_allocator(trainer.get_allocator_info(idx))

        # calculate the expected time at peak flops for speed-of-light estimation
        logger.set_expected_time_per_token(trainer)

        print(f"\nStarting training from step {latest_step}...\n")
        print(f"Total batch size: {total_batch_size} tokens")
        print(f"Steps per epoch: {steps_per_epoch}")

        run_training_loop(config, trainer, eval_loader, train_loader, latest_step, logger,
                          lr_schedule)

        # Save final model
        print(f"Saving model to {config.out_dir}...")
        trainer.export_model(str(out_dir))
        print("done")

        print(f"\nTraining complete! Logs saved to {out_dir / config.log_file}")


def run_training_loop(config: pyllmq.TrainingConfig, trainer: pyllmq.LLMQTrainer, eval_loader: pyllmq.DataLoader, train_loader: pyllmq.DataLoader,
                      latest_step: int, logger: pyllmq.TrainingRunLogger, lr_schedule: pyllmq.CosineLRSchedule):
    # preload first batch
    in_tokens = np.empty((config.gpus * config.batch_size, config.seq_len), dtype=np.int32)
    out_tokens = np.empty((config.gpus * config.batch_size, config.seq_len), dtype=np.int32)
    train_loader.load_batch(in_tokens, out_tokens)

    # Training loop
    for step in range(latest_step, config.steps):
        if not train_loader.has_next(config.grad_accumulation):
            train_loader.advance_epoch()
            train_loader.load_batch(in_tokens, out_tokens)

        run_eval = False
        if config.eval_every > 0 and step % config.eval_every == 0 and step > latest_step:
            run_eval = True

        # Checkpointing
        if config.ckpt_interval > 0 and step % config.ckpt_interval == 0 and step > latest_step:
            print(f"Saving checkpoint to {config.checkpoint_dir}...")
            start_time = time.time()
            ckpt_path = trainer.save_checkpoint(config.checkpoint_dir, step)
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.log_checkpoint(step, ckpt_path, elapsed_ms)
            print("done")

            # Clean old checkpoints
            if config.ckpt_keep_n > 0:
                removed = pyllmq.clean_old_checkpoints(config.checkpoint_dir, config.ckpt_keep_n, config.ckpt_major)
                if removed:
                    print(f"Cleaned {len(removed)} checkpoints")

        # Run evaluation
        if run_eval:
            val_loss, elapsed_ms = run_evaluation(trainer, eval_loader, in_tokens, out_tokens, config.eval_num_steps)
            epoch = train_loader.epoch() + 0.01 * train_loader.progress()
            eval_tokens = config.eval_num_steps * config.batch_size * config.seq_len * config.gpus
            logger.log_eval(step, epoch, eval_tokens, elapsed_ms, val_loss)

        # Training step
        step_start = time.time()

        for micro_step in range(config.grad_accumulation):
            trainer.step(in_tokens, out_tokens)
            if train_loader.has_next():
                train_loader.load_batch(in_tokens, out_tokens)

        # Log GPU info
        if config.log_gpu_util > 0 and step % config.log_gpu_util == 0:
            infos = trainer.get_gpu_info()
            for i, info in enumerate(infos):
                logger.log_gpu_state(step, i, info)

        # Optimizer update
        lr = lr_schedule.get_lr(step)
        result = trainer.update(lr, config.beta_1, config.beta_2, step + 1,
                                config.weight_decay, config.grad_clip)

        step_time = time.time() - step_start
        elapsed_ms = int(step_time * 1000)

        # Log step
        tokens_processed = config.batch_size * config.seq_len * config.grad_accumulation * config.gpus
        epoch = train_loader.epoch() + 0.01 * train_loader.progress()
        logger.log_step(step, epoch, tokens_processed, elapsed_ms,
                        result['norm'], result['loss'], lr)

    # Final evaluation
    print("\nRunning final evaluation...")
    final_loss, _ = run_evaluation(trainer, eval_loader, in_tokens, out_tokens, eval_loader.num_chunks)
    print(f"Final validation loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
