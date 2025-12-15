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
    options.attn_bwd_chunks = config.attn_bwd_chunks

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
    options.offload_grads = config.offload_grads
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
    pyllmq.add_training_args(parser)

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

        train_loader = pyllmq.DataLoader(train_files, config.batch_size * config.seq_len * config.gpus, seed=config.train_seed)
        eval_loader = pyllmq.DataLoader(eval_files, config.batch_size * config.seq_len * config.gpus, seed=config.eval_seed)

        # Log dataset information
        logger.log_dataset(train_loader, eval_loader)

        total_batch_size = config.batch_size * config.seq_len * config.gpus * config.grad_accumulation
        steps_per_epoch = train_loader.num_tokens // total_batch_size

        # Create trainer
        print("Creating trainer...")
        if config.continue_from_checkpoint:
            latest_step = pyllmq.find_latest_checkpoint(ckpt_dir)
            if latest_step >= 0:
                trainer = pyllmq.LLMQTrainer(ngpu=config.gpus, config=pyllmq.Config.from_pretrained(config.model, config.model_dtype),
                                             options=options, batch_size=config.batch_size, seq_len=config.seq_len, grad_accum=config.grad_accumulation,
                                             memcpy_all_gather=config.memcpy_all_gather, memcpy_send_recv=config.memcpy_send_recv)
                print(f"Loading checkpoint from step {latest_step}...")
                trainer.load_checkpoint(ckpt_dir, latest_step)
            else:
                print("No checkpoint found")
                exit(1)
        elif config.from_scratch:
            print(f"Creating {config.model} from scratch...")
            trainer = pyllmq.LLMQTrainer(ngpu=config.gpus, config=pyllmq.Config.from_name(config.model, config.model_dtype),
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
            logger.log_allocator(trainer, idx)

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
            trainer.save_checkpoint(config.checkpoint_dir, step)
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
