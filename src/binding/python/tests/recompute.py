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

import copy
import sys

from pyllmq.training import TrainingConfig
from pyllmq.tests.run import run_training, parse_args, compare_results


def disable_recompute(config: TrainingConfig):
    baseline_config = copy.deepcopy(config)
    baseline_config.recompute_swiglu = False
    baseline_config.recompute_rms_norm = False
    baseline_config.recompute_ffn = False
    baseline_config.recompute_qkv = False
    baseline_config.recompute_att = False
    baseline_config.recompute_block = False
    baseline_config.use_cuda_graphs = False
    baseline_config.offload_master = False
    baseline_config.offload_quants = False
    baseline_config.offload_opt_v = False
    baseline_config.offload_opt_m = False
    baseline_config.offload_grads = False
    baseline_config.attn_bwd_chunks = 1
    baseline_config.memcpy_all_gather = False
    baseline_config.shard_weights = False
    baseline_config.persistent_quants = False
    return baseline_config


def main():
    """Main entry point for command-line usage."""
    config = parse_args()
    test_run = run_training(config)
    ref_run = run_training(disable_recompute(config))
    sys.exit(0 if compare_results(test_run, ref_run) else -1)


if __name__ == "__main__":
    main()
