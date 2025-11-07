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

from pyllmq.tests.run import RunConfig, run_training, parse_args, compare_results


def disable_recompute(config: RunConfig):
    baseline_config = copy.deepcopy(config)
    baseline_config.recompute_swiglu = False
    baseline_config.recompute_rms_norm = False
    baseline_config.recompute_ffn = False
    baseline_config.recompute_qkv = False
    baseline_config.recompute_att = False
    baseline_config.recompute_block = False
    baseline_config.use_cuda_graphs = False
    return baseline_config


def main():
    """Main entry point for command-line usage."""
    config = parse_args()
    test_run = run_training(config)
    ref_run = run_training(disable_recompute(config))
    sys.exit(0 if compare_results(test_run, ref_run) else -1)


if __name__ == "__main__":
    main()