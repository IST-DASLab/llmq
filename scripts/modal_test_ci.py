#!/usr/bin/env python3
"""
Run tests on already-deployed Modal app without rebuilding.

Usage:
    python run_modal_tests.py [test args...]
"""
import argparse
import sys
import modal


def _get_gpu_arg(args: list[str]) -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default="1")
    parsed_args, rest = parser.parse_known_args(args)
    return parsed_args.gpus, rest


if __name__ == "__main__":
    # Reference the already-deployed app
    app = modal.App.lookup("llmq-test", create_if_missing=False)

    test_name = sys.argv[1]
    gpus, rest = _get_gpu_arg(sys.argv[2:])
    test_args_pos = []
    test_args_kw = {}

    if test_name == "recompute":
        # Get the run_recompute_test function from the deployed app
        if gpus == 2:
            test_fn = modal.Function.from_name("llmq-test", "run_recompute_test_x2")
        else:
            test_fn = modal.Function.from_name("llmq-test", "run_recompute_test")
        test_args_pos = [sys.argv[2:]]
    elif test_name == "fixed":
        parser = argparse.ArgumentParser()
        parser.add_argument("dtype", type=str)
        parser.add_argument("--shard-gradient", action="store_true")
        parsed_args, rest = parser.parse_known_args(rest)
        if gpus == 2:
            test_fn = modal.Function.from_name("llmq-test", "run_fixed_result_test_x2")
            test_args_kw = {"dtype": parsed_args.dtype, "shard_gradients": parsed_args.shard_gradient}
        else:
            assert not parsed_args.shard_gradient, "shard_gradient only supported for 2 gpus"
            test_fn = modal.Function.from_name("llmq-test", "run_fixed_result_test")
            test_args_kw = {"dtype": parsed_args.dtype}
    elif test_name == "torch-step":
        test_fn = modal.Function.from_name("llmq-test", "run_torch_compare_step")
        test_args_pos = [sys.argv[2:]]
    else:
        raise RuntimeError(f"Unknown test type {test_name}")

    # Get test arguments from command line

    print(f"Launching Modal test with args: {test_args_pos}, {test_args_kw}")
    result = test_fn.remote(*test_args_pos, **test_args_kw)

    # Print the comparison report
    print("\n" + result["report"])

    if not result["passed"]:
        sys.exit(1)
