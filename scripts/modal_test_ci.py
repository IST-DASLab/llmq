#!/usr/bin/env python3
"""
Run tests on already-deployed Modal app without rebuilding.

Usage:
    python run_modal_tests.py [test args...]
"""
import sys
import modal


if __name__ == "__main__":
    # Reference the already-deployed app
    app = modal.App.lookup("llmq-test", create_if_missing=False)

    test_name = sys.argv[1]

    if test_name == "recompute":
        # Get the run_recompute_test function from the deployed app
        test_fn = modal.Function.from_name("llmq-test", "run_recompute_test")
        test_args = sys.argv[2:]
    elif test_name == "fixed":
        test_fn = modal.Function.from_name("llmq-test", "run_fixed_result_test")
        test_args = sys.argv[2]
    else:
        raise RuntimeError(f"Unknown test type {test_name}")

    # Get test arguments from command line

    print(f"Launching Modal test with args: {test_args}")
    result = test_fn.remote(test_args)

    # Print the comparison report
    print("\n" + result["report"])

    if not result["passed"]:
        sys.exit(1)
