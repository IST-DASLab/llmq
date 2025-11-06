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

    # Get the run_recompute_test function from the deployed app
    run_recompute_test = modal.Function.from_name("llmq-test", "run_recompute_test")

    # Get test arguments from command line
    test_args = sys.argv[1:]

    print(f"Launching Modal test with args: {test_args}")
    result = run_recompute_test.remote(test_args)

    # Print the comparison report
    print("\n" + result["report"])

    if not result["passed"]:
        sys.exit(1)
