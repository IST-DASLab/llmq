#!/usr/bin/env python3
"""
Deploy locally-built pyllmq wheel to Modal and run recomputation tests.

Usage:
    modal run modal_test_app.py [-- test args...]
"""
import io
import sys
from pathlib import Path
import modal


def create_image(cuda_version: str = "12.8.1"):
    # Check where we can find some wheels
    wheelhouse_exists = Path("wheelhouse").exists() and list(Path("wheelhouse").glob("pyllmq*.whl"))
    dist_exists = Path("dist").exists() and list(Path("dist").glob("pyllmq*.whl"))

    if wheelhouse_exists:
        wheel_files = list(Path("wheelhouse").glob("pyllmq*.whl"))
    elif dist_exists:
        wheel_files = list(Path("dist").glob("pyllmq*.whl"))
    else:
        raise FileNotFoundError("No wheels found in wheelhouse/ or dist/")

    if len(wheel_files) > 1:
        print("Error: multiple wheels found:", file=sys.stderr)
        for wheel_file in wheel_files:
            print(f"  {wheel_file}", file=sys.stderr)
        sys.exit(1)

    wheel_file = wheel_files[0]
    print(f"Using wheel: {wheel_file}")

    """Create Modal image with the wheel and dependencies."""
    return (
        modal.Image.from_registry(f"nvidia/cuda:{cuda_version}-cudnn-devel-ubuntu24.04", add_python="3.12")
        .uv_pip_install("huggingface_hub", "transformers", "datasets", "numpy")
        # install dependencies for pyllmq, so that rebuild steps are faster
        .uv_pip_install("nvidia-cuda-runtime-cu12>=12.8", "nvidia-cudnn-cu12>=9.0", "nvidia-nccl-cu12>=2.0", "nvidia-cublas-cu12>=12.8")
        .run_commands("hf download Qwen/Qwen2.5-0.5B")
        .add_local_file("scripts/tokenize_data.py", "/root/tokenize_data.py", copy=True)
        .run_commands(
            "uv run /root/tokenize_data.py --dataset tiny-shakespeare --model qwen --out-dir /root/data"
        )
        .add_local_file(str(wheel_file), f"/tmp/{wheel_file.name}", copy=True)
        .pip_install(f"/tmp/{wheel_file.name}")
    )


if modal.is_local():
    image = create_image()
else:
    image = modal.Image.debian_slim()


app = modal.App("llmq-test")


@app.function(
    gpu="L4",
    memory=8192,
    timeout=300,
    image=image,
)
def run_recompute_test(test_args: list[str]):
    """Run recomputation tests on Modal."""
    from pyllmq.tests.recompute import parse_args, RecomputeTestRunner

    # Parse test arguments into config
    config = parse_args(test_args)
    config.train_file = "/root/data/tiny-shakespeare-qwen/train.bin"

    # Run test
    runner = RecomputeTestRunner(config)
    result = runner.run_test()
    report_buffer = io.StringIO()
    result.print_comparison(file=report_buffer)
    report = report_buffer.getvalue()

    return {
        "passed": result.passed,
        "report": report,
    }


@app.local_entrypoint()
def main(*test_args: str):
    """Run tests on Modal."""
    print(f"Launching Modal test with args: {test_args}")
    result = run_recompute_test.remote(list(test_args))

    # Print the comparison report
    print("\n" + result["report"])

    if not result["passed"]:
        sys.exit(1)
