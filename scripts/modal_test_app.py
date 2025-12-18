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
        # install dependencies for pyllmq, so that rebuild steps are faster
        .uv_pip_install("huggingface_hub", "transformers", "datasets", "numpy", "torch", "accelerate",
                        "nvidia-cuda-runtime-cu12>=12.8", "nvidia-cudnn-cu12>=9.0", "nvidia-nccl-cu12>=2.0", "nvidia-cublas-cu12>=12.8")
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


def compare_and_create_report(result, expected):
    from pyllmq.tests.run import compare_results
    report_buffer = io.StringIO()
    passed = compare_results(result, expected, file=report_buffer)
    report = report_buffer.getvalue()
    return {
        "passed": passed,
        "report": report,
    }


@app.function(
    gpu="L4",
    memory=8192,
    timeout=300,
    image=image,
)
def run_recompute_test(test_args: list[str]):
    """Run recomputation tests on Modal."""
    from pyllmq.tests.run import parse_args, run_training
    from pyllmq.tests.recompute import disable_recompute

    # Parse test arguments into config
    config = parse_args(test_args)
    config.train_file = "/root/data/tiny-shakespeare-qwen/train.bin"
    config.model = "Qwen/Qwen2.5-0.5B"

    test_run = run_training(config)
    ref_run = run_training(disable_recompute(config))
    return compare_and_create_report(test_run, ref_run)


def run_with_config(test_args: list[str]):
    from pyllmq.tests.run import parse_args, run_training
    config = parse_args(test_args)
    config.train_file = "/root/data/tiny-shakespeare-qwen/train.bin"
    config.model = "Qwen/Qwen2.5-0.5B"
    result = run_training(config)
    return result


@app.function(
    gpu="L4",
    memory=8192,
    timeout=300,
    image=image,
)
def run_fixed_result_test(dtype: str = "bf16"):
    from pyllmq.tests.run import RunResult

    print(f"Launching Modal fixed_result test with dtype: {dtype}")

    if dtype == "e5m2":
        args = [f"--matmul-dtype=e4m3", "--gradient-dtype=e5m2"]
    else:
        args = [f"--matmul-dtype={dtype}"]

    """Run tests on Modal."""
    result = run_with_config(args)
    if dtype == "bf16":
        expected = {
            "losses": [3.6691675186157227, 3.338416576385498, 3.4413886070251465, 3.4134111404418945, 3.1846399307250977, 3.581113338470459, 3.3045778274536133, 3.14133358001709, 3.3190853595733643, 3.14133882522583, 2.9201550483703613],
            "norms": [7.181778430938721, 6.117193698883057, 5.926026344299316, 5.785313129425049, 6.193798065185547, 6.783796310424805, 5.854440689086914, 5.702835559844971, 5.7921528816223145, 4.882175922393799],
        }
    elif dtype == "e4m3":
        expected = {
            "losses": [3.6896581649780273, 3.350105047225952, 3.4565865993499756, 3.4244627952575684, 3.1961753368377686, 3.5905539989471436, 3.314474582672119, 3.1519875526428223, 3.328141689300537, 3.1484572887420654, 2.935676097869873],
            "norms": [7.225314140319824, 6.242846965789795, 6.185355186462402, 5.921780586242676, 6.2074384689331055, 6.655130863189697, 5.875917911529541, 5.7703142166137695, 5.688600063323975, 4.981103897094727],
        }
    elif dtype == "e5m2":
        expected = {
            "losses": [3.6896581649780273, 3.352154016494751, 3.4541091918945312, 3.4255144596099854, 3.194918394088745, 3.591022491455078, 3.3119752407073975, 3.1553726196289062, 3.326538562774658, 3.150322675704956, 2.9374852180480957],
            "norms": [7.1560797691345215, 6.176127910614014, 6.131178855895996, 5.944278240203857, 6.135629653930664, 6.733249187469482, 5.816003322601318, 5.759800910949707, 5.704397201538086, 4.93273401260376],
        }
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    report = compare_and_create_report(result, RunResult(**expected))
    if not report["passed"]:
        import json
        import dataclasses
        # this helps with debugging/updating in case of failure
        print(json.dumps(dataclasses.asdict(result)))
    return report


@app.function(
    gpu="L4",
    memory=8192,
    timeout=300,
    image=image,
)
def run_torch_compare_step(test_args: list):
    from pyllmq.tests.torch_reference import compare_single_step
    from pyllmq.tests.run import parse_args

    config = parse_args(test_args)
    config.train_file = "/root/data/tiny-shakespeare-qwen/train.bin"
    config.model = "Qwen/Qwen2.5-0.5B"
    report_buffer = io.StringIO()
    passed = compare_single_step(config, file=report_buffer)
    report = report_buffer.getvalue()
    return {
        "passed": passed,
        "report": report,
    }


@app.local_entrypoint()
def test_recompute(*test_args: str):
    print(f"Launching Modal recomputation test with args: {test_args}")
    result = run_recompute_test.remote(list(test_args))

    # Print the comparison report
    print("\n" + result["report"])

    if not result["passed"]:
        sys.exit(1)


@app.local_entrypoint()
def test_torch_step(*test_args: str):
    print(f"Launching Modal torch_compare_step test with args: {test_args}")
    result = run_torch_compare_step.remote(list(test_args))
    print("\n" + result["report"])
    if not result["passed"]:
        sys.exit(1)


@app.local_entrypoint()
def test_fixed(dtype: str = "bf16"):
    print(f"Launching Modal test with dtype: {dtype}")
    result = run_fixed_result_test.remote(dtype)

    # Print the comparison report
    print("\n" + result["report"])

    if not result["passed"]:
        sys.exit(1)
