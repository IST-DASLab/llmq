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
            "losses": [3.6896581649780273, 3.3729734420776367, 3.4653379917144775, 3.442567825317383, 3.2096214294433594, 3.5971217155456543, 3.3238463401794434, 3.1644277572631836, 3.339334011077881, 3.159609317779541, 2.940225601196289],
            "norms": [7.225314140319824, 7.687681198120117, 5.967431545257568, 6.445042133331299, 6.517184734344482, 6.523508071899414, 5.884573936462402, 5.424758434295654, 5.607821941375732, 4.924382209777832],
        }
    elif dtype == "e5m2":
        expected = {
            "losses": [3.6896581649780273, 3.374328136444092, 3.4656238555908203, 3.442816734313965, 3.212712287902832, 3.597747325897217, 3.326484203338623, 3.1685662269592285, 3.3410708904266357, 3.1599009037017822, 2.9377570152282715],
            "norms": [7.1560797691345215, 7.673510551452637, 5.905216693878174, 6.354756832122803, 6.574455738067627, 6.515244483947754, 5.910398960113525, 5.479896545410156, 5.57126522064209, 4.886902332305908],
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
