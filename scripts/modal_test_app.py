#!/usr/bin/env python3
"""
Deploy locally-built pyllmq wheel to Modal and run recomputation tests.

Usage:
    modal run modal_test_app.py [-- test args...]
"""
import argparse
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


def _run_recompute_test(test_args: list[str]):
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


@app.function(
    gpu="L4",
    memory=8192,
    timeout=300,
    image=image,
)
def run_recompute_test(test_args: list[str]):
    return _run_recompute_test(test_args)


@app.function(
    gpu="L4:2",
    memory=8192,
    timeout=300,
    image=image,
)
def run_recompute_test_x2(test_args: list[str]):
    return _run_recompute_test(test_args)


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
            "losses":  [3.39782977104187, 3.4833269119262695, 3.1216840744018555, 3.3116273880004883, 3.20223331451416, 3.1785783767700195, 3.047924518585205, 3.088921070098877, 3.364147663116455, 2.9013497829437256, 2.617171287536621],
            "norms": [6.54591178894043, 5.693244934082031, 5.692412853240967, 6.199832916259766, 5.243079662322998, 5.554076194763184, 5.778393745422363, 5.7696404457092285, 5.275724411010742, 4.784895896911621],
        }
    elif dtype == "e4m3":
        expected = {
            "losses": [3.4156551361083984, 3.510408401489258, 3.1464829444885254, 3.335897922515869, 3.2293882369995117, 3.1958303451538086, 3.0667757987976074, 3.1137216091156006, 3.386855125427246, 2.920886754989624, 2.6443042755126953],
            "norms":  [6.743361949920654, 6.504960536956787, 5.839552879333496, 6.555571556091309, 5.403191089630127, 5.436037063598633, 5.6087870597839355, 5.9490203857421875, 5.442231178283691, 4.907982349395752],
        }
    elif dtype == "e5m2":
        expected = {
            "losses": [3.4156551361083984, 3.5104832649230957, 3.1469316482543945, 3.3347127437591553, 3.2285878658294678, 3.1958675384521484, 3.0641281604766846, 3.112499713897705, 3.3814735412597656, 2.919327735900879, 2.6460676193237305],
            "norms": [6.693141460418701, 6.478551387786865, 5.733729362487793, 6.543521881103516, 5.360874652862549, 5.400944709777832, 5.6198601722717285, 5.792316436767578, 5.3820109367370605, 4.932021141052246]
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
    gpu="L4:2",
    memory=8192,
    timeout=300,
    image=image,
)
def run_fixed_result_test_x2(dtype: str = "bf16", shard_gradients: bool = False):
    from pyllmq.tests.run import RunResult

    print(f"Launching Modal fixed_result test with dtype: {dtype}")

    if dtype == "e5m2":
        args = [f"--matmul-dtype=e4m3", "--gradient-dtype=e5m2"]
    else:
        args = [f"--matmul-dtype={dtype}"]

    if shard_gradients:
        args += ["--shard-gradients"]

    args += ["--gpus=2"]

    """Run tests on Modal."""
    result = run_with_config(args)
    if dtype == "bf16":
        expected = {
            "losses": [3.4119365215301514, 3.394049882888794, 3.4545254707336426, 3.0694894790649414, 3.007321834564209, 3.3855042457580566, 3.368359088897705, 3.421376943588257, 3.1316380500793457, 3.2092301845550537, 3.01995849609375],
            "norms": [5.42860746383667, 5.231578826904297, 5.656546115875244, 4.69525146484375, 4.644282341003418, 5.210570812225342, 5.396310806274414, 4.417316913604736, 4.4374165534973145, 4.28884220123291],
        }
    elif dtype == "e4m3":
        expected = {
            "losses": [3.4303817749023438, 3.43670392036438, 3.483766555786133, 3.0972299575805664, 3.0326924324035645, 3.409470558166504, 3.3872318267822266, 3.4421865940093994, 3.152552843093872, 3.229149341583252, 3.0453014373779297],
            "norms": [5.8067474365234375, 8.371203422546387, 5.1532464027404785, 4.662567615509033, 4.763641834259033, 4.693724632263184, 5.259921073913574, 4.645272731781006, 4.207671165466309, 4.346331596374512]
        }
    elif dtype == "e5m2":
        expected = {
            "losses": [3.4303817749023438, 3.4341166019439697, 3.4837355613708496, 3.09706711769104, 3.0316996574401855, 3.410259962081909, 3.3873462677001953, 3.441790819168091, 3.1511523723602295, 3.2284598350524902, 3.0418832302093506],
            "norms": [5.7736382484436035, 8.317730903625488, 5.149673938751221, 4.641636371612549, 4.685691833496094, 4.650301933288574, 5.228470325469971, 4.605687618255615, 4.183129787445068, 4.276437759399414],
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


def _get_gpu_arg(args: tuple[str, ...]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default="1")
    parsed_args, _ = parser.parse_known_args(args)
    return parsed_args.gpus


@app.local_entrypoint()
def test_recompute(*test_args: str):
    print(f"Launching Modal recomputation test with args: {test_args}")
    gpus = _get_gpu_arg(test_args)
    if gpus == 2:
        result = run_recompute_test_x2.remote(list(test_args))
    else:
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
def test_fixed(dtype: str = "bf16", gpus: int = 1, shard_gradients: bool = False):
    print(f"Launching Modal test with dtype: {dtype}")
    if  gpus == 2:
        result = run_fixed_result_test_x2.remote(dtype, shard_gradients)
    else:
        assert shard_gradients == False, "shard_gradient only supported for 2 gpus"
        result = run_fixed_result_test.remote(dtype)

    # Print the comparison report
    print("\n" + result["report"])

    if not result["passed"]:
        sys.exit(1)
