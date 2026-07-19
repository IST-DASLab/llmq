"""End-to-end multi-GPU gradient parity tests.

Each case runs torch_reference.py in a subprocess (fresh NCCL state per case) on the
tiny-qwen3 test model, comparing llmq gradients on 2 GPUs against a single-GPU torch
reference. Requires 2 GPUs and the tiny test model
(create with `scripts/create_tiny_test_model.py --arch qwen3`).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "src" / "binding" / "python" / "tests" / "torch_reference.py"


def _tiny_model_available() -> bool:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return False
    return (Path(HF_HUB_CACHE) / "models--test--tiny-qwen3").exists()


pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs"),
    pytest.mark.skipif(not _tiny_model_available(),
                       reason="tiny-qwen3 missing; run scripts/create_tiny_test_model.py"),
]


@pytest.mark.parametrize("extra", [
    pytest.param([], id="data-parallel"),
    pytest.param(["--shard-weights"], id="shard-weights"),
    pytest.param(["--shard-gradients"], id="shard-gradients"),
    pytest.param(["--shard-weights", "--shard-gradients"], id="shard-both"),
])
def test_two_gpu_gradient_parity(extra):
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--model", "test/tiny-qwen3", "--gpus", "2",
         "--seq-len", "512", "--grad-accum", "2",
         "--model-dtype", "bf16", "--matmul-dtype", "bf16",
         *extra],
        cwd=REPO_ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
