"""Gradient-parity checks for the model architectures llmq claims to support.

Each case builds a tiny random-weight model with `scripts/create_tiny_test_model.py`
and compares every parameter gradient after one forward+backward against
transformers; forward-only agreement would not catch a mis-wired backward.

Needs a GPU and tokenized data:
    uv run --extra scripts python scripts/tokenize_data.py --dataset tiny-shakespeare --model llama
    uv run --extra scripts python scripts/tokenize_data.py --dataset tiny-shakespeare --model qwen
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "src" / "binding" / "python" / "tests" / "torch_reference.py"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1"}

# fixture architecture -> tokenizer whose vocabulary it reuses
SUPPORTED = {
    "llama": "llama",
    "llama-tied": "llama",
    "mistral": "llama",
    "qwen3": "qwen",
}

# fixtures llmq must refuse, and the text the error has to mention
REFUSED = {
    "llama-bias": "attention_bias",
    "llama-rope-scaling": "rope_scaling",
}


@pytest.fixture(scope="session")
def tiny_models():
    script = REPO / "scripts" / "create_tiny_test_model.py"
    done = subprocess.run([sys.executable, str(script), "--arch", "all"],
                          capture_output=True, text=True, env=ENV)
    if done.returncode != 0:
        pytest.skip(f"could not create tiny models:\n{done.stderr}")


def _compare(arch: str, tokenizer: str):
    train_file = REPO / "data" / f"tiny-shakespeare-{tokenizer}" / "train.bin"
    if not train_file.exists():
        pytest.skip(f"missing {train_file}; see this module's docstring")
    return subprocess.run(
        [sys.executable, str(REFERENCE),
         "--model", f"test/tiny-{arch}", "--train-file", str(train_file),
         "--seq-len", "256", "--grad-accum", "2",
         # fp32 keeps this about the architecture rather than quantization
         "--model-dtype", "fp32", "--matmul-dtype", "fp32", "--gpus", "1"],
        capture_output=True, text=True, env=ENV)


@pytest.mark.parametrize("arch,tokenizer", sorted(SUPPORTED.items()))
def test_matches_transformers(tiny_models, arch, tokenizer):
    done = _compare(arch, tokenizer)
    assert done.returncode == 0, f"gradients diverge:\n{done.stdout[-3000:]}{done.stderr[-2000:]}"


@pytest.mark.parametrize("arch,expected", sorted(REFUSED.items()))
def test_rejects_unrepresentable(tiny_models, arch, expected):
    done = _compare(arch, "llama")
    assert done.returncode != 0, "should have been refused, but loaded fine"
    assert expected in done.stdout + done.stderr, \
        f"refused, but the error never mentions {expected!r}:\n{done.stderr[-2000:]}"
