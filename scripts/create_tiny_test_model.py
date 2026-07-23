# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers"]
# ///
"""Create a tiny random-weight model in the local HF cache for integration tests.

The generated model lands under models--test--tiny-<arch> in the HF hub cache, so
both `transformers` (with HF_HUB_OFFLINE=1) and llmq can load it as `test/tiny-<arch>`.

The vocabulary is chosen to match one of the tokenizers that `tokenize_data.py`
supports, so an existing tokenized dataset can be reused:
  * qwen3            -> data/tiny-shakespeare-qwen
  * llama* / mistral -> data/tiny-shakespeare-llama
"""
import argparse
from pathlib import Path

import torch
import transformers

# tiny-shakespeare-llama is tokenized with the llama-2 tokenizer
LLAMA_VOCAB = 32000
QWEN_VOCAB = 151936


def qwen3_config():
    # head_dim != hidden_size / num_attention_heads, to exercise the decoupled path
    return transformers.Qwen3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=2048,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        vocab_size=QWEN_VOCAB,
        bos_token_id=151643,
        eos_token_id=151645,
        torch_dtype=torch.bfloat16,
    )


def _llama_config(*, attention_bias: bool, tie_word_embeddings: bool):
    # 4 heads over 256 channels gives head_dim 64; the cuDNN attention backend
    # rejects the head_dim 32 that 8 heads would produce.
    return transformers.LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=10_000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=tie_word_embeddings,
        vocab_size=LLAMA_VOCAB,
        bos_token_id=1,
        eos_token_id=2,
        attention_bias=attention_bias,
        mlp_bias=False,
        torch_dtype=torch.bfloat16,
    )


def llama_config():
    return _llama_config(attention_bias=False, tie_word_embeddings=False)


def llama_bias_config():
    # negative fixture: attention_bias also biases o_proj, which we cannot represent
    return _llama_config(attention_bias=True, tie_word_embeddings=False)


def llama_tied_config():
    return _llama_config(attention_bias=False, tie_word_embeddings=True)


def llama_rope_scaling_config():
    # negative fixture: Llama-3.1 style scaling, where we only implement plain rope_theta
    config = _llama_config(attention_bias=False, tie_word_embeddings=False)
    config.rope_scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 1024,
    }
    return config


def mistral_config():
    # sliding_window must stay disabled; llmq rejects an active one
    return transformers.MistralConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=2048,
        rope_theta=10_000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        vocab_size=LLAMA_VOCAB,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=None,
        torch_dtype=torch.bfloat16,
    )


CONFIGS = {
    "qwen3": qwen3_config,
    "llama": llama_config,
    "llama-bias": llama_bias_config,
    "llama-tied": llama_tied_config,
    "llama-rope-scaling": llama_rope_scaling_config,
    "mistral": mistral_config,
}


def create(arch: str, seed: int = 42) -> Path:
    torch.manual_seed(seed)
    config = CONFIGS[arch]()
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    from huggingface_hub.constants import HF_HUB_CACHE
    hub = Path(HF_HUB_CACHE)
    base = hub / f"models--test--tiny-{arch}"
    snapshot = base / "snapshots" / "main"
    snapshot.mkdir(parents=True, exist_ok=True)
    (base / "refs").mkdir(exist_ok=True)
    (base / "refs" / "main").write_text("main")

    model.save_pretrained(snapshot)
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=[*sorted(CONFIGS), "all"], default="qwen3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arches = sorted(CONFIGS) if args.arch == "all" else [args.arch]
    for arch in arches:
        snapshot = create(arch, args.seed)
        print(f"saved test/tiny-{arch} to {snapshot}")


if __name__ == "__main__":
    main()
