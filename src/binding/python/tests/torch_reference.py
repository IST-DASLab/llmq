#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyllmq", "numpy", "torch", "transformers"]
# ///

import numpy as np
import torch
import transformers

import pyllmq
from pyllmq.tests.run import parse_args, _create_options
from pyllmq.training import TrainingConfig


def torch_grad_one_step(config: TrainingConfig):
    torch_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model, device_map="cuda", torch_dtype=torch.float32)

    data_loader = pyllmq.DataLoader(
        [config.train_file],
        config.batch_size * config.seq_len,
        seed=0x83b45442
    )

    in_tokens = np.empty((config.batch_size, config.seq_len), dtype=np.int32)
    out_tokens = np.empty((config.batch_size, config.seq_len), dtype=np.int32)

    result = {}

    torch_model.zero_grad()
    for j in range(config.grad_accumulation):
        data_loader.load_batch(in_tokens, out_tokens)
        logits = torch_model(torch.tensor(in_tokens).to("cuda")).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits[0].shape[-1]).to(torch.float32),
                                                 torch.tensor(out_tokens).to("cuda").to(torch.int64).reshape(-1),
                                                 reduction="none")
        loss = loss.reshape(out_tokens.shape)
        loss = loss.sum() / (out_tokens.shape[0] * out_tokens.shape[1] * config.grad_accumulation)
        loss.backward()

    for k, v in torch_model.named_parameters():
        result[k] = v.grad.to("cpu").detach().numpy()

    return result


def llmq_grad_one_step(config: TrainingConfig):
    options = _create_options(config)

    # Create trainer
    trainer = pyllmq.LLMQTrainer.from_pretrained(
        name=config.model,
        ngpu=config.gpus,
        dtype=config.model_dtype,
        options=options,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        grad_accum=config.grad_accumulation,
        memcpy_all_gather=config.memcpy_all_gather,
        memcpy_send_recv=config.memcpy_send_recv
    )

    # Create data loader
    train_loader = pyllmq.DataLoader(
        [config.train_file],
        config.batch_size * config.seq_len,
        seed=0x83b45442
    )

    # Prepare input/output buffers
    in_tokens = np.empty((config.batch_size, config.seq_len), dtype=np.int32)
    out_tokens = np.empty((config.batch_size, config.seq_len), dtype=np.int32)

    for j in range(config.grad_accumulation):
        train_loader.load_batch(in_tokens, out_tokens)
        trainer.step(in_tokens, out_tokens)
    return {k: torch.from_dlpack(v).cpu().to(torch.float32).numpy() for k, v in trainer.get_gradients(0).items()}


def compare_single_step(config, file=None):
    config.steps = 1

    torch_grads = torch_grad_one_step(config)
    torch.cuda.empty_cache()
    # fuse gradients so torch matches llmq
    fused_torch_grads = {}
    for k, v in torch_grads.items():
        if "up_proj" in k:
            up = v
            gate = torch_grads[k.replace("up_proj", "gate_proj")]
            fused_torch_grads[k.replace("up_proj", "up")] = np.concat([up, gate], axis=0)
        elif "q_proj" in k:
            q_ = v
            k_ = torch_grads[k.replace("q_proj", "k_proj")]
            v_ = torch_grads[k.replace("q_proj", "v_proj")]
            fused_torch_grads[k.replace("q_proj", "qkv")] = np.concat([q_, k_, v_], axis=0)
        elif "gate_proj" in k:
            pass
        elif "k_proj" in k:
            pass
        elif "v_proj" in k:
            pass
        else:
            fused_torch_grads[k] = v
    torch_grads = sorted(fused_torch_grads.items(), key=lambda x: x[0])
    llmq_grads = sorted(llmq_grad_one_step(config).items(), key=lambda x: x[0])

    passed = 0
    total = 0
    avg_cosing_similarity = 0
    avg_rel_norm_error = 0
    print(f"   {'tensor':40} {'cos':5}  {'norm':5}", file=file)
    for (kt, vt), (kl, vl) in zip(torch_grads, llmq_grads):
        assert kt == kl, (kl, kt)
        vt = vt.flatten()
        vl = vl.flatten()
        nt = np.linalg.norm(vt)
        nl = np.linalg.norm(vl)
        cosine_similarity = np.dot(vt, vl) / (nt * nl)
        rel_norm_error = np.abs(nt - nl) / max(nt, nl)
        short_k = kt.replace("model.", "").replace("layers.", "l.")
        verdict = "\033[1;32m✓\033[0m"
        total += 1
        if cosine_similarity < 0.95 or rel_norm_error > 0.05:
            verdict = "\033[1;31m✗\033[0m"
        else:
            passed += 1

        avg_cosing_similarity += cosine_similarity
        avg_rel_norm_error += rel_norm_error

        print(f" {verdict} {short_k:40} {cosine_similarity:5.3f}  {100*np.abs(nt - nl) / max(nt, nl):5.2f}", file=file)

    avg_cosing_similarity /= total
    avg_rel_norm_error /= total

    print("", file=file)
    result_str = f"{passed} / {total}  cos {avg_cosing_similarity:.3f} norm {100*avg_rel_norm_error:5.2f}"
    if passed > 94 * total // 100 and avg_cosing_similarity > 0.99 and avg_rel_norm_error < 0.012:
        print(f"\033[1;32mPASS\033[0m {result_str}", file=file)
        return True
    else:
        print(f"\033[1;31mFAIL\033[0m {result_str}", file=file)
        return False


def run_compare_single_step():
    config = parse_args()
    return compare_single_step(config)


if __name__ == "__main__":
    exit(0 if run_compare_single_step() else 1)
