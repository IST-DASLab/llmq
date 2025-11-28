#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers", "datasets", "numpy"]
# ///

# Note: naming this script tokenize.py *breaks* transformers
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
import datasets
import numpy as np
import multiprocessing as mp
import argparse


worker_tokenizer = None


def load_or_create_dataset(name: str):
    if name == "tiny-shakespeare":
        return datasets.load_dataset("Trelis/tiny-shakespeare")
    elif name == "tiny-stories":
        return datasets.load_dataset("roneneldan/TinyStories")
    elif name == "gsm8k":
        return datasets.load_dataset("openai/gsm8k", "main")
    elif name == "fineweb-1b":
        return datasets.load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True)
    elif name in ["climb-1b", "climb-10b", "climb-20b"]:
        cluster_datasets = []
        cluster_sizes = [86, 121, 145, 386, 189, 1778, 1673, 117, 81, 734, 156, 2568, 90, 28, 23, 728, 702, 227, 116, 51]
        cluster_sizes = [c / sum(cluster_sizes) for c in cluster_sizes]
        for cluster_id in range(1, 21):  # clusters 1-20
            cluster_ds = datasets.load_dataset(
                "gvlassis/ClimbMix",
                f"cluster_id={cluster_id}",
                split="train",
                streaming=True
            )
            cluster_datasets.append(cluster_ds)
        mixed_dataset = datasets.interleave_datasets(
            cluster_datasets,
            probabilities=cluster_sizes,
            seed=42,
            stopping_strategy="all_exhausted"
        )

        return {"train": mixed_dataset}

    elif name == "fineweb-10b":
        return datasets.load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True)
    elif name == "limo":
        return datasets.load_dataset("GAIR/LIMO")["train"].train_test_split(test_size=40, seed=42)

    raise ValueError(f"unknown dataset {name}")

class TokenizedDataFileWriter:
    def __init__(self, file_name: str,  vocab_size: int, masking: bool = False):
        self.file_name = file_name
        self.file_handle = None
        self.n_tokens = 0
        self.vocab_size = vocab_size
        self.has_masks = masking
        self.mask_list = []
        self.mask_rest = None

    def __enter__(self):
        self.file_handle = open(self.file_name, "wb+")
        # reserve space for the file header
        self.file_handle.write(('*' * 1023 + '\n').encode("ascii"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.has_masks:
            self._write_masks()
        self._write_header()
        self.file_handle.close()
        self.file_handle = None

    def add_document(self, tokens: np.ndarray, mask: Optional[np.ndarray] = None):
        assert self.file_handle is not None
        if mask is not None and self.has_masks is False:
            raise ValueError("Cannot add masking to a file that was not created with masking enabled")
        elif mask is None and self.has_masks is True:
            raise ValueError("Cannot add maskless tokens to a file that was created with masking enabled")

        tokens = np.array(tokens , dtype=np.int32)
        assert tokens.ndim == 1

        if mask is not None:
            self._record_mask(mask)

        self.file_handle.write(tokens.tobytes())
        self.n_tokens += len(tokens)
        if self.n_tokens >= 2**31:
            raise RuntimeError("cannot have more than 2**31 tokens in a single file")

    def _record_mask(self, mask: np.ndarray):
        mask = mask.astype(np.bool)
        if self.mask_rest is not None:
            full_mask = np.concatenate([self.mask_rest, mask], dtype=np.bool)
        else:
            full_mask = mask

        full_bytes = len(full_mask) // 8 * 8
        mask_bytes = full_mask[:full_bytes]
        self.mask_rest = full_mask[full_bytes:]
        self.mask_list.append(np.packbits(mask_bytes, bitorder='little'))

    def _write_masks(self):
        if self.mask_rest is not None and len(self.mask_rest) > 0:
            self.mask_list.append(np.packbits(self.mask_rest, bitorder='little'))
        for part in self.mask_list:
            self.file_handle.write(part.tobytes())

    def _write_header(self):
        assert self.file_handle is not None
        self.file_handle.seek(0)
        header_str = "BIN.TOK\n"  # 8 bytes
        version = 2
        bytes_per_token = 4
        self.file_handle.write(header_str.encode("ascii"))
        self.file_handle.write(np.array([version, bytes_per_token, self.n_tokens, self.vocab_size, self.has_masks], dtype=np.int32).tobytes())
        self.file_handle.seek(256*4)


def init_worker(model_name_arg):
    from transformers import AutoTokenizer
    global worker_tokenizer
    worker_tokenizer = AutoTokenizer.from_pretrained(model_name_arg)


def tokenize_example_worker(args: tuple) -> dict:
    example, key_func, seq_len = args
    """Worker function for multiprocessing"""
    if callable(key_func):
        example = key_func(example)
    else:
        example = example[key_func]
    if isinstance(example, str):
        tokens = worker_tokenizer(example, return_tensors='np', split_special_tokens=True).input_ids[0, ...]
        tokens = np.concatenate([tokens, [worker_tokenizer.eos_token_id]])
        return {"tokens": tokens}
    elif isinstance(example, tuple):
        assert len(example) == 2
        prompt = worker_tokenizer(example[0], return_tensors='np', split_special_tokens=True).input_ids[0, ...]
        response = worker_tokenizer(example[1], return_tensors='np', split_special_tokens=True).input_ids[0, ...]
        tokens = np.concatenate([prompt, response, [worker_tokenizer.eos_token_id]])
        mask = np.concatenate([np.zeros_like(prompt), np.ones_like(response)])
        if seq_len is not None:
            if len(tokens) > seq_len:
                # truncate, but ensure last token remains EOS
                tokens = tokens[:seq_len]
                tokens[seq_len-1] = worker_tokenizer.eos_token_id
                mask = mask[:seq_len]
                mask[seq_len-1] = 1
            else:
                tokens = np.pad(tokens, (0, seq_len - len(tokens)), mode="constant")
                mask = np.pad(mask, (0, seq_len - len(mask)), mode="constant")
    else:
        raise ValueError(f"unknown example type {type(example)}")
    return {"tokens": tokens, "mask": mask}

def process_single_file(ds_iter, key, file_name: str, vocab_size: int, max_tokens: int, masking: bool, seq_len: int, pool) -> tuple[bool, int]:
    def example_generator(f: TokenizedDataFileWriter):
        try:
            while f.n_tokens < max_tokens:
                yield next(ds_iter), key, seq_len
        except StopIteration:
            return

    has_more_data = False
    with TokenizedDataFileWriter(file_name, vocab_size, masking=masking) as f:
        tokenized_examples = pool.imap(
            tokenize_example_worker,
            example_generator(f),
            chunksize=256
        )

        for tokens in tokenized_examples:
            f.add_document(**tokens)

            if f.n_tokens >= max_tokens:
                has_more_data = True
                break

        return has_more_data, f.n_tokens


def process_dataset(file_name: str, ds, tokenizer, key, split_name: str, max_tokens: int = None, *, model_name: str, is_tiny: bool = False, first_is_eval: int = -1, masking: bool = False, seq_len: int = None):
    num_processes = max(1, min(mp.cpu_count() // 2, 8))

    ds_iter = iter(ds)
    file_index = 0
    total_tokens = 0
    max_size = 100_000_000
    if max_tokens and max_tokens < max_size:
        max_size = max_tokens

    Path(f"{file_name}").mkdir(parents=True, exist_ok=True)

    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(model_name,)) as pool:
        while True:
            if first_is_eval > 0:
                if file_index == 0:
                    max_size = first_is_eval
                    output_filename = f"{file_name}/eval.bin"
                else:
                    max_size = 100_000_000
                    output_filename = f"{file_name}/{split_name}-{file_index-1:03d}.bin"
            elif is_tiny:
                output_filename = f"{file_name}/{split_name}.bin"
            else:
                output_filename = f"{file_name}/{split_name}-{file_index:03d}.bin"

            has_more_data, tokens_written = process_single_file(ds_iter, key, output_filename, tokenizer.vocab_size, max_size, masking, seq_len, pool)

            total_tokens += tokens_written
            print(f"Completed file {output_filename} with {tokens_written:,} tokens")
            file_index += 1

            if not has_more_data:
                break

            if max_tokens and total_tokens >= max_tokens:
                break

def _extract_gsm8k_example(example):
    return example["question"], example["answer"]


def _extract_limo_example(example):
    return example["question"], example["solution"]


def generate_tokenized_dataset(dataset: str, model: str, out_dir: str = "data"):
    subsample = None
    is_tiny = False
    masking = False
    seq_len = None

    if dataset == "tiny-shakespeare":
        dst = "tiny-shakespeare"
        key = "Text"
        test_split = "test"
        is_tiny = True
    elif dataset == "tiny-stories":
        dst = "tiny-stories"
        key = "text"
        test_split = "validation"
    elif dataset == "gsm8k":
        dst = "gsm8k"
        key = _extract_gsm8k_example
        test_split = "test"
        is_tiny = True
        masking = True
        seq_len = 512
    elif dataset == "fineweb-1b":
        dst = "fineweb-1b"
        key = "text"
        test_split = 10_000_000
        subsample = 1_000_000_000
    elif dataset == "fineweb-10b":
        dst = "fineweb-10b"
        key = "text"
        test_split = 10_000_000
    elif dataset == "climb-1b":
        dst = "climb-1b"
        key = "text"
        test_split = 10_000_000
        subsample = 1_000_000_000
    elif dataset == "climb-10b":
        dst = "climb-10b"
        key = "text"
        test_split = 10_000_000
        subsample = 10_000_000_000
    elif dataset == "climb-20b":
        dst = "climb-20b"
        key = "text"
        test_split = 20_000_000
        subsample = 20_000_000_000
    elif dataset == "limo":
        dst = "limo"
        key = _extract_limo_example
        test_split = "test"
        is_tiny = True
        masking = True
        seq_len = 8192
    else:
        assert False, f"unknown dataset {dataset}"

    if model.lower() == "qwen":
        model_name = "Qwen/Qwen2.5-0.5B"
        dst += "-qwen"
    elif model.lower() == "llama":
        model_name = "unsloth/llama-2-7b"
        dst += "-llama"
    else:
        assert False, f"unknown model {model}"

    d = load_or_create_dataset(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dst = out_dir + "/" + dst

    if isinstance(test_split, int):
        process_dataset(dst, d["train"], tokenizer, key, "train", subsample, is_tiny=is_tiny, first_is_eval=test_split, model_name=model_name, masking=masking, seq_len=seq_len)
    else:
        process_dataset(dst, d["train"], tokenizer, key, "train", subsample, is_tiny=is_tiny, model_name=model_name, masking=masking, seq_len=seq_len)
        process_dataset(dst, d[test_split], tokenizer, key, "eval", None, is_tiny=True, model_name=model_name, masking=masking, seq_len=seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="data")

    args = parser.parse_args()
    generate_tokenized_dataset(dataset=args.dataset, model=args.model, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
