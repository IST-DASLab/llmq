# Note: naming this script tokenize.py *breaks* transformers
from pathlib import Path

from transformers import AutoTokenizer
import datasets
import numpy as np
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--model")
parser.add_argument("--out-dir", default="data")

args = parser.parse_args()
version = None
worker_tokenizer = None
subsample = None
is_tiny = False

if args.dataset == "tiny-shakespeare":
    dst = "tiny-shakespeare"
    key = "Text"
    test_split = "test"
    is_tiny = True
elif args.dataset == "tiny-stories":
    dst = "tiny-stories"
    key = "text"
    test_split = "validation"
elif args.dataset == "gsm8k":
    dst = "gsm8k"
    def extract_example(example):
        return example["question"] + "\n" + example["answer"]
    key = extract_example
    test_split = "test"
    is_tiny = True
elif args.dataset == "fineweb-1b":
    dst = "fineweb-1b"
    key = "text"
    test_split = 10_000_000
    subsample = 1_000_000_000
elif args.dataset == "fineweb-10b":
    dst = "fineweb-10b"
    key = "text"
    test_split = 10_000_000
elif args.dataset == "climb-1b":
    src = "OptimalScale/ClimbMix"
    dst = "climb-1b"
    key = "text"
    test_split = 10_000_000
    subsample = 1_000_000_000
elif args.dataset == "climb-10b":
    src = "OptimalScale/ClimbMix"
    dst = "climb-1b"
    key = "text"
    test_split = 10_000_000
    subsample = 10_000_000_000
elif args.dataset == "limo":
    dst = "limo"
    def extract_example(example):
        return example["question"] + "\n" + example["solution"]
    key = extract_example
    test_split = "test"
    is_tiny = True
else:
    assert False, f"unknown dataset {args.dataset}"

if args.model.lower() == "qwen":
    model_name = "Qwen/Qwen2.5-0.5B"
    dst += "-qwen"
elif args.model.lower() == "llama":
    model_name = "unsloth/llama-2-7b"
    dst += "-llama"
else:
    assert False, f"unknown model {args.model}"


def load_or_create_dataset(name: str):
    if name == "tiny-shakespeare":
        return datasets.load_dataset("Trelis/tiny-shakespeare")
    elif name == "tiny-stories":
        return datasets.load_dataset("roneneldan/TinyStories")
    elif name == "gsm8k":
        return datasets.load_dataset("openai/gsm8k", "main")
    elif name == "fineweb-1b":
        return datasets.load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True)
    elif name in ["climb-1b", "climb-10b"]:
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
        return datasets.load_dataset("GAIR/LIMO").train_test_split(test_size=40, seed=42)

    raise ValueError(f"unknown dataset {name}")

def tokenize_example(example):
    return tokenizer(example, return_tensors='np', split_special_tokens=True).input_ids[0, ...]


class TokenizedDataFileWriter:
    def __init__(self, file_name: str,  vocab_size: int):
        self.file_name = file_name
        self.file_handle = None
        self.n_tokens = 0
        self.vocab_size = vocab_size

    def __enter__(self):
        self.file_handle = open(self.file_name, "wb+")
        # reserve space for the file header
        self.file_handle.write(('*' * 1023 + '\n').encode("ascii"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._write_header()
        self.file_handle.close()
        self.file_handle = None

    def add_document(self, tokens: np.ndarray):
        assert self.file_handle is not None

        tokens = np.array(tokens , dtype=np.int32)
        assert tokens.ndim == 1

        self.file_handle.write(tokens.tobytes())
        self.n_tokens += len(tokens)
        if self.n_tokens >= 2**31:
            raise RuntimeError("cannot have more than 2**31 tokens in a single file")

    def _write_header(self):
        assert self.file_handle is not None
        self.file_handle.seek(0)
        header_str = "BIN.TOK\n"  # 8 bytes
        version = 2
        bytes_per_token = 4
        self.file_handle.write(header_str.encode("ascii"))
        self.file_handle.write(np.array([version, bytes_per_token, self.n_tokens, self.vocab_size], dtype=np.int32).tobytes())
        self.file_handle.seek(256*4)


def init_worker(model_name_arg):
    from transformers import AutoTokenizer
    global worker_tokenizer
    worker_tokenizer = AutoTokenizer.from_pretrained(model_name_arg)


def tokenize_example_worker(args: tuple):
    example, key_func = args
    """Worker function for multiprocessing"""
    if callable(key_func):
        example = key_func(example)
    else:
        example = example[key_func]
    return worker_tokenizer(example, return_tensors='np', split_special_tokens=True).input_ids[0, ...]


def process_single_file(ds_iter, file_name: str, max_tokens: int, pool) -> tuple[bool, int]:
    def example_generator(f: TokenizedDataFileWriter):
        try:
            while f.n_tokens < max_tokens:
                yield next(ds_iter), key
        except StopIteration:
            return

    has_more_data = False
    with TokenizedDataFileWriter(file_name, tokenizer.vocab_size) as f:
        tokenized_examples = pool.imap(
            tokenize_example_worker,
            example_generator(f),
            chunksize=256
        )

        for tokens in tokenized_examples:
            f.add_document(tokens)

            if f.n_tokens >= max_tokens:
                has_more_data = True
                break

        return has_more_data, f.n_tokens


def tokenize_dataset(file_name: str, ds, split_name: str, max_tokens: int = None, is_tiny: bool = False, first_is_eval: int = -1):
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

            has_more_data, tokens_written = process_single_file(ds_iter, output_filename, max_size, pool)

            total_tokens += tokens_written
            print(f"Completed file {output_filename} with {tokens_written:,} tokens")
            file_index += 1

            if not has_more_data:
                break

            if max_tokens and total_tokens >= max_tokens:
                break


if __name__ == "__main__":
    d = load_or_create_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dst = args.out_dir + "/" + dst

    if isinstance(test_split, int):
        tokenize_dataset(dst, d["train"], "train", subsample, is_tiny, test_split)
    else:
        tokenize_dataset(dst, d["train"], "train", subsample, is_tiny)
        tokenize_dataset(dst, d[test_split], "eval", None, True)
