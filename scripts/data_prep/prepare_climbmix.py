"""
Offline data preparation for ClimbMix-400B with the GPT-4 cl100k_base tokenizer.

Pipeline:
  1. Stream karpathy/climbmix-400b-shuffle from Hugging Face
  2. Tokenize every document with the cl100k_base tokenizer (tiktoken)
  3. Prepend <|endoftext|> (id 100257) before every document as a boundary token
  4. Slice the flat token stream into shards whose length is an exact
     multiple of block_size  (no padding, no masking at train time)
  5. Write each shard as a dense uint32 .bin file

Usage:
    python scripts/data_prep/prepare_climbmix.py \
        --block_size 4096 \
        --shard_size 100000000 \
        --output_dir data/climbmix_400b

    # Resume an interrupted run:
    python scripts/data_prep/prepare_climbmix.py \
        --output_dir data/climbmix_400b \
        --resume
"""

import argparse
import json
import multiprocessing as mp
import os
from typing import Optional

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "karpathy/climbmix-400b-shuffle"
SHARD_SIZE = int(1e8)  # 100M tokens per shard
CHECKPOINT_FILE = "checkpoint.json"

EOT_TOKEN = "<|endoftext|>"
EOT_ID = 100257  # cl100k_base / o200k_base <|endoftext|> id

# ── multiprocessing worker state ──────────────────────────────────────────────
_worker_enc: Optional[tiktoken.Encoding] = None


def _init_worker() -> None:
    global _worker_enc
    _worker_enc = tiktoken.get_encoding("cl100k_base")


def _tokenize(doc: dict) -> np.ndarray:
    """Prepend <|endoftext|> then tokenize the document body."""
    assert _worker_enc is not None
    tokens = [EOT_ID]
    tokens.extend(_worker_enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint32)


# ── shard I/O ─────────────────────────────────────────────────────────────────


def _write_shard(
    output_dir: str,
    split: str,
    shard_index: int,
    buf: np.ndarray,
    n_tokens: int,
    block_size: int,
) -> int:
    usable = (n_tokens // block_size) * block_size
    if usable == 0:
        return n_tokens
    fname = os.path.join(output_dir, f"climbmix_{split}_{shard_index:06d}.bin")
    buf[:usable].tofile(fname)
    print(f"  → {fname}  |  {usable:>12,} tokens  ({usable // block_size} chunks)")
    return n_tokens - usable


def _save_checkpoint(output_dir: str, shard_idx: int, docs_processed: int) -> None:
    path = os.path.join(output_dir, CHECKPOINT_FILE)
    with open(path, "w") as f:
        json.dump({"shard_idx": shard_idx, "docs_processed": docs_processed}, f)


def _load_checkpoint(output_dir: str) -> tuple[int, int]:
    path = os.path.join(output_dir, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return 0, 0
    with open(path) as f:
        data = json.load(f)
    return data["shard_idx"], data["docs_processed"]


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ClimbMix-400B shards with cl100k_base tokenizer"
    )
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("--output_dir", type=str, default="data/climbmix_400b")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument(
        "--val_shards",
        type=int,
        default=1,
        help="Number of leading shards reserved for validation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Tokenization workers (0 = use all CPUs)",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Optional cap on written shards, useful for smoke tests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint.json in output_dir",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── resume state ──────────────────────────────────────────────────────────
    start_shard, skip_docs = 0, 0
    if args.resume:
        start_shard, skip_docs = _load_checkpoint(args.output_dir)
        if skip_docs > 0:
            print(f"Resuming from shard {start_shard}, skipping {skip_docs:,} documents …")
        else:
            print("No checkpoint found — starting from the beginning.")

    # ── print config ──────────────────────────────────────────────────────────
    print(f"Tokenizer     : cl100k_base (tiktoken)")
    print(f"Vocab size    : 100277  (padded to 100352 for tensor cores)")
    print(f"EOT token     : {EOT_TOKEN!r} id={EOT_ID}  (used as document boundary)")
    print(f"Block size    : {args.block_size}")
    print(f"Shard size    : {args.shard_size:,} tokens")
    print(f"Output dir    : {args.output_dir}")
    print()

    # ── stream dataset ────────────────────────────────────────────────────────
    ds = load_dataset(args.dataset, split="train", streaming=True)
    if skip_docs > 0:
        ds = ds.skip(skip_docs)

    nprocs = args.num_workers if args.num_workers > 0 else (os.cpu_count() or 1)
    print(f"Tokenising with {nprocs} workers …")

    shard_idx = start_shard
    docs_processed = skip_docs
    buf = np.empty((args.shard_size,), dtype=np.uint32)
    token_count = 0
    progress: Optional[tqdm] = None

    with mp.Pool(nprocs, initializer=_init_worker) as pool:
        for tokens in pool.imap(_tokenize, ds, chunksize=16):
            docs_processed += 1
            while len(tokens) > 0:
                space = args.shard_size - token_count
                take = min(len(tokens), space)
                buf[token_count : token_count + take] = tokens[:take]
                token_count += take
                tokens = tokens[take:]

                if progress is None:
                    progress = tqdm(
                        total=args.shard_size, unit="tok", desc=f"Shard {shard_idx}"
                    )
                progress.update(take)

                if token_count >= args.shard_size:
                    progress.close()
                    progress = None

                    split = "val" if shard_idx < args.val_shards else "train"
                    leftover = _write_shard(
                        args.output_dir,
                        split,
                        shard_idx,
                        buf,
                        token_count,
                        args.block_size,
                    )

                    if leftover > 0:
                        buf[:leftover] = buf[token_count - leftover : token_count]
                    token_count = leftover
                    shard_idx += 1

                    _save_checkpoint(args.output_dir, shard_idx, docs_processed)

                    if args.max_shards is not None and shard_idx >= args.max_shards:
                        print(f"\nReached --max_shards={args.max_shards}.")
                        return

    if token_count > 0:
        if progress is not None:
            progress.close()
        split = "val" if shard_idx < args.val_shards else "train"
        _write_shard(
            args.output_dir, split, shard_idx, buf, token_count, args.block_size
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
