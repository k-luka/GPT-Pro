"""
Offline data preparation for ClimbMix-400B with DeepSeek V3 tokenizer.

Pipeline:
  1. Stream karpathy/climbmix-400b-shuffle from Hugging Face
  2. Tokenize every document with the DeepSeek V3 tokenizer
  3. Concatenate documents separated by the EOS token  (sequence packing)
  4. Slice the flat token stream into shards whose length is an exact
     multiple of block_size  (no padding, no masking at train time)
  5. Write each shard as a dense uint32 .bin file

Usage:
    python scripts/data_prep/prepare_climbmix.py \
        --block_size 2048 \
        --shard_size 100000000 \
        --output_dir data/climbmix_400b
"""

import argparse
import multiprocessing as mp
import os
from typing import Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_NAME = "deepseek-ai/DeepSeek-V3-Base"
DATASET_NAME = "karpathy/climbmix-400b-shuffle"
SHARD_SIZE = int(1e8)  # 100M tokens per shard

# ── multiprocessing worker state ──────────────────────────────────────────────
_worker_tokenizer: Optional[AutoTokenizer] = None
_worker_eot_id: Optional[int] = None


def _init_worker(tokenizer_name: str) -> None:
    global _worker_tokenizer, _worker_eot_id
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )
    _worker_eot_id = _worker_tokenizer.eos_token_id


def _tokenize(doc: dict) -> np.ndarray:
    """Prepend the EOS token then tokenize the document body."""
    assert _worker_tokenizer is not None and _worker_eot_id is not None
    tokens = [_worker_eot_id]
    tokens.extend(_worker_tokenizer.encode(doc["text"], add_special_tokens=False))
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
    """
    Truncate *buf[:n_tokens]* to the largest multiple of block_size,
    write the result as a raw uint32 .bin file, and return the number
    of leftover tokens that did NOT fit.
    """
    usable = (n_tokens // block_size) * block_size
    if usable == 0:
        return n_tokens  # nothing to write yet
    fname = os.path.join(output_dir, f"climbmix_{split}_{shard_index:06d}.bin")
    buf[:usable].tofile(fname)
    print(f"  → {fname}  |  {usable:>12,} tokens  ({usable // block_size} chunks)")
    return n_tokens - usable  # leftover count


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ClimbMix-400B shards with Qwen 3.5 tokenizer"
    )
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("--output_dir", type=str, default="data/climbmix_400b")
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_NAME)
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── print config ──────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eot_id = tok.eos_token_id
    print(f"Tokenizer     : {args.tokenizer}")
    print(f"Vocab size    : {tok.vocab_size}")
    print(f"EOS Token ID  : {eot_id} ({tok.eos_token})")
    print(f"Block size    : {args.block_size}")
    print(f"Shard size    : {args.shard_size:,} tokens")
    print(f"Output dir    : {args.output_dir}")
    print()

    # ── stream dataset ────────────────────────────────────────────────────────
    ds = load_dataset(args.dataset, split="train", streaming=True)

    nprocs = args.num_workers if args.num_workers > 0 else (os.cpu_count() or 1)
    print(f"Tokenising with {nprocs} workers …")

    shard_idx = 0
    buf = np.empty((args.shard_size,), dtype=np.uint32)
    token_count = 0
    progress: Optional[tqdm] = None

    with mp.Pool(nprocs, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for tokens in pool.imap(_tokenize, ds, chunksize=16):
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

                    # carry leftover tokens into the next shard buffer
                    if leftover > 0:
                        buf[:leftover] = buf[token_count - leftover : token_count]
                    token_count = leftover
                    shard_idx += 1

    # ── flush remaining tokens ────────────────────────────────────────────────
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
