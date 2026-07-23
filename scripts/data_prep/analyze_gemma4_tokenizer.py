#!/usr/bin/env python3
"""Count original Gemma 4 token use on decoded ClimbMix documents.

The existing ``climbmix_arch_64k`` shards contain losslessly encoded documents
separated by ``<|bos|>``.  This scanner samples deterministic windows across all
shards, decodes documents with their source tokenizer, and counts how the full
Gemma 4 tokenizer represents the same text.

The output NPZ is the only corpus-dependent input to the child-tokenizer
builder.  It contains a dense uint64 count array indexed by original Gemma ID
plus scalar audit statistics.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-tokenizer", required=True)
    parser.add_argument("--source-tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--data-root", default="data/climbmix_arch_64k")
    parser.add_argument("--output", default="data/tokenizer_gemma4/gemma_counts.npz")
    parser.add_argument("--source-dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--source-bos-id", type=int, default=65527)
    parser.add_argument("--sample-docs", type=int, default=200_000)
    parser.add_argument("--windows-per-shard", type=int, default=3)
    parser.add_argument("--window-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def document_spans(window: np.ndarray, bos_id: int) -> list[tuple[int, int]]:
    positions = np.flatnonzero(window == bos_id)
    return [
        (int(left + 1), int(right))
        for left, right in zip(positions[:-1], positions[1:])
        if 8 <= right - left - 1 <= 100_000
    ]


def flush_batch(
    texts: list[str], gemma: Tokenizer, counts: np.ndarray
) -> tuple[int, int]:
    if not texts:
        return 0, 0
    encodings = gemma.encode_batch(texts, add_special_tokens=False)
    token_count = 0
    byte_count = 0
    for text, encoding in zip(texts, encodings):
        ids = np.asarray(encoding.ids, dtype=np.int64)
        if len(ids):
            counts += np.bincount(ids, minlength=len(counts)).astype(np.uint64)
        token_count += len(ids)
        byte_count += len(text.encode("utf-8"))
    texts.clear()
    return token_count, byte_count


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source = Tokenizer.from_file(args.source_tokenizer)
    gemma = Tokenizer.from_file(args.gemma_tokenizer)
    vocab_size = gemma.get_vocab_size()
    counts = np.zeros(vocab_size, dtype=np.uint64)

    shards = sorted(glob.glob(os.path.join(args.data_root, "*.bin")))
    if not shards:
        raise FileNotFoundError(f"no .bin shards found under {args.data_root}")

    # Divide the requested sample evenly; the final shard absorbs the remainder.
    base_per_shard, remainder = divmod(args.sample_docs, len(shards))
    dtype = np.dtype(args.source_dtype)
    documents = gemma_tokens = source_tokens = byte_count = 0
    batch: list[str] = []

    for shard_index, shard_path in enumerate(shards):
        target = base_per_shard + (1 if shard_index < remainder else 0)
        shard = np.memmap(shard_path, dtype=dtype, mode="r")
        candidates: list[tuple[np.ndarray, int, int]] = []

        for _ in range(args.windows_per_shard):
            width = min(args.window_tokens, len(shard))
            start = rng.randrange(0, max(1, len(shard) - width + 1))
            window = np.asarray(shard[start : start + width])
            candidates.extend((window, left, right) for left, right in document_spans(window, args.source_bos_id))

        if len(candidates) < target:
            raise RuntimeError(
                f"only found {len(candidates):,} complete documents in sampled "
                f"windows from {shard_path}; need {target:,}. Increase "
                "--windows-per-shard or --window-tokens."
            )

        for window, left, right in rng.sample(candidates, target):
            token_ids = window[left:right].tolist()
            batch.append(source.decode(token_ids))
            source_tokens += right - left
            documents += 1
            if len(batch) >= args.batch_size:
                new_tokens, new_bytes = flush_batch(batch, gemma, counts)
                gemma_tokens += new_tokens
                byte_count += new_bytes

        print(
            f"[{shard_index + 1:>2}/{len(shards)}] docs={documents:,} "
            f"Gemma_tokens={gemma_tokens:,}",
            flush=True,
        )

    new_tokens, new_bytes = flush_batch(batch, gemma, counts)
    gemma_tokens += new_tokens
    byte_count += new_bytes

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        counts=counts,
        documents=np.asarray(documents, dtype=np.uint64),
        source_tokens=np.asarray(source_tokens, dtype=np.uint64),
        gemma_tokens=np.asarray(gemma_tokens, dtype=np.uint64),
        utf8_bytes=np.asarray(byte_count, dtype=np.uint64),
    )
    metadata = {
        "documents": documents,
        "source_tokens": source_tokens,
        "gemma_tokens": gemma_tokens,
        "utf8_bytes": byte_count,
        "gemma_bytes_per_token": byte_count / max(gemma_tokens, 1),
        "source_bytes_per_token": byte_count / max(source_tokens, 1),
        "unique_gemma_ids": int(np.count_nonzero(counts)),
        "gemma_tokenizer_sha256": file_sha256(args.gemma_tokenizer),
        "source_tokenizer_sha256": file_sha256(args.source_tokenizer),
        "seed": args.seed,
        "shards": len(shards),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Saved counts: {output}")


if __name__ == "__main__":
    main()
