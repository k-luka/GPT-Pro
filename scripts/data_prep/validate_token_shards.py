#!/usr/bin/env python3
"""Exhaustively validate metadata-described pretraining token shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.shard_format import (  # noqa: E402
    parse_shard_dtype,
    read_metadata,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--report",
        default=None,
        help="Optionally write the final JSON report to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    metadata = read_metadata(root, allow_legacy=False)
    dtype = parse_shard_dtype(metadata["dtype"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.get_vocab_size() != metadata["vocab_size"]:
        raise ValueError("tokenizer and shard metadata vocabulary sizes differ")
    if sha256_file(args.tokenizer) != metadata["tokenizer_sha256"]:
        raise ValueError("tokenizer SHA-256 does not match shard metadata")
    if tokenizer.token_to_id(metadata["bos_token"]) != metadata["bos_id"]:
        raise ValueError("tokenizer BOS mapping does not match shard metadata")

    paths = sorted(glob.glob(str(root / "*.bin")))
    if args.max_shards is not None:
        paths = paths[: args.max_shards]
    if not paths:
        raise FileNotFoundError(f"no .bin shards under {root}")

    total_tokens = 0
    total_boundaries = 0
    observed_max = 0
    for index, path in enumerate(paths, start=1):
        byte_size = Path(path).stat().st_size
        if byte_size % dtype.itemsize:
            raise ValueError(f"shard byte size is invalid for {dtype.name}: {path}")
        tokens = np.memmap(path, dtype=dtype, mode="r")
        if len(tokens) % metadata["block_size"]:
            raise ValueError(f"shard is not block-aligned: {path}")
        shard_max = int(tokens.max(initial=0))
        if shard_max >= metadata["vocab_size"]:
            raise ValueError(
                f"token ID {shard_max} exceeds vocabulary in shard {path}"
            )
        total_tokens += len(tokens)
        total_boundaries += int(np.count_nonzero(tokens == metadata["bos_id"]))
        observed_max = max(observed_max, shard_max)
        print(
            f"[{index:>4}/{len(paths)}] {Path(path).name}: "
            f"{len(tokens):,} tokens, max_id={shard_max:,}",
            flush=True,
        )

    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "shards": len(paths),
        "tokens": total_tokens,
        "document_boundaries": total_boundaries,
        "observed_max_id": observed_max,
        "dtype": dtype.name,
        "vocab_size": metadata["vocab_size"],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
