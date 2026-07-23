"""Create self-describing, resumable ClimbMix token shards.

The tokenizer determines the BOS ID and the smallest safe shard dtype.  Text is
encoded with special-token processing disabled and exactly one document-boundary
BOS is prepended manually.  Shard metadata prevents a resume with a different
tokenizer or binary layout.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.shard_format import (  # noqa: E402
    assert_metadata_matches,
    build_metadata,
    dtype_for_vocab_size,
    parse_shard_dtype,
    read_metadata,
    sha256_file,
    write_metadata,
)


DATASET_NAME = "karpathy/climbmix-400b-shuffle"
SHARD_SIZE = 100_000_000
CHECKPOINT_FILE = "checkpoint.json"
TOKENIZER_PATH = "tokenizers/gemma4_80k/tokenizer.json"

_worker_enc: Optional[Tokenizer] = None
_worker_bos_id: Optional[int] = None
_worker_dtype: Optional[np.dtype] = None


def _init_worker(tokenizer_path: str, bos_id: int, dtype_name: str) -> None:
    global _worker_enc, _worker_bos_id, _worker_dtype
    _worker_enc = Tokenizer.from_file(tokenizer_path)
    _worker_bos_id = bos_id
    _worker_dtype = np.dtype(dtype_name)


def _tokenize(doc: dict[str, Any]) -> np.ndarray:
    """Encode one document with exactly one manually inserted BOS."""

    assert _worker_enc is not None
    assert _worker_bos_id is not None
    assert _worker_dtype is not None
    body = _worker_enc.encode(doc["text"], add_special_tokens=False).ids
    tokens = np.empty(len(body) + 1, dtype=_worker_dtype)
    tokens[0] = _worker_bos_id
    tokens[1:] = body
    return tokens


def _write_full_shard(
    output_dir: str,
    split: str,
    shard_index: int,
    tokens: np.ndarray,
    block_size: int,
) -> None:
    if len(tokens) % block_size:
        raise ValueError("a full shard must contain an exact number of blocks")
    path = os.path.join(output_dir, f"climbmix_{split}_{shard_index:06d}.bin")
    temporary = f"{path}.tmp"
    tokens.tofile(temporary)
    os.replace(temporary, path)
    print(
        f"  → {path}  |  {len(tokens):>12,} tokens  "
        f"({len(tokens) // block_size} chunks)"
    )


def _checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir) / CHECKPOINT_FILE


def _save_checkpoint(
    output_dir: str,
    *,
    next_shard: int,
    completed_docs: int,
    doc_token_offset: int,
    metadata: dict[str, Any],
    complete: bool = False,
) -> None:
    state = {
        "next_shard": next_shard,
        "completed_docs": completed_docs,
        "doc_token_offset": doc_token_offset,
        "metadata": metadata,
        "complete": complete,
    }
    path = _checkpoint_path(output_dir)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_checkpoint(
    output_dir: str, expected_metadata: dict[str, Any]
) -> dict[str, Any]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"cannot resume without checkpoint: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    required = {"next_shard", "completed_docs", "doc_token_offset", "metadata"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    assert_metadata_matches(state["metadata"], expected_metadata)
    if state.get("complete", False):
        raise RuntimeError("the checkpoint marks this dataset preparation complete")
    if min(
        state["next_shard"],
        state["completed_docs"],
        state["doc_token_offset"],
    ) < 0:
        raise ValueError("checkpoint positions must be non-negative")
    return state


def _resolve_bos(tokenizer: Tokenizer, requested: str | None) -> tuple[str, int]:
    candidates = [requested] if requested else ["<bos>", "<|bos|>"]
    for token in candidates:
        if token is None:
            continue
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return token, token_id
    raise ValueError(
        f"none of the BOS candidates exist in the tokenizer: {candidates!r}"
    )


def _apply_resume_offset(
    tokens: np.ndarray,
    *,
    doc_index: int,
    completed_docs: int,
    resume_offset: int,
) -> tuple[np.ndarray, int]:
    """Reconstruct the unpersisted tail of a boundary-spanning document."""

    if doc_index != completed_docs or resume_offset == 0:
        return tokens, 0
    if resume_offset >= len(tokens):
        raise ValueError(
            f"resume offset {resume_offset} is outside document length {len(tokens)}"
        )
    return tokens[resume_offset:], resume_offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("--output_dir", default="data/climbmix_gemma4_80k")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--bos-token", default=None)
    parser.add_argument(
        "--dtype",
        choices=("auto", "uint16", "uint32"),
        default="auto",
        help="Default auto chooses the smallest dtype that can hold every ID.",
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--val_shards", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=0, help="0 uses all available CPUs"
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Stop after writing this many shards during the current invocation.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size <= 0 or args.shard_size < args.block_size:
        raise ValueError("shard_size must be at least one positive block")
    if args.val_shards < 0:
        raise ValueError("val_shards must be non-negative")
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("max_shards must be positive")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    max_token_id = max(vocab.values())
    if (
        len(vocab) != vocab_size
        or len(set(vocab.values())) != vocab_size
        or max_token_id != vocab_size - 1
    ):
        raise ValueError("tokenizer vocabulary IDs must be unique and dense")
    bos_token, bos_id = _resolve_bos(tokenizer, args.bos_token)

    minimum_dtype = dtype_for_vocab_size(max_token_id + 1)
    dtype = minimum_dtype if args.dtype == "auto" else parse_shard_dtype(args.dtype)
    if max_token_id > np.iinfo(dtype).max:
        raise ValueError(
            f"tokenizer max ID {max_token_id} does not fit in {dtype.name}"
        )

    effective_shard_size = (args.shard_size // args.block_size) * args.block_size
    metadata = build_metadata(
        dtype=dtype,
        vocab_size=vocab_size,
        max_token_id=max_token_id,
        bos_token=bos_token,
        bos_id=bos_id,
        block_size=args.block_size,
        shard_size=effective_shard_size,
        tokenizer_sha256=sha256_file(args.tokenizer),
        dataset=args.dataset,
        val_shards=args.val_shards,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.iterdir())
    if existing and not args.resume:
        raise FileExistsError(
            f"output directory is not empty: {output}; use --resume or a new directory"
        )

    if args.resume:
        existing_metadata = read_metadata(output, allow_legacy=False)
        assert_metadata_matches(existing_metadata, metadata)
        state = _load_checkpoint(str(output), metadata)
        shard_index = int(state["next_shard"])
        completed_docs = int(state["completed_docs"])
        resume_offset = int(state["doc_token_offset"])
        bin_files = list(output.glob("*.bin"))
        if len(bin_files) != shard_index:
            raise ValueError(
                f"checkpoint expects {shard_index} shards, found {len(bin_files)}"
            )
    else:
        write_metadata(output, metadata)
        shard_index = completed_docs = resume_offset = 0

    print(f"Tokenizer     : {args.tokenizer}")
    print(f"Tokenizer SHA : {metadata['tokenizer_sha256']}")
    print(f"Vocab / max ID: {vocab_size:,} / {max_token_id:,}")
    print(f"BOS token     : {bos_token!r} id={bos_id}")
    print(f"Shard dtype   : {dtype.name}")
    print(f"Block size    : {args.block_size:,}")
    print(f"Shard size    : {effective_shard_size:,} tokens")
    print(f"Output dir    : {output}")
    if args.resume:
        print(
            f"Resume        : shard={shard_index}, completed_docs={completed_docs:,}, "
            f"doc_token_offset={resume_offset:,}"
        )

    dataset = load_dataset(args.dataset, split="train", streaming=True)
    if completed_docs:
        dataset = dataset.skip(completed_docs)

    workers = args.num_workers if args.num_workers > 0 else (os.cpu_count() or 1)
    buffer = np.empty(effective_shard_size, dtype=dtype)
    buffer_count = 0
    shards_written = 0
    final_completed_docs = completed_docs
    progress: Optional[tqdm] = None

    with mp.Pool(
        workers,
        initializer=_init_worker,
        initargs=(args.tokenizer, bos_id, dtype.name),
    ) as pool:
        for doc_index, full_tokens in enumerate(
            pool.imap(_tokenize, dataset, chunksize=16), start=completed_docs
        ):
            original_length = len(full_tokens)
            full_tokens, consumed = _apply_resume_offset(
                full_tokens,
                doc_index=doc_index,
                completed_docs=completed_docs,
                resume_offset=resume_offset,
            )

            remaining = full_tokens
            while len(remaining):
                take = min(len(remaining), effective_shard_size - buffer_count)
                buffer[buffer_count : buffer_count + take] = remaining[:take]
                buffer_count += take
                consumed += take
                remaining = remaining[take:]

                if progress is None:
                    progress = tqdm(
                        total=effective_shard_size,
                        initial=buffer_count - take,
                        unit="tok",
                        desc=f"Shard {shard_index}",
                    )
                progress.update(take)

                if buffer_count == effective_shard_size:
                    progress.close()
                    progress = None
                    split = "val" if shard_index < args.val_shards else "train"
                    _write_full_shard(
                        str(output), split, shard_index, buffer, args.block_size
                    )
                    shard_index += 1
                    shards_written += 1
                    buffer_count = 0

                    if consumed < original_length:
                        checkpoint_docs = doc_index
                        checkpoint_offset = consumed
                    else:
                        checkpoint_docs = doc_index + 1
                        checkpoint_offset = 0
                    _save_checkpoint(
                        str(output),
                        next_shard=shard_index,
                        completed_docs=checkpoint_docs,
                        doc_token_offset=checkpoint_offset,
                        metadata=metadata,
                    )
                    if (
                        args.max_shards is not None
                        and shards_written >= args.max_shards
                    ):
                        print(f"Reached --max_shards={args.max_shards}.")
                        return

            resume_offset = 0
            final_completed_docs = doc_index + 1

    if progress is not None:
        progress.close()
    usable = (buffer_count // args.block_size) * args.block_size
    if usable:
        split = "val" if shard_index < args.val_shards else "train"
        _write_full_shard(
            str(output), split, shard_index, buffer[:usable], args.block_size
        )
        shard_index += 1
    _save_checkpoint(
        str(output),
        next_shard=shard_index,
        completed_docs=final_completed_docs,
        doc_token_offset=0,
        metadata=metadata,
        complete=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
