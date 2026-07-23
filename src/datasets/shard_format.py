"""Metadata and dtype helpers for flat pretraining-token shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_VERSION = 1
METADATA_FILENAME = "metadata.json"
SUPPORTED_DTYPES = {"uint16", "uint32"}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dtype_for_vocab_size(vocab_size: int) -> np.dtype:
    """Return the smallest supported integer dtype for dense token IDs."""

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if vocab_size <= 2**16:
        return np.dtype(np.uint16)
    if vocab_size <= 2**32:
        return np.dtype(np.uint32)
    raise ValueError(f"vocab_size {vocab_size:,} exceeds uint32 capacity")


def parse_shard_dtype(value: str) -> np.dtype:
    if value not in SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported shard dtype {value!r}; expected one of "
            f"{sorted(SUPPORTED_DTYPES)}"
        )
    return np.dtype(value)


def build_metadata(
    *,
    dtype: np.dtype,
    vocab_size: int,
    max_token_id: int,
    bos_token: str,
    bos_id: int,
    block_size: int,
    shard_size: int,
    tokenizer_sha256: str,
    dataset: str,
    val_shards: int,
) -> dict[str, Any]:
    dtype = np.dtype(dtype)
    if dtype.name not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported shard dtype: {dtype}")
    if not 0 <= bos_id < vocab_size:
        raise ValueError(f"BOS ID {bos_id} is outside vocab size {vocab_size}")
    if max_token_id != vocab_size - 1:
        raise ValueError(
            "tokenizer IDs must be dense: "
            f"max ID {max_token_id}, vocab size {vocab_size}"
        )
    if vocab_size > np.iinfo(dtype).max + 1:
        raise ValueError(f"vocab size {vocab_size} does not fit in {dtype.name}")
    if block_size <= 0 or shard_size < block_size:
        raise ValueError("shard_size must be at least one positive block")
    return {
        "format_version": FORMAT_VERSION,
        "dtype": dtype.name,
        "vocab_size": vocab_size,
        "max_token_id": max_token_id,
        "bos_token": bos_token,
        "bos_id": bos_id,
        "block_size": block_size,
        "shard_size": shard_size,
        "tokenizer_sha256": tokenizer_sha256,
        "dataset": dataset,
        "val_shards": val_shards,
    }


def metadata_path(data_root: str | os.PathLike[str]) -> Path:
    return Path(data_root) / METADATA_FILENAME


def write_metadata(data_root: str | os.PathLike[str], metadata: dict[str, Any]) -> None:
    path = metadata_path(data_root)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_metadata(
    data_root: str | os.PathLike[str], *, allow_legacy: bool = True
) -> dict[str, Any]:
    path = metadata_path(data_root)
    if not path.exists():
        if allow_legacy:
            return {"format_version": 0, "dtype": "uint16", "legacy": True}
        raise FileNotFoundError(f"missing shard metadata: {path}")

    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "dtype",
        "vocab_size",
        "max_token_id",
        "bos_token",
        "bos_id",
        "block_size",
        "shard_size",
        "tokenizer_sha256",
        "dataset",
        "val_shards",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"shard metadata is missing fields: {sorted(missing)}")
    if metadata["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"unsupported shard format version {metadata['format_version']}; "
            f"expected {FORMAT_VERSION}"
        )
    dtype = parse_shard_dtype(metadata["dtype"])
    if metadata["vocab_size"] > np.iinfo(dtype).max + 1:
        raise ValueError("metadata vocabulary does not fit its declared dtype")
    if metadata["max_token_id"] != metadata["vocab_size"] - 1:
        raise ValueError("metadata tokenizer IDs are not dense")
    if not 0 <= metadata["bos_id"] < metadata["vocab_size"]:
        raise ValueError("metadata BOS ID is outside its vocabulary")
    return metadata


def assert_metadata_matches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Reject unsafe resume attempts with a different shard encoding."""

    mismatches = {
        key: (actual.get(key), expected.get(key))
        for key in expected
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}: existing={old!r}, requested={new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ValueError(f"output metadata is incompatible with this run: {details}")
