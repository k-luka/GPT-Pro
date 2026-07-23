#!/usr/bin/env python3
"""Fail-fast validation for the final 8xB200 Gemma-80K pretraining launch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from omegaconf import OmegaConf
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.shard_format import read_metadata, sha256_file  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "config/config_dense_6p5b_gemma4_80k.yaml"
EXPECTED_WORLD_SIZE = 8
JOB_SECONDS = 14 * 24 * 60 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--require-full-validation",
        action="store_true",
        help="Require a successful exhaustive shard-validation report.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(f"final preflight failed: {message}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = OmegaConf.load(config_path)

    tokenizer_path = (PROJECT_ROOT / cfg.data.tokenizer_path).resolve()
    data_root = (PROJECT_ROOT / cfg.data.train_data_root).resolve()
    metadata = read_metadata(data_root, allow_legacy=False)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_sha = sha256_file(tokenizer_path)

    if tokenizer.get_vocab_size() != cfg.model.vocab_size:
        fail("tokenizer and model vocabulary sizes differ")
    if metadata["vocab_size"] != cfg.model.vocab_size:
        fail("dataset and model vocabulary sizes differ")
    if metadata["tokenizer_sha256"] != tokenizer_sha:
        fail("dataset metadata and tokenizer SHA-256 differ")
    if tokenizer.token_to_id(metadata["bos_token"]) != metadata["bos_id"]:
        fail("dataset and tokenizer BOS mappings differ")
    if cfg.model.sandwich_norm is not False:
        fail("the selected final architecture requires sandwich_norm=false")
    expected_model = {
        "sliding_window": 512,
        "global_attn_every_n": 5,
        "global_attn_placement": "end",
        "local_attn_impl": "fa4",
        "global_attn_impl": "sdpa",
        "qk_norm_mode": "after",
    }
    for key, expected in expected_model.items():
        if cfg.model.get(key) != expected:
            fail(f"model.{key} must be {expected!r}, got {cfg.model.get(key)!r}")

    val_shards = sorted(data_root.glob("*_val_*.bin"))
    train_shards = sorted(data_root.glob("*_train_*.bin"))
    all_shards = val_shards + train_shards
    if len(val_shards) != 1 or len(train_shards) != 1599:
        fail(
            f"expected 1 validation + 1599 train shards, found "
            f"{len(val_shards)} + {len(train_shards)}"
        )

    dtype_bytes = 4 if metadata["dtype"] == "uint32" else 2
    lengths: list[int] = []
    for path in all_shards:
        byte_size = path.stat().st_size
        if byte_size % dtype_bytes:
            fail(f"{path.name} has a partial {metadata['dtype']} value")
        token_count = byte_size // dtype_bytes
        if token_count % metadata["block_size"]:
            fail(f"{path.name} is not block-aligned")
        lengths.append(token_count)

    total_tokens = sum(lengths)
    expected_total = 159_999_590_400
    if total_tokens != expected_total:
        fail(f"expected {expected_total:,} raw tokens, found {total_tokens:,}")

    micro_tokens = (
        int(cfg.training.batch_size)
        * int(cfg.model.block_size)
        * EXPECTED_WORLD_SIZE
    )
    grad_accum = int(cfg.training.grad_accum_steps)
    train_microsteps = sum((length - 1) // micro_tokens for length in lengths[1:])
    requested_microsteps = int(cfg.training.max_steps) * grad_accum
    if requested_microsteps > train_microsteps:
        fail(
            f"{cfg.training.max_steps:,} steps would wrap the train loader; "
            f"at most {train_microsteps // grad_accum:,} complete steps fit"
        )

    trained_tokens = requested_microsteps * micro_tokens
    required_tps = trained_tokens / JOB_SECONDS
    warmdown_steps = round(
        float(cfg.training.warmdown_ratio) * int(cfg.training.max_steps)
    )
    warmdown_start = int(cfg.training.max_steps) - warmdown_steps
    interval = int(cfg.training.checkpoint_interval)
    if warmdown_start % interval or int(cfg.training.max_steps) % interval:
        fail("checkpoint cadence does not land on warmdown start and final step")
    if cfg.training.eval_interval != interval:
        fail("eval_interval must match checkpoint_interval for final-run efficiency")

    report_path = data_root / "validation_report.json"
    if args.require_full_validation:
        if not report_path.is_file():
            fail(
                f"missing {report_path}; run scripts/data_prep/"
                "validate_token_shards.py with --report first"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("shards") != len(all_shards):
            fail("full-validation report has the wrong shard count")
        if report.get("tokens") != total_tokens:
            fail("full-validation report has the wrong token count")
        if report.get("vocab_size") != cfg.model.vocab_size:
            fail("full-validation report has the wrong vocabulary size")
        if report.get("tokenizer_sha256") != tokenizer_sha:
            fail("full-validation report has the wrong tokenizer SHA-256")
        if report.get("observed_max_id", cfg.model.vocab_size) >= cfg.model.vocab_size:
            fail("full-validation report contains an out-of-range token ID")
        report_mtime = report_path.stat().st_mtime
        newest_input = max(
            tokenizer_path.stat().st_mtime,
            (data_root / "metadata.json").stat().st_mtime,
            max(path.stat().st_mtime for path in all_shards),
        )
        if report_mtime < newest_input:
            fail("full-validation report is older than a tokenizer/dataset input")

    print("---| Final Gemma-80K preflight passed |---")
    print(f"Config             : {config_path}")
    print(f"Tokenizer SHA-256  : {tokenizer_sha}")
    print(f"Vocabulary / BOS   : {cfg.model.vocab_size:,} / {metadata['bos_id']}")
    print(f"Shards             : {len(train_shards):,} train + 1 val")
    print(f"Raw tokens         : {total_tokens:,}")
    print(f"Training steps     : {int(cfg.training.max_steps):,}")
    print(f"Training tokens    : {trained_tokens:,}")
    print(f"Tokens / parameter : {trained_tokens / 6.54e9:.2f}")
    print(f"Warmdown           : steps {warmdown_start:,}..{int(cfg.training.max_steps):,}")
    print(f"Required 14d rate  : {math.ceil(required_tps):,} tok/s aggregate")
    if args.require_full_validation:
        print(f"Full validation    : {report_path}")


if __name__ == "__main__":
    main()
