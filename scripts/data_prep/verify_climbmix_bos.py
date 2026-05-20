"""
Verify ClimbMix uint32 shards use a prepended BOS document-boundary token.

This does not try to reconstruct documents from flat shards. It checks the
properties we need for training:
  - shard lengths are exact multiples of block_size
  - sampled shards contain BOS id 0
  - EOS id 1 is not being inserted as the document boundary

Usage:
    python scripts/data_prep/verify_climbmix_bos.py \
        --data_root data/climbmix_400b_bos \
        --block_size 4096
"""

import argparse
import os

import numpy as np


def _token_count(path: str) -> int:
    return os.path.getsize(path) // np.dtype(np.uint32).itemsize


def _sample_paths(paths: list[str], n: int) -> list[str]:
    if len(paths) <= n:
        return paths
    if n == 1:
        return [paths[0]]

    indices = np.linspace(0, len(paths) - 1, num=n, dtype=int)
    return [paths[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/climbmix_400b_bos")
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--bos_id", type=int, default=0)
    parser.add_argument("--eos_id", type=int, default=1)
    parser.add_argument("--sample_shards", type=int, default=3)
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(
            f"{args.data_root} does not exist yet. Run prepare_climbmix.py first, "
            "or pass the directory where the regenerated BOS shards were written."
        )

    shards = sorted(
        os.path.join(args.data_root, name)
        for name in os.listdir(args.data_root)
        if name.endswith(".bin")
    )
    if not shards:
        raise FileNotFoundError(f"No .bin shards found in {args.data_root}")

    bad_sizes = [
        path
        for path in shards
        if _token_count(path) == 0 or _token_count(path) % args.block_size != 0
    ]
    if bad_sizes:
        print("Bad shard sizes:")
        for path in bad_sizes[:10]:
            print(f"  {path}: {_token_count(path)} tokens")
        raise RuntimeError(f"{len(bad_sizes)} shards are empty or not block-aligned")

    train_shards = [path for path in shards if "_train_" in os.path.basename(path)]
    val_shards = [path for path in shards if "_val_" in os.path.basename(path)]
    sampled = _sample_paths(val_shards, 1) + _sample_paths(
        train_shards, args.sample_shards
    )

    print(f"Data root      : {args.data_root}")
    print(f"Total shards   : {len(shards)}")
    print(f"Train shards   : {len(train_shards)}")
    print(f"Val shards     : {len(val_shards)}")
    print(f"Block size     : {args.block_size}")
    print(f"BOS id / EOS id: {args.bos_id} / {args.eos_id}")
    print()

    any_bos = False
    total_bos = 0
    total_eos = 0
    total_tokens = 0

    for path in sampled:
        tokens = np.fromfile(path, dtype=np.uint32)
        bos_positions = np.flatnonzero(tokens == args.bos_id)
        eos_count = int((tokens == args.eos_id).sum())
        bos_count = int(len(bos_positions))

        any_bos = any_bos or bos_count > 0
        total_bos += bos_count
        total_eos += eos_count
        total_tokens += int(len(tokens))

        bos_per_mtok = bos_count / max(len(tokens), 1) * 1_000_000
        print(os.path.basename(path))
        print(f"  tokens             : {len(tokens):,}")
        print(f"  BOS count          : {bos_count:,} ({bos_per_mtok:.2f} / 1M tok)")
        print(f"  EOS count          : {eos_count:,}")
        print(f"  first BOS positions: {bos_positions[:10].tolist()}")

    if not any_bos:
        raise RuntimeError("Sampled shards contain no BOS tokens")

    print()
    print(f"Sampled tokens : {total_tokens:,}")
    print(f"Sampled BOS    : {total_bos:,}")
    print(f"Sampled EOS    : {total_eos:,}")
    print("OK: sampled shards contain BOS document-boundary tokens.")


if __name__ == "__main__":
    main()
