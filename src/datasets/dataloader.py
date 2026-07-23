import os
import warnings

import numpy as np
import torch

from src.datasets.shard_format import parse_shard_dtype, read_metadata


def load_tokens(filename: str, dtype: np.dtype = np.dtype(np.uint16)) -> torch.Tensor:
    """Memory-map one dense shard without expanding the whole file."""

    dtype = np.dtype(dtype)
    byte_size = os.path.getsize(filename)
    if byte_size % dtype.itemsize:
        raise ValueError(
            f"shard byte length {byte_size} is not divisible by {dtype.name} "
            f"item size {dtype.itemsize}: {filename}"
        )
    tokens = np.memmap(filename, dtype=dtype, mode="c")
    return torch.from_numpy(tokens)


class DataLoader:
    """Read metadata-described flat token shards in DDP lockstep."""

    def __init__(
        self,
        data_root,
        batch_size,
        block_size,
        split,
        rank=0,
        world_size=1,
        expected_vocab_size=None,
    ):
        self.B = batch_size
        self.T = block_size
        self.rank = rank
        self.world_size = world_size
        self.global_stride = self.B * self.T * self.world_size
        if self.B <= 0 or self.T <= 0 or self.world_size <= 0:
            raise ValueError("batch size, block size, and world size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if split not in {"train", "val"}:
            raise ValueError(f"invalid split: {split!r}")

        self.metadata = read_metadata(data_root, allow_legacy=True)
        self.dtype = parse_shard_dtype(self.metadata["dtype"])
        if self.metadata.get("legacy"):
            warnings.warn(
                f"{data_root} has no metadata.json; assuming legacy uint16 shards",
                RuntimeWarning,
                stacklevel=2,
            )
        elif (
            expected_vocab_size is not None
            and self.metadata["vocab_size"] != expected_vocab_size
        ):
            raise ValueError(
                f"dataset vocab size {self.metadata['vocab_size']} does not match "
                f"expected model/tokenizer vocab size {expected_vocab_size}"
            )

        shards = sorted(
            s
            for s in os.listdir(data_root)
            if f"_{split}_" in s and s.endswith(".bin")
        )
        self.shards = [os.path.join(data_root, s) for s in shards]
        if not self.shards:
            raise FileNotFoundError(
                f"no .bin shards found for split {split!r} in {data_root}"
            )
        self._shard_lengths = [self._token_length(path) for path in self.shards]
        self._shard_steps = [
            max(0, (length - 1) // self.global_stride)
            for length in self._shard_lengths
        ]
        if not any(self._shard_steps):
            raise ValueError(
                "no shard contains one complete global next-token batch: "
                f"need at least {self.global_stride + 1:,} tokens"
            )
        self.reset()

    def _token_length(self, path: str) -> int:
        byte_size = os.path.getsize(path)
        if byte_size % self.dtype.itemsize:
            raise ValueError(
                f"shard byte length is invalid for {self.dtype.name}: {path}"
            )
        return byte_size // self.dtype.itemsize

    def _load_current_shard(self) -> None:
        self.tokens = load_tokens(self.shards[self.current_shard], self.dtype)
        expected_length = self._shard_lengths[self.current_shard]
        if len(self.tokens) != expected_length:
            raise RuntimeError("shard size changed after DataLoader initialization")
        vocab_size = self.metadata.get("vocab_size")
        if vocab_size and len(self.tokens):
            # A bounded, evenly spaced corruption check keeps shard opening fast;
            # the writer and full smoke validator perform exhaustive checks.
            step = max(1, len(self.tokens) // 4096)
            sampled_max = self.tokens[::step].to(torch.int64).max()
            if int(sampled_max) >= vocab_size:
                raise ValueError(
                    f"sampled token ID exceeds metadata vocab size in "
                    f"{self.shards[self.current_shard]}"
                )

    def _position_for_microstep(self, microstep: int) -> int:
        return microstep * self.global_stride + self.rank * self.B * self.T

    def _advance_to_nonempty_shard(self) -> None:
        for _ in range(len(self.shards)):
            if self._shard_steps[self.current_shard] > 0:
                self._load_current_shard()
                self.current_microstep = 0
                self.current_position = self._position_for_microstep(0)
                return
            self.current_shard = (self.current_shard + 1) % len(self.shards)
        raise RuntimeError("no shard can provide a global batch")

    def reset(self):
        self.current_shard = 0
        self._advance_to_nonempty_shard()

    def set_step(self, step, grad_accum_steps):
        """Reproduce the exact shard/microstep position for checkpoint resume."""

        remaining = step * grad_accum_steps
        if remaining < 0:
            raise ValueError("step and grad_accum_steps must be non-negative")
        total_cycle_steps = sum(self._shard_steps)
        if total_cycle_steps <= 0:
            raise RuntimeError("dataset contains no complete global batches")
        remaining %= total_cycle_steps

        for shard_idx, shard_steps in enumerate(self._shard_steps):
            if remaining < shard_steps:
                self.current_shard = shard_idx
                self._load_current_shard()
                self.current_microstep = remaining
                self.current_position = self._position_for_microstep(remaining)
                print(
                    f"Rank {self.rank}: Resuming at Shard {shard_idx}, "
                    f"global microstep {remaining}, index {self.current_position}"
                )
                return
            remaining -= shard_steps
        raise AssertionError("failed to locate a normalized resume position")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_microstep >= self._shard_steps[self.current_shard]:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self._advance_to_nonempty_shard()

        B, T = self.B, self.T
        self.current_position = self._position_for_microstep(self.current_microstep)
        end = self.current_position + B * T + 1
        buf = self.tokens[self.current_position:end]
        if len(buf) != B * T + 1:
            raise RuntimeError("rank-independent shard step calculation is inconsistent")

        x = buf[:-1].view(B, T).to(torch.int64)
        y = buf[1:].view(B, T).to(torch.int64)
        self.current_microstep += 1
        self.current_position = self._position_for_microstep(self.current_microstep)
        return x.pin_memory(), y.pin_memory()
