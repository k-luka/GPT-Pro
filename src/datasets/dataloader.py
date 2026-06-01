import os
import numpy as np
import torch


def load_tokens(filename: str) -> torch.Tensor:
    """Load a dense uint16 .bin shard into a long tensor."""
    tokens = np.fromfile(filename, dtype=np.uint16)
    return torch.from_numpy(tokens.astype(np.int64))


class DataLoader:
    """
    Reads pre-packed .bin shards produced by prepare_climbmix.py.

    Each shard is a flat uint16 token stream whose length is an exact
    multiple of block_size, so no padding or masking is needed.
    """

    def __init__(self, data_root, batch_size, block_size, split, rank=0, world_size=1):
        self.B = batch_size
        self.T = block_size
        self.rank = rank
        self.world_size = world_size
        assert split in {"train", "val"}

        shards = sorted(
            s for s in os.listdir(data_root) if split in s and s.endswith(".bin")
        )
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert (
            len(self.shards) > 0
        ), f"no .bin shards found for split '{split}' in {data_root}"

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.rank

    def set_step(self, step, grad_accum_steps):
        """Fast-forward the loader to resume from a checkpoint."""
        total_micro_steps = step * grad_accum_steps
        offset = total_micro_steps * (self.B * self.T * self.world_size) + (
            self.rank * self.B * self.T
        )

        for shard_idx, shard_path in enumerate(self.shards):
            shard_len = os.path.getsize(shard_path) // np.dtype(np.uint16).itemsize
            if offset < shard_len:
                self.current_shard = shard_idx
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = offset
                print(
                    f"Rank {self.rank}: Resuming at Shard {shard_idx}, "
                    f"Index {self.current_position}"
                )
                return
            offset -= shard_len

        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        B, T = self.B, self.T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.rank

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.world_size
        return x.pin_memory(), y.pin_memory()
