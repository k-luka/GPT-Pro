import os
import torch
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, data_root, batch_size, block_size, split, rank=0, world_size=1):
        self.B = batch_size
        self.T = block_size
        self.rank = rank
        self.world_size = world_size
        assert split in {"train", "val"}

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # Start at the offset for this specific GPU
        self.current_position = self.B * self.T * self.rank

    def __iter__(self):
        return self

    def __next__(self):
        B, T = self.B, self.T
        
        # Check if we need to switch shards
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            # Reset position: Rank 0 starts at 0, Rank 1 at B*T, etc.
            self.current_position = B * T * self.rank

        buff = self.tokens[self.current_position : self.current_position + B*T + 1]

        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)
        
        # Advance position by total batch size across all GPUs
        self.current_position += B * T * self.world_size
        
        return x, y