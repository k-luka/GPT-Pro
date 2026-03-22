import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=5),
        device_id=device,
    )

    x = torch.tensor([rank], device=device, dtype=torch.int32)
    gathered = torch.empty(world_size, device=device, dtype=torch.int32)
    print(
        f"[NCCL_SMOKE] rank={rank} local_rank={local_rank} before_all_gather",
        flush=True,
    )
    dist.all_gather_into_tensor(gathered, x)
    print(
        f"[NCCL_SMOKE] rank={rank} gathered={gathered.tolist()}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
