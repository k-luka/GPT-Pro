import sys
import traceback
import torch
from src.models.gpt_dense import GPT
from src.training.trainer_dense import Trainer, TrainerConfig
from src.utils.helpers import print_trainable_parameters, estimate_flops
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, cast
import os
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import timedelta


@hydra.main(version_base=None, config_name="config_dense", config_path="config")
def main(cfg: DictConfig):
    local_rank = int(os.environ["LOCAL_RANK"])
    device_obj = torch.device(f"cuda:{local_rank}")

    dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=device_obj)
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    if rank == 0:
        master_rank = True
        print(OmegaConf.to_yaml(cfg))
    else:
        master_rank = False

    try:
        _run_training(cfg, device_obj, local_rank, master_rank)
    except Exception:
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)


def _run_training(
    cfg: DictConfig, device_obj: torch.device, local_rank: int, master_rank: bool
):
    # init wandb
    wandb_run = None
    if master_rank:
        wandb_run = wandb.init(
            project=cfg.experiment.project,
            name=cfg.experiment.run_name,
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            dir=os.getcwd(),
        )

    # define tokenizer
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # speed up
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA unsupported on pre-Hopper GPUs
    torch._dynamo.config.optimize_ddp = False     # don't trace into DDP internals

    # Define dense model
    model = GPT(
        n_embd=cfg.model.n_embd,
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_heads=cfg.model.n_heads,
        n_kv_heads=cfg.model.n_kv_heads,
        n_layers=cfg.model.n_layers,
        ffn_hidden_size=cfg.model.ffn_hidden_size,
        mtp_depth=cfg.model.get("mtp_depth", 0),
        dtype=torch.bfloat16,
    )
    model.to(device_obj)

    # DDP wrapping (dense model -- every rank holds the full model)
    model = DDP(model, device_ids=[local_rank])

    # Compile for throughput — enable profiler annotation capture to avoid dynamo warnings
    torch._dynamo.config.capture_profiler_record_function = True
    model = torch.compile(model)

    # trainer
    trainer_config = TrainerConfig(
        run_name=cfg.experiment.run_name,
        batch_size=cfg.training.batch_size,
        block_size=cfg.model.block_size,
        grad_accum_steps=cfg.training.grad_accum_steps,
        max_steps=cfg.training.max_steps,
        warmup_steps=cfg.training.warmup_steps,
        warmdown_ratio=cfg.training.get("warmdown_ratio", 0.3),
        min_lr=cfg.training.min_lr,
        max_lr=cfg.training.max_lr,
        weight_decay=cfg.training.weight_decay,
        logging_steps=cfg.training.logging_steps,
        checkpoint_interval=cfg.training.checkpoint_interval,
        generation_interval=cfg.training.generation_interval,
        eval_interval=cfg.training.eval_interval,
        eval_steps=cfg.training.eval_steps,
        eval_batch_size=cfg.training.eval_batch_size,
        eval_block_size=cfg.training.eval_block_size,
        device=str(device_obj),
    )

    trainer = Trainer(
        model=model,
        wandb_run=wandb_run,
        train_data_root=cfg.data.train_data_root,
        val_data_root=cfg.data.val_data_root,
        config=trainer_config,
        tokenizer=enc,
    )

    if master_rank:
        print_trainable_parameters(cfg, model)
        estimate_flops(cfg)

    # check if training from checkpoint
    resume_path = cfg.get("resume_checkpoint", None)

    # train the model
    trainer.train(resume_from_checkpoint=resume_path)

    dist.barrier()
    if master_rank:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
