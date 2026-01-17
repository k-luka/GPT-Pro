import torch
from src.te_versions.model_te import GPT, Block
from src.te_versions.trainer_te import Trainer, TrainerConfig
from src.helpers import print_trainable_parameters, estimate_flops
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, cast
import functools
import os
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

def get_auto_wrap_policy():
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )


@hydra.main(version_base=None, config_name="config_pretrain", config_path="config")
def main(cfg: DictConfig):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device_obj = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        master_rank = True
        print(OmegaConf.to_yaml(cfg))
    else:
        master_rank = False

    # init wandb
    wandb_run = None
    if master_rank:
        wandb_run = wandb.init(
            project=cfg.experiment.project,
            name=cfg.experiment.run_name, 
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            dir=os.getcwd() 
        )

    # define tokenizer
    import tiktoken
    enc = tiktoken.encoding_for_model('gpt2')

    # speed up
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )

    # Define model
    model = GPT(
        n_embd = cfg.model.n_embd,
        vocab_size = cfg.model.vocab_size,
        block_size = cfg.model.block_size,
        n_heads = cfg.model.n_heads,
        head_size = cfg.model.head_size,
        rope_head_size = cfg.model.rope_head_size,
        kv_latent_size = cfg.model.kv_latent_size,
        q_latent_size = cfg.model.q_latent_size,
        n_layers = cfg.model.n_layers,
        n_shared_experts = cfg.model.n_shared_experts,
        n_routed_experts = cfg.model.n_routed_experts,
        topk_experts = cfg.model.topk_experts,
        expert_hidden_size = cfg.model.expert_hidden_size,
        dtype = torch.bfloat16
    )
    
    # Move to device BEFORE FSDP wrapping
    model.to(device_obj)

    model = FSDP(
        model,
        auto_wrap_policy=get_auto_wrap_policy(),
        device_id=device_obj,
        use_orig_params=True,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD
    )

    # model = torch.compile(model)

    # trainer
    trainer_config = TrainerConfig(
        run_name=cfg.experiment.run_name,
        batch_size=cfg.training.batch_size,
        block_size=cfg.model.block_size,
        grad_accum_steps=cfg.training.grad_accum_steps,
        max_steps=cfg.training.max_steps,
        warmup_steps=cfg.training.warmup_steps,
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

    if master_rank:
        wandb.finish()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()