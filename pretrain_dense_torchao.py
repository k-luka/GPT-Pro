"""Pretraining entry point for the torchao MXFP8 dense stack."""

import os
import sys
import traceback
from datetime import timedelta
from typing import Any, cast

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchao.quantization import quantize_
from torchao.prototype.moe_training.config import (
    MXFP8TrainingOpConfig,
    MXFP8TrainingRecipe,
)

from src.models.gpt_dense_torchao import GPT
from src.training.trainer_dense_torchao import Trainer, TrainerConfig
from src.utils.helpers import estimate_flops, print_trainable_parameters


@hydra.main(version_base=None, config_name="config_dense_torchao", config_path="config")
def main(cfg: DictConfig):
    local_rank = int(os.environ["LOCAL_RANK"])
    device_obj = torch.device(f"cuda:{local_rank}")

    dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=device_obj)
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    master_rank = rank == 0
    if master_rank:
        print(OmegaConf.to_yaml(cfg))

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
    wandb_run = None
    if master_rank:
        wandb_run = wandb.init(
            project=cfg.experiment.project,
            name=cfg.experiment.run_name,
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            dir=os.getcwd(),
        )

    from tokenizers import Tokenizer

    enc = Tokenizer.from_file("data/tokenizer/tokenizer.json")

    # Seed all ranks identically so controlled A/B runs (e.g. sandwich-norm on
    # vs off) start from the same weight init. Data order is already
    # deterministic (shard slicing by rank), so this makes the only difference
    # the thing under test. Override with `seed=...` on the CLI.
    seed = cfg.get("seed", 1337)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cuDNN's fused attention is the fastest SDPA backend on B200 (measured
    # ~1.6-3x faster than the FlashAttention-2 backend for our GQA shapes, and
    # numerically equal within bf16 noise). It also handles GQA natively. Leave
    # it enabled — disabling it forces the slower flash backend.
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch._dynamo.config.optimize_ddp = False

    model = GPT(
        n_embd=cfg.model.n_embd,
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_heads=cfg.model.n_heads,
        n_kv_heads=cfg.model.n_kv_heads,
        n_layers=cfg.model.n_layers,
        ffn_hidden_size=cfg.model.ffn_hidden_size,
        # Gemma-style interleaved local/global attention. Both default to 0
        # (every layer global = original dense model) when absent from config.
        sliding_window=cfg.model.get("sliding_window", 0),
        global_attn_every_n=cfg.model.get("global_attn_every_n", 0),
        # DeepSeek-V3 multi-token prediction (training-only auxiliary heads).
        mtp_depth=cfg.model.get("mtp_depth", 0),
        mtp_lambda=cfg.model.get("mtp_lambda", 0.3),
        dtype=torch.bfloat16,
    )
    model.to(device_obj)

    # MXFP8 conversion: swap weight Parameters of all hidden Linears with a
    # tensor subclass that dispatches matmuls to MXFP8 forward/backward kernels.
    # lm_head is excluded so the output projection stays BF16.
    config = MXFP8TrainingOpConfig.from_recipe(
        MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP
    )
    # torchao logs one INFO line per swapped Linear ("Swapped .weight to ...") —
    # one per layer, so dozens-to-hundreds of lines on a real model. That's
    # confirmation, not a warning. Raise the whole `torchao` logger to WARNING so
    # the per-layer spam is gone; we print a single count below instead. Anything
    # genuinely wrong still surfaces (WARNING/ERROR). Done after Hydra has already
    # configured logging (main() -> here) so it isn't reset out from under us.
    import logging

    logging.getLogger("torchao").setLevel(logging.WARNING)
    quantize_(
        model,
        config,
        filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear)
        and "lm_head" not in fqn,
    )
    if master_rank:
        from torchao.prototype.moe_training.tensor import (
            TrainingWeightWrapperBaseTensor,
        )

        n_mxfp8 = sum(
            isinstance(p.data, TrainingWeightWrapperBaseTensor)
            for p in model.parameters()
        )
        print(
            f"---| torchao: MXFP8 training enabled (RCEIL, wgrad in BF16) | "
            f"{n_mxfp8} Linear weights swapped |---"
        )

    # Pre-build sliding-window BlockMasks for the train and eval sequence lengths
    # now (on-device, eager) so the compiled forward only reads cached masks. A
    # no-op unless local attention is configured.
    model.build_attention_masks(
        seq_lens=[cfg.model.block_size, cfg.training.eval_block_size],
        device=device_obj,
    )
    if master_rank and model.has_local_layers:
        n_global = sum(model.is_global)
        print(
            f"---| Attention: sliding-window local (W={model.sliding_window}) "
            f"with {n_global}/{model.n_layers} global layers "
            f"(global at {[i for i, g in enumerate(model.is_global) if g]}) |---"
        )
    if master_rank and model.mtp_depth > 0:
        print(
            f"---| MTP: DeepSeek-V3 multi-token prediction, depth={model.mtp_depth} "
            f"(predicts +2..+{model.mtp_depth + 1} tokens), lambda={model.mtp_lambda} |---"
        )

    model = DDP(model, device_ids=[local_rank])

    torch._dynamo.config.capture_profiler_record_function = True
    model = torch.compile(model)

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
        eval_hellaswag=cfg.training.get("eval_hellaswag", True),
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

    resume_path = cfg.get("resume_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_path)

    dist.barrier()
    if master_rank:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
