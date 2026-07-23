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
from src.datasets.shard_format import read_metadata, sha256_file
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

    tokenizer_path = cfg.data.get("tokenizer_path", "data/tokenizer/tokenizer.json")
    enc = Tokenizer.from_file(tokenizer_path)
    tokenizer_vocab_size = enc.get_vocab_size()
    if tokenizer_vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"tokenizer vocab size {tokenizer_vocab_size} does not match "
            f"model.vocab_size {cfg.model.vocab_size}"
        )
    tokenizer_sha256 = sha256_file(tokenizer_path)
    for data_root in {cfg.data.train_data_root, cfg.data.val_data_root}:
        metadata = read_metadata(
            data_root, allow_legacy=tokenizer_vocab_size <= 2**16
        )
        if metadata.get("legacy"):
            continue
        if metadata["vocab_size"] != tokenizer_vocab_size:
            raise ValueError(
                f"dataset {data_root} uses vocab size {metadata['vocab_size']}, "
                f"but the configured tokenizer uses {tokenizer_vocab_size}"
            )
        if metadata["tokenizer_sha256"] != tokenizer_sha256:
            raise ValueError(
                f"dataset {data_root} was built with tokenizer SHA "
                f"{metadata['tokenizer_sha256']}, not {tokenizer_sha256}"
            )
        bos_id = enc.token_to_id(metadata["bos_token"])
        if bos_id != metadata["bos_id"]:
            raise ValueError(
                f"dataset {data_root} BOS mapping does not match the tokenizer"
            )

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
        # p-RoPE (Gemma): fraction of frequency pairs that get rotated (1.0=full).
        rope_p=cfg.model.get("rope_p", 1.0),
        # Per-layer-type attention geometry (Gemma-4 style asymmetric heads).
        # head_dim=0 derives from n_embd/n_heads; global_* default to the local
        # values. Keys are optional in the YAML — pass via Hydra +model.<key>=.
        head_dim=cfg.model.get("head_dim", 0),
        global_n_heads=cfg.model.get("global_n_heads", 0),
        global_n_kv_heads=cfg.model.get("global_n_kv_heads", 0),
        global_head_dim=cfg.model.get("global_head_dim", 0),
        rope_theta=cfg.model.get("rope_theta", 10000.0),
        global_rope_theta=cfg.model.get("global_rope_theta", 0.0),
        global_rope_p=cfg.model.get("global_rope_p", 0.0),
        # First-class normalization choice. Older experiment configs may omit
        # this and continue to use the SANDWICH_NORM environment fallback.
        sandwich_norm=cfg.model.get("sandwich_norm", None),
        # First-class forms of the attention A/B environment toggles. Explicit
        # YAML values win; absent keys retain compatibility with old scripts.
        qk_norm_mode=cfg.model.get("qk_norm_mode", None),
        local_attn_impl=cfg.model.get("local_attn_impl", None),
        global_attn_impl=cfg.model.get("global_attn_impl", None),
        global_attn_placement=cfg.model.get("global_attn_placement", None),
        # Weight tying between token embedding and lm_head. Default True; set
        # model.tie_embeddings=false to untie (large-vocab experiments).
        tie_embeddings=cfg.model.get("tie_embeddings", True),
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
        print(
            f"---| Attention kernels: local={model.local_attn_impl}, "
            f"global={model.global_attn_impl}; QK norm={model.qk_norm_mode} |---"
        )
    if master_rank and model.mtp_depth > 0:
        print(
            f"---| MTP: DeepSeek-V3 multi-token prediction, depth={model.mtp_depth} "
            f"(predicts +2..+{model.mtp_depth + 1} tokens), lambda={model.mtp_lambda} |---"
        )
    if master_rank:
        norm_name = "sandwich (pre+post)" if model.sandwich_norm else "pre-norm"
        print(f"---| Normalization: {norm_name} RMSNorm |---")

    torch._dynamo.config.capture_profiler_record_function = True
    model = torch.compile(model)

    # DDP wraps the COMPILED module (not the reverse). This keeps DDP's internal
    # buffer/param broadcast (_broadcast_coalesced, a pybind C++ collective) OUTSIDE
    # the compiled graph, so Dynamo no longer tries to trace it and graph-break at
    # the first forward (the "does not know how to trace ... _broadcast_coalesced"
    # UserWarning). no_sync() (used for grad-accum) needs DDP as the OUTER wrapper —
    # satisfied here — and _unwrap() (DDP.module -> ._orig_mod) still reaches the GPT.
    model = DDP(model, device_ids=[local_rank])

    trainer_config = TrainerConfig(
        run_name=cfg.experiment.run_name,
        batch_size=cfg.training.batch_size,
        block_size=cfg.model.block_size,
        grad_accum_steps=cfg.training.grad_accum_steps,
        max_steps=cfg.training.max_steps,
        stop_after_steps=cfg.training.get("stop_after_steps", None),
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
        eval_core=cfg.training.get("eval_core", False),
        core_max_examples=cfg.training.get("core_max_examples", 1000),
        fp32_grad_accum=cfg.training.get("fp32_grad_accum", False),
        periodic_ckpt_keep_from_frac=cfg.training.get(
            "periodic_ckpt_keep_from_frac", 0.667
        ),
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
