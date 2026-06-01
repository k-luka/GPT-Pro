"""
Trainer for the torchao MXFP8 dense stack.

Twin of src/training/trainer_dense.py. Differences:
  - No transformer_engine imports.
  - No te.autocast / DelayedScaling / amax_reduction_group. torchao computes
    the MXFP8 (e8m0 microscale) factors per forward pass — the quantize_()
    call in the launcher (pretrain_dense_torchao.py) swaps each hidden Linear's
    weight with an MXFP8TrainingWeightWrapperTensor whose matmul dispatches to
    MXFP8 fwd/bwd kernels. wgrad is kept in BF16 (MXFP8_RCEIL_WGRAD_WITH_HP).
  - Model forward signature has no is_first_microbatch kwarg (no FP8 weight
    cache to prime — scaling is dynamic per microbatch).
  - No torch.autocast wrapper: params and intermediates are already BF16, and
    the MXFP8 forward kernel only accepts BF16 inputs, so autocast upcasting
    through RMSNorm would break it. lm_head is left out of quantize_ and stays
    BF16 by design.

Requires the torch-nightly env (LLM_torchao_nightly): the MXFP8 training tensor
is gated on torch.distributed.tensor internals absent from torch 2.11 release.
"""

import contextlib
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.checkpoint

from src.datasets.dataloader import DataLoader
from src.eval.metrics import estimate_loss, evaluate_hella_swag, evaluate_core

# DCP loads tensors with torch.load(weights_only=True) (the PyTorch 2.6+ secure
# default), which refuses to unpickle custom classes unless they are allowlisted.
# The MXFP8 weights pickle as MXFP8TrainingWeightWrapperTensor, which in turn
# stores an MXFP8TrainingOpConfig (holding KernelPreference / ScaleCalculationMode
# enums). All of these must be allowlisted or resume-from-checkpoint fails. Our
# own checkpoints are trusted, so this is safe.
def _allowlist_torchao_globals():
    safe = []
    try:
        from torchao.prototype.moe_training.tensor import (
            MXFP8TrainingWeightWrapperTensor,
        )

        safe.append(MXFP8TrainingWeightWrapperTensor)
    except Exception:
        pass
    try:
        from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig

        safe.append(MXFP8TrainingOpConfig)
    except Exception:
        pass
    try:
        from torchao.prototype.mx_formats.config import ScaleCalculationMode

        safe.append(ScaleCalculationMode)
    except Exception:
        pass
    try:
        from torchao.quantization.quantize_.common.kernel_preference import (
            KernelPreference,
        )

        safe.append(KernelPreference)
    except Exception:
        pass
    if safe:
        torch.serialization.add_safe_globals(safe)


_allowlist_torchao_globals()


@dataclass
class TrainerConfig:
    run_name: str = "test1"
    batch_size: int = 64
    grad_accum_steps: int = 4
    block_size: int = 1024
    max_steps: int = 1000
    warmup_steps: int = 100
    warmdown_ratio: float = 0.3
    max_lr: float = 6e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    logging_steps: int = 1
    checkpoint_interval: int = 1000
    generation_interval: int = 50
    eval_interval: int = 200
    eval_steps: int = 200
    eval_batch_size: int = 64
    eval_block_size: int = 1024
    eval_hellaswag: bool = True
    eval_core: bool = False
    core_max_examples: int = 1000
    device: str = "cuda"


def _unwrap(model):
    """Strip DDP (and torch.compile) wrappers to reach the underlying GPT module."""
    inner = model.module if hasattr(model, "module") else model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod
    return inner


class Trainer:
    def __init__(
        self,
        model,
        train_data_root,
        val_data_root,
        config: TrainerConfig,
        tokenizer=None,
        wandb_run=None,
    ):
        self.config = config
        self.model = model
        self.wandb_run = wandb_run

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.train_loader = DataLoader(
            train_data_root,
            config.batch_size,
            config.block_size,
            "train",
            rank=self.rank,
            world_size=self.world_size,
        )
        self.val_loader = DataLoader(
            val_data_root,
            config.eval_batch_size,
            config.eval_block_size,
            "val",
            rank=self.rank,
            world_size=self.world_size,
        )

        self.optimizer = _unwrap(self.model).configure_optimizers(
            self.config.weight_decay, self.config.max_lr, self.config.device
        )
        self.tokenizer = tokenizer
        self.step = 0
        self._prefetch_stream = torch.cuda.Stream()
        self._prefetched = None

        # Average UTF-8 bytes per token on the val set, used to report val_bpb
        # (bits-per-byte = val_loss_nats / (ln2 * bytes_per_token)) — a
        # vocab-invariant loss metric (nanochat-style). Computed once from a
        # sample of the loaded val shard.
        self.bytes_per_token = None
        try:
            from scripts.data_prep.hellaswag import _get_enc

            sample = self.val_loader.tokens[:100000].tolist()
            text = _get_enc().decode(sample)
            n_bytes = len(text.encode("utf-8"))
            if n_bytes > 0:
                self.bytes_per_token = n_bytes / len(sample)
        except Exception as e:
            if self.rank == 0:
                print(f"[val_bpb] disabled (could not measure bytes/token: {e})")

    def get_lr(self, it):
        warmdown_start = self.config.max_steps - round(
            self.config.warmdown_ratio * self.config.max_steps
        )
        if it < self.config.warmup_steps:
            return self.config.max_lr * it / self.config.warmup_steps
        elif it < warmdown_start:
            return self.config.max_lr
        else:
            progress = (self.config.max_steps - it) / (
                self.config.max_steps - warmdown_start
            )
            return self.config.min_lr + progress * (
                self.config.max_lr - self.config.min_lr
            )

    def _prefetch(self):
        x, y = next(self.train_loader)
        with torch.cuda.stream(self._prefetch_stream):
            self._prefetched = (
                x.to(self.config.device, non_blocking=True),
                y.to(self.config.device, non_blocking=True),
            )

    def _get_prefetched(self):
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)
        return self._prefetched

    def _train_global_batch(self, grad_accum_steps):
        self.optimizer.zero_grad()
        loss_accum = 0.0

        self._prefetch()

        for micro_step in range(grad_accum_steps):
            x, y = self._get_prefetched()
            is_last = micro_step == grad_accum_steps - 1

            if not is_last:
                self._prefetch()

            torch.compiler.cudagraph_mark_step_begin()

            ctx = contextlib.nullcontext() if is_last else self.model.no_sync()
            with ctx:
                # No autocast: model params and intermediates are already bf16,
                # and autocast can upcast through RMSNorm in ways that break the
                # MXFP8 forward kernel (which only accepts bf16 inputs).
                _, loss = self.model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

        if hasattr(self.optimizer, "get_adamw_params"):
            torch.nn.utils.clip_grad_norm_(self.optimizer.get_adamw_params(), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        return loss_accum

    def train(self, resume_from_checkpoint=None):
        self.model.train()

        start_step = 1
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)
            start_step = self.step + 1
            if self.rank == 0:
                print(
                    f"---| Resuming training from step {start_step} until {self.config.max_steps} |---"
                )
            self.train_loader.set_step(self.step, self.config.grad_accum_steps)

        else:
            if self.rank == 0:
                print(f"---| Starting training for {self.config.max_steps} |---")

        best_val_loss = 100
        for step in range(start_step, self.config.max_steps + 1):
            self.step = step
            t0 = time.time()
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            loss = self._train_global_batch(self.config.grad_accum_steps)
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            tps = (
                self.train_loader.B
                * self.train_loader.T
                * self.config.grad_accum_steps
                * self.world_size
            ) / dt

            if self.rank == 0:
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "train loss": float(loss),
                            "tokens/sec": float(tps),
                            "train step time (ms)": dt * 1000,
                        },
                        step=step,
                    )
                if step % self.config.logging_steps == 0:
                    print(
                        f"Step: {step} | loss: {loss:.6f} | dt: {dt * 1000:.4f} ms | tokens/sec: {tps:.4f}"
                    )
            val_loss = None
            if step % self.config.eval_interval == 0:
                val_loss = estimate_loss(
                    self.model,
                    self.val_loader,
                    self.config.eval_steps,
                    self.config.device,
                    use_autocast=False,  # torchao MXFP8: bf16-only, no autocast
                )
                val_loss_tensor = torch.tensor(val_loss, device=self.config.device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_loss_tensor.item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if step % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(val_loss, step, is_best=True)
                hella_acc = None
                if self.config.eval_hellaswag:
                    hella_acc = evaluate_hella_swag(
                        self.model, self.config.device, use_autocast=False
                    )
                core_results = None
                if self.config.eval_core:
                    core_results = evaluate_core(
                        self.model,
                        self.config.device,
                        use_autocast=False,
                        max_examples=self.config.core_max_examples,
                    )
                val_bpb = (
                    val_loss / (math.log(2) * self.bytes_per_token)
                    if self.bytes_per_token
                    else None
                )
                if self.rank == 0:
                    hella_str = (
                        f"{hella_acc:.4f}" if hella_acc is not None else "skipped"
                    )
                    bpb_str = f"{val_bpb:.4f}" if val_bpb is not None else "n/a"
                    core_str = (
                        f" | CORE: {core_results['core']:.4f}"
                        if core_results and core_results.get("core") is not None
                        else ""
                    )
                    print(
                        f"Step: {step} | val loss: {val_loss:.6f} | val bpb: {bpb_str} | "
                        f"HellaSwag: {hella_str}{core_str} | best val: {best_val_loss:.6f}"
                    )
                    if self.wandb_run is not None:
                        log_dict = {"val loss": val_loss}
                        if val_bpb is not None:
                            log_dict["val bpb"] = val_bpb
                        if hella_acc is not None:
                            log_dict["HellaSwag accuracy"] = hella_acc
                        if core_results is not None:
                            for k, v in core_results.items():
                                if v is not None:
                                    log_dict[f"core/{k}"] = v
                        self.wandb_run.log(log_dict, step=step)
            if step % self.config.checkpoint_interval == 0:
                if val_loss is None:
                    val_loss = estimate_loss(
                        self.model,
                        self.val_loader,
                        self.config.eval_steps,
                        self.config.device,
                        use_autocast=False,  # torchao MXFP8: bf16-only, no autocast
                    )
                    val_loss_tensor = torch.tensor(val_loss, device=self.config.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                self.save_checkpoint(val_loss, step, is_best=False)

    def save_checkpoint(self, val_loss, step, is_best=False):
        if is_best:
            checkpoint_path = f"output/checkpoints/{self.config.run_name}/best_val"
            if self.rank == 0 and os.path.exists(checkpoint_path):
                import shutil

                shutil.rmtree(checkpoint_path)
            dist.barrier()
        else:
            checkpoint_path = f"output/checkpoints/{self.config.run_name}/step_{step}"

        os.makedirs(checkpoint_path, exist_ok=True)
        if self.rank == 0:
            print(f"---| Saving checkpoint to {checkpoint_path} |---")

        base_model = _unwrap(self.model)

        step_tensor = torch.tensor(step)
        state_dict = {
            "model": base_model,
            "optimizer": self.optimizer,
            "step": step_tensor,
        }
        torch.distributed.checkpoint.save(
            state_dict=state_dict,
            storage_writer=torch.distributed.checkpoint.FileSystemWriter(
                checkpoint_path
            ),
        )

    def load_checkpoint(self, checkpoint_path):
        if self.rank == 0:
            print(f"---| Loading checkpoint from {checkpoint_path} |---")

        base_model = _unwrap(self.model)

        step_tensor = torch.tensor(0)
        state_dict = {
            "model": base_model,
            "optimizer": self.optimizer,
            "step": step_tensor,
        }
        torch.distributed.checkpoint.load(
            state_dict=state_dict,
            storage_reader=torch.distributed.checkpoint.FileSystemReader(
                checkpoint_path
            ),
        )
        self.step = step_tensor.item()
