import torch
from dataclasses import dataclass
from src.data import DataLoader
from src.evaluator import estimate_loss, evaluate_hella_swag
import time
import math
import os
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


@dataclass
class TrainerConfig:
    run_name: str = "test1"
    batch_size: int = 64
    grad_accum_steps: int = 4
    block_size: int = 1024
    max_steps: int = 1000
    warmup_steps: int = 100
    min_lr: float = 1e-4
    max_lr: float = 6e-4
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    logging_steps: int = 1
    checkpoint_interval: int = 1000
    generation_interval: int = 50
    eval_interval: int = 200
    eval_steps: int = 200
    eval_batch_size: int = 64
    eval_block_size: int = 1024
    device: str = "cuda"


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

        # Pass rank and world_size to DataLoader to fix data duplication
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

        self.optimizer = self.model.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            self.config.device,
        )
        self.tokenizer = tokenizer
        self.step = 0

        self.fp8_format = Format.HYBRID
        self.fp8_recipe = DelayedScaling(
            fp8_format=self.fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        # The model is in bf16 so gradients are calculated in bf16. 
        # This is fine but I have high grad accumulation steps which means the accumulated grad can overflow
        # So I calculate in bf16 but accumulate in float32
        for param in self.model.parameters():
            if param.requires_grad:
                param.main_grad = torch.zeros_like(param, dtype=torch.float32)

    def get_lr(self, it):
        if it < self.config.warmup_steps:
            return self.config.max_lr * it / self.config.warmup_steps
        if it > self.config.max_steps:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_steps) / (
            self.config.max_steps - self.config.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)

    def _train_global_batch(self):
        self.optimizer.zero_grad()
        loss_accum = 0.0

        for param in self.model.parameters():
            if param.requires_grad and hasattr(param, "main_grad"):
                param.main_grad.zero_()

        for step in range(self.config.grad_accum_steps):
            x, y = next(self.train_loader)
            x, y = x.to(self.config.device), y.to(self.config.device)
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                with te.autocast(enabled=True, recipe=self.fp8_recipe):
                    _, loss = self.model(x, y)
            loss = (
                loss / self.config.grad_accum_steps
            )  # scale loss as otherwise it would accumulate
            loss_accum += loss.detach()
            loss.backward()
            # Accumulate the grad in float32
            for param in self.model.parameters():
                if param.requires_grad and hasattr(param, "main_grad") and param.main_grad is not None and param.grad is not None:
                    param.main_grad.add_(param.grad.float())
                    param.grad = None
        
        for param in self.model.parameters():
            if param.requires_grad and hasattr(param, "main_grad") and param.grad is None and param.main_grad is not None:
                param.grad = param.main_grad.to(param.dtype)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0
        )  # gradient clipping
        self.optimizer.step()
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        return loss_accum

    def _generate(self):
        """Prints some generations to see how the model is doing"""
        if self.tokenizer is None:
            return

        with FSDP.summon_full_params(self.model, writeback=False, rank0_only=True):
            if self.rank == 0:
                print("\n--- [Generation] -------------------------")
                self.model.eval()
                gen_prompt = self.tokenizer.encode("Hello ")
                context = torch.tensor(
                    gen_prompt, dtype=torch.long, device=self.config.device
                )
                with torch.no_grad():
                    response = self.model.generate(context)
                print(">", self.tokenizer.decode(response[0].tolist()))
                print("------------------------------------------\n")
                self.model.train()

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

        else:
            if self.rank == 0:
                print(f"---| Starting training for {self.config.max_steps} |---")

        best_val_loss = 100
        for step in range(start_step, self.config.max_steps + 1):
            self.step = step
            t0 = time.time()
            # set the learning_rate
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            loss = self._train_global_batch()
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
                # report to wandb
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "train loss": float(loss),
                            "tokens/sec": float(tps),
                            "train step time (ms)": dt * 1000,
                        }
                    )
                # print train loss and stats to console
                if step % self.config.logging_steps == 0:
                    print(
                        f"Step: {step} | loss: {loss:.6f} | dt: {dt * 1000:.4f} ms | tokens/sec: {tps:.4f}"
                    )
            # eval loss and report it
            val_loss = None
            if step % self.config.eval_interval == 0:
                val_loss = estim     ate_loss(
                    self.model,
                    self.val_loader,
                    self.config.eval_steps,
                    self.config.device,
                )
                val_loss_tensor = torch.tensor(val_loss, device=self.config.device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_loss_tensor.item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(val_loss, step, is_best=True)
                hella_acc = evaluate_hella_swag(self.model, self.config.device)
                if self.rank == 0:
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {"val loss": val_loss, "HellaSwag accuracy": hella_acc}
                        )
            # once in a while save checkpoint
            if step % self.config.checkpoint_interval == 0:
                if val_loss is None:
                    val_loss = estimate_loss(
                        self.model,
                        self.val_loader,
                        self.config.eval_steps,
                        self.config.device,
                    )
                    val_loss_tensor = torch.tensor(val_loss, device=self.config.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                self.save_checkpoint(val_loss, step, is_best=False)
            # # once in a while generate from model
            # if step % self.config.generation_interval == 0:
            #     self._generate()

    def save_checkpoint(self, val_loss, step, is_best=False):
        """Saves a checkpoint using PyTorch DCP"""

        checkpoint_path = f"output/checkpoints/{self.config.run_name}/step_{step}"
        os.makedirs(checkpoint_path, exist_ok=True)
        if self.rank == 0:
            print(f"---| Saving checkpoint to {checkpoint_path} |---")

        # UNWRAP torch.compile
        # We must save the underlying FSDP module, not the OptimizedModule wrapper.
        fsdp_model = self.model
        if hasattr(self.model, "_orig_mod"):
            fsdp_model = self.model._orig_mod

        # Create the Stateful Dictionary
        # Wrap 'step' in a tensor so it is a valid stateful object
        step_tensor = torch.tensor(step)
        state_dict = {
            "model": fsdp_model,
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
        """Loads a checkpoint using PyTorch DCP"""
        if self.rank == 0:
            print(f"---| Loading checkpoint from {checkpoint_path} |---")

        # UNWRAP torch.compile
        fsdp_model = self.model
        if hasattr(self.model, "_orig_mod"):
            fsdp_model = self.model._orig_mod

        step_tensor = torch.tensor(0)
        state_dict = {
            "model": fsdp_model,
            "optimizer": self.optimizer,
            "step": step_tensor,
        }
        torch.distributed.checkpoint.load(
            state_dict=state_dict,
            storage_reader=torch.distributed.checkpoint.FileSystemReader(
                checkpoint_path
            ),
        )
