import math
import os
import random
import time
from typing import Any, cast

import hydra
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf

from src.te_versions.model_te import GPT


def load_sft_data(file_path: str):
	return torch.load(file_path, map_location="cpu")


def sample_batch(data, batch_size: int, block_size: int, device: torch.device):
	x = torch.zeros((batch_size, block_size), dtype=torch.long)
	y = torch.zeros((batch_size, block_size), dtype=torch.long)
	loss_mask = torch.zeros((batch_size, block_size), dtype=torch.bool)

	for i in range(batch_size):
		item = data[random.randrange(0, len(data))]
		tokens = item["tokens"]
		if not torch.is_tensor(tokens):
			tokens = torch.tensor(tokens, dtype=torch.long)
		tokens = tokens.to(dtype=torch.long)

		if tokens.numel() < 2:
			continue

		seq_len = min(block_size, tokens.numel() - 1)
		x[i, :seq_len] = tokens[:seq_len]
		y[i, :seq_len] = tokens[1 : seq_len + 1]

		# With shifted labels, supervision begins when target index reaches mask_len.
		start = max(0, min(seq_len, int(item["mask_len"]) - 1))
		loss_mask[i, start:seq_len] = True

	return x.to(device), y.to(device), loss_mask.to(device)


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor):
	bsz, seq_len, vocab = logits.shape
	losses = F.cross_entropy(
		logits.view(bsz * seq_len, vocab),
		targets.view(bsz * seq_len),
		reduction="none",
	).view(bsz, seq_len)

	mask = loss_mask.float()
	denom = mask.sum().clamp(min=1.0)
	return (losses * mask).sum() / denom


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
	if step < warmup_steps:
		return max_lr * float(step + 1) / float(max(1, warmup_steps))
	if step >= max_steps:
		return min_lr
	decay_ratio = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return min_lr + coeff * (max_lr - min_lr)


def load_pretrained_checkpoint(model, optimizer, checkpoint_path: str, device: torch.device):
	if os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, ".metadata")):
		try:
			state = {"model": model.state_dict()}
			dcp.load(
				state_dict=state,
				storage_reader=dcp.FileSystemReader(checkpoint_path),
				no_dist=True,
			)
			model.load_state_dict(state["model"], strict=False)
			print(f"Loaded model weights from DCP checkpoint: {checkpoint_path}")
			return 0, float("inf")
		except Exception as exc:
			fallback_pt = f"{checkpoint_path}_single.pt"
			if os.path.exists(fallback_pt):
				print(
					f"DCP load failed ({exc}). Falling back to converted checkpoint: {fallback_pt}"
				)
				checkpoint_path = fallback_pt
			else:
				raise

	checkpoint = torch.load(checkpoint_path, map_location=device)
	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		model.load_state_dict(checkpoint["model_state_dict"], strict=False)
		if "optimizer_state_dict" in checkpoint:
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		step = int(checkpoint.get("step", 0))
		best_val = float(checkpoint.get("best_val_loss", float("inf")))
		print(f"Loaded training checkpoint: {checkpoint_path}")
		return step, best_val

	if isinstance(checkpoint, dict) and "model" in checkpoint:
		model_payload = checkpoint["model"]
		if isinstance(model_payload, dict) and "model_state_dict" in model_payload:
			model_payload = model_payload["model_state_dict"]
		if isinstance(model_payload, dict):
			model.load_state_dict(model_payload, strict=False)
			print(f"Loaded converted checkpoint (model key): {checkpoint_path}")
			return 0, float("inf")

	if isinstance(checkpoint, dict):
		model.load_state_dict(checkpoint, strict=False)
		print(f"Loaded state_dict checkpoint: {checkpoint_path}")
		return 0, float("inf")

	raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def save_sft_checkpoint(model, optimizer, step: int, best_val_loss: float, run_name: str, is_best: bool):
	ckpt_dir = os.path.join("output", "checkpoints", run_name)
	os.makedirs(ckpt_dir, exist_ok=True)

	payload = {
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"step": step,
		"best_val_loss": best_val_loss,
	}

	latest_path = os.path.join(ckpt_dir, "latest.pt")
	torch.save(payload, latest_path)

	if is_best:
		best_path = os.path.join(ckpt_dir, "best.pt")
		torch.save(payload, best_path)


@torch.no_grad()
def evaluate(model, val_data, cfg: DictConfig, device: torch.device):
	model.eval()
	losses = []

	for _ in range(cfg.training.eval_steps):
		x, y, loss_mask = sample_batch(
			val_data,
			cfg.training.eval_batch_size,
			cfg.model.block_size,
			device,
		)
		with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
			logits, _ = model(x)
			loss = masked_ce_loss(logits, y, loss_mask)
		losses.append(loss.item())

	model.train()
	return sum(losses) / len(losses)


@hydra.main(version_base=None, config_name="config_sft", config_path="config")
def main(cfg: DictConfig):
	dist.init_process_group("nccl")
	rank = dist.get_rank()
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	device = torch.device(f"cuda:{local_rank}")
	master_rank = rank == 0

	if master_rank:
		print(OmegaConf.to_yaml(cfg))
	torch.set_float32_matmul_precision("high")
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	model = GPT(
		n_embd=cfg.model.n_embd,
		vocab_size=cfg.model.vocab_size,
		block_size=cfg.model.block_size,
		n_heads=cfg.model.n_heads,
		head_size=cfg.model.head_size,
		rope_head_size=cfg.model.rope_head_size,
		kv_latent_size=cfg.model.kv_latent_size,
		q_latent_size=cfg.model.q_latent_size,
		n_layers=cfg.model.n_layers,
		n_shared_experts=cfg.model.n_shared_experts,
		n_routed_experts=cfg.model.n_routed_experts,
		topk_experts=cfg.model.topk_experts,
		expert_hidden_size=cfg.model.expert_hidden_size,
		dtype=torch.bfloat16,
	).to(device)

	optimizer = model.configure_optimizers(
		weight_decay=cfg.training.weight_decay,
		learning_rate=cfg.training.max_lr,
		device_type=device.type,
	)

	start_step, best_val_loss = load_pretrained_checkpoint(
		model, optimizer, str(cfg.resume_checkpoint), device
	)

	train_data = load_sft_data(cfg.data.train_file)
	val_data = load_sft_data(cfg.data.val_file)
	wandb_run = None
	if master_rank:
		wandb_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
		wandb_run = wandb.init(
			project=cfg.experiment.project,
			name=cfg.experiment.run_name,
			config=wandb_config,
			dir=os.getcwd(),
		)

	model.train()
	if master_rank:
		print(f"Starting SFT from step {start_step + 1} to {cfg.training.max_steps}")

	for step in range(start_step + 1, cfg.training.max_steps + 1):
		t0 = time.time()

		lr = get_lr(
			step=step,
			warmup_steps=cfg.training.warmup_steps,
			max_steps=cfg.training.max_steps,
			max_lr=cfg.training.max_lr,
			min_lr=cfg.training.min_lr,
		)
		for group in optimizer.param_groups:
			group["lr"] = lr

		optimizer.zero_grad(set_to_none=True)
		train_loss = 0.0

		for _ in range(cfg.training.grad_accum_steps):
			x, y, loss_mask = sample_batch(
				train_data,
				cfg.training.batch_size,
				cfg.model.block_size,
				device,
			)
			with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
				logits, _ = model(x)
				loss = masked_ce_loss(logits, y, loss_mask)

			(loss / cfg.training.grad_accum_steps).backward()
			train_loss += loss.item()

		torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
		optimizer.step()

		dt = time.time() - t0
		train_loss = train_loss / cfg.training.grad_accum_steps

		if step % cfg.training.logging_steps == 0:
			tps = (
				cfg.training.batch_size
				* cfg.model.block_size
				* cfg.training.grad_accum_steps
				/ max(dt, 1e-6)
			)
			if master_rank:
				print(
					f"step {step:6d} | train_loss {train_loss:.6f} | lr {lr:.3e} | "
					f"dt {dt*1000:.1f} ms | tokens/sec {tps:.1f}"
				)

		if master_rank and wandb_run is not None:
			wandb_run.log(
				{
					"train/loss": train_loss,
					"train/lr": lr,
					"train/step_time_ms": dt * 1000,
				},
				step=step,
			)

		if step % cfg.training.eval_interval == 0:
			val_loss = evaluate(model, val_data, cfg, device)
			is_best = val_loss < best_val_loss
			if is_best:
				best_val_loss = val_loss
			if master_rank:
				print(f"step {step:6d} | val_loss {val_loss:.6f} | best {best_val_loss:.6f}")

			if master_rank and wandb_run is not None:
				wandb_run.log(
					{
						"val/loss": val_loss,
						"val/best_loss": best_val_loss,
					},
					step=step,
				)

			save_sft_checkpoint(
				model=model,
				optimizer=optimizer,
				step=step,
				best_val_loss=best_val_loss,
				run_name=cfg.experiment.run_name,
				is_best=is_best,
			)
		elif step % cfg.training.checkpoint_interval == 0:
			save_sft_checkpoint(
				model=model,
				optimizer=optimizer,
				step=step,
				best_val_loss=best_val_loss,
				run_name=cfg.experiment.run_name,
				is_best=False,
			)

	if master_rank and wandb_run is not None:
		wandb_run.finish()

	dist.destroy_process_group()


if __name__ == "__main__":
	main()
