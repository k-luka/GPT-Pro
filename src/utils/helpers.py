import math
import humanize
import torch
import torch.distributed as dist
import os
import torch.distributed.checkpoint as dcp


def print_trainable_parameters(cfg, model):
    """
    Estimates parameters based on configuration.
    """
    # 1. Architecture Constants
    n_layers = cfg.model.n_layers
    n_embd = cfg.model.n_embd
    vocab_size = cfg.model.vocab_size
    n_heads = cfg.model.n_heads
    n_kv_heads = cfg.model.n_kv_heads
    head_dim = n_embd // n_heads

    # 2. Embedding + Head
    params_emb = vocab_size * n_embd
    params_ln_f = n_embd

    # 3. Parameters per Block
    # --- GQA Attention ---
    params_q = n_embd * (n_heads * head_dim)
    params_k = n_embd * (n_kv_heads * head_dim)
    params_v = n_embd * (n_kv_heads * head_dim)
    params_proj = (n_heads * head_dim) * n_embd
    params_qk_norms = 2 * head_dim

    params_gqa = params_q + params_k + params_v + params_proj + params_qk_norms

    # --- MoE (Shared + Routed) ---
    s_hidden_req = cfg.model.get("n_shared_experts", 0) * cfg.model.get(
        "expert_hidden_size", 0
    )
    s_hidden = (s_hidden_req + 255) // 256 * 256
    params_shared = (n_embd * s_hidden) + (n_embd * s_hidden) + (s_hidden * n_embd)

    n_routed = cfg.model.get("n_routed_experts", 0)
    topk = cfg.model.get("topk_experts", 0)
    expert_hidden = cfg.model.get("expert_hidden_size", 0)

    params_per_expert = 3 * n_embd * expert_hidden

    params_gate = n_embd * n_routed

    params_moe_total = params_shared + params_gate + (n_routed * params_per_expert)
    params_moe_active = params_shared + params_gate + (topk * params_per_expert)

    # Block Layer Norms
    params_block_ln = 2 * n_embd

    # --- Layer Totals ---
    params_layer_total = params_gqa + params_moe_total + params_block_ln
    params_layer_active = params_gqa + params_moe_active + params_block_ln

    # 4. Final Sums
    total_params = params_emb + (n_layers * params_layer_total) + params_ln_f
    active_params = params_emb + (n_layers * params_layer_active) + params_ln_f

    params_per_shard = 0
    for i, param in enumerate(model.parameters()):
        params_per_shard += param.numel()

    print("| --------------------------------------------------------------------")
    print(f"| Config: {cfg.experiment.run_name}")
    print(f"| Architecture: {n_layers} layers, {n_heads} heads ({n_kv_heads} KV), {n_embd} dim")
    print(
        f"| Experts: {n_routed} routed, {cfg.model.get('n_shared_experts', 0)} shared, TopK: {topk}"
    )
    print("| --------------------------------------------------------------------")
    print(
        f"| Total Params (Storage):      {humanize.intword(total_params)} ({total_params:,})"
    )
    print(
        f"| Active Params (Forward):     {humanize.intword(active_params)} ({active_params:,})"
    )
    print(
        f"| True Params per GPU:         {humanize.intword(params_per_shard)} ({params_per_shard:,})"
    )
    print(f"| Utilization:                 {active_params/total_params:.1%}")


def estimate_flops(cfg):
    """Prints the estimated number of FLOPs per token for the model and for the run."""
    n_layers = cfg.model.n_layers
    n_embd = cfg.model.n_embd
    n_heads = cfg.model.n_heads
    n_kv_heads = cfg.model.n_kv_heads
    head_dim = n_embd // n_heads

    # GQA Params per layer
    params_q = n_embd * (n_heads * head_dim)
    params_k = n_embd * (n_kv_heads * head_dim)
    params_v = n_embd * (n_kv_heads * head_dim)
    params_proj = (n_heads * head_dim) * n_embd
    params_qk_norms = 2 * head_dim
    params_gqa = params_q + params_k + params_v + params_proj + params_qk_norms

    # MoE Active Params per layer
    s_hidden_req = cfg.model.get("n_shared_experts", 0) * cfg.model.get(
        "expert_hidden_size", 0
    )
    s_hidden = (s_hidden_req + 255) // 256 * 256
    params_shared = (n_embd * s_hidden) * 3

    topk = cfg.model.get("topk_experts", 0)
    expert_hidden = cfg.model.get("expert_hidden_size", 0)
    params_per_expert = 3 * n_embd * expert_hidden

    params_gate = n_embd * cfg.model.get("n_routed_experts", 0)

    params_moe_active = params_shared + params_gate + (topk * params_per_expert)
    params_block_ln = 2 * n_embd

    active_params_per_layer = params_gqa + params_moe_active + params_block_ln
    active_body_params = n_layers * active_params_per_layer

    l, t = n_layers, cfg.model.block_size

    num_flops_per_token = 6 * active_body_params + 12 * l * n_heads * head_dim * t

    print("| --------------------------------------------------------------------")

    # Account for world_size (number of GPUs)
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()

    # Assuming cfg.training.batch_size is PER-DEVICE batch size
    total_tokens = (
        cfg.training.max_steps
        * cfg.training.batch_size
        * cfg.model.block_size
        * cfg.training.grad_accum_steps
        * world_size
    )

    print(
        f"| Total tokens to be used for training: {humanize.intword(total_tokens)} ({total_tokens:,})"
    )
    print(
        f"| FLOPs per token: {humanize.intword(num_flops_per_token)} ({num_flops_per_token:,})."
    )
    total_flops = num_flops_per_token * total_tokens
    print(
        f"| Total FLOPs for the training run: {humanize.intword(total_flops)} ({total_flops:,})."
    )
    print("| --------------------------------------------------------------------")


def apply_rotary_emb(x, sin, cos):
    """
    Standard RoPE application.
    Expects x to be (B, H, T, D) or broadcastable.
    sin, cos are precomputed and passed in.
    """
    # x is (B, H, T, head_dim)
    # chunk into two halves for the rotation
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]

    # Standard RoPE rotation formula
    # [-x2, x1] * sin + [x1, x2] * cos
    return torch.cat((-x2, x1), dim=-1) * sin + x * cos


def save_checkpoint(self, val_loss, step, is_best=False):
    checkpoint_path = f"output/checkpoints/{self.config.run_name}/step_{step}"
    os.makedirs(checkpoint_path, exist_ok=True)

    if self.rank == 0:
        print(f"---| Saving checkpoint to {checkpoint_path} |---")

    # Unwrap OptimizedModule if torch.compile is used
    fsdp_model = self.model
    if hasattr(self.model, "_orig_mod"):
        fsdp_model = self.model._orig_mod

    step_tensor = torch.tensor(step)
    state_dict = {"model": fsdp_model, "optimizer": self.optimizer, "step": step_tensor}

    dcp.save(
        state_dict=state_dict, storage_writer=dcp.FileSystemWriter(checkpoint_path)
    )


def load_checkpoint(self, checkpoint_path):
    if self.rank == 0:
        print(f"---| Loading checkpoint from {checkpoint_path} |---")

    # Unwrap OptimizedModule if torch.compile is used
    fsdp_model = self.model
    if hasattr(self.model, "_orig_mod"):
        fsdp_model = self.model._orig_mod

    step_tensor = torch.tensor(0)
    state_dict = {"model": fsdp_model, "optimizer": self.optimizer, "step": step_tensor}

    dcp.load(
        state_dict=state_dict, storage_reader=dcp.FileSystemReader(checkpoint_path)
    )

    self.step = step_tensor.item()
    if self.rank == 0:
        print(f"---| Loaded successfully from step {self.step} |---")
