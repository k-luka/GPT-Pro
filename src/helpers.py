import torch
import torch.nn.functional as F
import os
import humanize
import torch.distributed as dist

def _count_unique_params(params):
    seen = set()
    total = 0
    for param in params:
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        total += param.numel()
    return total

def _unwrap_fsdp(module):
    return getattr(module, "_fsdp_wrapped_module", module)

def _iter_moe_modules(model):
    seen = set()
    root = _unwrap_fsdp(model)
    for module in root.modules():
        candidate = _unwrap_fsdp(module)
        mid = id(candidate)
        if mid in seen:
            continue
        seen.add(mid)
        if (
            hasattr(candidate, "n_routed_experts")
            and hasattr(candidate, "topk")
            and hasattr(candidate, "routed_fused_proj")
            and hasattr(candidate, "routed_down_proj")
        ):
            yield candidate

def _count_routed_params(moe_module):
    routed_params = []
    for name, param in moe_module.named_parameters(recurse=True):
        if "routed_fused_proj" in name or "routed_down_proj" in name:
            routed_params.append(param)
    return _count_unique_params(routed_params)

def print_trainable_parameters(model):
    total_params = _count_unique_params(model.parameters())
    trainable_params = _count_unique_params(
        p for p in model.parameters() if p.requires_grad
    )
    inactive_routed_params = 0.0
    world_size = dist.get_world_size()
    for moe in _iter_moe_modules(model):
        try:
            n_routed = int(moe.n_routed_experts)
            topk = int(moe.topk)
        except (TypeError, ValueError):
            continue
        if n_routed <= 0 or topk <= 0:
            continue
        routed_params = _count_routed_params(moe)
        active_frac = min(topk / n_routed, 1.0)
        inactive_routed_params += routed_params * (1.0 - active_frac)
    active_params = total_params - inactive_routed_params
    active_params_int = int(round(active_params))
    inactive_params_int = max(total_params - active_params_int, 0)
    print("| -----------------------------")
    print(
        f"| Total parameters: {humanize.intword(total_params * world_size)} ({total_params * world_size:,})"
    )
    print(
        f"| Active parameters (MoE top-k): {humanize.intword(active_params_int * world_size)} ({active_params_int * world_size:,})"
    )
    print(
        f"| Inactive parameters (MoE): {humanize.intword(inactive_params_int * world_size)} ({inactive_params_int * world_size:,})"
    )
    print(
        f"| Trainable parameters: {humanize.intword(trainable_params * world_size)} ({trainable_params * world_size:,})"
    )
    print(
        f"| Total parameters per GPU: {humanize.intword(total_params)} ({total_params:,})"
    )
    print(
        f"| Active parameters per GPU (MoE top-k): {humanize.intword(active_params_int)} ({active_params_int:,})"
    )
    print(
        f"| Inactive parameters per GPU (MoE): {humanize.intword(inactive_params_int)} ({inactive_params_int:,})"
    )
    print(
        f"| Trainable parameters per GPU: {humanize.intword(trainable_params)} ({trainable_params:,})"
    )

def estimate_flops(model, cfg):
    """ Prints the estimated number of FLOPs per token for the model and for the run. Ref: https://arxiv.org/abs/2204.02311 """
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = model.wte.weight.numel()
    l, h, q, t = cfg.model.n_layers, cfg.model.n_heads, cfg.model.n_embd // cfg.model.n_heads, cfg.model.block_size
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
    print("| -----------------------------")
    total_tokens = cfg.training.max_steps * cfg.training.batch_size * cfg.model.block_size * cfg.training.grad_accum_steps
    print(f"| Total tokens to be used for training: {humanize.intword(total_tokens)} ({total_tokens:,}) out of 350 billion in the dataset.")
    print(f"| FLOPs per token: {humanize.intword(num_flops_per_token)} ({num_flops_per_token:,}).")
    total_flops = num_flops_per_token * total_tokens
    print(f"| Total FLOPs for the training run: {humanize.intword(total_flops)} ({total_flops:,}).")
    print("| -----------------------------")

def norm(x):
    # RMS norm
    return F.rms_norm(x, (x.size(-1),))

# Applies RoPE
def apply_rotary_emb(x, sin, cos):
    assert x.ndim == 4 # must be attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up the time into two halves
    with torch.autocast(device_type=x.device.type, dtype=torch.float32):
        y1 = cos * x1 + sin * x2
        y2 = (-sin) * x1 + cos * x2
    out = torch.cat([y1,y2], 3)
    out = out.to(x.dtype)
    return out


def save_checkpoint(model, run_name, step, val_loss):
    checkpoint_dir = "output/checkpoints"
    checkpoint_dir = os.path.join(checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:05d}.pt")
    checkpoint = {
                    'model': model.state_dict(),
                    'step': step,
                    'val_loss': val_loss
    }
    # TODO: add optimizer state
    torch.save(checkpoint, checkpoint_path)

def save_best_checkpoint(model, run_name, step, val_loss):
    checkpoint_dir = "output/checkpoints"
    checkpoint_dir = os.path.join(checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_best_val.pt")
    checkpoint = {
                    'model': model.state_dict(),
                    'step': step,
                    'val_loss': val_loss
    }
    # TODO: add optimizer state
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, checkpoint_path, device="cpu"):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']

    prefix = '_orig_mod.'
    new_state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)

# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
