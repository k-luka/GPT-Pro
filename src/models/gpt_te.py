import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
from src.utils.helpers import apply_rotary_emb
import torch.distributed as dist
import transformer_engine.pytorch as te
import json
import os


class MLA(nn.Module):
    def __init__(
        self,
        n_embd,
        n_heads,
        head_size,
        rope_head_size,
        kv_latent_size,
        q_latent_size,
        dtype=None,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_size = head_size
        self.rope_head_size = rope_head_size
        self.latent_head_size = head_size - rope_head_size
        self.kv_latent_size = kv_latent_size
        self.q_latent_size = q_latent_size

        # Fused Down-Projection
        # Projects x into ALL latent vectors at once: (Q_latent | KV_latent | K_rope)
        self.w_down = te.Linear(
            n_embd,
            q_latent_size + kv_latent_size + rope_head_size,
            bias=False,
            params_dtype=dtype,
        )

        self.q_norm = nn.RMSNorm(q_latent_size, dtype=dtype)
        self.kv_norm = nn.RMSNorm(kv_latent_size, dtype=dtype)

        # Up-projections
        self.w_up_qr = te.Linear(
            q_latent_size,
            n_heads * (self.latent_head_size + rope_head_size),
            bias=False,
            params_dtype=dtype,
        )
        self.w_up_kv = te.Linear(
            kv_latent_size,
            n_heads * (self.latent_head_size + head_size),
            bias=False,
            params_dtype=dtype,
        )

        self.proj = te.Linear(
            n_heads * head_size, n_embd, bias=False, params_dtype=dtype
        )
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True

    def forward(self, x, sin, cos):
        B, T, _ = x.shape
        H = self.n_heads
        d_c = self.latent_head_size

        # Fused Projection & Split
        fused_down = self.w_down(x)
        c_q, c_kv, k_rope = fused_down.split(
            [self.q_latent_size, self.kv_latent_size, self.rope_head_size], dim=-1
        )

        # Normalization
        c_q = self.q_norm(c_q)
        c_kv = self.kv_norm(c_kv)

        # RoPE Shared Key (Optimization: Rotate the tiny shared key once)
        # Reshape to (B, 1, T, 64) so it broadcasts to all heads later
        k_rope = apply_rotary_emb(k_rope.unsqueeze(1), sin, cos)

        # Generate Query (Q)
        # Project up -> Split into Content & RoPE parts
        q_lr = self.w_up_qr(c_q).view(B, T, H, -1).transpose(1, 2)
        q_l = q_lr[..., :d_c]
        q_r = q_lr[..., d_c:]

        q_r = apply_rotary_emb(q_r, sin, cos)
        q = torch.cat((q_l, q_r), dim=-1)  # (B, H, T, head_size)

        # Generate Key/Value (K, V)
        # Project up -> Split into Key-Content & Value
        kv = self.w_up_kv(c_kv).view(B, T, H, -1).transpose(1, 2)

        k_l = kv[..., :d_c]
        v = kv[..., d_c:]  # Value is ready

        # Combine Key-Content with the Shared RoPE Key
        k = torch.cat((k_l, k_rope.expand(B, H, T, -1)), dim=-1)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(out)


# Attention (Kept as fallback/reference)
class Attention(nn.Module):
    def __init__(self, n_embd, n_heads, dtype=None):
        super().__init__()
        assert (
            n_embd % n_heads == 0
        ), f"Embedding dim ({n_embd}) must be divisible by number of heads ({n_heads})."
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.H = n_embd // n_heads  # head size
        self.attn = te.Linear(n_embd, 3 * n_embd, bias=False, params_dtype=dtype)
        self.proj = te.Linear(n_embd, n_embd, bias=False, params_dtype=dtype)
        self.q_norm = te.RMSNorm(self.H, params_dtype=dtype)
        self.k_norm = te.RMSNorm(self.H, params_dtype=dtype)
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True

    def forward(self, x, sin, cos):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_heads, self.H).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.H).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.H).transpose(1, 2)
        # Apply RoPE
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)
        q, k = self.q_norm(q), self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# Standard MLP (Keeps te.Linear for speed on full batches)
class MLP(nn.Module):
    def __init__(self, n_embd, dtype=None):
        super().__init__()
        self.n_embd = n_embd
        hidden_dim = int(2 * n_embd)
        self.hidden_dim = (hidden_dim + 255) // 256 * 256

        self.gate_proj = te.Linear(
            n_embd, self.hidden_dim, bias=False, params_dtype=dtype
        )
        self.up_proj = te.Linear(
            n_embd, self.hidden_dim, bias=False, params_dtype=dtype
        )
        self.down_proj = te.Linear(
            self.hidden_dim, n_embd, bias=False, params_dtype=dtype
        )
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        x = gate * value
        x = self.down_proj(x)
        return x


# Same as MLP but takes a parameter for hidden_size
class SharedExpert(nn.Module):
    def __init__(self, n_embd, hidden_size, dtype=None):
        super().__init__()
        self.n_embd = n_embd
        self.hidden_dim = (hidden_size + 255) // 256 * 256

        self.gate_proj = te.Linear(
            n_embd, self.hidden_dim, bias=False, params_dtype=dtype
        )
        self.up_proj = te.Linear(
            n_embd, self.hidden_dim, bias=False, params_dtype=dtype
        )
        self.down_proj = te.Linear(
            self.hidden_dim, n_embd, bias=False, params_dtype=dtype
        )
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        x = gate * self.up_proj(x)
        return self.down_proj(x)


# Decides which experts will be used
class Gate(nn.Module):
    def __init__(self, n_embd, topk, n_routed_experts, route_scale=1.0, dtype=None):
        super().__init__()
        self.n_embd = n_embd
        self.topk = topk
        self.route_scale = route_scale
        self.weight = nn.Linear(n_embd, n_routed_experts, bias=False, dtype=dtype)
        self.register_buffer("bias", torch.zeros(n_routed_experts, dtype=dtype))

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.weight(x)
        scores = logits.sigmoid()

        # Break the link between the computation and the storage buffer.
        bias_term = self.bias.detach()  # pyrefly: ignore

        topk_idx = torch.topk(scores + bias_term, self.topk, dim=-1)[1].to(
            dtype=torch.int32
        )

        weights = torch.gather(scores, -1, topk_idx)
        weights = (weights / weights.sum(-1, keepdim=True)) * self.route_scale
        return topk_idx, weights


# Processes all experts in parallel
class MoE(nn.Module):
    def __init__(
        self,
        n_embd,
        n_shared_experts,
        n_routed_experts,
        topk,
        expert_hidden_size,
        dtype=None,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.gate = Gate(n_embd, topk, n_routed_experts, dtype=dtype)
        self.shared_experts = SharedExpert(
            n_embd, hidden_size=n_shared_experts * expert_hidden_size, dtype=dtype
        )
        self.routed_fused_proj = te.GroupedLinear(
            in_features=n_embd,
            out_features=expert_hidden_size * n_shared_experts,
            num_gemms=n_routed_experts,
            bias=False,
            params_dtype=dtype,
        )
        self.routed_down_proj = te.GroupedLinear(
            in_features=expert_hidden_size,
            out_features=n_embd,
            num_gemms=n_routed_experts,
            bias=False,
            params_dtype=dtype,
        )
        # Store global counts for the logger
        self.last_global_counts = None

    def update_bias(self, global_count, update_rate=0.001):
        """
        DeepSeek Auxiliary-Loss-Free Load Balancing.
        """
        with torch.no_grad():
            total_tokens = global_count.sum()
            actual_load = (
                global_count.float() / total_tokens if total_tokens != 0 else 0
            )
            target_load = 1 / self.n_routed_experts

            # Determine correction direction
            correction = torch.sign(target_load - actual_load)
            self.gate.bias.add_(update_rate * correction)  # pyrefly: ignore

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        shared = self.shared_experts(x)

        topk_idx, weights = self.gate(x_flat)

        permuted_x, permuted_map = te.moe_permute(x_flat, topk_idx, map_type="index")

        # Calculate Local Counts
        tokens_per_expert = torch.bincount(
            topk_idx.flatten(), minlength=self.n_routed_experts
        )
        local_tokens_per_expert_list = tokens_per_expert.tolist()

        # Global Sync & Update
        if self.training:
            if dist.is_initialized():
                dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM)

            # Save for Logger (Detached CPU list for safety/speed)
            self.last_global_counts = tokens_per_expert.detach().cpu().tolist()

            # Update bias
            self.update_bias(tokens_per_expert)

        routed_up_proj, routed_gate = self.routed_fused_proj(
            permuted_x, m_splits=local_tokens_per_expert_list
        ).chunk(2, dim=-1)
        permuted_up_x = routed_up_proj * F.silu(routed_gate)
        permuted_y = self.routed_down_proj(
            permuted_up_x, m_splits=local_tokens_per_expert_list
        )

        routed = te.moe_unpermute(
            permuted_y,
            permuted_map,
            merging_probs=weights,
            restore_shape=x_flat.shape,
            map_type="index",
        )
        return shared + routed.view(B, T, C)


class ExpertParallelMoE(nn.Module):
    def __init__(
        self,
        n_embd,
        n_shared_experts,
        n_routed_experts,
        topk,
        expert_hidden_size,
        dtype=None,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.dtype = dtype

        # Parallel Setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # 1. Split Experts: Each GPU gets a slice of the total experts
        assert n_routed_experts % self.world_size == 0, "Experts must divide evenly"
        self.num_local_experts = n_routed_experts // self.world_size

        # Gate remains global (predicts for all experts)
        self.gate = Gate(n_embd, topk, n_routed_experts, dtype=dtype)

        # Shared experts are dense, processed locally (handled by FSDP usually)
        self.shared_experts = SharedExpert(
            n_embd, hidden_size=n_shared_experts * expert_hidden_size, dtype=dtype
        )

        # 2. Local Experts: Size is reduced to (Total / World_Size)
        self.routed_fused_proj = te.GroupedLinear(
            in_features=n_embd,
            out_features=expert_hidden_size * 2,
            num_gemms=self.num_local_experts,  # Local count only
            bias=False,
            params_dtype=dtype,
        )
        self.routed_down_proj = te.GroupedLinear(
            in_features=expert_hidden_size,
            out_features=n_embd,
            num_gemms=self.num_local_experts,  # Local count only
            bias=False,
            params_dtype=dtype,
        )
        self.last_global_counts = None

    def update_bias(self, global_count, update_rate=0.001):
        with torch.no_grad():
            total_tokens = global_count.sum()
            actual_load = (
                global_count.float() / total_tokens if total_tokens != 0 else 0
            )
            target_load = 1 / self.n_routed_experts
            correction = torch.sign(target_load - actual_load)
            self.gate.bias.add_(update_rate * correction)  # pyrefly: ignore

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        # Shared experts
        shared = self.shared_experts(x)

        # Routing (Global)
        topk_idx, weights = self.gate(x_flat)
        topk_idx, weights = topk_idx.to(torch.int32), weights.to(torch.bfloat16)

        # Load Balancing (Global Sync)
        if self.training:
            tokens_per_expert = torch.bincount(
                topk_idx.flatten(), minlength=self.n_routed_experts
            )
            if dist.is_initialized():
                dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM)
            self.update_bias(tokens_per_expert)
            if self.rank == 0:
                self.last_global_counts = tokens_per_expert.detach().cpu().tolist()

        # 3. Dispatch Preparation
        # Calculate which rank owns the chosen experts
        dest_ranks = topk_idx // self.num_local_experts
        local_expert_indices = topk_idx % self.num_local_experts

        # Flatten and expand for top-k
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.topk, -1).reshape(-1, C)
        dest_ranks_flat = dest_ranks.view(-1)
        local_expert_idx_flat = local_expert_indices.view(-1)
        weights_flat = weights.view(-1)

        # Sort by destination rank for all_to_all
        sort_idx = torch.argsort(dest_ranks_flat)
        dest_ranks_sorted = dest_ranks_flat[sort_idx]
        x_sorted = x_expanded[sort_idx]
        local_expert_idx_sorted = local_expert_idx_flat[sort_idx]
        weights_sorted = weights_flat[sort_idx]

        # Calculate Send/Recv Counts
        send_counts = torch.bincount(dest_ranks_sorted, minlength=self.world_size)
        send_splits = send_counts.tolist()
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        recv_splits = recv_counts.tolist()

        # 4. Dispatch (All-to-All)
        total_recv = sum(recv_splits)
        recv_x = torch.empty((total_recv, C), dtype=x.dtype, device=x.device)
        recv_expert_idx = torch.empty((total_recv,), dtype=torch.int32, device=x.device)
        recv_weights = torch.empty((total_recv,), dtype=weights.dtype, device=x.device)

        dist.all_to_all_single(
            recv_x,
            x_sorted,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_expert_idx,
            local_expert_idx_sorted,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_weights,
            weights_sorted,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )

        # 5. Local Compute
        # TE GroupedLinear requires inputs sorted by expert ID
        recv_expert_idx = recv_expert_idx.to(torch.int32)
        te_sort_idx = torch.argsort(recv_expert_idx)
        recv_x_te = recv_x[te_sort_idx]

        # Prepare splits for TE
        tokens_per_local_expert = torch.bincount(
            recv_expert_idx, minlength=self.num_local_experts
        ).tolist()

        # Forward Pass through Experts
        up = self.routed_fused_proj(recv_x_te, m_splits=tokens_per_local_expert)
        gate = up.chunk(2, dim=-1)[1]
        up = up.chunk(2, dim=-1)[0]

        h = up * F.silu(gate)
        out_te = self.routed_down_proj(h, m_splits=tokens_per_local_expert)

        # Apply routing weights
        out_te = out_te * recv_weights[te_sort_idx].unsqueeze(-1)
        out_te = out_te.to(self.dtype)

        # 6. Return Dispatch (All-to-All)
        # Unsort TE order
        inv_te_sort_idx = torch.argsort(te_sort_idx)
        out_ready = out_te[inv_te_sort_idx]

        return_x = torch.empty_like(x_sorted)
        dist.all_to_all_single(
            return_x,
            out_ready,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
        )

        # 7. Un-Permute / Reduce
        # We need to map the sorted returned tokens back to their original positions in x_flat
        # sort_idx maps: Sorted Position -> Expanded Position
        # We need to sum results for the same token (since topk > 1)

        output = torch.zeros(x_flat.shape, dtype=x.dtype, device=x.device)

        # Create indices mapping each returned token to its original row in x_flat
        row_indices = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(
            self.topk
        )
        row_indices_sorted = row_indices[sort_idx]

        # Accumulate results
        output.index_add_(0, row_indices_sorted, return_x)

        return shared + output.view(B, T, C)


# Block
class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_heads,
        head_size,
        rope_head_size,
        kv_latent_size,
        q_latent_size,
        n_shared_experts,
        n_routed_experts,
        topk,
        expert_hidden_size,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = te.RMSNorm(n_embd, params_dtype=dtype)
        self.sa = MLA(
            n_embd,
            n_heads,
            head_size,
            rope_head_size,
            kv_latent_size,
            q_latent_size,
            dtype=dtype,
        )
        # self.sa = Attention(n_embd, n_heads, dtype=dtype)
        self.sa_compiled = torch.compile(self.sa)  # pyrefly:ignore
        self.ln2 = te.RMSNorm(n_embd, params_dtype=dtype)
        self.moe = ExpertParallelMoE(
            n_embd,
            n_shared_experts,
            n_routed_experts,
            topk,
            expert_hidden_size,
            dtype=dtype,
        )

    def forward(self, x, sin, cos):
        if self.training:
            x = x + self.sa_compiled(self.ln1(x), sin, cos)
        else:
            x = x + self.sa(self.ln1(x), sin, cos)
        x = x + self.moe(self.ln2(x))
        return x


# LLM
class GPT(nn.Module):
    def __init__(
        self,
        n_embd,
        vocab_size,
        block_size,
        n_heads,
        head_size,
        rope_head_size,
        kv_latent_size,
        q_latent_size,
        n_layers,
        n_shared_experts,
        n_routed_experts,
        topk_experts,
        expert_hidden_size,
        dtype,
    ):
        super().__init__()
        self.dtype = dtype
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.wte = nn.Embedding(vocab_size, n_embd)

        sin, cos = self._precompute_rotary_embeddings(block_size, rope_head_size)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.transformer = nn.ModuleList(
            [
                Block(
                    n_embd,
                    n_heads,
                    head_size,
                    rope_head_size,
                    kv_latent_size,
                    q_latent_size,
                    n_shared_experts,
                    n_routed_experts,
                    topk_experts,
                    expert_hidden_size,
                    dtype=dtype,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = te.RMSNorm(n_embd, params_dtype=dtype)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False, dtype=dtype)
        self.wte.weight = (
            self.lm_head.weight
        )  # Embedding layer and final calssifier are the same
        self.apply(self._init_weights)
        self.rank = dist.get_rank()
        self.step_counter = 0

    def _log_moe_stats(self):
        """
        Gathers stored stats from layers and logs them to file with Step/Layer info.
        Only called on Rank 0.
        """
        if self.rank != 0:
            return

        os.makedirs("output/expert_stats", exist_ok=True)

        # Collect data
        batch_log = []
        for i, block in enumerate(self.transformer):
            if block.moe.last_global_counts is not None:  # pyrefly: ignore
                entry = {
                    "step": self.step_counter,
                    "layer": i,
                    "counts": block.moe.last_global_counts,
                }
                batch_log.append(json.dumps(entry))

        # Bulk write
        if batch_log:
            with open("output/expert_stats/layer_loads.jsonl", "a") as f:
                f.write("\n".join(batch_log) + "\n")

    # initialize weights
    def _init_weights(self, module):
        std = 0.015
        # Handle standard layers (Linear, TE Linear, Embedding)
        if isinstance(module, (nn.Linear, te.Linear, nn.Embedding)):
            if hasattr(module, "RESIDUAL_SCALE_INIT_FACTOR"):
                std *= 1 / (math.sqrt(2 * self.n_layers))

            # These modules all have a standard .weight attribute
            if hasattr(module, "weight"):
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=std  # pyrefly: ignore
                )

            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # pyrefly: ignore

        # 2. Handle GroupedLinear (MoE Experts) - It uses 'weight0'
        elif isinstance(module, te.GroupedLinear):
            if hasattr(module, "weight"):
                # Just in case future versions use 'weight'
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=std  # pyrefly: ignore
                )
            elif hasattr(module, "weight0"):
                # THIS IS THE FIX: Initialize weight0
                torch.nn.init.normal_(
                    module.weight0, mean=0.0, std=std  # pyrefly: ignore
                )

            # It usually doesn't have bias in MoE, but safe
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # pyrefly: ignore

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"Sequence length ({T}) is longer than the block_size ({self.block_size})."
        x = self.wte(idx)
        sin = self.sin[:, :, :T, :]  # pyrefly: ignore
        cos = self.cos[:, :, :T, :]  # pyrefly: ignore

        for block in self.transformer:
            x = block(x, sin, cos)
        x = self.ln(x)

        if self.training:
            self.step_counter += 1
            if self.rank == 0:
                self._log_moe_stats()

        if targets is not None:
            logits = self.lm_head(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            return None, loss
        else:
            logits = self.lm_head(x)
            return logits, None

    def generate(
        self,
        idx,
        num_sequences=5,
        max_tokens=200,
        topk=50,
        chat_mode=False,
        eos_token=50256,
    ):
        idx = torch.repeat_interleave(idx.unsqueeze(0), num_sequences, dim=0)

        for _ in range(max_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]  # pyrefly: ignore
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=topk)
            idx_next = torch.multinomial(topk_probs, num_samples=1)
            idx_next = torch.gather(topk_indices, -1, idx_next)

            if chat_mode and (idx_next == eos_token).all():
                break

            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        decay_params = []
        nodecay_params = []
        seen = set()

        for n, p in self.named_parameters():
            if p.requires_grad:
                if id(p) in seen:
                    continue
                seen.add(id(p))

                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    nodecay_params.append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        if self.rank == 0:
            print(f"Decayed params: {sum(p.numel() for p in decay_params):,}")
            print(f"No-decay params: {sum(p.numel() for p in nodecay_params):,}")

        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        if self.rank == 0:
            print(f"Using fused AdamW: {use_fused}")

        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused
        )

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        sin, cos = freqs.sin(), freqs.cos()
        sin, cos = sin.bfloat16(), cos.bfloat16()
        sin, cos = sin[None, None, :, :], cos[None, None, :, :]
        return sin, cos
