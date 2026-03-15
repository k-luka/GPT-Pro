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


class SharedExpert(nn.Module):
    def __init__(self, n_embd, n_experts, hidden_dim):
        super().__init__()
        self.n_embd = n_embd
        self.fused_up = te.Linear(n_embd, n_experts * hidden_dim * 2)
        self.proj_down = te.Linear(n_experts * hidden_dim, n_embd)

    def forward(self, x):
        fused = self.fused_up(x)
        proj_up, gate = fused.chunk(2, -1)
        return self.proj_down(F.silu(gate) * proj_up)


class Gate(nn.Module):
    def __init__(self, n_embd, n_routed_experts, topk, route_scale=1.0):
        super().__init__()
        self.n_embd = n_embd
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.route_scale = route_scale
        self.gate = te.Linear(n_embd, n_routed_experts)
        self.register_buffer("bias", torch.zeros(n_routed_experts))

    def forward(self, x):
        logits = self.gate(x)
        scores = torch.sigmoid(logits)
        bias = self.bias.detach()  # pyrefly: ignore

        topk_idx = torch.topk(scores + bias, self.topk, dim=-1)[1].to(dtype=torch.int32)

        weights = torch.gather(scores, -1, topk_idx)
        weights = (weights / weights.sum(-1, keepdim=True)) * self.route_scale
        return topk_idx, weights

    class ParallelExperts(nn.Module):
        def __init__(
            self,
            n_embd,
            n_shared_experts,
            n_routed_experts,
            topk,
            hidden_size,
            route_scale=1.0,
        ):
            super().__init__()
            self.n_embd = n_embd
            self.n_shared_experts = n_shared_experts
            self.n_routed_experts = n_routed_experts
            self.topk = topk
            self.gate = Gate(n_embd, n_routed_experts, topk, route_scale=route_scale)
            self.shared_expert = SharedExpert(n_embd, n_shared_experts, hidden_size)
            # Parallel setup
            self.world_rank = dist.get_world_size() if dist.is_initialized() else 1
            self.rank = dist.get_rank() if dist.is_initialized() else 0

            assert n_routed_experts % self.world_size == 0
            self.num_local_experts = n_routed_experts // self.world_size

            self.fused_up = te.GroupedLinear(
                num_gemms=self.num_local_experts,
                in_features=n_embd,
                out_features=hidden_size * 2,
                bias=False,
            )
            self.down_proj = te.GroupedLinear(
                num_gemms=self.num_local_experts,
                in_features=hidden_size,
                out_features=n_embd,
                bias=False,
            )
            self.last_global_counts = 0

        def update_bias(self, global_counts, update_rate=0.001):
            with torch.no_grad():
                total_tokens = global_counts.sum()
                actual_load = (
                    global_counts.float() / total_tokens if total_tokens != 0 else 0
                )
                target_load = 1 / total_tokens
                update_direction = torch.sign(target_load - actual_load)
                self.gate.bias.add_(update_direction * update_rate)  # pyrefly: ignore

        def forward(self, x):
            B, T, C = x.shape

            shared = self.shared_expert(x)

            topk_idx, weights = self.gate(x)
