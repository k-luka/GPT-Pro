import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
from src.helpers import apply_rotary_emb, norm
import torch.distributed as dist
import transformer_engine.pytorch as te

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

        self.w_down_q = te.Linear(
            n_embd, q_latent_size, bias=False, params_dtype=dtype
        )
        self.w_kva = te.Linear(
            n_embd,
            kv_latent_size + rope_head_size,
            bias=False,
            params_dtype=dtype,
        )

        self.q_norm = te.RMSNorm(q_latent_size, params_dtype=dtype)
        self.kv_norm = te.RMSNorm(kv_latent_size, params_dtype=dtype)

        # q = (q_content and q_rope) per head
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
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True  # pyrefly: ignore

    def forward(self, x, sin, cos):
        B, T, _ = x.shape
        H = self.n_heads
        d_c = self.latent_head_size
        d_r = self.rope_head_size
        d = self.head_size

        # --- Q ---
        c_q = self.q_norm(self.w_down_q(x))

        q_lr = self.w_up_qr(c_q).view(B, T, H, d).transpose(1, 2)
        q_l = q_lr[..., :d_c]
        q_r = q_lr[..., d_c:]
        q_r = apply_rotary_emb(q_r, sin, cos)
        q = torch.cat((q_l, q_r), dim=-1).contiguous()

        # --- KV ---
        c_kv_rope = self.w_kva(x)
        c_kv = c_kv_rope[..., : self.kv_latent_size]
        c_kr = c_kv_rope[..., self.kv_latent_size :]

        # shared k_rope accross all heads
        k_r = apply_rotary_emb(c_kr.unsqueeze(1), sin, cos)

        c_kv = self.kv_norm(c_kv)
        kv = (
            self.w_up_kv(c_kv).view(B, T, H, d_c + d).transpose(1, 2)
        )  # (B, H, T, d_c + d)
        k_l = kv[..., :d_c]
        v = kv[..., d_c:]

        k = torch.cat(
            (k_l, k_r.expand(B, H, T, d_r)), dim=-1
        ).contiguous()  # (B, H, T, d)
        v = v.contiguous()

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, H * d)
        return self.proj(out)


# Attention (Regular Attention but fast)
class Attention(nn.Module):
    def __init__(self, n_embd, n_heads, dtype=None):
        super().__init__()
        assert (
            n_embd % n_heads == 0
        ), f"Embedding dim ({n_embd}) must be divisible by number of heads ({n_heads})."
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.H = n_embd // n_heads  # head size
        self.attn = te.Linear(
            n_embd, 3 * n_embd, bias=False, params_dtype=dtype
        )
        self.proj = te.Linear(
            n_embd, n_embd, bias=False, params_dtype=dtype
        )
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True  # pyrefly: ignore
        # self.register_buffer("tril", torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size))

    def forward(self, x, sin, cos):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=-1)  # q,k,v each is (B,T,C)
        q = q.view(B, T, self.n_heads, self.H).transpose(1, 2)  # (B,n_heads,T,H)
        k = k.view(B, T, self.n_heads, self.H).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.H).transpose(1, 2)
        # Apply RoPE
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)
        q, k = norm(q), norm(k)
        # att = q @ k.tranpose(-2,-1) / (1 * math.sqrt(self.H)) # (B,n_heads,T,T)
        # att = att.masked_fill(self.tril[:,:,:T,:T], float("-inf"))
        # out = att @ v # (B,n_heds,T,H)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # Abstraction but uses flash att for 20% faster training
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
        self.weight = te.Linear(n_embd, n_routed_experts, bias=False, params_dtype=dtype)
        self.register_buffer("bias", torch.zeros(n_routed_experts, dtype=dtype))

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.weight(x)
        scores = logits.sigmoid()

        # Break the link between the computation and the storage buffer.
        # Now, modifying self.bias later will not trigger "ViewBackward0" error.
        bias_term = self.bias.detach()  # pyrefly: ignore

        topk_idx = torch.topk(scores + bias_term, self.topk, dim=-1)[1].to(
            dtype=torch.int32
        )

        weights = torch.gather(scores, -1, topk_idx)
        weights = (weights / weights.sum(-1, keepdim=True)) * self.route_scale
        return topk_idx, weights

# Processes all experts in parallel (I guess the shared ones are first)
class MoE(nn.Module):
    def __init__(self, n_embd, n_shared_experts, n_routed_experts, topk, dtype=None):
        super().__init__()
        self.n_embd = n_embd
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.gate = Gate(n_embd, topk, n_routed_experts, dtype=dtype)
        self.shared_experts = SharedExpert(
            n_embd, hidden_size=n_shared_experts * 2 * n_embd, dtype=dtype
        )
        self.routed_fused_proj = te.GroupedLinear(
            in_features=n_embd,
            out_features=n_embd * 2 * 2,
            num_gemms=n_routed_experts,
            bias=False,
            params_dtype=dtype,
        )
        self.routed_down_proj = te.GroupedLinear(
            in_features=n_embd * 2,
            out_features=n_embd,
            num_gemms=n_routed_experts,
            bias=False,
            params_dtype=dtype,
        )

    def update_bias(self, count, update_rate=0.001):
        """
        DeepSeek Auxiliary-Loss-Free Load Balancing
        """
        total_tokens = count.sum()

        actual_load = count.float() / total_tokens if total_tokens != 0 else 0
        target_load = 1 / self.n_routed_experts

        self.gate.bias.add_(update_rate * torch.sign(target_load - actual_load)) # pyrefly: ignore

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        shared = self.shared_experts(x)

        topk_idx, weights = self.gate(x_flat)

        permuted_x, permuted_map = te.moe_permute(x_flat, topk_idx, map_type="index")

        tokens_per_expert = torch.bincount(
            topk_idx.flatten(), minlength=self.n_routed_experts
        ).tolist()

        routed_up_proj, routed_gate = self.routed_fused_proj(
            permuted_x, m_splits=tokens_per_expert
        ).chunk(2, dim=-1)
        permuted_up_x = routed_up_proj * F.silu(routed_gate)
        permuted_y = self.routed_down_proj(permuted_up_x, m_splits=tokens_per_expert)
        routed = te.moe_unpermute(
            permuted_y,
            permuted_map,
            merging_probs=weights,
            restore_shape=x_flat.shape,
            map_type="index",
        )
        return shared + routed.view(B, T, C)


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
        dtype=None,
    ):
        super().__init__()
        self.ln1 = te.RMSNorm(n_embd, params_dtype=dtype)
        # self.sa = Attention(n_embd, n_heads)
        self.sa = MLA(
            n_embd,
            n_heads,
            head_size,
            rope_head_size,
            kv_latent_size,
            q_latent_size,
            dtype=dtype,
        )
        self.ln2 = te.RMSNorm(n_embd, params_dtype=dtype)
        self.moe = MoE(
            n_embd, n_shared_experts=2, n_routed_experts=4, topk=2, dtype=dtype
        )

    def forward(self, x, sin, cos):
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
        dtype,
    ):
        super().__init__()
        self.dtype=dtype
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        # self.wpe = nn.Embedding(block_size, n_embd)  # old learned positional embeddings
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
                    dtype=dtype,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = te.RMSNorm(n_embd, params_dtype=dtype)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = (
            self.lm_head.weight
        )  # Embedding layer and final calssifier are the same
        self.apply(self._init_weights)
        self.rank = dist.get_rank()

    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # std = 1 / math.sqrt(self.n_embd) is what GPT-3 says
            # But DeepSeek says to go lower!
            std = 0.01
            if hasattr(module, "RESIDUAL_SCALE_INIT_FACTOR"):
                std *= 1 / (math.sqrt(2 * self.n_layers))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, idx, targets=None):
        # select the index and put them together
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"Sequence length ({T}) is longer than the block_size ({self.block_size})."
        x = self.wte(idx)  # (B,T,C)
        # # old code with learned positional embedding
        # pos_emb = self.wpe(torch.arange(0, T, dtype=torch.long, device=idx.device)) # (T,C)
        # x = tok_emb + pos_emb
        sin = self.sin[:, :, :T, :]  # pyrefly: ignore
        cos = self.cos[:, :, :T, :]  # pyrefly: ignore

        for block in self.transformer:
            x = block(x, sin, cos)
        x = self.ln(x)

        if targets is not None:
            logits = self.lm_head(x)
            B, T, C = logits.shape
            # cross_entropy expects shape (N,C)
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Dictionary to handle tied weights (wte and lm_head sharing the same tensor)
        # We iterate named_parameters to inspect names, but track object IDs to avoid duplicates
        decay_params = []
        nodecay_params = []
        seen = set()

        for n, p in self.named_parameters():
            if p.requires_grad:
                if p in seen:
                    continue
                seen.add(p)

                # HEURISTIC: Filter by name
                # Biases typically don't decay
                # Norm layers (ln, ln1, ln2, q_norm, kv_norm) don't decay
                if n.endswith(".bias") or "norm" in n or "ln" in n:
                    nodecay_params.append(p)
                else:
                    decay_params.append(p)

        # Create the optimizer groups
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Debug print
        if self.rank == 0:
            num_decay = sum(p.numel() for p in decay_params)
            num_nodecay = sum(p.numel() for p in nodecay_params)
            print(
                f"Decayed params: {len(decay_params)} tensors, {num_decay:,} parameters"
            )
            print(
                f"Non-decayed params: {len(nodecay_params)} tensors, {num_nodecay:,} parameters"
            )

        # Configure fused AdamW
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in str(device)
        if self.rank == 0:
            print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequency at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        sin, cos = freqs.sin(), freqs.cos()
        sin, cos = sin.bfloat16(), cos.bfloat16()
        sin, cos = sin[None, None, :, :], cos[None, None, :, :]
        return sin, cos
