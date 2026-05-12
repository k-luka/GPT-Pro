import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
from src.utils.helpers import apply_rotary_emb
import torch.distributed as dist
import transformer_engine.pytorch as te


class GQA(nn.Module):
    def __init__(self, n_embd, n_heads, n_kv_heads, dtype=None):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = n_embd // n_heads

        self.w_q = te.Linear(
            n_embd, n_heads * self.head_dim, bias=False, params_dtype=dtype
        )
        self.w_k = te.Linear(
            n_embd, n_kv_heads * self.head_dim, bias=False, params_dtype=dtype
        )
        self.w_v = te.Linear(
            n_embd, n_kv_heads * self.head_dim, bias=False, params_dtype=dtype
        )
        self.proj = te.Linear(
            n_heads * self.head_dim, n_embd, bias=False, params_dtype=dtype
        )
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True

        self.q_norm = nn.RMSNorm(self.head_dim, dtype=dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, dtype=dtype)

    def forward(self, x, sin, cos, is_first_microbatch=None):
        B, T, _ = x.shape

        q = (
            self.w_q(x, is_first_microbatch=is_first_microbatch)
            .view(B, T, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.w_k(x, is_first_microbatch=is_first_microbatch)
            .view(B, T, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.w_v(x, is_first_microbatch=is_first_microbatch)
            .view(B, T, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = apply_rotary_emb(q, sin, cos).to(x.dtype)
        k = apply_rotary_emb(k, sin, cos).to(x.dtype)
        q, k = self.q_norm(q), self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(out, is_first_microbatch=is_first_microbatch)


class MLP(nn.Module):
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

    def forward(self, x, is_first_microbatch=None):
        gate = F.silu(self.gate_proj(x, is_first_microbatch=is_first_microbatch))
        value = self.up_proj(x, is_first_microbatch=is_first_microbatch)
        x = gate * value
        return self.down_proj(x, is_first_microbatch=is_first_microbatch)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, n_kv_heads, ffn_hidden_size, dtype=None):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd, dtype=dtype)
        self.sa = GQA(n_embd, n_heads, n_kv_heads, dtype=dtype)
        self.ln2 = nn.RMSNorm(n_embd, dtype=dtype)
        self.mlp = MLP(n_embd, hidden_size=ffn_hidden_size, dtype=dtype)

    def forward(self, x, sin, cos, is_first_microbatch=None):
        x = x + self.sa(
            self.ln1(x), sin, cos, is_first_microbatch=is_first_microbatch
        )
        x = x + self.mlp(self.ln2(x), is_first_microbatch=is_first_microbatch)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        n_embd,
        vocab_size,
        block_size,
        n_heads,
        n_kv_heads,
        n_layers,
        ffn_hidden_size,
        dtype,
    ):
        super().__init__()
        self.dtype = dtype
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.wte = nn.Embedding(vocab_size, n_embd, dtype=dtype)

        head_dim = n_embd // n_heads
        sin, cos = self._precompute_rotary_embeddings(block_size, head_dim)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.transformer = nn.ModuleList(
            [
                Block(
                    n_embd,
                    n_heads,
                    n_kv_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    dtype=dtype,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.RMSNorm(n_embd, dtype=dtype)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False, dtype=dtype)
        self.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        self.rank = dist.get_rank()

    def _init_weights(self, module):
        std = 0.015
        if isinstance(module, (nn.Linear, te.Linear, nn.Embedding)):
            if hasattr(module, "RESIDUAL_SCALE_INIT_FACTOR"):
                std *= 1 / (math.sqrt(2 * self.n_layers))
            if hasattr(module, "weight"):
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=std  # pyrefly: ignore
                )
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # pyrefly: ignore

    def forward(self, idx, targets=None, is_first_microbatch=None):
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"Sequence length ({T}) is longer than the block_size ({self.block_size})."
        x = self.wte(idx)
        sin = self.sin[:, :, :T, :]  # pyrefly: ignore
        cos = self.cos[:, :, :T, :]  # pyrefly: ignore

        for block in self.transformer:
            x = block(x, sin, cos, is_first_microbatch=is_first_microbatch)
        x = self.ln(x)

        logits = self.lm_head(x).float()
        logits = 30.0 * torch.tanh(logits / 30.0)  # soft-cap to prevent logit explosion

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            return None, loss
        else:
            return logits, None

    def generate(
        self,
        idx,
        num_sequences=5,
        max_tokens=200,
        topk=50,
        chat_mode=False,
        eos_token=151643,
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
        from src.utils.optimizers import DualOptimizer

        muon_params = []
        adamw_decay_params = []
        adamw_nodecay_params = []
        seen = set()

        for pn, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))

            if p.dim() >= 2 and "wte" not in pn and "lm_head" not in pn:
                muon_params.append(p)
            elif p.dim() >= 2:
                adamw_decay_params.append(p)
            else:
                adamw_nodecay_params.append(p)

        if self.rank == 0:
            print(
                f"Muon params (2D hidden): {len(muon_params)} tensors, {sum(p.numel() for p in muon_params):,} parameters"
            )
            print(
                f"AdamW decay params (Embed/Head): {len(adamw_decay_params)} tensors, {sum(p.numel() for p in adamw_decay_params):,} parameters"
            )
            print(
                f"AdamW no-decay params (1D norms): {len(adamw_nodecay_params)} tensors, {sum(p.numel() for p in adamw_nodecay_params):,} parameters"
            )

        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        if self.rank == 0:
            print(f"Using fused AdamW: {use_fused}")

        adam_opt = torch.optim.AdamW(
            [
                {"params": adamw_decay_params, "weight_decay": weight_decay},
                {"params": adamw_nodecay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )

        muon_opt = torch.optim.Muon(
            muon_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )

        return DualOptimizer(adam_opt, muon_opt)

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
