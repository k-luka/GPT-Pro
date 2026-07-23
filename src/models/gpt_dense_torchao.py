"""
Dense GPT model — torchao MXFP8 backend variant.

Structural twin of src/models/gpt_dense.py. Differences:
  - Uses torch.nn.Linear everywhere (no transformer_engine).
  - No is_first_microbatch arg threaded through forward — MXFP8 microscale
    factors are recomputed dynamically each microbatch (no cached FP8 weight).
  - MXFP8 conversion happens in the launcher (pretrain_dense_torchao.py) by
    calling quantize_(model, MXFP8TrainingOpConfig.from_recipe(...)), which
    swaps each hidden Linear's weight with an MXFP8TrainingWeightWrapperTensor.
    Master weights stay BF16; the MXFP8 cast happens at GEMM time only, and the
    weight gradient stays BF16 (wgrad_with_hp recipe). lm_head is left out of
    the swap so the output projection stays BF16.
"""

import inspect
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from src.models.fa4_attn import fa4_attention
from src.utils.helpers import apply_rotary_emb


def _resolve_sandwich_norm(sandwich_norm):
    """Resolve the explicit config value, retaining env support for old scripts."""
    if sandwich_norm is not None:
        if not isinstance(sandwich_norm, bool):
            raise TypeError(
                "sandwich_norm must be a boolean when set explicitly, "
                f"got {type(sandwich_norm).__name__}"
            )
        return sandwich_norm

    env_value = os.environ.get("SANDWICH_NORM", "on")
    if env_value not in {"on", "off"}:
        raise ValueError(
            "SANDWICH_NORM must be 'on' or 'off' when model.sandwich_norm "
            f"is absent, got {env_value!r}"
        )
    return env_value == "on"


def _resolve_choice(value, env_name, default, allowed):
    """Resolve an explicit config string before its legacy environment fallback."""
    resolved = value if value is not None else os.environ.get(env_name, default)
    if resolved not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{env_name} must be one of {{{choices}}}, got {resolved!r}")
    return resolved


class GQA(nn.Module):
    def __init__(
        self,
        n_embd,
        n_heads,
        n_kv_heads,
        dtype=None,
        is_global=True,
        sliding_window=0,
        head_dim=0,
        qk_norm_mode=None,
        local_attn_impl=None,
        global_attn_impl=None,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        # head_dim decouples head width from n_embd (Gemma 4 over-provisions:
        # e.g. E2B projects hidden 1536 -> 8 heads x 512 = 4096 on global
        # layers). 0 = derive from n_embd as before.
        self.head_dim = head_dim or n_embd // n_heads

        self.w_q = nn.Linear(n_embd, n_heads * self.head_dim, bias=False, dtype=dtype)
        self.w_k = nn.Linear(
            n_embd, n_kv_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.w_v = nn.Linear(
            n_embd, n_kv_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.proj = nn.Linear(n_heads * self.head_dim, n_embd, bias=False, dtype=dtype)
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True

        # Overnight A/B toggle. Read once at construct time; each run is a fresh
        # process so torch.compile bakes it in as a constant. "after" (default)
        # = current behavior (QK-norm after RoPE); "before" = QK-norm before RoPE;
        # "off" = no QK-norm. Set via QK_NORM_MODE env var. Create the norm layers
        # only when used — otherwise their params are unused in forward and DDP
        # raises "parameters that were not used in producing loss".
        self.qk_norm_mode = _resolve_choice(
            qk_norm_mode, "QK_NORM_MODE", "after", {"after", "before", "off"}
        )
        if self.qk_norm_mode != "off":
            self.q_norm = nn.RMSNorm(self.head_dim, dtype=dtype)
            self.k_norm = nn.RMSNorm(self.head_dim, dtype=dtype)

        # Attention kernel selection (env A/B toggles, read at construct time
        # like the norm toggles above — torch.compile bakes them in).
        #   ATTN_IMPL        = flex (default) | fa4  — LOCAL (sliding-window) layers
        #   GLOBAL_ATTN_IMPL = sdpa | fa4 | sdpa_eff — GLOBAL (full-causal) layers
        #     (default: sdpa, or sdpa_eff when head_dim > 256)
        # FA4 (flash_attn.cute) is the fastest kernel on B200 for windowed
        # head_dim <= 128 and for full-causal head_dim 256; cuDNN SDPA remains
        # fastest for full-causal head_dim 128. FA4 has no windowed kernel for
        # head_dim 256 on SM100 — local layers there must stay on flex.
        # head_dim > 256 (Gemma-4 global layers are 512): the mem-efficient SDPA
        # backend is the ONLY kernel on B200 that runs it fwd+bwd (flash/cuDNN
        # cap at 256, FA4 caps at 256, flex overflows shared memory) — measured
        # 84 ms vs 238 ms math fallback at B16/S4096/8Q/1KV.
        if is_global:
            default = "sdpa_eff" if self.head_dim > 256 else "sdpa"
            impl = _resolve_choice(
                global_attn_impl,
                "GLOBAL_ATTN_IMPL",
                default,
                {"sdpa", "sdpa_eff", "fa4"},
            )
        else:
            impl = _resolve_choice(
                local_attn_impl, "ATTN_IMPL", "flex", {"flex", "fa4"}
            )
        self.attn_impl = impl
        self.use_fa4 = impl == "fa4"
        self.use_sdpa_eff = impl == "sdpa_eff"
        self.fa4_window = 0 if is_global else sliding_window

    def forward(self, x, sin, cos, block_mask=None):
        B, T, _ = x.shape

        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim)
        if self.use_fa4:
            # FA4 uses the (B, T, H, D) layout the projections already produce;
            # only the rotary tables need re-broadcasting: (1,1,T,D) -> (1,T,1,D).
            sin, cos = sin.transpose(1, 2), cos.transpose(1, 2)
        else:
            # SDPA / FlexAttention use (B, H, T, D).
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        if self.qk_norm_mode == "before":
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, sin, cos).to(x.dtype)
        k = apply_rotary_emb(k, sin, cos).to(x.dtype)
        if self.qk_norm_mode == "after":
            q, k = self.q_norm(q), self.k_norm(k)

        if self.use_fa4:
            out = fa4_attention(q, k, v, sliding_window=self.fa4_window)
            out = out.view(B, T, -1)
        elif self.use_sdpa_eff:
            # mem-efficient backend has no GQA support: expand KV to the query
            # head count (free view when n_kv_heads == 1, else a copy). Without
            # enable_gqa + with head_dim > 256, SDPA auto-dispatch lands on the
            # mem-efficient kernel (flash/cuDNN reject the head dim).
            if self.n_kv_heads == 1:
                k = k.expand(B, self.n_heads, T, self.head_dim)
                v = v.expand(B, self.n_heads, T, self.head_dim)
            elif self.n_kv_heads != self.n_heads:
                rep = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        elif block_mask is None:
            # Global layer: full causal attention via the (cuDNN) SDPA backend —
            # fastest dense kernel on B200 at head_dim 128.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        else:
            # Local layer: sliding-window causal attention via FlexAttention. The
            # block-sparse BlockMask skips off-window key blocks, cutting the K/V
            # working set (~40% less attention memory than full causal).
            out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd, hidden_size, dtype=None):
        super().__init__()
        self.n_embd = n_embd
        self.hidden_dim = (hidden_size + 255) // 256 * 256

        self.gate_proj = nn.Linear(n_embd, self.hidden_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(n_embd, self.hidden_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(self.hidden_dim, n_embd, bias=False, dtype=dtype)
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)


class Block(nn.Module):
    """Transformer block with Gemma-style sandwich norm: a pre-norm before each
    sub-layer AND a post-norm on each sub-layer's output before the residual add.
    """

    def __init__(
        self,
        n_embd,
        n_heads,
        n_kv_heads,
        ffn_hidden_size,
        dtype=None,
        is_global=True,
        sliding_window=0,
        head_dim=0,
        sandwich_norm=None,
        qk_norm_mode=None,
        local_attn_impl=None,
        global_attn_impl=None,
    ):
        super().__init__()
        # is_global=True  -> full causal attention (SDPA)
        # is_global=False -> sliding-window local attention (FlexAttention or FA4)
        self.is_global = is_global
        # An explicit model config value wins. The environment fallback remains
        # only so older A/B scripts continue to work unchanged.
        # Create the post-norms ONLY when on — otherwise their params are never used
        # in forward and DDP raises "parameters that were not used in producing loss".
        self.sandwich = _resolve_sandwich_norm(sandwich_norm)
        self.ln1 = nn.RMSNorm(n_embd, dtype=dtype)
        self.sa = GQA(
            n_embd,
            n_heads,
            n_kv_heads,
            dtype=dtype,
            is_global=is_global,
            sliding_window=sliding_window,
            head_dim=head_dim,
            qk_norm_mode=qk_norm_mode,
            local_attn_impl=local_attn_impl,
            global_attn_impl=global_attn_impl,
        )
        self.ln2 = nn.RMSNorm(n_embd, dtype=dtype)
        self.mlp = MLP(n_embd, hidden_size=ffn_hidden_size, dtype=dtype)
        if self.sandwich:
            self.post_attn_norm = nn.RMSNorm(n_embd, dtype=dtype)
            self.post_mlp_norm = nn.RMSNorm(n_embd, dtype=dtype)

    def forward(self, x, sin, cos, local_block_mask=None):
        # Global blocks ignore the mask (None -> SDPA causal); local blocks use
        # the shared sliding-window BlockMask.
        block_mask = None if self.is_global else local_block_mask
        if self.sandwich:
            x = x + self.post_attn_norm(
                self.sa(self.ln1(x), sin, cos, block_mask)
            )
            x = x + self.post_mlp_norm(self.mlp(self.ln2(x)))
        else:
            x = x + self.sa(self.ln1(x), sin, cos, block_mask)
            x = x + self.mlp(self.ln2(x))
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
        sliding_window=0,
        global_attn_every_n=0,
        mtp_depth=0,
        mtp_lambda=0.3,
        rope_p=1.0,
        tie_embeddings=True,
        head_dim=0,
        global_n_heads=0,
        global_n_kv_heads=0,
        global_head_dim=0,
        rope_theta=10000.0,
        global_rope_theta=0.0,
        global_rope_p=0.0,
        sandwich_norm=None,
        qk_norm_mode=None,
        local_attn_impl=None,
        global_attn_impl=None,
        global_attn_placement=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.sandwich_norm = _resolve_sandwich_norm(sandwich_norm)
        self.qk_norm_mode = _resolve_choice(
            qk_norm_mode, "QK_NORM_MODE", "after", {"after", "before", "off"}
        )
        self.local_attn_impl = _resolve_choice(
            local_attn_impl, "ATTN_IMPL", "flex", {"flex", "fa4"}
        )
        # p-RoPE (Gemma): rotate only the first `rope_p` fraction of frequency
        # pairs (the high-frequency, position-carrying ones); leave the
        # low-frequency tail un-rotated so it can encode semantics. rope_p=1.0
        # is standard full RoPE.
        self.rope_p = rope_p
        # DeepSeek-V3 Multi-Token Prediction: `mtp_depth` sequential modules that
        # predict the 2nd..(mtp_depth+1)-th future tokens during training, weighted
        # by `mtp_lambda`. Training-only (eval/generate use the main head only).
        self.mtp_depth = mtp_depth
        self.mtp_lambda = mtp_lambda
        self.wte = nn.Embedding(vocab_size, n_embd, dtype=dtype)

        # Gemma-style interleaved attention. A layer is GLOBAL (full causal) when
        # its index is a multiple of `global_attn_every_n` (so layer 0 is global)
        # or it is the last layer; all other layers are LOCAL sliding-window of
        # width `sliding_window`. With global_attn_every_n=5 this is 1 global per
        # 5 layers (4 local : 1 global), first and last always global.
        # sliding_window <= 0 OR global_attn_every_n <= 0 disables locality
        # entirely (every layer is global) — recovers the original dense model.
        self.sliding_window = sliding_window
        self.global_attn_placement = _resolve_choice(
            global_attn_placement,
            "GLOBAL_PLACEMENT",
            "start",
            {"start", "end"},
        )
        if sliding_window > 0 and global_attn_every_n > 0:
            # Global-layer placement (env A/B toggle, read at construct time):
            #   "start" (default): global where i % n == 0, plus the last layer
            #     -> for n=5: globals at 0,5,10,15,19 (layer 0 is a full layer).
            #   "end": global where i % n == n-1
            #     -> for n=5: globals at 4,9,14,19 (layer 0 is a WINDOW layer;
            #        the pattern is 4 window : 1 full and ends on a full layer).
            if self.global_attn_placement == "end":
                self.is_global = [
                    (i % global_attn_every_n == global_attn_every_n - 1)
                    for i in range(n_layers)
                ]
            else:
                self.is_global = [
                    (i % global_attn_every_n == 0) or (i == n_layers - 1)
                    for i in range(n_layers)
                ]
        else:
            self.is_global = [True] * n_layers
        # Lazily-built, cached sliding-window BlockMasks keyed by sequence length.
        # Populate via build_attention_masks() before torch.compile so the hot
        # path only reads them (no per-step graph break from create_block_mask).
        self._local_block_masks = {}

        # Per-layer-type attention geometry (Gemma 4 style). Local (sliding)
        # and global (full-causal) layers may differ in head count, KV head
        # count, head width, RoPE base and p-RoPE fraction — e.g. Gemma-4 E2B:
        # local 8Q/1KV x 256 theta 10k full-rotary, global 8Q/1KV x 512 theta 1M
        # partial 0.25. All global_* params default to the local values, so
        # existing configs are unchanged.
        self.head_dim = head_dim or n_embd // n_heads
        self.g_n_heads = global_n_heads or n_heads
        self.g_n_kv_heads = global_n_kv_heads or n_kv_heads
        self.g_head_dim = global_head_dim or self.head_dim
        default_global_impl = "sdpa_eff" if self.g_head_dim > 256 else "sdpa"
        self.global_attn_impl = _resolve_choice(
            global_attn_impl,
            "GLOBAL_ATTN_IMPL",
            default_global_impl,
            {"sdpa", "sdpa_eff", "fa4"},
        )
        g_rope_theta = global_rope_theta or rope_theta
        g_rope_p = global_rope_p or rope_p

        sin, cos = self._precompute_rotary_embeddings(
            block_size, self.head_dim, base=rope_theta, rope_p=rope_p
        )
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        g_sin, g_cos = self._precompute_rotary_embeddings(
            block_size, self.g_head_dim, base=g_rope_theta, rope_p=g_rope_p
        )
        self.register_buffer("g_sin", g_sin, persistent=False)
        self.register_buffer("g_cos", g_cos, persistent=False)
        self.transformer = nn.ModuleList(
            [
                Block(
                    n_embd,
                    self.g_n_heads if self.is_global[i] else n_heads,
                    self.g_n_kv_heads if self.is_global[i] else n_kv_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    dtype=dtype,
                    is_global=self.is_global[i],
                    sliding_window=sliding_window,
                    head_dim=self.g_head_dim if self.is_global[i] else self.head_dim,
                    sandwich_norm=self.sandwich_norm,
                    qk_norm_mode=self.qk_norm_mode,
                    local_attn_impl=self.local_attn_impl,
                    global_attn_impl=self.global_attn_impl,
                )
                for i in range(n_layers)
            ]
        )
        self.ln = nn.RMSNorm(n_embd, dtype=dtype)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False, dtype=dtype)
        # Weight tying (default). Untie for large-vocab experiments: separate wte
        # and lm_head roughly double the embedding params + optimizer state (at
        # 248k vocab / 4096 dim ~ +1.0B params, +~10 GiB measured) but cost
        # ~no throughput. Set model.tie_embeddings=false to untie.
        if tie_embeddings:
            self.wte.weight = self.lm_head.weight  # weight tying

        # DeepSeek-V3 MTP modules. Module k (0-indexed) combines the previous
        # depth's hidden with the embedding of the next input token via
        # proj_k([RMSNorm(h); RMSNorm(emb)]) -> 2d->d, then a full transformer
        # block. The embedding (wte), final norm (self.ln) and head (lm_head) are
        # shared with the main model.
        if mtp_depth > 0:
            self.mtp_hidden_norm = nn.ModuleList(
                [nn.RMSNorm(n_embd, dtype=dtype) for _ in range(mtp_depth)]
            )
            self.mtp_embed_norm = nn.ModuleList(
                [nn.RMSNorm(n_embd, dtype=dtype) for _ in range(mtp_depth)]
            )
            self.mtp_proj = nn.ModuleList(
                [
                    nn.Linear(2 * n_embd, n_embd, bias=False, dtype=dtype)
                    for _ in range(mtp_depth)
                ]
            )
            self.mtp_block = nn.ModuleList(
                [
                    Block(
                        n_embd,
                        n_heads,
                        n_kv_heads,
                        ffn_hidden_size=ffn_hidden_size,
                        dtype=dtype,
                        is_global=True,  # standalone single layers -> full causal
                        head_dim=self.head_dim,  # local geometry: fed sin/cos
                        sandwich_norm=self.sandwich_norm,
                        qk_norm_mode=self.qk_norm_mode,
                        local_attn_impl=self.local_attn_impl,
                        global_attn_impl=self.global_attn_impl,
                    )
                    for _ in range(mtp_depth)
                ]
            )

        self.apply(self._init_weights)
        self.rank = dist.get_rank()

    @property
    def has_local_layers(self):
        return self.sliding_window > 0 and not all(self.is_global)

    @property
    def needs_local_block_mask(self):
        return self.has_local_layers and self.local_attn_impl == "flex"

    def _make_local_block_mask(self, T, device):
        """Build a causal sliding-window BlockMask for sequence length T.

        Broadcast across batch and heads (mask depends only on positions).
        """
        window = self.sliding_window

        def sliding_window_causal(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

        return create_block_mask(
            sliding_window_causal, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device
        )

    def build_attention_masks(self, seq_lens, device):
        """Pre-build and cache local BlockMasks for the given sequence lengths.

        Call this AFTER moving the model to its device and BEFORE torch.compile,
        so the compiled forward only reads cached masks (no create_block_mask in
        the hot path, which would otherwise force a graph break each step).
        """
        if not self.needs_local_block_mask:
            return
        for T in {int(t) for t in seq_lens}:
            self._local_block_masks[T] = self._make_local_block_mask(T, device)

    def _init_weights(self, module):
        std = 0.015
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module, "SKIP_GPT_INIT"):
                return
            if hasattr(module, "RESIDUAL_SCALE_INIT_FACTOR"):
                std *= 1 / (math.sqrt(2 * self.n_layers))
            if hasattr(module, "weight"):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"Sequence length ({T}) is longer than the block_size ({self.block_size})."
        emb_all = self.wte(idx)  # kept for MTP (future-token embeddings)
        x = emb_all
        sin = self.sin[:, :, :T, :]
        cos = self.cos[:, :, :T, :]
        g_sin = self.g_sin[:, :, :T, :]
        g_cos = self.g_cos[:, :, :T, :]

        # Shared sliding-window mask for all local layers at this T. Prefer the
        # cached mask (built before compile); fall back to building one for
        # sequence lengths not pre-registered (e.g. generate()).
        local_block_mask = None
        if self.needs_local_block_mask:
            local_block_mask = self._local_block_masks.get(T)
            if local_block_mask is None:
                local_block_mask = self._make_local_block_mask(T, idx.device)
                self._local_block_masks[T] = local_block_mask

        for block in self.transformer:
            if block.is_global:
                x = block(x, g_sin, g_cos, local_block_mask)
            else:
                x = block(x, sin, cos, local_block_mask)
        h_trunk = x  # main trunk representation (h^0 for the MTP chain)
        x = self.ln(h_trunk)

        logits = self.lm_head(x).float()
        logits = 30.0 * torch.tanh(logits / 30.0)

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

            # DeepSeek-V3 MTP auxiliary loss (training only). Skipped at eval so
            # the reported val loss stays a pure next-token metric, comparable
            # across mtp_depth settings.
            if self.mtp_depth > 0 and self.training:
                loss = loss + self.mtp_lambda * self._mtp_loss(
                    h_trunk, emb_all, targets, sin, cos
                )

            return None, loss
        else:
            return logits, None

    def _mtp_loss(self, h_trunk, emb_all, targets, sin, cos):
        """Mean cross-entropy over the MTP modules (DeepSeek-V3).

        Module k (0-indexed, depth d=k+1) predicts the token d steps beyond the
        main next-token target, using a transformer block over
            proj_k([RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+d}))]).
        Sequences shorten by d (drop the last d positions, which lack a target),
        and h^k feeds the next module to preserve the causal chain.
        """
        T = h_trunk.shape[1]
        h_prev = h_trunk
        total = 0.0
        for k in range(self.mtp_depth):
            d = k + 1
            Td = T - d
            if Td <= 0:
                break
            h_in = h_prev[:, :Td, :]
            emb_future = emb_all[:, d : d + Td, :]  # Emb(t_{i+d}) for i in 0..Td-1
            combined = torch.cat(
                [self.mtp_hidden_norm[k](h_in), self.mtp_embed_norm[k](emb_future)],
                dim=-1,
            )
            h_d = self.mtp_proj[k](combined)
            h_d = self.mtp_block[k](h_d, sin[:, :, :Td, :], cos[:, :, :Td, :])

            logits_d = self.lm_head(self.ln(h_d)).float()
            logits_d = 30.0 * torch.tanh(logits_d / 30.0)
            tgt_d = targets[:, d:].contiguous()  # predict t_{i+1+d}
            total = total + F.cross_entropy(
                logits_d.reshape(-1, logits_d.size(-1)), tgt_d.reshape(-1)
            )
            h_prev = h_d  # chain into the next depth (next slices [:Td-1])
        return total / self.mtp_depth

    def generate(
        self,
        idx,
        num_sequences=5,
        max_tokens=200,
        topk=50,
        chat_mode=False,
        eos_token=32000,
    ):
        idx = torch.repeat_interleave(idx.unsqueeze(0), num_sequences, dim=0)

        for _ in range(max_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]
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
                f"AdamW decay params (embed/head): {len(adamw_decay_params)} tensors, {sum(p.numel() for p in adamw_decay_params):,} parameters"
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

    def _precompute_rotary_embeddings(
        self, seq_len, head_dim, base=10000, rope_p=1.0, device=None
    ):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
        # p-RoPE: zero the rotation angle for the low-frequency tail pairs so
        # they pass through unchanged (sin(0)=0, cos(0)=1 -> identity). Only the
        # first n_rot = round(rope_p * head_dim/2) high-frequency pairs rotate.
        if rope_p < 1.0:
            n_pairs = inv_freq.shape[0]
            n_rot = max(1, int(round(rope_p * n_pairs)))
            if n_rot < n_pairs:
                freqs[:, n_rot:] = 0.0
        freqs = torch.cat((freqs, freqs), dim=-1)
        sin, cos = freqs.sin(), freqs.cos()
        sin, cos = sin.bfloat16(), cos.bfloat16()
        sin, cos = sin[None, None, :, :], cos[None, None, :, :]
        return sin, cos
