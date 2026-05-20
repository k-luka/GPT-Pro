import itertools
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformer_engine.pytorch as te

from src.models.gpt_dense import GQA, MLP  # unchanged sublayer modules
from src.utils.helpers import apply_rotary_emb

HC_N = 4  # number of parallel residual streams


class HyperConnection(nn.Module):
    """
    mHC-lite: replaces x = x + f(x) with doubly stochastic stream mixing.

    Residual stream x has shape [B, T, n, C].
    - H_res (n×n doubly stochastic): mixes the n streams via permutation decomposition
    - h_pre (n,): weighted sum of streams → single C-dim input for the sublayer
    - h_post (n,): distributes sublayer output back to n streams

    Initialised to recover a standard residual connection at step 0:
      stream 0 = stream 0 + sublayer(stream 0),  streams 1..n-1 pass through unchanged.
    """

    def __init__(self, n_embd: int, n: int = HC_N):
        super().__init__()
        self.n = n
        n_perms = math.factorial(n)

        # --- constant buffer: all n! permutation matrices ---
        perms = list(itertools.permutations(range(n)))
        eye = torch.eye(n)
        P = torch.stack([eye[list(p)] for p in perms])  # [n!, n, n]
        self.register_buffer("P", P)

        # find index of the identity permutation for initialisation
        identity_idx = perms.index(tuple(range(n)))

        # --- learned parameters ---
        # a_res: softmax weights over n! permutations → doubly stochastic H_res
        a_res_init = torch.zeros(n_perms)
        a_res_init[identity_idx] = 8.0  # concentrate on identity → H_res ≈ I
        self.a_res = nn.Parameter(a_res_init)

        # w_pre: sigmoid → h_pre weighting of n streams for sublayer input
        w_pre_init = torch.full((n,), -8.0)
        w_pre_init[0] = 8.0  # sigmoid ≈ [1, 0, 0, 0] → only stream 0 feeds sublayer
        self.w_pre = nn.Parameter(w_pre_init)

        # w_post: 2*sigmoid → h_post weighting of sublayer output to n streams
        w_post_init = torch.full((n,), -8.0)
        w_post_init[0] = 0.0  # 2*sigmoid(0)=1 → stream 0 gets output, others ~0
        self.w_post = nn.Parameter(w_post_init)

    def forward(self, x: torch.Tensor, sublayer_fn) -> torch.Tensor:
        # x: [B, T, n, C]
        n = self.n

        # build mixing matrices from learned scalars
        H_res = (torch.softmax(self.a_res, dim=0).view(-1, 1, 1) * self.P).sum(0)  # [n, n]
        h_pre = torch.sigmoid(self.w_pre)                    # [n]
        h_post = 2.0 * torch.sigmoid(self.w_post)            # [n]

        # weighted sum of n streams → single C-dim input for the sublayer
        x_in = (x * h_pre.view(1, 1, n, 1)).sum(dim=2)      # [B, T, C]

        # apply the sublayer (attention or MLP)
        y = sublayer_fn(x_in)                                 # [B, T, C]

        # mix streams via doubly stochastic H_res, then add sublayer output
        x_res = torch.einsum("ij,btjc->btic", H_res, x)      # [B, T, n, C]
        x_out = x_res + y.unsqueeze(2) * h_post.view(1, 1, n, 1)  # [B, T, n, C]
        return x_out


class Block_mHC(nn.Module):
    def __init__(self, n_embd, n_heads, n_kv_heads, ffn_hidden_size, dtype=None):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd, dtype=dtype)
        self.sa = GQA(n_embd, n_heads, n_kv_heads, dtype=dtype)
        self.ln2 = nn.RMSNorm(n_embd, dtype=dtype)
        self.mlp = MLP(n_embd, hidden_size=ffn_hidden_size, dtype=dtype)
        self.hc_attn = HyperConnection(n_embd)
        self.hc_mlp = HyperConnection(n_embd)

    def forward(self, x, sin, cos, is_first_microbatch=None):
        # x: [B, T, n, C]
        x = self.hc_attn(
            x,
            lambda h: self.sa(self.ln1(h), sin, cos, is_first_microbatch=is_first_microbatch),
        )
        x = self.hc_mlp(
            x,
            lambda h: self.mlp(self.ln2(h), is_first_microbatch=is_first_microbatch),
        )
        return x


class GPT(nn.Module):
    """
    Dense GPT with mHC-lite hyper-connections.
    Identical constructor signature to gpt_dense.GPT — same config and pretrain script work.
    """

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
                Block_mHC(
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
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)  # pyrefly: ignore
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # pyrefly: ignore
        # HyperConnection params are initialised in HyperConnection.__init__ — do not overwrite

    def forward(self, idx, targets=None, is_first_microbatch=None):
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length ({T}) is longer than block_size ({self.block_size})."
        )

        x = self.wte(idx)                                          # [B, T, C]
        x = x.unsqueeze(2).expand(-1, -1, HC_N, -1).clone()       # [B, T, n, C]

        sin = self.sin[:, :, :T, :]  # pyrefly: ignore
        cos = self.cos[:, :, :T, :]  # pyrefly: ignore

        for block in self.transformer:
            x = block(x, sin, cos, is_first_microbatch=is_first_microbatch)

        x = x.mean(dim=2)   # [B, T, C]  contract n streams back to C
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
        eos_token=100257,
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
                # 1-D params: norms + HyperConnection scalars (a_res, w_pre, w_post)
                adamw_nodecay_params.append(p)

        if self.rank == 0:
            print(
                f"Muon params (2D hidden): {len(muon_params)} tensors, "
                f"{sum(p.numel() for p in muon_params):,} parameters"
            )
            print(
                f"AdamW decay params (Embed/Head): {len(adamw_decay_params)} tensors, "
                f"{sum(p.numel() for p in adamw_decay_params):,} parameters"
            )
            print(
                f"AdamW no-decay params (1D norms + HC): {len(adamw_nodecay_params)} tensors, "
                f"{sum(p.numel() for p in adamw_nodecay_params):,} parameters"
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
