"""FlashAttention-4 (flash_attn.cute) causal attention as a torch custom op.

FA4 is the CuTe-DSL attention implementation from flash-attn-4 — the fastest
kernel on B200 (measured fwd+bwd at S=4096, B200: sliding-window head_dim 128
is 2.3x faster than the FlexAttention Triton path; full-causal head_dim 256 is
2.9x faster than SDPA). Its Python entry point launches JIT-compiled kernels
through tvm_ffi, which torch.compile cannot trace through — so it is exposed
here as an opaque custom op with an explicit autograd rule, the supported way
to put an external kernel inside a compiled graph.

Semantics match the model's other attention paths: causal, GQA-native
(n_kv_heads <= n_heads), softmax scale 1/sqrt(head_dim).

Layout is FA-style (B, S, H, D) — note this differs from SDPA/FlexAttention's
(B, H, S, D).

SM100 (B200) kernel coverage in flash-attn-4 4.0.0b15 (also true of b20):
head_dim <= 128 supports causal + sliding window; head_dim 256 supports
causal only ("does not support local attention yet").

The head_dim-256 SM100 kernel silently returns wrong values for strided
(e.g. transposed) inputs (verified against SDPA on b15), so the op impls
densify with .contiguous() — a no-op for already-packed tensors, which is
what the model provides.
"""

from typing import Optional

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


@torch.library.custom_op("gptpro::fa4_attn_fwd", mutates_args=())
def _fa4_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size_left: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = _flash_attn_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        causal=True,
        window_size_left=window_size_left,
        window_size_right=0,
        return_lse=True,
    )
    return out, lse


@_fa4_fwd.register_fake
def _(q, k, v, window_size_left):
    B, S, H, _ = q.shape
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    return out, lse


@torch.library.custom_op("gptpro::fa4_attn_bwd", mutates_args=())
def _fa4_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    window_size_left: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Positional args mirror FlashAttnFunc.backward:
    # (q, k, v, out, dout, lse, softmax_scale, causal, softcap)
    dq, dk, dv = _flash_attn_bwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        out.contiguous(),
        dout.contiguous(),
        lse,
        None,
        True,
        0.0,
        window_size_left=window_size_left,
        window_size_right=0,
    )
    return dq, dk, dv


@_fa4_bwd.register_fake
def _(dout, q, k, v, out, lse, window_size_left):
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
    )


def _setup_context(ctx, inputs, output):
    q, k, v, window_size_left = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.window_size_left = window_size_left


def _backward(ctx, dout, dlse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _fa4_bwd(dout, q, k, v, out, lse, ctx.window_size_left)
    return dq, dk, dv, None


_fa4_fwd.register_autograd(_backward, setup_context=_setup_context)


def fa4_attention(q, k, v, sliding_window=0):
    """Causal attention via FA4. q: (B, S, H, D); k, v: (B, S, H_kv, D).

    sliding_window > 0 limits each query to its previous `sliding_window`
    positions (inclusive of itself); 0 means full causal attention.
    Returns (B, S, H, D), contiguous.
    """
    window_size_left = sliding_window - 1 if sliding_window > 0 else None
    out, _ = _fa4_fwd(q, k, v, window_size_left)
    return out
