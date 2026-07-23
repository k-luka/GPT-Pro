# Architecture decisions

The production design was selected with matched-token proxy training and short
6.54B-parameter systems probes on B200 GPUs. Values below are measurements from
the July 2026 campaigns; they are comparative results, not benchmark claims.

## Selected configuration

| Component | Choice |
|---|---|
| Residual and normalization | Standard residual, pre-norm RMSNorm |
| Attention pattern | Four local layers followed by one full layer |
| Local attention | W512, FlashAttention-4 |
| Full attention | cuDNN SDPA |
| Attention geometry | 32 query / 8 KV heads, head dimension 128 |
| Positional encoding | 25% partial RoPE |
| Feed-forward | Dense SwiGLU, hidden size 11008 |
| Tokenizer | Gemma-derived English/code BPE, 81,920 tokens |

## Final attention comparison

At 3B proxy tokens, W512 local attention and full attention with head dimension
256 were effectively tied in validation BPB. On the exact 6.54B model, the
selected W512 configuration was 9.1% faster.

| Exact-model probe | Tokens/s, 2 GPUs | Peak GPU memory |
|---|---:|---:|
| Full attention, head dimension 128 | 59,508 | 135,360 MiB/GPU |
| Full attention, head dimension 256 | 58,092 | 129,472 MiB/GPU |
| 4:1 local/full, W512, head dimension 128 | **63,394** | **130,610 MiB/GPU** |
| 4:1 local/full, W1024 | 62,476 | 130,610 MiB/GPU |
| 4:1 local/full, W2048 | 61,123 | 130,610 MiB/GPU |

W512 was retained because larger windows reduced throughput without a measured
quality advantage. A 3:1 local/full cadence was only 1.1% faster and had
slightly worse proxy BPB.

## Rejected options

- Sandwich norm was slower, consumed more memory, and degraded the local
  attention proxy result. Pre-norm was selected.
- Dynamic mHC residuals improved proxy loss but approximately halved
  throughput in the unfused implementation. The implementation was removed
  from the production path.
- Multi-token prediction was excluded because its additional training compute
  was not justified under the fixed two-week budget.
- The existing MoE implementation was not selected because routing and expert
  dispatch reduced end-to-end throughput at this scale.

## Training budget

The final schedule uses 38,000 optimizer steps at 4,194,304 tokens per step:
159,383,552,000 training tokens, or about 24.4 tokens per parameter. It uses
1,000 warmup steps followed by a stable phase and a 10% linear warmdown.

The exact two-GPU W512 probe measured 63.4K tokens/s. The 14-day job requires
131.8K aggregate tokens/s on eight GPUs, leaving substantial scaling margin.
