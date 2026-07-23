# GPT-Pro

GPT-Pro is a from-scratch decoder-only language-model training project built
for NVIDIA B200 GPUs. The current production target is a 6.54B-parameter dense
model trained on 159.4B tokens with a custom 81,920-token Gemma-derived
English/code tokenizer.

## Production architecture

- 35 layers, width 4096, SwiGLU hidden size 11008
- 32 query heads and 8 KV heads with head dimension 128
- Four W512 local-attention layers per full-attention layer
- FlashAttention-4 for local attention and cuDNN SDPA for full attention
- QK RMSNorm, 25% partial RoPE, pre-norm residual blocks
- Tied token embeddings and output projection
- BF16 parameters with torchao MXFP8 linear layers
- Muon for matrix parameters and AdamW for embeddings and norms

The final configuration uses a 4.19M-token global batch on eight B200 GPUs.
Architecture selection and measured tradeoffs are summarized in
[`docs/architecture-decisions.md`](docs/architecture-decisions.md).

## Repository layout

```text
config/                              Hydra configurations
pretrain_dense_torchao.py            Production pretraining entry point
scripts/run_gemma80k_6p54b_final.sh  Eight-B200 Slurm launcher
scripts/data_prep/                    Tokenizer and shard construction
src/datasets/                         Self-describing token-shard loader
src/models/gpt_dense_torchao.py      Dense transformer
src/models/fa4_attn.py               FA4 autograd integration
src/training/                         Training and checkpoint loop
tokenizers/gemma4_80k/               Exact tokenizer and teacher-ID mappings
tests/                                Tokenizer and shard-pipeline tests
```

## Environment

The production stack was tested with Python 3.13, a PyTorch 2.12 nightly,
torchao 0.18 nightly, CUDA 12.8, and B200 GPUs. It also requires Hydra,
Hugging Face `tokenizers` and `datasets`, W&B, NumPy, and the FA4
`flash_attn_interface` package.

## Validate and train

The dataset is 1,599 training shards plus one validation shard. Before a final
run, exhaustively verify it and write the report required by the launcher:

```bash
python scripts/data_prep/validate_token_shards.py \
  data/climbmix_gemma4_80k \
  --tokenizer tokenizers/gemma4_80k/tokenizer.json \
  --report data/climbmix_gemma4_80k/validation_report.json

python scripts/preflight_gemma80k_final.py --require-full-validation
sbatch scripts/run_gemma80k_6p54b_final.sh
```

The launcher resumes the highest complete distributed checkpoint and reuses a
stable W&B run ID across Slurm requeues.

## Tests

```bash
python -m pytest -q
```

Generated datasets, checkpoints, W&B state, and experiment logs are excluded
from version control.
