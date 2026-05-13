# Project State

## Goal

Train a production-quality LLM to deploy on **Navigator**, a tool for University of Florida (UF) students. The model must be capable (strong reasoning, coding, math, general knowledge) and safe (aligned, helpful, harmless). Achieving this requires multiple training stages:

1. **Pretraining** — learn language and world knowledge from large-scale text
2. **Supervised Fine-Tuning (SFT)** — teach the model to follow instructions and have conversations
3. **Reinforcement Learning (RL)** — improve reasoning and alignment via feedback

---

## Cluster

- **System:** HiPerGator (University of Florida)
- **GPUs:** NVIDIA B200 (Blackwell), partition `hpg-b200`, account `weishao`
- **Environment:** conda env `LLM`, CUDA 12.8.1, Python 3.13
- **Data root:** `/blue/pinaki.sarder/kirill.luka/`

---

## Repository Structure

Two parallel model stacks exist side by side for comparison:

| | Dense | MoE |
|---|---|---|
| Model | `src/models/gpt_dense.py` | `src/models/gpt_moe.py` |
| Trainer | `src/training/trainer_dense.py` | `src/training/trainer_moe.py` |
| Entry point | `pretrain_dense.py` | `pretrain_moe.py` |
| SLURM script | `scripts/run_dense.sh` | `scripts/run_moe.sh` |
| Config (full) | `config/config_dense.yaml` | `config/config_moe.yaml` / `config_moe_big.yaml` |
| Config (nano) | `config/config_dense_nano.yaml` | — |
| Run script (nano) | `scripts/run_dense_nano.sh` | — |

The legacy PyTorch (non-TE) files `src/models/gpt.py` and `src/training/trainer.py` are kept as reference artifacts.

---

## Architecture

Both stacks share:
- **GQA** (Grouped Query Attention) with RoPE and per-head RMSNorm on Q/K
- **SwiGLU** MLP (dense) or **MoE** with shared + routed experts (MoE)
- **DeepSeek-V3-Base tokenizer** (vocab size 129024, padded to multiple of 256)
- **Muon + AdamW** dual optimizer
- **FP8 training** via NVIDIA Transformer Engine
- **DCP** (PyTorch Distributed Checkpoint) for saving/loading
- **WSD LR schedule** (linear warmup → constant → linear warmdown)
- **Logit soft-capping** (`30 * tanh(logits / 30)`) to prevent logit explosion

**Dense stack** uses DDP + `torch.compile`. Simpler, faster to iterate on.

**MoE stack** uses FSDP (dense parts) + Expert Parallelism via all-to-all dispatch. DeepSeek-V3-style aux-loss-free load balancing via gate bias correction.

---

## Pretraining Data

- **Dataset:** ClimbMix-400B (same dataset used by Karpathy's nanochat)
- **Location:** `data/climbmix_400b/` (binary uint32 shards, 100M tokens each)
- **Tokenizer:** DeepSeek-V3-Base (via HuggingFace `transformers`)
- **Note:** Shards are flat-packed with no document boundary markers. BOS token packing requires re-processing from raw source — deferred.

---

## Model Configs

### GatorLM2_nano (Dense) — `config/config_dense_nano.yaml`
- ~50M params, 8 layers, n_embd=512, 8 Q heads / 4 KV heads, block_size=1024
- 4 GPUs, 40k steps
- **Purpose:** fast iteration on pretraining quality — perfect this before scaling up
- Status: **ready to run**

### GatorLM2 (Dense) — `config/config_dense.yaml`
- ~3.5B params, 27 layers, n_embd=4096, 16 Q heads / 4 KV heads, ffn_hidden_size=5632
- 4 GPUs, 67k steps, ~70B tokens (Chinchilla-optimal)
- Status: **ready to train, confirmed running**

### LM_Neo_5B (MoE) — `config/config_moe.yaml`
- ~5B total / ~2B active params, 24 layers, n_embd=3072, 24 Q heads / 8 KV heads
- 16 routed experts, 2 shared experts, topK=4, expert_hidden_size=1024
- 8 GPUs, 52k steps
- Status: **ready to train, not yet tested after reorganization**

### LM_Neo_15B (MoE Large) — `config/config_moe_big.yaml`
- ~15B total / ~4B active params
- Status: **config exists, not yet run**

---

## What Has Been Done

### Architecture
- [x] GQA with RoPE and per-head Q/K RMSNorm
- [x] SwiGLU MLP (dense) and MoE with shared + routed experts
- [x] Expert parallelism via all-to-all (MoE)
- [x] Aux-loss-free MoE load balancing via gate bias correction
- [x] FP8 training via Transformer Engine
- [x] Muon + AdamW dual optimizer
- [x] WSD LR schedule (warmup → stable → linear warmdown)
- [x] Logit soft-capping (30 * tanh(x / 30))
- [x] DCP distributed checkpointing with best-val tracking

### Infrastructure
- [x] Two separate stacks (dense / MoE) runnable from separate scripts — no commenting/uncommenting
- [x] Nano config for fast iteration (`config_dense_nano.yaml`, `run_dense_nano.sh`)
- [x] ClimbMix-400B tokenized and sharded
- [x] SLURM scripts for single-node (4 GPU) and multi-node runs
- [x] W&B logging, HellaSwag eval during training
- [x] Conda env fixed after path migration to `/blue/pinaki.sarder/`

---

## Training Stages — TODO

### Stage 1: Pretraining
- [x] Dense stack implemented and confirmed working on 4× B200
- [x] MoE stack implemented (needs test run after reorganization)
- [ ] **Use nano model to perfect pretraining** (current focus)
  - [ ] Investigate BOS token packing (document boundaries in ClimbMix shards)
  - [ ] Evaluate adding value residual (ResFormer) to architecture
  - [ ] Consider sliding window attention (blocked on flash_attn 3)
- [ ] Decide dense vs MoE based on throughput comparison
- [ ] Run full pretraining on full-scale model (~70B tokens for dense)

### Stage 2: Supervised Fine-Tuning — not started
- [ ] Assemble SFT dataset (instruction following, conversations, math, coding, safety)
- [ ] Implement SFT training loop with loss masking on completions only
- [ ] `sft.py` exists but needs updating to match new model/trainer structure

### Stage 3: Reinforcement Learning — not started
- [ ] Implement GRPO or similar
- [ ] Define reward signal (math correctness, helpfulness, safety)
- [ ] Safety-specific RL pass (harmful request refusal, UF policy compliance)

### Evaluation — partial
- [x] HellaSwag (runs during training automatically)
- [ ] Add ARC, MMLU, GSM8K, HumanEval benchmarks
- [ ] Safety evaluation suite

---

## Known Issues / Notes

- `flash_attn` 2.x does not compile for B200 (sm_100 Blackwell). TE falls back to its internal attention backend. Handled by a patch in `transformer_engine/pytorch/attention/.../backends.py`.
- The conda env was relocated from `/blue/weishao/` to `/blue/pinaki.sarder/`. All paths fixed (shebangs, sysconfig, Jupyter kernel).
- `torch` pinned to `2.11.0+cu128`, `transformer_engine` to `2.10.0`.
- Sliding window attention requires flash_attn 3 — not feasible on current setup.
