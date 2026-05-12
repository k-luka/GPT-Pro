# CLAUDE.md -- GPT-Pro Codebase Guide

## Project Overview

GPT-Pro is a research LLM training framework implementing a **DeepSeek-V3-style architecture**. It evolved from a GPT-2 baseline and features:

- **Grouped Query Attention (GQA)** with RoPE and per-head RMSNorm
- **Auxiliary-loss-free Mixture of Experts (MoE)** with shared + routed experts
- **Expert Parallelism** via all-to-all dispatch/return across GPU ranks
- **FP8 training** via NVIDIA Transformer Engine (TE) with BF16 parameters
- **Muon + AdamW dual optimizer** for faster convergence
- Designed for NVIDIA B200 (Blackwell) GPUs on the HiPerGator supercomputer

Largest model trained: ~15B total params, ~4B active params.

---

## Repository Structure

```
GPT-Pro/
├── pretrain.py                  # Main pretraining entry point (Hydra + FSDP setup)
├── inference.py                 # Interactive inference/chat script
├── evaluate.py                  # HellaSwag benchmark evaluation
├── sft.py                       # Supervised fine-tuning script
│
├── config/
│   ├── config_pretrain.yaml     # Default pretraining config (~5B params)
│   ├── config_pretrain_big.yaml # Large pretraining config (~15B params)
│   ├── config_sft.yaml          # SFT configuration
│   ├── config_eval.yaml         # Evaluation configuration
│   ├── config_inference.yaml    # Inference configuration
│   └── config_convert_checkpoint.yaml
│
├── src/
│   ├── models/
│   │   ├── gpt_te.py            # PRIMARY model: GQA + MoE with Transformer Engine
│   │   └── gpt.py               # Legacy non-TE model (reference only)
│   ├── training/
│   │   ├── trainer_te.py        # PRIMARY trainer: FP8, FSDP, checkpointing
│   │   └── trainer.py           # Legacy trainer (reference only)
│   ├── datasets/
│   │   └── dataloader.py        # Shard-based distributed dataloader
│   ├── eval/
│   │   └── metrics.py           # Loss estimation + HellaSwag evaluation
│   └── utils/
│       ├── helpers.py           # Param counting, FLOP estimation, RoPE utilities
│       └── optimizers.py        # DualOptimizer (AdamW + Muon wrapper)
│
├── scripts/
│   ├── run_node.sh              # Single-node SLURM launcher (8 B200 GPUs)
│   ├── run_multi_node.sh        # Multi-node SLURM launcher
│   ├── convert_dist_checkpoint.py
│   └── data_prep/
│       ├── fineweb.py           # FineWeb-EDU tokenization + binary sharding
│       ├── hellaswag.py         # HellaSwag download + rendering
│       ├── prepare_climbmix.py  # ClimbMix-400B tokenization pipeline
│       ├── alpacha-cleaned.py   # Dataset preprocessing
│       ├── clean_SFT_data.py    # SFT data cleaning
│       └── run_climbmix_tokenizer.sh
│
└── assets/
    └── small_model_training_runs.png
```

---

## Key Source Files

| File | Role |
|---|---|
| `src/models/gpt_te.py` | Primary model implementation -- **edit here for architecture changes** |
| `src/training/trainer_te.py` | Primary training loop -- **edit here for training logic changes** |
| `pretrain.py` | Entry point: initializes distributed process group, wraps model with FSDP, creates Trainer |
| `config/config_pretrain.yaml` | Default hyperparameters (~5B params) -- **first place to look when adjusting a run** |
| `config/config_pretrain_big.yaml` | Large model config (~15B params) |
| `src/datasets/dataloader.py` | Distributed binary shard dataloader |
| `src/utils/optimizers.py` | DualOptimizer: wraps AdamW + Muon into a single interface |
| `src/utils/helpers.py` | `print_trainable_parameters()`, `estimate_flops()`, RoPE utilities |

The `gpt.py` and `trainer.py` files are **legacy non-TE implementations** kept for reference. Do not modify them; use the `_te` variants.

---

## Architecture

### Model (`src/models/gpt_te.py`)

Decoder-only Transformer with these key components:

**GQA (Grouped Query Attention)**
- Separate Q, K, V projections via `te.Linear` (fewer KV heads than query heads)
- Per-head RMSNorm on Q and K *after* RoPE application
- `F.scaled_dot_product_attention` with `is_causal=True, enable_gqa=True`
- Configurable via `n_heads` (query) and `n_kv_heads` (KV)

**Gate (MoE Router)**
- Top-k expert selection with sigmoid scoring + learnable bias correction
- Bias updated from global expert token counts (no auxiliary loss)
- Weights normalized per-token: `weights / weights.sum() * route_scale`

**SharedExpert**
- Dense FFN applied to all tokens unconditionally
- SiLU gating; hidden size aligned to 256

**MoE (single-GPU)**
- Shared experts + routed experts via TE `GroupedLinear`
- Uses `te.moe_permute` / `te.moe_unpermute` for efficient expert dispatch
- Updates routing bias from global expert token counts for load balancing
- Logs per-layer expert utilization to `output/expert_stats/layer_loads.jsonl`

**ExpertParallelMoE (multi-GPU)**
- Routed experts sharded across ranks (`n_routed_experts / world_size` per GPU)
- All-to-all dispatch (tokens + expert indices + weights) -> local expert compute -> all-to-all return
- Shared experts compiled with `torch.compile` during training

**Block**
- Pre-norm: `RMSNorm -> GQA`, `RMSNorm -> MoE`, with residual connections
- Uses `torch.compile`d attention during training for speed

**GPT**
- Token embeddings weight-tied with output projection (`lm_head`)
- Pre-computed RoPE sin/cos buffers
- `generate()` method for autoregressive text generation with top-k sampling

### MoE Load Balancing (Aux-Loss-Free)
- Each expert accumulates a global token count via `dist.all_reduce`
- The gate bias for over-loaded experts is decremented; under-loaded gets incremented
- No separate auxiliary loss term -- same approach as DeepSeek-V3

---

## Configuration System

All configuration uses [Hydra](https://hydra.cc/). Config files live in `config/`.

There are two pretraining configs:
- `config_pretrain.yaml` -- default (~5B param model)
- `config_pretrain_big.yaml` -- large (~15B param model)

**Key fields in `config/config_pretrain.yaml`:**

```yaml
experiment:
  project: "GPT_Max"          # W&B project name
  run_name: "LM_Neo_5B"       # W&B run name + checkpoint directory name

model:
  vocab_size: 129024           # Padded to multiple of 256 for tensor cores
  block_size: 4096             # Sequence length
  n_embd: 3072                 # Hidden dimension
  n_layers: 24
  n_heads: 24                  # Query heads
  n_kv_heads: 8                # KV heads (GQA)
  n_shared_experts: 2
  n_routed_experts: 16
  topk_experts: 4
  expert_hidden_size: 1024

data:
  train_data_root: "data/climbmix_400b/"
  val_data_root:   "data/climbmix_400b/"

training:
  batch_size: 8                # Per-device batch size
  grad_accum_steps: 8          # Gradient accumulation
  batch_warmup_steps: 1000     # Steps to ramp grad_accum from 1 to full
  max_steps: 52000
  warmup_steps: 1000           # LR warmup
  max_lr: 2.0e-4
  min_lr: 2.0e-5
  weight_decay: 0.1
  checkpoint_interval: 5000
  eval_interval: 1000
  eval_steps: 80

# Uncomment to resume:
# resume_checkpoint: output/checkpoints/LM_Neo_5B/best_val
```

**The big config** (`config_pretrain_big.yaml`) differs mainly in:
- `n_embd: 4096`, `n_heads: 32`, `n_routed_experts: 24`, `topk_experts: 8`
- `max_steps: 160000`, `warmup_steps: 2000`

Hydra overrides work on the CLI: `torchrun ... pretrain.py training.max_lr=1e-4`

---

## Training

### FSDP Wrapping (in `pretrain.py`)
- Uses `transformer_auto_wrap_policy` wrapping each `Block` independently
- `ShardingStrategy.SHARD_GRAD_OP` -- shards optimizer states + gradients, not params
- MoE modules are excluded from FSDP via `ignored_modules` (they are expert-parallel, not data-parallel)
- `MixedPrecision`: BF16 params, FP8 compute (from TE autocast inside trainer), float32 reduce

### FP8 (in `src/training/trainer_te.py`)
- TE autocast with `HYBRID` format and `DelayedScaling` (`amax_history_len=16`)
- BF16 model parameters stored; FP8 used only for GEMM operations
- `is_first_microbatch` flag lets TE cache FP8-cast weights across gradient accumulation steps
- `amax_reduction_group=dist.group.WORLD` syncs FP8 scaling across all ranks

### Optimizer (`src/utils/optimizers.py` + `gpt_te.py:configure_optimizers`)
- **DualOptimizer**: wraps two separate optimizers into a single interface
  - **AdamW** (`torch.optim.AdamW`): embeddings (`wte`, `lm_head`), norms, and biases. Betas=(0.9, 0.95), fused when available.
  - **Muon** (`torch.optim.Muon`): all 2D weight matrices except embeddings. Uses Newton-Schulz orthogonalization with `momentum=0.95`, `nesterov=True`, `adjust_lr_fn="match_rms_adamw"`.
- Gradient clipping (norm 1.0) applied only to AdamW params when using DualOptimizer (Muon handles its own normalization).
- Parameter routing logic in `GPT.configure_optimizers()`: 2D params -> Muon (unless embedding/head), 1D params -> AdamW without decay.

### Learning Rate Schedule
- Linear warmup for `warmup_steps` steps
- Cosine decay from `max_lr` to `min_lr` over `max_steps`

### Gradient Accumulation with Batch Warmup
- Starts with `grad_accum_steps=1`, linearly ramps to configured value over `batch_warmup_steps`
- FSDP `no_sync()` context used for all microbatches except the last (single all-reduce over fully-accumulated gradients)
- Effective batch size = `batch_size * grad_accum_steps * world_size`

### Data Prefetching
- Uses a separate CUDA stream to prefetch the next microbatch while the current one is computing
- Overlap of data transfer and compute for better GPU utilization

### Checkpointing
- Uses PyTorch DCP (`torch.distributed.checkpoint`) for distributed saves
- Saves at `output/checkpoints/<run_name>/step_XXXXX/` periodically
- Best validation loss checkpoint at `output/checkpoints/<run_name>/best_val/`
- Resume by setting `resume_checkpoint` in config or CLI

---

## Running Training

### Prerequisites
1. Activate conda environment: `conda activate LLM`
2. Load CUDA: `module load cuda/12.8.1`
3. Prepare data (see Data Pipeline section below)

### Single Node (8 GPUs)
```bash
# On HiPerGator -- submits a SLURM job
sbatch scripts/run_node.sh

# Or directly (if on a GPU node):
torchrun --nproc-per-node=8 pretrain.py
```

**SLURM config in `scripts/run_node.sh`:**
- Partition: `hpg-b200`, Account: `weishao`
- 8 GPUs, 256 GB RAM, 14-day wall time
- Sets `NCCL_NVLS_ENABLE=0` to avoid NVLink issues

### Multi-Node
```bash
sbatch scripts/run_multi_node.sh
# 2 nodes x 4 GPUs = 8 GPUs total, c10d backend
```

### Resuming a Run
```bash
torchrun --nproc-per-node=8 pretrain.py resume_checkpoint=output/checkpoints/LM_Neo_5B/best_val
```

---

## Data Pipeline

### Pretraining Data (ClimbMix-400B)
```bash
# Tokenize and shard ClimbMix-400B into binary files
bash scripts/data_prep/run_climbmix_tokenizer.sh
# Or directly:
python scripts/data_prep/prepare_climbmix.py
# Output: data/climbmix_400b/*.bin (uint32, 100M tokens per shard)
```

### Alternative: FineWeb-EDU
```bash
python scripts/data_prep/fineweb.py
# Output: data/edu_fineweb350B/edufineweb_{train,val}_XXXXXX (uint16)
```

### HellaSwag (Auto-downloaded on first eval)
```bash
python scripts/data_prep/hellaswag.py
# Output: data/hellaswag/{train,val,test}.jsonl
```

### Data Format
- Binary files containing packed uint32 token IDs
- `DataLoader` (`src/datasets/dataloader.py`) loads shards, slices by `(rank, world_size)` to prevent duplication
- Supports resuming mid-shard via `set_step()`

### Tokenizer
- **DeepSeek-V3-Base tokenizer** (via `transformers.AutoTokenizer`)
- Vocab size: 128815 (padded to 129024 for tensor core alignment)

---

## Inference

```bash
torchrun --nproc-per-node=1 inference.py
# Config: config/config_inference.yaml
# Interactive prompt loop; loads checkpoint and generates text
```

---

## Evaluation

```bash
torchrun --nproc-per-node=8 evaluate.py
# Config: config/config_eval.yaml
# Runs HellaSwag 4-way multiple choice; prints accuracy
```

HellaSwag evaluation is also run automatically during training every `eval_interval` steps.

---

## Supervised Fine-Tuning

```bash
torchrun --nproc-per-node=8 sft.py
# Config: config/config_sft.yaml
# Loads a pretrained checkpoint, fine-tunes on instruction data
# Loss is masked: only loss_mask positions are supervised
```

---

## Outputs Directory Layout

```
output/
├── checkpoints/<run_name>/
│   ├── best_val/              # Best val-loss checkpoint (DCP format)
│   └── step_XXXXX/            # Periodic checkpoints (DCP format)
├── expert_stats/
│   └── layer_loads.jsonl      # Per-layer expert token counts (JSON lines, appended each eval)
└── hydra/<YYYY-MM-DD>/<HH-MM-SS>/
    ├── .hydra/                # Config snapshot for this run
    └── outputs.log            # Training stdout
```

---

## Logging

- **W&B**: Logged from rank 0 only. Project/run name from `experiment.project`/`experiment.run_name` in config.
- **stdout**: Loss, LR, tokens/sec, HellaSwag accuracy printed every `logging_steps`.
- **Expert stats**: `output/expert_stats/layer_loads.jsonl` -- one JSON object per layer per eval step with expert token count arrays.

---

## Key Conventions

### Distributed Training
- Always use `dist.get_rank()` / `LOCAL_RANK` env var for rank checks. `LOCAL_RANK` is set by `torchrun`.
- Rank 0 is "master rank" -- only it logs to W&B, prints summaries, and writes expert stats.
- All collective ops (`all_reduce`, `all_to_all`) must be called on **all ranks** simultaneously.
- Barrier at end of training (`dist.barrier()`) before `wandb.finish()`.

### Model Changes
- Primary model is `src/models/gpt_te.py`. Do not edit `src/models/gpt.py` (legacy).
- If adding new layers, make sure they are either included in FSDP wrapping or explicitly added to `ignored_modules`.
- Expert-parallel modules (MoE) are intentionally excluded from FSDP -- they manage their own distribution.
- Vocab size must remain a multiple of 256 for tensor core efficiency.

### Configuration Changes
- All hyperparameters go in the appropriate `config/*.yaml` file.
- Use `config_pretrain.yaml` for default runs, `config_pretrain_big.yaml` for the large model.
- Use Hydra CLI overrides for one-off experiment changes rather than editing the YAML.
- `resume_checkpoint` is commented out by default; uncomment or pass via CLI to resume.

### Checkpoints
- Checkpoints use PyTorch DCP format -- they are **directories**, not single files.
- To convert a distributed checkpoint for single-GPU use, run `scripts/convert_dist_checkpoint.py`.

### Optimizer Parameter Groups
- Do not accidentally assign embedding/norm parameters to the Muon group -- Muon is designed for 2D weight matrices only.
- `GPT.configure_optimizers()` routes parameters: 2D hidden weights -> Muon, embeddings -> AdamW with decay, 1D params -> AdamW without decay.
- `DualOptimizer` wraps both into a unified interface with `zero_grad()`, `step()`, `state_dict()`, `load_state_dict()`.

### Data
- Shards are uint32 binary files. New datasets must be preprocessed into this format before use.
- Update `data.train_data_root` / `data.val_data_root` in config to switch datasets.
- Sequence packing is handled at preprocessing time (ClimbMix pipeline); `DataLoader` simply reads sequential tokens.

---

## Technology Stack

| Component | Library/Tool |
|---|---|
| Core framework | PyTorch 2.x |
| FP8 / Fused ops | NVIDIA Transformer Engine (TE) |
| Distributed training | PyTorch FSDP + NCCL |
| Distributed checkpointing | `torch.distributed.checkpoint` (DCP) |
| Configuration | Hydra + OmegaConf |
| Experiment tracking | Weights & Biases (W&B) |
| Tokenizer | HuggingFace `transformers` (DeepSeek-V3-Base) |
| Data download | HuggingFace `datasets` |
| Compute | NVIDIA B200 (Blackwell), CUDA 12.8.1 |
| Cluster scheduler | SLURM (HiPerGator, account: `weishao`) |

---

## Common Tasks

**Change learning rate:** Edit `training.max_lr` / `training.min_lr` in `config/config_pretrain.yaml`.

**Change model size:** Edit `model.n_embd`, `model.n_layers`, `model.n_heads`, etc. Or switch to `config_pretrain_big.yaml`.

**Switch dataset:** Update `data.train_data_root` and `data.val_data_root`.

**Resume training:** Uncomment `resume_checkpoint` in config or pass `resume_checkpoint=<path>` on CLI.

**Check expert utilization:** Parse `output/expert_stats/layer_loads.jsonl` -- each line has per-expert token counts per layer.

**Add a new optimizer parameter group:** Modify `DualOptimizer` in `src/utils/optimizers.py` and the `configure_optimizers()` logic in `gpt_te.py`.

**Run a quick experiment (no cluster):** Use a smaller config (reduce `n_layers`, `n_embd`, `max_steps`) and run `torchrun --nproc-per-node=1 pretrain.py` directly.
