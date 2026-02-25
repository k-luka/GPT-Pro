# GPT-Pro: DeepSeek-Style MoE LLM

A research-oriented LLM codebase that started from a GPT-2 style decoder-only Transformer and evolved toward a DeepSeek-V3 style architecture. The current model uses Multihead Latent Attention (MLA) and an auxiliary-loss-free Mixture of Experts (MoE) with shared and routed experts, plus expert parallelism.

This repo is built for large-scale training on multi-GPU clusters and includes data preprocessing, evaluation, logging, and distributed training utilities.

## Highlights
- DeepSeek-style **Multihead Latent Attention (MLA)** with RoPE.
- **Auxiliary-loss-free MoE** using bias-corrected routing for load balancing.
- **Expert parallelism** (all-to-all dispatch/return) plus FSDP sharding.
- **FP8 training** via NVIDIA Transformer Engine (TE) with BF16 params.
- HellaSwag evaluation and dataset preprocessing scripts.

## Project Layout
- `pretrain.py`: main entrypoint for distributed pretraining with Hydra config.
- `src/te_versions/model_te.py`: MLA, MoE, and the TE-optimized GPT model.
- `src/te_versions/trainer_te.py`: training loop, LR schedule, eval, checkpoints.
- `src/data.py`: shard-based dataloader with multi-rank offset logic.
- `dataset_preprocessing/fineweb.py`: FineWeb-EDU tokenizer + sharding.
- `dataset_preprocessing/hellaswag.py`: download + eval helpers.
- `config/config_pretrain.yaml`: model/training/data configuration.
- `run.sh` / `multi_node_run.sh`: SLURM single-node and multi-node launchers.

## Architecture Summary
The model is a decoder-only Transformer with:
- **MLA**: fused down-projection into latent Q/KV + shared RoPE key, followed by per-head up-projections.
- **MoE block**: shared dense expert + routed experts with top-k gating.
- **Aux-loss-free load balancing**: routing bias updated from global expert token counts.
- **Expert parallelism**: routed experts sharded across ranks with all-to-all dispatch.

## Training Stack
- **Distributed**: PyTorch DDP + FSDP, sharding gradients (SHARD_GRAD_OP).
- **Mixed precision**: BF16 params and TE FP8 autocast for GEMMs.
- **Logging**: W&B on rank 0, plus JSONL expert-load logs.

## Scale Achieved (From Project Notes)
- Largest trained model: **~15B total params, ~4B active params**.
- Training hardware: **NVIDIA B200 (Blackwell)** with FP8 (Transformer Engine) on HiPerGator.
- Parallelism: **Data Parallelism + Expert Parallelism + Optimizer Sharding**.

## Quickstart
### 1) Prepare data
FineWeb-EDU sharding (writes to `data/edu_fineweb350B/` by default):

```bash
python dataset_preprocessing/fineweb.py
```

HellaSwag is downloaded on first eval run automatically.

### 2) Configure
Edit `config/config_pretrain.yaml` for model shape, MoE settings, and training schedule.

### 3) Run (single node)
```bash
bash run.sh
```

### 4) Run (multi-node)
```bash
bash multi_node_run.sh
```

## Outputs
- Checkpoints: `output/checkpoints/<run_name>/...`
- Expert-load stats: `output/expert_stats/layer_loads.jsonl`
- Hydra logs: `output/hydra/<date>/<time>/...`

## Suggested Graphs to Add
These are natural “progressive” visuals that show model evolution and MoE health:
- **Training loss vs step** (W&B or stdout logs).
- **Validation loss vs step** and **HellaSwag accuracy vs step**.
- **Tokens/sec vs step** for throughput regression tracking.
- **Expert load balance over time** using `output/expert_stats/layer_loads.jsonl`.
- **Per-layer expert utilization heatmap** (layer x expert, average tokens).
- **Router entropy or top-k utilization** to detect mode collapse.
- **Active vs total parameter utilization** (from `src/helpers.py` prints).

## Notes
- `sft.py` is currently a placeholder for supervised fine-tuning.
- The TE model is the primary implementation; older non-TE versions live in `src/model.py`.

## Citation / Credit
This project is inspired by GPT-2 and DeepSeek-V3 style design choices, adapted for large-scale, high-throughput training.
