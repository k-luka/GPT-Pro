#!/bin/bash
# MEMORY PROBE — diagnose the slow host-RAM climb that OOM-killed the 6.5B run at
# step ~13.8k. Single GPU, small (shallow) model, but the SAME data path, prefetch,
# torchao MXFP8 stack, eval, and per-step loop as the real run. The trainer logs
# host RSS (sum/peak/max-rank) to W&B every logging_steps — watch "host RSS sum
# (GiB)" vs step in the "mem_probe" run:
#   - flat after warmup        -> no per-process leak (200 GB was just a tight cap)
#   - linear climb             -> real single-process leak; slope gives steps-to-OOM
# Caveat: world_size=1, so NCCL/collective all-reduces are no-ops. If this probe
# stays flat but the 8-GPU run still climbs, the leak is in the multi-rank path.
#
#   sbatch scripts/run_mem_probe.sh
#
#SBATCH --job-name=mem_probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --account=pinaki.sarder
#SBATCH --qos=pinaki.sarder
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=output/logs/mem_probe_%j.log
#SBATCH --constraint=el9

set -euo pipefail

module load conda 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
source /apps/conda/25.3.1/etc/profile.d/conda.sh 2>/dev/null || true
conda activate LLM_torchao_nightly

export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p output/logs

# Shallow model (fast steps) but identical batch_size/block_size/grad_accum churn
# and data path. No checkpoints (huge interval). Evals kept on so we also see
# whether RSS jumps at eval (the variable-shape recompile hypothesis). Long step
# budget so a small slope still shows; the 12h wall or a 64 GB OOM ends it.
torchrun --nproc-per-node=1 pretrain_dense_torchao.py \
  --config-name=config_dense_6p5b \
  experiment.run_name=mem_probe \
  model.n_layers=6 \
  training.grad_accum_steps=8 \
  training.max_steps=60000 \
  training.warmup_steps=200 \
  training.logging_steps=20 \
  training.eval_interval=2000 \
  training.eval_steps=20 \
  training.checkpoint_interval=100000000 \
  2>&1
