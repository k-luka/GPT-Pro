#!/bin/bash
# Single training run of the 600M dense torchao model with MTP n=2
# (model.mtp_depth=2). Sandwich norm + sliding-window 512 + MXFP8, batch 32 /
# grad_accum 4. Everything else comes from config/config_dense_torchao.yaml.
#
# Run interactively:   bash scripts/run_dense_1b_torchao.sh
# Submit to SLURM:     sbatch scripts/run_dense_1b_torchao.sh   (survives logout)
#
#SBATCH --job-name=dense_600m_n2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --account=pinaki.sarder
#SBATCH --qos=pinaki.sarder
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/dense_600m_n2.log
#SBATCH --constraint=el9

set -euo pipefail

module load conda 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
# Make `conda activate` work in a non-interactive shell (fallback if `module`
# didn't initialize conda).
source /apps/conda/25.3.1/etc/profile.d/conda.sh 2>/dev/null || true
conda activate LLM_torchao_nightly

export NCCL_NVLS_ENABLE=0
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p output/logs

torchrun --nproc-per-node=1 pretrain_dense_torchao.py \
  --config-name=config_dense_torchao \
  experiment.run_name=dense_600m_n2 \
  model.mtp_depth=2 \
  2>&1
