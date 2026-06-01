#!/bin/bash
# MAIN BIG TRAINING RUN — 6.5B dense (config_dense_6p5b.yaml) on 8x B200, 14 days.
# Auto-resumes from the latest checkpoint on SLURM requeue, so the run survives
# preemption/restarts without a loss spike (DCP resume verified).
#
#   sbatch scripts/run_dense_torchao.sh
#
#SBATCH --job-name=dense_6p5b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512gb
#SBATCH --account=pinaki.sarder
#SBATCH --qos=pinaki.sarder
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --time=14-00:00:00
#SBATCH --output=output/logs/dense_6p5b_%j.log
#SBATCH --constraint=el9
#SBATCH --requeue

set -euo pipefail

module load conda 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
source /apps/conda/25.3.1/etc/profile.d/conda.sh 2>/dev/null || true
conda activate LLM_torchao_nightly

export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p output/logs

RUN=dense_6p5b
CKPT_DIR="output/checkpoints/${RUN}"

# Auto-resume: pick the highest-numbered step_* checkpoint if one exists.
RESUME=""
if compgen -G "${CKPT_DIR}/step_*" > /dev/null; then
  LATEST=$(ls -d ${CKPT_DIR}/step_* | sed 's/.*step_//' | sort -n | tail -1)
  RESUME="+resume_checkpoint=${CKPT_DIR}/step_${LATEST}"
  echo "---| Auto-resuming from ${CKPT_DIR}/step_${LATEST} |---"
else
  echo "---| Fresh start (no checkpoint found in ${CKPT_DIR}) |---"
fi

torchrun --nproc-per-node=8 pretrain_dense_torchao.py \
  --config-name=config_dense_6p5b \
  experiment.run_name=${RUN} \
  ${RESUME} \
  2>&1
