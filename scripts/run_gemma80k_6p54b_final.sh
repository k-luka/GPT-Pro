#!/bin/bash
# 6.54B dense pretraining on one 8xB200 node.
# Submit from the repository root:
#   sbatch scripts/run_gemma80k_6p54b_final.sh

#SBATCH --job-name=gptpro_6p54b_159b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300gb
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --time=14-00:00:00
#SBATCH --constraint=el9
#SBATCH --requeue
#SBATCH --output=gptpro_6p54b_159b_%j.log

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
RUN=gptpro_6p54b_gemma80k_159b
CONFIG=config_dense_6p5b_gemma4_80k
FINAL_STEP=38000
CKPT_DIR="$PROJECT_ROOT/output/checkpoints/$RUN"

cd "$PROJECT_ROOT"
mkdir -p output/logs "$CKPT_DIR"

module load conda 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
# shellcheck source=/dev/null
source /apps/conda/25.3.1/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torchao

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN_ID="$RUN"
export WANDB_RESUME=allow

unset SANDWICH_NORM QK_NORM_MODE ATTN_IMPL GLOBAL_ATTN_IMPL GLOBAL_PLACEMENT || true

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [[ "$GPU_COUNT" -ne 8 ]]; then
    echo "Expected 8 visible B200 GPUs, found $GPU_COUNT" >&2
    exit 1
fi

python scripts/preflight_gemma80k_final.py --require-full-validation

if [[ -f "$CKPT_DIR/step_${FINAL_STEP}/.metadata" ]]; then
    echo "Final checkpoint already exists: $CKPT_DIR/step_${FINAL_STEP}"
    exit 0
fi

# Resume only checkpoints whose DCP metadata commit exists.
LATEST=""
LATEST_STEP=-1
shopt -s nullglob
for METADATA in "$CKPT_DIR"/step_*/.metadata; do
    STEP_DIR=${METADATA%/.metadata}
    STEP=${STEP_DIR##*/step_}
    if [[ "$STEP" =~ ^[0-9]+$ ]] && (( STEP > LATEST_STEP )); then
        LATEST_STEP=$STEP
        LATEST=$STEP_DIR
    fi
done
shopt -u nullglob

RESUME_ARGS=()
if [[ -n "$LATEST" ]]; then
    echo "---| Auto-resuming from $LATEST |---"
    RESUME_ARGS=("+resume_checkpoint=$LATEST")
fi

exec torchrun \
    --standalone \
    --nproc-per-node=8 \
    pretrain_dense_torchao.py \
    --config-name="$CONFIG" \
    experiment.run_name="$RUN" \
    "${RESUME_ARGS[@]}"
