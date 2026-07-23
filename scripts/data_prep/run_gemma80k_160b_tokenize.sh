#!/bin/bash
#SBATCH --job-name=gemma80k_160b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48gb
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --partition=hpg-default
#SBATCH --time=24:00:00
#SBATCH --output=gemma80k_160b_%j.log

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
DATA_ROOT="$PROJECT_ROOT/data/climbmix_gemma4_80k"
TOKENIZER="$PROJECT_ROOT/tokenizers/gemma4_80k/tokenizer.json"
TARGET_SHARDS=1600

cd "$PROJECT_ROOT"

module load conda 2>/dev/null || true
# shellcheck source=/dev/null
source /apps/conda/25.3.1/etc/profile.d/conda.sh 2>/dev/null || true
conda activate LLM

export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

CURRENT_SHARDS=$(find "$DATA_ROOT" -maxdepth 1 -type f -name '*.bin' | wc -l)
if (( CURRENT_SHARDS >= TARGET_SHARDS )); then
    echo "Target already met: $CURRENT_SHARDS/$TARGET_SHARDS shards"
    exit 0
fi
REMAINING_SHARDS=$((TARGET_SHARDS - CURRENT_SHARDS))

echo "Starting Gemma-80K ClimbMix tokenization"
echo "Slurm job       : ${SLURM_JOB_ID:-unknown}"
echo "Current shards  : $CURRENT_SHARDS"
echo "Target shards   : $TARGET_SHARDS"
echo "Remaining       : $REMAINING_SHARDS"
echo "Workers / memory: ${SLURM_CPUS_PER_TASK:-32} CPUs / 48 GiB"

exec python scripts/data_prep/prepare_climbmix.py \
    --tokenizer "$TOKENIZER" \
    --output_dir "$DATA_ROOT" \
    --block_size 4096 \
    --shard_size 100000000 \
    --num_workers "${SLURM_CPUS_PER_TASK:-32}" \
    --max_shards "$REMAINING_SHARDS" \
    --resume
