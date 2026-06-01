#!/bin/bash
# LR sweep for the 6.5B dense run — pick max_lr before the 14-day launch.
#
# KEY IDEA: the optimal LR depends on the model and the EFFECTIVE BATCH (tokens/
# step), NOT the number of GPUs. DDP just data-parallelizes; the optimization is
# identical on 4 or 8 GPUs as long as tokens/step matches. The real 8-GPU run is
# batch 4 x grad_accum 16 x 8 = 512 seqs = ~2.1M tokens/step. Here we reproduce
# that EXACT effective batch on 1 GPU via grad_accum 128 (4 x 128 x 1 = 512), so
# the winning LR transfers directly to the 8-GPU run.
#
# We sweep 4 LRs in PARALLEL, one per GPU (real 6.5B model, ~314M tokens each),
# and compare val loss. ~6h wall on 4 B200s.
#
#   nohup bash scripts/lr_sweep_6p5b.sh > output/logs/lrsweep_driver.log 2>&1 &
#
set -uo pipefail

CONFIG="config_dense_6p5b"
PROJECT="${PROJECT:-lr_sweep_6p5b}"
# 450 steps x 2.1M tok = ~944M tokens per LR. On 1 GPU (~14.7k tok/s) that's
# ~18h wall in parallel -> a ~3pm launch finishes ~9am. More tokens = a more
# reliable LR ranking than a quick few-hundred-M proxy.
STEPS="${STEPS:-450}"
GA="${GA:-128}"                  # 4 x 128 x 1 GPU = 512 seqs = the 8-GPU eff. batch

# max_lr ladder (2x spacing around the 3e-4 default). min_lr = max_lr/10.
LRS=(1.5e-4 3e-4 6e-4 1.2e-3)

module load conda 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
source /apps/conda/25.3.1/etc/profile.d/conda.sh 2>/dev/null || true
conda activate LLM_torchao_nightly

export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
mkdir -p output/logs

declare -A PID NAME
gpu=0
for lr in "${LRS[@]}"; do
  minlr=$(awk "BEGIN{printf \"%.6g\", $lr/10}")
  name="lr_${lr}"; NAME[$lr]="$name"
  echo "=== launch ${name}: max_lr=${lr} min_lr=${minlr} on GPU ${gpu} (eff.batch=512 seq) ==="
  CUDA_VISIBLE_DEVICES="$gpu" torchrun --nproc-per-node=1 --master_port=$((29600+gpu)) \
    pretrain_dense_torchao.py --config-name="$CONFIG" \
    experiment.project="$PROJECT" experiment.run_name="$name" \
    training.batch_size=4 training.grad_accum_steps="$GA" \
    training.max_steps="$STEPS" training.warmup_steps=30 \
    training.max_lr="$lr" training.min_lr="$minlr" \
    training.eval_interval=50 training.eval_steps=20 \
    training.checkpoint_interval=100000 training.eval_hellaswag=false training.eval_core=false \
    training.logging_steps=1 \
    > "output/logs/${name}.log" 2>&1 &
  PID[$lr]=$!
  gpu=$((gpu+1))
done

echo "Launched ${#LRS[@]} LR runs in parallel; waiting..."
for lr in "${LRS[@]}"; do
  if wait "${PID[$lr]}"; then echo "  ${NAME[$lr]} -> done"; else echo "  ${NAME[$lr]} -> FAILED/diverged (see log)"; fi
done

# ---- compare final val loss -------------------------------------------------
echo
echo "===================== LR SWEEP RESULT (6.5B) ========================="
printf "%-10s %-12s %-10s\n" "max_lr" "final_val" "note"
best=""; bestloss=""
for lr in "${LRS[@]}"; do
  log="output/logs/${NAME[$lr]}.log"
  # last "val loss: X" in the log
  vl=$(grep -oE "val loss: [0-9.]+" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  note=""
  grep -qiE "nan|inf|diverg|out of memory" "$log" 2>/dev/null && note="UNSTABLE"
  [ -z "$vl" ] && { vl="NA"; [ -z "$note" ] && note="no val loss"; }
  printf "%-10s %-12s %-10s\n" "$lr" "${vl:-NA}" "$note"
  if [ "$vl" != "NA" ] && [ -z "$note" ]; then
    if [ -z "$bestloss" ] || awk "BEGIN{exit !($vl < $bestloss)}"; then bestloss="$vl"; best="$lr"; fi
  fi
done
echo "-----------------------------------------------------------------------"
echo "Lowest val loss: max_lr=${best:-?} (val=${bestloss:-?})  <- set this in config_dense_6p5b.yaml"
echo "Compare full curves in W&B project: ${PROJECT}"
echo "Per-run logs: output/logs/lr_<value>.log"
