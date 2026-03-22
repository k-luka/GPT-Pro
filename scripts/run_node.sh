#!/bin/bash

#SBATCH --job-name=LLM_2_pretraining
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --time=336:00:00 # example 8 hrs
#SBATCH --output=LLM_2_pretrain.log
#SBATCH --constraint=el9
hostname;date;pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

module load conda
module load cuda/12.8.1

conda activate LLM

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_MERGE_LEVEL="${NCCL_NET_MERGE_LEVEL:-LOC}"

EXTRA_ARGS=()
if [[ "${DEBUG_DIST:-0}" == "1" ]]; then
  mkdir -p output/logs
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
  export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL}"
  export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-output/logs/nccl.%h.%p.log}"

  if [[ "${CUDA_LAUNCH_BLOCKING:-0}" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
  fi
fi

if [[ "${DEBUG_ONE_STEP:-0}" == "1" ]]; then
  EXTRA_ARGS+=("training.max_steps=1" "training.grad_accum_steps=1")
fi

torchrun --nproc-per-node=8 pretrain.py "${EXTRA_ARGS[@]}"
