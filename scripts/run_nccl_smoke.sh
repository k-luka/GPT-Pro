#!/bin/bash

#SBATCH --job-name=nccl_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --time=00:30:00
#SBATCH --output=output/logs/nccl_smoke.log
#SBATCH --constraint=el9
hostname; date; pwd
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

module load conda
module load cuda/12.8.1

conda activate LLM

mkdir -p output/logs

export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ENABLE_TIMING="${TORCH_NCCL_ENABLE_TIMING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-120}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL}"
export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-output/logs/nccl_smoke.%h.%p.log}"

# Single-node defaults.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_MERGE_LEVEL="${NCCL_NET_MERGE_LEVEL:-LOC}"

torchrun --nproc-per-node=8 scripts/nccl_smoke.py
