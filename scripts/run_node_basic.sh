#!/bin/bash

#SBATCH --job-name=GatorLM2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200
#SBATCH --gpus=4
#SBATCH --time=180:00:00
#SBATCH --output=output/logs/GatorLM2.log
#SBATCH --constraint=el9

module load conda
module load cuda/12.8.1
conda activate LLM

export NCCL_NVLS_ENABLE=0
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1

torchrun --nproc-per-node=4 pretrain_basic.py 2>&1
