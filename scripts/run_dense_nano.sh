#!/bin/bash

#SBATCH --job-name=GatorLM2_nano
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --output=output/logs/GatorLM2_nano.log
#SBATCH --constraint=el9

module load conda
module load cuda/12.8.1
conda activate LLM

export NCCL_NVLS_ENABLE=0
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1

torchrun --nproc-per-node=4 pretrain_dense.py --config-name=config_dense_nano 2>&1
