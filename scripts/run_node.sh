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

torchrun --nproc-per-node=8 pretrain.py