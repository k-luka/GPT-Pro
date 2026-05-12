#!/bin/bash
#SBATCH --job-name=tokenize_climbmix
#SBATCH --output=/home/kirill.luka/weishao/kirill.luka/learning/GPT-Pro/scripts/data_prep/tokenize_climbmix.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

# Activate your environment
conda activate LLM

# Navigate to the project directory
cd /home/kirill.luka/weishao/kirill.luka/learning/GPT-Pro

# Run the script
python scripts/data_prep/prepare_climbmix.py --block_size 4096 --num_workers 8