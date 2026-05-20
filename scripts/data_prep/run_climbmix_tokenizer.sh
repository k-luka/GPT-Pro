#!/bin/bash
#SBATCH --job-name=tokenize_climbmix_bos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --account=pinaki.sarder
#SBATCH --qos=pinaki.sarder
#SBATCH --partition=hpg-milan
#SBATCH --time=48:00:00
#SBATCH --output=/blue/pinaki.sarder/kirill.luka/learning/GPT-Pro/scripts/data_prep/tokenize_climbmix_bos.log

# Activate your environment
module load conda
conda activate LLM

# Navigate to the project directory
cd /blue/pinaki.sarder/kirill.luka/learning/GPT-Pro

# Run the script
python scripts/data_prep/prepare_climbmix.py \
  --block_size 4096 \
  --num_workers 16 \
  --output_dir data/climbmix_400b
