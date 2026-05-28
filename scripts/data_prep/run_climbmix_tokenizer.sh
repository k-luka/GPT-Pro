#!/bin/bash
#SBATCH --job-name=tokenize_climbmix_64k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --account=pinaki.sarder
#SBATCH --qos=pinaki.sarder
#SBATCH --partition=hpg-milan
#SBATCH --time=48:00:00
#SBATCH --output=/blue/pinaki.sarder/kirill.luka/learning/GPT-Pro/scripts/data_prep/tokenize_climbmix_64k.log

# Activate your environment
module load conda
conda activate LLM

# Navigate to the project directory
cd /blue/pinaki.sarder/kirill.luka/learning/GPT-Pro

# Re-tokenize ClimbMix-400B with the new 64k BPE tokenizer.
# Output goes to a new directory so the existing 32k shards in
# data/climbmix_400b/ remain usable for ongoing experimentation.
python scripts/data_prep/prepare_climbmix.py \
  --block_size 4096 \
  --num_workers 16 \
  --tokenizer data/tokenizer/tokenizer.json \
  --output_dir data/climbmix_400b_64k \
  --resume
