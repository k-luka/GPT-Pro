#!/bin/bash
#SBATCH --job-name=train_tokenizer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-milan
#SBATCH --time=02:00:00
#SBATCH --output=output/logs/train_tokenizer.log
#SBATCH --constraint=el9

module load conda
conda activate LLM

cd /blue/pinaki.sarder/kirill.luka/learning/GPT-Pro

python scripts/data_prep/train_tokenizer.py \
    --vocab_size 32000 \
    --sample_docs 1000000 \
    --output_dir data/tokenizer
