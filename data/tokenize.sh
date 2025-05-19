#!/bin/bash
#SBATCH --job-name=tokenize_job
#SBATCH --output=logs/tokenize_%A_%a.out
#SBATCH --error=logs/tokenize_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --array=0-8   # Adjust this to number_of_nodes - 1

source /lustre/orion/bif151/scratch/istabrak/env_rebuttal/setup.sh

INPUT_DIR=/lustre/orion/bif151/proj-shared/arabic/cluster_jsonl
OUTPUT_DIR=/lustre/orion/bif151/proj-shared/arabic/arabic_tokenized

# Run your Python tokenizer script with job array index and count
srun python /lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/data/3_tokenize.py \
    --input-path $INPUT_DIR \
    --output-path $OUTPUT_DIR \
    --merge --start 0 --end 20
    # add other args as needed
