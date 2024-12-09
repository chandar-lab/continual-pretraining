#!/bin/bash
#SBATCH -A bif151
#SBATCH -t 2:00:00
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=istabrak.abbes@mila.quebec
#SBATCH --mail-type=all
#SBATCH --output=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/logs/ex/train_%A.out
#SBATCH --error=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/logs/ex/train_%A.err

srun pkill python

# Setup environment
source /lustre/orion/bif151/scratch/istabrak/latest_env/setup.sh

# Prepare SLURM environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# NCCL settings
export NCCL_TIMEOUT_MS=172800000
export NCCL_TIMEOUT=172800
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3

# Run the job
bash /lustre/orion/bif151/scratch/istabrak/new/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/hostfiles/hosts_$SLURM_JOBID
cd /lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox

# Clean up old lock files
rm -rf ./megatron/fused_kernels/build/lock
rm -rf /autofs/nccs-svm1_home2/istabrak/.cache/torch_extensions/py312_cpu/fused_adam/lock

# Call the training script with the passed prefix (parameter from the chain)
python ./deepy.py train_new.py ./configs/1B_no_buffer.yml
