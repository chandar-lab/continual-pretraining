#!/bin/bash
#SBATCH -A bif151
#SBATCH -t 24:00:00
#SBATCH --nodes=64
#SBATCH --gpus-per-node=8
#SBATCH -p extended
#SBATCH --mail-user=istabrak.abbes@mila.quebec
#SBATCH --mail-type=all
#SBATCH -o /lustre/orion/bif151/scratch/istabrak/ben/continual_neox/gpt-neox/logs/train_%A_%a.out
#SBATCH -e /lustre/orion/bif151/scratch/istabrak/ben/continual_neox/gpt-neox/logs/train_%A_%a.err
source /lustre/orion/bif151/scratch/istabrak/ben/setup.sh

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`



bash /lustre/orion/bif151/scratch/istabrak/ben/continual_neox/gpt-neox/write_hostfiles.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/istabrak/ben/continual_neox/gpt-neox/hostfiles/hosts_$SLURM_JOBID
cd /lustre/orion/bif151/scratch/istabrak/ben/continual_neox/gpt-neox
rm -rf ./megatron/fused_kernels/build/lock
rm -rf svm1_home2/istabrak/.cache/torch_extensions/py312_cpu/fused_adam/lock

python ./deepy.py train.py ./configs/slurm_125M.yml
