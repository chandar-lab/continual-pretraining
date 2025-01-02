#!/bin/bash
#SBATCH -A bif151
#SBATCH -t 2:00:00
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=istabrak.abbes@mila.quebec
#SBATCH --output=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/logs/50_replay/train_%A.out
#SBATCH --error=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/logs/50_replay/train_%A.err
#SBATCH --signal=SIGUSR1@600  # Send SIGUSR1 signal 10 minutes before time limit

# Function to handle SIGUSR1 signal
function handle_signal {
    echo "SIGUSR1 received. Requeuing job."
    
    # Requeue the job
    scontrol requeue $SLURM_JOBID
    exit 0
}

# Trap SIGUSR1 signal
trap 'handle_signal' SIGUSR1

# Job commands
srun pkill python

source /lustre/orion/bif151/scratch/istabrak/latest_env/setup.sh

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

bash /lustre/orion/bif151/scratch/istabrak/new/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/hostfiles/hosts_$SLURM_JOBID
cd /lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox
rm -rf ./megatron/fused_kernels/build/lock
rm -rf /autofs/nccs-svm1_home2/istabrak/.cache/torch_extensions/py312_cpu/fused_adam/lock

# Start the training script
python ./deepy.py train_new_replay.py ./configs/replay_50_410.yml
