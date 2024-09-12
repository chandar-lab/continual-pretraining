#!/bin/bash
#SBATCH -A bif151
#SBATCH -t 2:00:00
#SBATCH -N 4
#SBATCH --mail-user=gopeshhraaj@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -o /lustre/orion/bif151/scratch/gopeshh/gpt-neox/logs/train_%A_%a.out
#SBATCH -e /lustre/orion/bif151/scratch/gopeshh/gpt-neox/logs/train_%A_%a.err




module load rocm/6.0.0
module load gcc/12.2.0

cd "/lustre/orion/bif151/scratch/gopeshh/gpt-neox/"


source activate base
conda activate neox

export TORCH_EXTENSIONS_DIR=/lustre/orion/bif151/scratch/gopeshh/latest_install/cache
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Write the hostfile for this job
bash /lustre/orion/bif151/scratch/gopeshh/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/gopeshh/hostfiles/hosts_$SLURM_JOBID


./deepy.py train.py ./configs/local_setup.yml




