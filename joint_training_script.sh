#!/bin/bash
#SBATCH -A bif151
#SBATCH -t 24:00:00
#SBATCH --nodes=32
#SBATCH -p extended
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nizar.isl17@gmail.com
#SBATCH --mail-type=all
#SBATCH --output=/lustre/orion/bif151/scratch/nizar17/gpt-neox/logs/train_%A.out
#SBATCH --error=/lustre/orion/bif151/scratch/nizar17/gpt-neox/logs/train_%A.err
# srun pkill python

ROCM_VERSION=6.0.0
module reset
module load rocm/$ROCM_VERSION
module load ums/default  ums012/default
module load gcc/12.2.0

export ROCM_HOME=/opt/rocm-6.0.0
export ROCM_ROOT=/opt/rocm-6.0.0
export ROCM_PATH=/opt/rocm-6.0.0
###


#export PATH=$CONDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CONDA_HOME/lib:$LD_LIBRARY_PATH
#export CPATH=$CONDA_HOME/include:$CPATH
#export LD_LIBRARY_PATH=/lib64:/usr/lib64:$LD_LIBRARY_PATH
#source $CONDA_HOME/bin/activate
# conda activate neox
###
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=0 # INFO
export NCCL_DEBUG_SUBSYS=ALL

conda activate neox

bash /lustre/orion/bif151/scratch/nizar17/gpt-neox/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/nizar17/gpt-neox/hostfiles/hosts_$SLURM_JOBID
# cd /lustre/orion/bif151/scratch/nizar17/gpt-neox
# rm -rf ./megatron/fused_kernels/build/lock
# rm -rf /autofs/nccs-svm1_home2/nizar17/.cache/torch_extensions/py312_cpu/fused_adam/lock

python ./deepy.py train.py ./configs/spectra_560M_joint.yml
