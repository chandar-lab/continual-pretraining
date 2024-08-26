#!/bin/bash
#BSUB -nnodes 46
#BSUB -W 6:00
#BSUB -q batch
#BSUB -o /lustre/orion/bif151/scratch/istabrak/gpt-neox/training_logs/gpt_neox_out.%J
#BSUB -e /lustre/orion/bif151/scratch/istabrak/gpt-neox/training_logs/gpt_neox_err.%J
#BSUB -alloc_flags gpudefault
#BSUB -P bif151
#BSUB -N istabrak.abbes@mila.quebec
#BSUB -B istabrak.abbes@mila.quebec


module purge
module load rocm/5.6.0
module load gcc/12.2.0
module load cmake
source $HOME/my_envs/mpi4py_env/bin/activate


# Move to the gpt-neox install
export TRAIN_PATH=/lustre/orion/bif151/scratch/istabrak/gpt-neox
cd $TRAIN_PATH



python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py ./configs/125M.yml ./configs/local_setup.yml --wandb_group="istabrak.abbes"
