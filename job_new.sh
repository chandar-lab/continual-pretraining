#!/bin/bash
#!/bin/bash
#SBATCH -A bif151
#SBATCH -t 6:00:00
#SBATCH --nodes=128
#SBATCH -p batch
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=istabrak.abbes@mila.quebec
#SBATCH --mail-type=all
#SBATCH --output=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/logs/ex_ha9i9i/train_%A.out
#SBATCH --error=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/logs/ex_ha9i9i/train_%A.err

srun pkill python

source /lustre/orion/bif151/scratch/istabrak/latest_env/setup.sh

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`


bash /lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/hostfiles/hosts_$SLURM_JOBID
cd /lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox
rm -rf ./megatron/fused_kernels/build/lock
rm -rf /autofs/nccs-svm1_home2/istabrak/.cache/torch_extensions/py312_cpu/fused_adam/lock


python ./deepy.py train_new.py ./configs/new_local_setup_from_ex.yml