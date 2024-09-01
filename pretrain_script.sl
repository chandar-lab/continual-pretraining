#!/bin/bash
#SBATCH -A bif151
#SBATCH -t 6:00:00
#SBATCH -N 100
#SBATCH --mail-user=istabrak.abbes@mila.quebec
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE


#SBATCH -o /lustre/orion/bif151/scratch/istabrak/gpt-neox/logs/train_%A_%a.out
#SBATCH -e /lustre/orion/bif151/scratch/istabrak/gpt-neox/logs/train_%A_%a.err

module load rocm/5.6.0
module load gcc/12.2.0

source activate base
conda activate my_env
./deepy.py train.py ./configs/local_setup.yml
