#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 15:00:00
#SBATCH -C gpu
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH -q regular
#SBATCH -J Ca
#SBATCH --output=%x.out
#SBATCH -A m4444

module load python
srun python train.py Ca 9 2 0
srun python gen.py Ca 9 2 0
