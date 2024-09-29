#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:30:00
#SBATCH -C gpu
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH -q regular
#SBATCH -J Mg-0.5-1-d
#SBATCH --output=%x.out
#SBATCH -A m4444

module load python
srun python train.py 6 Mg 0
srun python generate.py 6 Mg 1