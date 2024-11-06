#!/bin/sh

#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:00:00 
#SBATCH --job-name=user_task
#SBATCH --error=jobs.%J.err 
#SBATCH --output=jobs.%J.out 
#SBATCH --reservation=nitk_res 
#SBATCH --partition=cpu
cd /home/nitk211it067/Major\ Project

python user_embedding.py

