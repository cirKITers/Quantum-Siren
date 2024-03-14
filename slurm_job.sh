#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=quantum-siren
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=8
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=30:00:00
# 
# partition the job will run on
#SBATCH --partition single
# 
# expected memory requirements
#SBATCH --mem=28000MB
#
# infos
#
# output path
#SBATCH --output="logs/slurm/slurm-%j-%x.out"

module load devel/python/3.10.5_intel_2021.4.0
~/quantum-siren/.venv/bin/python -m kedro run --params=$1 --pipeline $2

# Done
exit 0


