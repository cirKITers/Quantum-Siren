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
#SBATCH --time=20:00:00
# 
# partition the job will run on
#SBATCH --partition single
# 
# expected memory requirements
#SBATCH --mem=16000MB
#
# output path
#SBATCH --output="logs/slurm/slurm-%j.out"

/home/kit/scc/lc3267/quantum-siren/.venv/bin/python -m kedro run --params=$1

# Done
exit 0


