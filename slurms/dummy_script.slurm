#!/bin/bash
#SBATCH --job-name=DUMMY
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=??
#SBATCH --time=12:00:00
#SBATCH --output=dummy_script.out
#SBATCH --error=dummy_script.err

# TASK TO DEFINE

export TMPDIR=$JOBSCRATCH

# OTHER EXPORTS HERE

srun #run script
