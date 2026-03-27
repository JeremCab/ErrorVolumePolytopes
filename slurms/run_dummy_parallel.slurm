#!/bin/bash
#SBATCH --job-name=dummy_parallel
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/dummy_parallel_%j.out
#SBATCH --error=logs/dummy_parallel_%j.err

export TMPDIR=$JOBSCRATCH

# Activate project virtual environment
source $WORK/ErrorVolumePolytopes/.venv/bin/activate

srun python $WORK/ErrorVolumePolytopes/scripts/dummy_parallel.py \
    --data_path $WORK/ErrorVolumePolytopes/data/fashionMNIST_correct_mlp.pt \
    --output_dir $WORK/ErrorVolumePolytopes/results
