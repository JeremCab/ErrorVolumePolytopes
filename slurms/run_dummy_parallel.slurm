#!/bin/bash
#SBATCH --job-name=dummy_parallel
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/dummy_parallel_%j.out
#SBATCH --error=logs/dummy_parallel_%j.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH
export DATAPATH=/lustre/fswork/projects/rech/aik/uhr51mx/ErrorVolumePolytopes/data/fashionMNIST_correct_mlp.pt


srun python $WORK/ErrorVolumePolytopes/scripts/dummy_parallel.py \
    --data_path $DATAPATH \
    --output_dir $SCRATCH/dummy_results