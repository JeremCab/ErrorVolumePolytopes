#!/bin/bash
#SBATCH --job-name=volumes
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-999          # first batch (samples 0–999); resubmit with 1000-1999 for the second
#SBATCH --time=00:30:00
#SBATCH --output=logs/volumes_%A_%a.out
#SBATCH --error=logs/volumes_%A_%a.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

mkdir -p $WORK/ErrorVolumePolytopes/results/volumes
mkdir -p $WORK/ErrorVolumePolytopes/logs

srun python $WORK/ErrorVolumePolytopes/scripts/run_volumes.py \
    --sample_idx    $SLURM_ARRAY_TASK_ID \
    --model_path    $WORK/ErrorVolumePolytopes/checkpoints/fashion_mlp_best.pth \
    --data_path     $WORK/ErrorVolumePolytopes/data/fashionMNIST_correct_mlp.pt \
    --n_directions  200 \
    --output_dir    $WORK/ErrorVolumePolytopes/results/volumes
