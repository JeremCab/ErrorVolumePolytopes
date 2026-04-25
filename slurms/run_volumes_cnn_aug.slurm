#!/bin/bash
#SBATCH --job-name=volumes_cnn_aug
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-1999         # one task per augmented sample (2000 total)
#SBATCH --time=16:00:00
#SBATCH --output=logs/volumes_cnn_aug_%A_%a.out
#SBATCH --error=logs/volumes_cnn_aug_%A_%a.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

mkdir -p $WORK/ErrorVolumePolytopes/results/volumes_cnn_aug
mkdir -p $WORK/ErrorVolumePolytopes/logs

srun python $WORK/ErrorVolumePolytopes/scripts/run_volumes.py \
    --model_type    cnn \
    --sample_idx    $SLURM_ARRAY_TASK_ID \
    --model_path    $WORK/ErrorVolumePolytopes/checkpoints/fashion_cnn_best.pth \
    --data_path     $WORK/ErrorVolumePolytopes/data/fashionMNIST_augmented_cnn_seed42.pt \
    --n_directions  200 \
    --output_dir    $WORK/ErrorVolumePolytopes/results/volumes_cnn_aug
