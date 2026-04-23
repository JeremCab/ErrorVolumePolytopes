#!/bin/bash
#SBATCH --job-name=vol_aug_cnn_qincorrect
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-99          # 100 augmented points (10 per original sample x 10 samples)
#SBATCH --time=16:00:00
#SBATCH --output=logs/data_aug/vol_aug_cnn_qincorrect_%A_%a.out
#SBATCH --error=logs/data_aug/vol_aug_cnn_qincorrect_%A_%a.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

mkdir -p $WORK/ErrorVolumePolytopes/results/volumes_aug_cnn_qincorrect
mkdir -p $WORK/ErrorVolumePolytopes/logs/data_aug

srun python $WORK/ErrorVolumePolytopes/scripts/run_volumes.py \
    --model_type    cnn \
    --sample_idx    $SLURM_ARRAY_TASK_ID \
    --model_path    $WORK/ErrorVolumePolytopes/checkpoints/fashion_cnn_best.pth \
    --data_path     $WORK/ErrorVolumePolytopes/data/fashionMNIST_augmented_cnn_seed42_walk_qincorrect.pt \
    --n_directions  200 \
    --output_dir    $WORK/ErrorVolumePolytopes/results/volumes_aug_cnn_qincorrect
