#!/bin/bash
#SBATCH --job-name=volumes_v3k_cnn
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-999
#SBATCH --time=16:00:00       # est. ~9.5h (5.5× the 1.73h of old run_volumes_cnn)
#SBATCH --output=logs/volumes_v3k_cnn_%A_%a.out
#SBATCH --error=logs/volumes_v3k_cnn_%A_%a.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

mkdir -p $WORK/ErrorVolumePolytopes/results/volumes_v3k_cnn
mkdir -p $WORK/ErrorVolumePolytopes/logs

srun python $WORK/ErrorVolumePolytopes/scripts/run_volumes_v3k.py \
    --model_type    cnn \
    --sample_idx    $SLURM_ARRAY_TASK_ID \
    --model_path    $WORK/ErrorVolumePolytopes/checkpoints/fashion_cnn_best.pth \
    --data_path     $WORK/ErrorVolumePolytopes/data/fashionMNIST_correct_cnn.pt \
    --n_directions  200 \
    --output_dir    $WORK/ErrorVolumePolytopes/results/volumes_v3k_cnn
