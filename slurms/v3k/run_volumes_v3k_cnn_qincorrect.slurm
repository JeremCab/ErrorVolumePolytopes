#!/bin/bash
#SBATCH --job-name=v3k_cnn_qi
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-9          # 10 q-incorrect CNN samples
#SBATCH --time=04:00:00
#SBATCH --output=logs/v3k/v3k_cnn_qincorrect_%A_%a.out
#SBATCH --error=logs/v3k/v3k_cnn_qincorrect_%A_%a.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

mkdir -p $WORK/ErrorVolumePolytopes/results/volumes_v3k_cnn_qincorrect
mkdir -p $WORK/ErrorVolumePolytopes/logs/v3k

srun python $WORK/ErrorVolumePolytopes/scripts/run_volumes_v3k.py \
    --model_type    cnn \
    --sample_idx    $SLURM_ARRAY_TASK_ID \
    --model_path    $WORK/ErrorVolumePolytopes/checkpoints/fashion_cnn_best.pth \
    --data_path     $WORK/ErrorVolumePolytopes/data/qmodel_qincorrect_cnn_b4.pt \
    --n_directions  200 \
    --output_dir    $WORK/ErrorVolumePolytopes/results/volumes_v3k_cnn_qincorrect
