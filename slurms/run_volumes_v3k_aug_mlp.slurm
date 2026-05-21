#!/bin/bash
# =============================================================================
# run_volumes_v3k_aug_mlp.slurm
#
# Computes V3k volumes (P1/P2/P3(k) mean widths, all bit-widths) for each
# point in the MCMC-augmented MLP dataset.  One SLURM task per augmented point.
#
# HOW TO ADAPT FOR YOUR CLUSTER
# ------------------------------
# 1. --account      Replace "aik@cpu" with your own allocation account name.
# 2. module load    Replace "pytorch-gpu/py3/2.8.0" with the PyTorch module
#                   available on your cluster (needs torch, numpy, scipy).
#                   If no module system is used, remove the line and activate
#                   your conda/virtualenv instead (e.g. "source ~/venv/bin/activate").
# 3. $WORK          Replace every "$WORK/ErrorVolumePolytopes" with the absolute
#                   path to the cloned repo on your cluster
#                   (e.g. /home/youruser/ErrorVolumePolytopes).
# 4. $JOBSCRATCH    Replace "$JOBSCRATCH" with your cluster's per-job temp dir
#                   (e.g. $TMPDIR, $SCRATCH, or just /tmp).  If unsure, remove
#                   the "export TMPDIR=..." line entirely.
# 5. --cpus-per-task  40 CPUs is optimal for parallelising the LP solves.
#                   Reduce if your cluster has fewer cores per node, but keep
#                   at least 8 for reasonable performance.
# =============================================================================

#SBATCH --job-name=v3k_aug_mlp
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu          # <-- ADAPT: your allocation account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-8234             # 8235 augmented points (fashionMNIST_augmented_mlp_seed42_walk_aug200.pt)
#SBATCH --time=04:00:00
#SBATCH --output=logs/v3k_aug_mlp_%A_%a.out
#SBATCH --error=logs/v3k_aug_mlp_%A_%a.err

module load pytorch-gpu/py3/2.8.0  # <-- ADAPT: your cluster's PyTorch module

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH           # <-- ADAPT: your cluster's per-job temp dir

mkdir -p $WORK/ErrorVolumePolytopes/results/volumes_v3k_mlp_aug  # <-- ADAPT: $WORK
mkdir -p $WORK/ErrorVolumePolytopes/logs                          # <-- ADAPT: $WORK

srun python $WORK/ErrorVolumePolytopes/scripts/run_volumes_v3k.py \  # <-- ADAPT: $WORK
    --model_type    mlp \
    --sample_idx    $SLURM_ARRAY_TASK_ID \
    --model_path    $WORK/ErrorVolumePolytopes/checkpoints/fashion_mlp_best.pth \  # <-- ADAPT: $WORK
    --data_path     $WORK/ErrorVolumePolytopes/data/fashionMNIST_augmented_mlp_seed42_walk_aug200.pt \  # <-- ADAPT: $WORK
    --n_directions  200 \
    --output_dir    $WORK/ErrorVolumePolytopes/results/volumes_v3k_mlp_aug  # <-- ADAPT: $WORK
