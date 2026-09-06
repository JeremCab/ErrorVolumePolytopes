#!/bin/bash
# =============================================================================
# run_rho_p2_gen150_cnn.slurm
#
# rho(P2) for EVERY tile of the campaign — one LP each.
#
# Purpose: decide which tiles are degenerate. A tile whose correct polytope P2
# is flat has zero volume, so it carries zero weight in the volume-based
# generalized accuracy — yet a full one in the mean-width surrogate, because a
# flat polytope still has a mean width of order 1. Including such tiles is a
# defect of the surrogate, not a property of the quantity being estimated.
#
# This is not hypothetical: at b=4 the MEDIAN rho(P2) over the stage-2 sample is
# exactly 0, so about half the b=4 tiles are degenerate and currently weigh in.
#
# It is also the measurement Jiri asked for when he suggested selecting
# augmentation points by relevance rather than keeping all of them.
#
# One LP per tile instead of the ~3 of stage 2, which is what makes it
# affordable over all 1423 tiles. It reuses the SAME task table as stage 1, so
# the array bounds are the ones make_volume_tasks.py printed:
#
#     sbatch --array=0-1422 slurms/run_rho_p2_gen150_cnn.slurm
#
# CAUTION when using the result: rho(P2) grows strongly with b (0 at b=4,
# 1.2e-4 at b=6, 3.0e-4 at b=10, 1.3e-3 at b=16 on the stage-2 sample), so a
# FIXED absolute cut keeps nearly everything at b=16 and nearly nothing at b=4 —
# a b-dependent selection that would manufacture a trend rather than measure one.
# Report the retained fraction per b, and prefer a relative criterion.
# =============================================================================

#SBATCH --job-name=rho_p2_gen150
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/rho_p2_gen150_%A_%a.out
#SBATCH --error=logs/rho_p2_gen150_%A_%a.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

ROOT=$WORK/ErrorVolumePolytopes
TASK_FILE=$ROOT/data/gen150/volume_tasks.txt

if [ ! -f "$TASK_FILE" ]; then
    echo "[ABORT] $TASK_FILE not found — run scripts/make_volume_tasks.py first." >&2
    exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
if [ -z "$LINE" ]; then
    echo "[ABORT] task $SLURM_ARRAY_TASK_ID beyond the table." >&2
    exit 1
fi
BITS=$(echo    "$LINE" | cut -d' ' -f1)
AUG_IDX=$(echo "$LINE" | cut -d' ' -f2)
BITS_PAD=$(printf "%02d" "$BITS")

OUT_DIR=$ROOT/results/rho_p2_cnn_gen150/b${BITS_PAD}
mkdir -p "$OUT_DIR" "$ROOT/logs"

echo "[INFO] task=$SLURM_ARRAY_TASK_ID  bits=$BITS  aug_idx=$AUG_IDX"

srun python $ROOT/scripts/run_chebyshev.py \
    --model_type    cnn \
    --sample_idx    $AUG_IDX \
    --bits          $BITS \
    --p2_only \
    --model_path    $ROOT/checkpoints/fashion_cnn_best.pth \
    --data_path     $ROOT/data/fashionMNIST_augmented_cnn_seed42_walk_gen150_b${BITS}.pt \
    --stage1_dir    $ROOT/results/volumes_v3k_cnn_gen150 \
    --output_dir    $OUT_DIR \
    --time_limit    3600
