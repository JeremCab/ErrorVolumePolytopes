#!/bin/bash
# =============================================================================
# run_volumes_v3k_aug_sample760_cnn.slurm
#
# Stage 1 of the augmented-GACC experiment (Jiri's email of 2026-06-07).
# Focused volume computation on a single original CNN sample (#760), using
# b-specific MCMC-augmented datasets generated locally by
# scripts/build_augmented_dataset.py.
#
# One Slurm array — 57 tasks — covers all (b, augmented_point) pairs:
#   b=4  → 51 points (x_0 + 50 walk reps)
#   b=6  →  2 points (x_0 +  1 walk rep)
#   b=8  →  1 point  (x_0 alone — single P2 tile covers all of P1)
#   b=10 →  1 point
#   b=12 →  1 point
#   b=16 →  1 point
# Total: 51 + 2 + 1 + 1 + 1 + 1 = 57 tasks
#
# Each task writes one JSON file to
#   $WORK/ErrorVolumePolytopes/results/volumes_v3k_cnn_aug_sample760/b{BB}/volumes_sample{I}.json
# with bits_grid=[B] and widths_correct/widths_both keyed by "B" only.
# Aggregation per b is then a one-liner in the analysis notebook.
# =============================================================================

#SBATCH --job-name=v3k_aug_s760_cnn
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --array=0-56                # 57 tasks (51+2+1+1+1+1)
#SBATCH --time=2:00:00              # ~20 min/task expected; 1 h safety margin
#SBATCH --output=logs/v3k_aug_s760_cnn_%A_%a.out
#SBATCH --error=logs/v3k_aug_s760_cnn_%A_%a.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

# -----------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID → (BITS, AUG_IDX)
#
# Dataset sizes for sample 760 (generated locally with seed=42, max_steps=10000):
#   b=4  → 51 reps    (cumulative range:  0..50)
#   b=6  →  2 reps    (                  51..52)
#   b=8  →  1 rep     (                      53)
#   b=10 →  1 rep     (                      54)
#   b=12 →  1 rep     (                      55)
#   b=16 →  1 rep     (                      56)
# -----------------------------------------------------------------------------
TASK_ID=$SLURM_ARRAY_TASK_ID

if   [ $TASK_ID -lt 51 ]; then BITS=4;  AUG_IDX=$TASK_ID
elif [ $TASK_ID -lt 53 ]; then BITS=6;  AUG_IDX=$((TASK_ID - 51))
elif [ $TASK_ID -lt 54 ]; then BITS=8;  AUG_IDX=$((TASK_ID - 53))
elif [ $TASK_ID -lt 55 ]; then BITS=10; AUG_IDX=$((TASK_ID - 54))
elif [ $TASK_ID -lt 56 ]; then BITS=12; AUG_IDX=$((TASK_ID - 55))
else                           BITS=16; AUG_IDX=$((TASK_ID - 56))
fi

# Zero-padded bit-width for tidy directory names (b04, b06, ..., b16)
BITS_PAD=$(printf "%02d" $BITS)

OUT_DIR=$WORK/ErrorVolumePolytopes/results/volumes_v3k_cnn_aug_sample760/b${BITS_PAD}
mkdir -p $OUT_DIR
mkdir -p $WORK/ErrorVolumePolytopes/logs

echo "[INFO] task=$TASK_ID  bits=$BITS  aug_idx=$AUG_IDX  out=$OUT_DIR"

srun python $WORK/ErrorVolumePolytopes/scripts/run_volumes_v3k.py \
    --model_type    cnn \
    --sample_idx    $AUG_IDX \
    --bits_grid     $BITS \
    --n_directions  100 \
    --model_path    $WORK/ErrorVolumePolytopes/checkpoints/fashion_cnn_best.pth \
    --data_path     $WORK/ErrorVolumePolytopes/data/fashionMNIST_augmented_cnn_seed42_walk_sample760_b${BITS}.pt \
    --output_dir    $OUT_DIR
