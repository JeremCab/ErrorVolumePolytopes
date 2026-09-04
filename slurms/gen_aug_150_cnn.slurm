#!/bin/bash
# =============================================================================
# gen_aug_150_cnn.slurm
#
# Generation stage of the augmented-GACC campaign (2026-09).
# Builds the b-specific MCMC-augmented datasets for the first 150 original
# CNN points of data/fashionMNIST_correct_cnn.pt, for b in {4, 6, 10, 16}.
#
# One task = one (original point, b) pair  ->  150 x 4 = 600 tasks.
# Each task writes ONE .pt + ONE _log.json into data/gen150/b{BB}/.
# They are merged into four datasets afterwards by
#   scripts/merge_aug_shards.py           (run locally or on a login node).
#
# WHY one task per point: the walk is a sequential Markov chain, so a task
# cannot be parallelised internally. Splitting by point is the only axis.
#
# WHY --cpus-per-task=4 and not 40: the walk is single-threaded (one small
# forward pass per step). Asking for 40 would bill ~4000 core-hours for ~100
# actually used.
#
# COST (measured on a laptop, MPS; CPU here may differ by a factor of a few):
#   b=4   ~26 s/point  (the walk hits the 500 cap after ~450 steps)
#   b>=6  ~580 s/point (the walk burns all 10000 steps and finds 0-1 reps)
#   => ~600 tasks, worst case ~10 min each.
#
# max_steps=10000 is deliberate, NOT a budget compromise: at b=4 the walk
# already stops on its own after ~70 steps, so a larger cap changes nothing
# there while multiplying the b>=6 cost tenfold.
# =============================================================================

#SBATCH --job-name=gen_aug150_cnn
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-599                # 150 originals x 4 bit-widths
#SBATCH --time=2:00:00               # ~10 min expected; wide margin for CPU
#SBATCH --output=logs/gen_aug150_cnn_%A_%a.out
#SBATCH --error=logs/gen_aug150_cnn_%A_%a.err

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

ROOT=$WORK/ErrorVolumePolytopes
N_ORIG=150
BITS_LIST=(4 6 10 16)

# ── Map SLURM_ARRAY_TASK_ID -> (BITS, ORIG_IDX) ──────────────────────────────
#   task = bits_slot * N_ORIG + orig_idx
TASK_ID=$SLURM_ARRAY_TASK_ID
BITS_SLOT=$(( TASK_ID / N_ORIG ))
ORIG_IDX=$((  TASK_ID % N_ORIG ))
BITS=${BITS_LIST[$BITS_SLOT]}
BITS_PAD=$(printf "%02d" $BITS)

OUT_DIR=$ROOT/data/gen150/b${BITS_PAD}
mkdir -p $OUT_DIR $ROOT/logs

TAG=gen150_b${BITS}_i${ORIG_IDX}
OUT_PT=$OUT_DIR/fashionMNIST_augmented_cnn_seed42_walk_${TAG}.pt

# Idempotence: a relaunched array only recomputes what is missing.
if [ -f "$OUT_PT" ]; then
    echo "[SKIP] task=$TASK_ID bits=$BITS orig=$ORIG_IDX already done"
    exit 0
fi

echo "[INFO] task=$TASK_ID  bits=$BITS  orig_idx=$ORIG_IDX  out=$OUT_DIR"

srun python $ROOT/scripts/build_augmented_dataset.py \
    --model_type     cnn \
    --model_path     $ROOT/checkpoints/fashion_cnn_best.pth \
    --data_path      $ROOT/data/fashionMNIST_correct_cnn.pt \
    --output_dir     $OUT_DIR \
    --bits           $BITS \
    --strategy       walk \
    --walk_mode      projected \
    --nb_aug_points  500 \
    --nb_diverse     5 \
    --max_steps      10000 \
    --p1_filter_tol  1e-4 \
    --seed           42 \
    --start_idx      $ORIG_IDX \
    --n_samples      1 \
    --tag            $TAG
