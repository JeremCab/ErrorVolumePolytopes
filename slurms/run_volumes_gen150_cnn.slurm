#!/bin/bash
# =============================================================================
# run_volumes_gen150_cnn.slurm
#
# STAGE 1 of the augmented-GACC campaign: mean widths.
#
# One task = one (augmented point, b) pair. Each computes the mean width of P2
# and of every non-empty P3(k), which is what the numerator and denominator of
# (19) need — and, as a by-product, the |V3^c - V2| test that tells case A from
# case B for stage 2.
#
# The array size is NOT hard-coded: it depends on how many representatives the
# generation found, which is only known afterwards. Run
#     python scripts/make_volume_tasks.py --bits 4 6 10 16
# first; it writes data/gen150/volume_tasks.txt (one "<bits> <aug_idx>" line per
# task) and prints the exact sbatch command with the right array bounds.
#
# Each task writes one JSON to
#   results/volumes_v3k_cnn_gen150/b{BB}/volumes_sample{I}.json
# and run_volumes_v3k.py skips a task whose output already exists, so a
# relaunched array only fills the gaps.
#
# --skip_base: P1 appears nowhere in (19) and cost 2 LPs per direction, i.e.
# 21-27% of each task. Direction validity comes from P2 instead. Run a separate
# subsample WITHOUT the flag if the V2/V1 ratio is wanted for the paper.
#
# TIME: with the pre-screen now parallelised, a task is ~750 LPs over 40 cores
# plus ~10 pre-screen LPs, i.e. ~50 min expected. The 4 h limit is deliberate
# margin: the previous b=4 array died on a 2 h limit, and a timeout here means
# resubmitting the whole thing.
# =============================================================================

#SBATCH --job-name=vol_gen150_cnn
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --output=logs/vol_gen150_cnn_%A_%a.out
#SBATCH --error=logs/vol_gen150_cnn_%A_%a.err
# NOTE: no #SBATCH --array here on purpose — pass it on the command line,
#       using the bounds printed by make_volume_tasks.py.

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

ROOT=$WORK/ErrorVolumePolytopes
TASK_FILE=$ROOT/data/gen150/volume_tasks.txt

if [ ! -f "$TASK_FILE" ]; then
    echo "[ABORT] $TASK_FILE not found — run scripts/make_volume_tasks.py first." >&2
    exit 1
fi

# ── Line SLURM_ARRAY_TASK_ID+1 of the task table -> (BITS, AUG_IDX) ──────────
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
if [ -z "$LINE" ]; then
    echo "[ABORT] task $SLURM_ARRAY_TASK_ID beyond the table ($(wc -l < "$TASK_FILE") lines)." >&2
    exit 1
fi
BITS=$(echo "$LINE" | cut -d' ' -f1)
AUG_IDX=$(echo "$LINE" | cut -d' ' -f2)
BITS_PAD=$(printf "%02d" "$BITS")

OUT_DIR=$ROOT/results/volumes_v3k_cnn_gen150/b${BITS_PAD}
mkdir -p "$OUT_DIR" "$ROOT/logs"

echo "[INFO] task=$SLURM_ARRAY_TASK_ID  bits=$BITS  aug_idx=$AUG_IDX  out=$OUT_DIR"

srun python $ROOT/scripts/run_volumes_v3k.py \
    --model_type    cnn \
    --sample_idx    $AUG_IDX \
    --bits_grid     $BITS \
    --n_directions  100 \
    --skip_base \
    --model_path    $ROOT/checkpoints/fashion_cnn_best.pth \
    --data_path     $ROOT/data/fashionMNIST_augmented_cnn_seed42_walk_gen150_b${BITS}.pt \
    --output_dir    $OUT_DIR
