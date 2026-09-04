#!/bin/bash
# =============================================================================
# run_chebyshev_gen150_cnn.slurm
#
# STAGE 2 of the augmented-GACC campaign: Chebyshev radii, i.e. the zero-volume
# condition of paper eq. (18).
#
# Deliberately SEPARATE from stage 1, on two measured grounds: a Chebyshev LP
# costs ~17-19x a mean-width LP, and its cost is unpredictable (at b=4 the
# unscaled formulation did not converge at all). Running it inside stage 1 would
# put the primary, predictable computation at the mercy of the unpredictable one.
#
# Only a MINORITY of tiles need it. The lemma settles (18) wherever some class
# has d(P3^k) = d(P2) — 79-94% of cases on the non-augmented data. So the array
# covers the case-B tiles plus a sample of case-A tiles as a cross-check, and
# that list is produced by
#     python scripts/select_case_b.py --bits 4 6 10 16
# which writes data/gen150/chebyshev_tasks.txt and prints the sbatch command
# with the right array bounds.
#
# --cpus-per-task=8: a task holds ~3 polytopes solved in parallel, one LP each,
# and an LP is essentially single-threaded. 40 would be billed for nothing.
#
# --time_limit 3600 per LP: HiGHS honours it, and a capped LP is reported as
# 'failed', never 'empty' — the difference matters, since 'empty' would zero the
# polytope in (19) while 'failed' says the value is unusable. Non-convergence is
# itself a result to report: it bears on whether (18) is computable at all.
# =============================================================================

#SBATCH --job-name=cheb_gen150_cnn
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/cheb_gen150_cnn_%A_%a.out
#SBATCH --error=logs/cheb_gen150_cnn_%A_%a.err
# NOTE: no #SBATCH --array on purpose — pass the bounds printed by
#       scripts/select_case_b.py on the command line.

module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH

ROOT=$WORK/ErrorVolumePolytopes
TASK_FILE=$ROOT/data/gen150/chebyshev_tasks.txt

if [ ! -f "$TASK_FILE" ]; then
    echo "[ABORT] $TASK_FILE not found — run scripts/select_case_b.py first." >&2
    exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
if [ -z "$LINE" ]; then
    echo "[ABORT] task $SLURM_ARRAY_TASK_ID beyond the table ($(wc -l < "$TASK_FILE") lines)." >&2
    exit 1
fi
BITS=$(echo    "$LINE" | cut -d' ' -f1)
AUG_IDX=$(echo "$LINE" | cut -d' ' -f2)
REASON=$(echo  "$LINE" | cut -d' ' -f3)     # caseB | xcheck — recorded for traceability
BITS_PAD=$(printf "%02d" "$BITS")

OUT_DIR=$ROOT/results/chebyshev_cnn_gen150/b${BITS_PAD}
mkdir -p "$OUT_DIR" "$ROOT/logs"

echo "[INFO] task=$SLURM_ARRAY_TASK_ID  bits=$BITS  aug_idx=$AUG_IDX  reason=$REASON"

srun python $ROOT/scripts/run_chebyshev.py \
    --model_type    cnn \
    --sample_idx    $AUG_IDX \
    --bits          $BITS \
    --model_path    $ROOT/checkpoints/fashion_cnn_best.pth \
    --data_path     $ROOT/data/fashionMNIST_augmented_cnn_seed42_walk_gen150_b${BITS}.pt \
    --stage1_dir    $ROOT/results/volumes_v3k_cnn_gen150 \
    --output_dir    $OUT_DIR \
    --time_limit    3600
