#!/bin/bash
# =============================================================================
# complete_b04_ipm.slurm — recompute the b=4 Chebyshev radii with interior point.
#
# The campaign used the dual simplex and converged on 274 of 589 polytopes at
# b=4. The two-arm retry showed those failures are reproducible with the same
# solver and 97% recoverable with interior point, so this redoes them with
# --lp_method auto.
#
#     python scripts/list_b04_tiles.py --all
#     sbatch --array=0-185 slurms/complete_b04_ipm.slurm
#
# Writes to results/chebyshev_cnn_gen150_ipm/ — the original campaign is left
# untouched, so the two can be compared and gamma recomputed from either.
#
# ⚠ Completing b=4 CHANGES gamma at b=4: entries recorded as 'failed' become
# real radii and alter which polytopes enter (19). More correct, but not neutral.
# =============================================================================
#SBATCH --job-name=b04_ipm
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/b04_ipm_%A_%a.out
#SBATCH --error=logs/b04_ipm_%A_%a.err

module load pytorch-gpu/py3/2.8.0
export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH
ROOT=$WORK/ErrorVolumePolytopes

TASK_FILE=$ROOT/data/gen150/b04_ipm_tasks.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
[ -z "$LINE" ] && { echo "[ABORT] tache hors table"; exit 1; }
BITS=$(echo "$LINE" | cut -d' ' -f1); IDX=$(echo "$LINE" | cut -d' ' -f2)
BP=$(printf "%02d" "$BITS")

OUT_DIR=$ROOT/results/chebyshev_cnn_gen150_ipm/b${BP}
mkdir -p "$OUT_DIR" "$ROOT/logs"
[ -f "$OUT_DIR/chebyshev_sample${IDX}.json" ] && { echo "[$IDX] deja fait"; exit 0; }

srun python $ROOT/scripts/run_chebyshev.py \
    --model_type    cnn \
    --sample_idx    $IDX \
    --bits          $BITS \
    --lp_method     auto \
    --model_path    $ROOT/checkpoints/fashion_cnn_best.pth \
    --data_path     $ROOT/data/fashionMNIST_augmented_cnn_seed42_walk_gen150_b${BITS}.pt \
    --stage1_dir    $ROOT/results/volumes_v3k_cnn_gen150 \
    --output_dir    "$OUT_DIR" \
    --time_limit    3600
