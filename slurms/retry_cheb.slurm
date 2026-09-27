#!/bin/bash
# =============================================================================
# retry_cheb.slurm — retry the Chebyshev LPs that did not converge.
#
# Two arms, same task list, different solver, separate output directories:
#   METHOD=highs TAG=armA   are the failures reproducible?
#   METHOD=auto  TAG=armB   does interior point recover them?
#
#   python scripts/list_failed_cheb.py --bits 4 --limit 20
#   sbatch --array=0-19 --export=ALL,METHOD=highs,TAG=armA slurms/retry_cheb.slurm
#   sbatch --array=0-19 --export=ALL,METHOD=auto,TAG=armB slurms/retry_cheb.slurm
# =============================================================================
#SBATCH --job-name=retry_cheb
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/retry_cheb_%A_%a.out
#SBATCH --error=logs/retry_cheb_%A_%a.err

module load pytorch-gpu/py3/2.8.0
export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH
ROOT=$WORK/ErrorVolumePolytopes
: "${METHOD:=highs}" ; : "${TAG:=armA}"

TASK_FILE=$ROOT/data/gen150/retry_cheb_tasks.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
[ -z "$LINE" ] && { echo "[ABORT] tache hors table"; exit 1; }
BITS=$(echo "$LINE" | cut -d' ' -f1); IDX=$(echo "$LINE" | cut -d' ' -f2)
BP=$(printf "%02d" "$BITS")

OUT_DIR=$ROOT/results/retry_cheb_${TAG}/b${BP}
mkdir -p "$OUT_DIR" "$ROOT/logs"
echo "[INFO] $TAG  method=$METHOD  bits=$BITS  aug=$IDX"

srun python $ROOT/scripts/run_chebyshev.py \
    --model_type    cnn \
    --sample_idx    $IDX \
    --bits          $BITS \
    --lp_method     $METHOD \
    --model_path    $ROOT/checkpoints/fashion_cnn_best.pth \
    --data_path     $ROOT/data/fashionMNIST_augmented_cnn_seed42_walk_gen150_b${BITS}.pt \
    --stage1_dir    $ROOT/results/volumes_v3k_cnn_gen150 \
    --output_dir    "$OUT_DIR" \
    --time_limit    3600
