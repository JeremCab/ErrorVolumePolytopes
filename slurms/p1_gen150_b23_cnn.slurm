#!/bin/bash
# =============================================================================
# p1_gen150_b23_cnn.slurm — echantillonnage direct dans P1 des 150 points originaux
# de la campagne gen150, pour calibrer la tolerance de (18).
#
# Une tache = un point original. P1 ne depend pas de b, donc une seule marche
# sert les quatre largeurs de bits.
#
#     sbatch --array=0-149 slurms/p1_gen150_b23_cnn.slurm
#
# ~4 min par tache (1 LP de Chebyshev ~20 s, puis 800 tirages).
# =============================================================================
#SBATCH --job-name=p1_gen150_b23
#SBATCH --hint=nomultithread
#SBATCH --account=aik@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=logs/p1_gen150_b23_%A_%a.out
#SBATCH --error=logs/p1_gen150_b23_%A_%a.err

module load pytorch-gpu/py3/2.8.0
export PYTHONUSERBASE=$WORK/.local_python
export TMPDIR=$JOBSCRATCH
ROOT=$WORK/ErrorVolumePolytopes
mkdir -p $ROOT/results/volume_ratio $ROOT/logs

IDX=$SLURM_ARRAY_TASK_ID
OUT=$ROOT/results/volume_ratio/cnn_s${IDX}_P1_b23.json
if [ -f "$OUT" ]; then echo "[$IDX] deja fait"; exit 0; fi

srun python $ROOT/scripts/volume_ratio_hitrun.py \
    --model_type cnn --sample_idx $IDX --polytope p1 \
    --bits 16 --bits_grid 2 3 4 6 10 16 \
    --n_samples 800 --thin 784 --burn_in 7840 \
    --lp_method auto --time_limit 2400 --out "$OUT"
