#!/bin/bash
# Reprend la campagne d'echantillonnage volumique la ou elle s'est arretee.
# Idempotent : un point dont le JSON existe deja est saute.
#   bash scripts/resume_volume_ratio.sh
cd "$(dirname "$0")/.." || exit 1
LIST=results/volume_ratio/random_points.txt
[ -f "$LIST" ] || { echo "[ABORT] $LIST manquant"; exit 1; }
for i in $(cat "$LIST"); do
  OUT=results/volume_ratio/cnn_s${i}_P1.json
  if [ -f "$OUT" ]; then echo "[$i] deja fait — saute"; continue; fi
  echo "[$i] en cours..."
  python3 -u scripts/volume_ratio_hitrun.py --model_type cnn --sample_idx "$i" \
    --polytope p1 --bits 16 --bits_grid 4 6 8 10 12 16 \
    --n_samples 800 --thin 784 --burn_in 7840 \
    --lp_method auto --time_limit 2400 --out "$OUT" \
    >> logs/vr_cnn_random_P1.log 2>&1
done
echo "termine : $(ls results/volume_ratio/cnn_s*_P1.json 2>/dev/null | wc -l) points au total"
