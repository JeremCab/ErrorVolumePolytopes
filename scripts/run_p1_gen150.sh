#!/bin/bash
# Echantillonnage direct dans P1 pour les 150 points originaux de gen150.
# Idempotent : un point dont le JSON existe est saute.
cd "$(dirname "$0")/.." || exit 1
N=${1:-150}
for i in $(seq 0 $((N-1))); do
  OUT=results/volume_ratio/cnn_s${i}_P1.json
  [ -f "$OUT" ] && continue
  python3 -u scripts/volume_ratio_hitrun.py --model_type cnn --sample_idx "$i" \
    --polytope p1 --bits 16 --bits_grid 4 6 10 16 \
    --n_samples 800 --thin 784 --burn_in 7840 \
    --lp_method auto --time_limit 2400 --out "$OUT" \
    >> logs/vr_gen150_originals.log 2>&1
  echo "[$i] fait"
done
echo "TERMINE_P1_GEN150"
