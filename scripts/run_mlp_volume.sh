#!/bin/bash
# GACC volumique directe pour le MLP — meme protocole que le CNN.
# Idempotent : un point dont le JSON existe est saute.
cd "$(dirname "$0")/.." || exit 1
for i in $(cat results/volume_ratio/mlp_points.txt); do
  OUT=results/volume_ratio/mlp_s${i}_P1.json
  [ -f "$OUT" ] && { echo "[$i] deja fait"; continue; }
  python3 -u scripts/volume_ratio_hitrun.py --model_type mlp --sample_idx "$i" \
    --polytope p1 --bits 16 --bits_grid 4 6 8 10 12 16 \
    --n_samples 800 --thin 784 --burn_in 7840 \
    --lp_method auto --time_limit 2400 --out "$OUT" \
    >> logs/vr_mlp_P1.log 2>&1
  echo "[$i] fait"
done
echo "TERMINE_MLP"
