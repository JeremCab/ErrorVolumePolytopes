#!/bin/bash
# 1) validation du melange : memes points, thin x4 et graine differente
# 2) precision : 50 points aleatoires supplementaires, reglages standard
# Idempotent : un point dont le JSON existe est saute.
cd "$(dirname "$0")/.." || exit 1
echo "=== 1. VALIDATION DU MELANGE (thin 3136, seed 7) ==="
for i in $(cat results/volume_ratio/mixing_points.txt); do
  OUT=results/volume_ratio/mix_cnn_s${i}_P1.json
  [ -f "$OUT" ] && { echo "[$i] deja fait"; continue; }
  python3 -u scripts/volume_ratio_hitrun.py --model_type cnn --sample_idx "$i" \
    --polytope p1 --bits 16 --bits_grid 4 6 8 10 12 16 \
    --n_samples 800 --thin 3136 --burn_in 31360 --seed 7 \
    --lp_method auto --time_limit 2400 --out "$OUT" >> logs/vr_mixing.log 2>&1
  echo "[$i] fait"
done
echo "=== 2. PRECISION : 50 points supplementaires ==="
for i in $(cat results/volume_ratio/random_points2.txt); do
  OUT=results/volume_ratio/cnn_s${i}_P1.json
  [ -f "$OUT" ] && { echo "[$i] deja fait"; continue; }
  python3 -u scripts/volume_ratio_hitrun.py --model_type cnn --sample_idx "$i" \
    --polytope p1 --bits 16 --bits_grid 4 6 8 10 12 16 \
    --n_samples 800 --thin 784 --burn_in 7840 \
    --lp_method auto --time_limit 2400 --out "$OUT" >> logs/vr_cnn_random2_P1.log 2>&1
  echo "[$i] fait"
done
echo "TERMINE"
