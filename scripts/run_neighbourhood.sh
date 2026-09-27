#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
R="0.001 0.003 0.01 0.03 0.1 0.3 0.5 1.0"
for M in mlp cnn; do
  OUT=results/volume_ratio/neighbourhood_${M}.json
  [ -f "$OUT" ] && { echo "[$M] deja fait"; continue; }
  echo "=== $M ==="
  python3 -u scripts/neighbourhood_size.py --model_type $M --n_points 74 \
      --n_samples 500 --radii $R --out "$OUT"
done
echo "TERMINE_VOISINAGE"
