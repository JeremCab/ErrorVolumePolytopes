#!/bin/bash
# Balayage en rayon avec b = 2 et 3 ajoutes, pour repondre a la question de Jiri.
cd "$(dirname "$0")/.." || exit 1
B="2 3 4 6 8 10 12 16"
R="0.001 0.003 0.01 0.03 0.1 0.3 0.5 1.0"
for M in mlp cnn; do
  OUT=results/volume_ratio/neighbourhood_${M}_b23.json
  [ -f "$OUT" ] && { echo "[$M] deja fait"; continue; }
  PTS=results/volume_ratio/$([ $M = mlp ] && echo mlp_points.txt || echo cnn_points74.txt)
  echo "=== $M ==="
  python3 -u scripts/neighbourhood_size.py --model_type $M --n_points 74 \
      --points_file "$PTS" --n_samples 500 --bits_grid $B --radii $R --out "$OUT"
done
echo "TERMINE_B23"
