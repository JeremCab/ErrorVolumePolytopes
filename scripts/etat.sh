#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
echo "MLP : $(ls results/volume_ratio/mlp_s*_P1.json 2>/dev/null | wc -l | tr -d ' ')/74"
grep -q TERMINE_MLP logs/mlp_driver.log 2>/dev/null && echo "  -> TERMINE" \
  || { pgrep -f run_mlp_volume >/dev/null && echo "  -> en cours" || echo "  -> ARRETE (relancer : bash scripts/run_mlp_volume.sh)"; }
