#!/usr/bin/env python3
"""
show_p1_volume.py — aggregate the direct volume measurements in P1.

Reads the JSONs written by `volume_ratio_hitrun.py --polytope p1` and prints the
mean volume ratio per bitwidth with its standard error, next to the pointwise
accuracy on the same points. Reads the JSON field `gacc_by_bits`, not the
terminal output, so a SLURM array needs no log wrangling.

    python scripts/show_p1_volume.py --pattern 'results/volume_ratio/cnn_s*_P1_b23.json'
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.networks import FashionCNN_Small, FashionMLP_Large   # noqa: E402
from src.quantization.quantize import quantize_model                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="results/volume_ratio/cnn_s*_P1_b23.json")
    ap.add_argument("--model_type", default="cnn", choices=["cnn", "mlp"])
    ap.add_argument("--no_acc", action="store_true", help="skip the alpha_T row")
    a = ap.parse_args()

    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit(f"[ABORT] aucun fichier pour {a.pattern}")

    vals, pts, skipped = {}, [], 0
    for f in files:
        d = json.loads(Path(f).read_text())
        g = d.get("gacc_by_bits")
        if not g:                       # older runs stored the grid only in stdout
            skipped += 1
            continue
        pts.append(int(d["sample_idx"]))
        for b, v in g.items():
            vals.setdefault(int(b), []).append(float(v))
    if not vals:
        raise SystemExit(f"[ABORT] {len(files)} fichiers lus, aucun ne contient "
                         f"'gacc_by_bits' (runs anterieurs au correctif)")
    bits = sorted(vals)

    print(f"\n{len(pts)} points  ({skipped} sautes, sans grille dans le JSON)")
    print(f"{'':>16}" + "".join(f"{'b=' + str(b):>9}" for b in bits))
    print("-" * (16 + 9 * len(bits)))
    m = {b: float(np.mean(vals[b])) for b in bits}
    se = {b: float(np.std(vals[b]) / np.sqrt(len(vals[b]))) for b in bits}
    print(f"{'volume dans P1':>16}" + "".join(f"{m[b]:>9.4f}" for b in bits))
    print(f"{'  erreur-type':>16}" + "".join(f"{se[b]:>9.4f}" for b in bits))

    if not a.no_acc:
        Net = FashionCNN_Small if a.model_type == "cnn" else FashionMLP_Large
        fp = Net(); fp.load_state_dict(torch.load(
            f"checkpoints/fashion_{a.model_type}_best.pth", map_location="cpu",
            weights_only=True)); fp.eval()
        ds = torch.load(f"data/fashionMNIST_correct_{a.model_type}.pt",
                        map_location="cpu", weights_only=False)
        X = torch.stack([ds[i][0] for i in pts])
        if a.model_type == "mlp":
            X = X.flatten(1)
        C = np.array([int(ds[i][1]) for i in pts])
        acc = {}
        for b in bits:
            q = quantize_model(fp, bits=b).eval()
            with torch.no_grad():
                acc[b] = float((q(X).argmax(1).numpy() == C).mean())
        print(f"{'alpha_T':>16}" + "".join(f"{acc[b]:>9.4f}" for b in bits))
        print("-" * (16 + 9 * len(bits)))
        print(f"{'ecart':>16}" + "".join(f"{m[b]-acc[b]:>+9.4f}" for b in bits))
        print("\n   Un ecart positif signifie que le VOISINAGE est mieux classe que le"
              "\n   point lui-meme — les voisinages des points mal classes etant en"
              "\n   partie corrects, ils tirent la moyenne vers le haut.")


if __name__ == "__main__":
    main()
