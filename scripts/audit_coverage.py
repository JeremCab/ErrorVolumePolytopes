#!/usr/bin/env python3
"""
audit_coverage.py — are the augmented tiles comparable in size to the original one?

The augmentation exists to cover P1 with the linearity tiles of the quantized
network. If the representatives found by the walk turn out to be slivers next to
the original point's tile, the covering is nominal rather than real: the extra
points would add little volume while adding a lot of weight to a mean-width
average, which is not additive.

Compares, per bit-width, the mean width V2 of the original point's tile against
those of its augmented representatives — per original point, so each is its own
control. Reads stage-1 results and the merge provenance; no LP.

On a Jean Zay login node: module load pytorch-gpu/py3/2.8.0  (numpy only)

Usage
-----
    python scripts/audit_coverage.py --bits 4 6 10 16
"""
import argparse, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results/volumes_v3k_cnn_gen150")
    ap.add_argument("--prov_prefix",
                    default="data/fashionMNIST_augmented_cnn_seed42_walk_gen150")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 10, 16])
    a = ap.parse_args()

    print(f"{'b':>3} {'originaux':>10} {'reps':>7} {'V2 orig (med)':>14} "
          f"{'V2 aug (med)':>13} {'V2aug/V2orig p10/p50/p90':>28}")
    print("-" * 82)

    for b in a.bits:
        prov = Path(f"{a.prov_prefix}_b{b}_provenance.json")
        if not prov.exists():
            print(f"{b:>3}   provenance absente : {prov}"); continue
        meta = json.loads(prov.read_text())

        V2 = {}
        for f in (Path(a.results_dir) / f"b{b:02d}").glob("volumes_sample*.json"):
            d = json.loads(f.read_text())
            v = d.get("widths_correct", {}).get(str(b))
            if v is not None and np.isfinite(v):
                V2[int(f.stem.split("sample")[1])] = float(v)

        by_orig = {}
        for e in meta["points"]:
            v = V2.get(e["aug_idx"])
            if v is None:
                continue
            by_orig.setdefault(e["orig_idx"], {"orig": None, "aug": []})
            if e["is_original"]:
                by_orig[e["orig_idx"]]["orig"] = v
            else:
                by_orig[e["orig_idx"]]["aug"].append(v)

        origs = [g["orig"] for g in by_orig.values() if g["orig"] is not None]
        augs  = [v for g in by_orig.values() for v in g["aug"]]
        # ratio computed WITHIN each original point, so each is its own control
        ratios = [v / g["orig"] for g in by_orig.values()
                  if g["orig"] and g["orig"] > 0 for v in g["aug"]]
        if not origs:
            print(f"{b:>3}   aucun resultat"); continue
        r = (f"{np.percentile(ratios,10):.2f} / {np.percentile(ratios,50):.2f} / "
             f"{np.percentile(ratios,90):.2f}" if ratios else "  (aucun representant)")
        print(f"{b:>3} {len(origs):>10} {len(augs):>7} {np.median(origs):>14.4f} "
              f"{(np.median(augs) if augs else float('nan')):>13.4f} {r:>28}")

    print("\nLecture : un rapport proche de 1 signifie que les representants sont des"
          "\ntuiles de taille comparable a l'originale — la couverture de P1 est reelle."
          "\nUn rapport tres inferieur a 1 signifierait que la marche ne trouve que des"
          "\neclats, qui pesent lourd dans une moyenne de largeurs sans apporter de volume.")


if __name__ == "__main__":
    main()
