#!/usr/bin/env python3
"""
gamma_measured.py — gamma (19) computed with the radii we actually measured.

The question this answers: once the zero-volume subpolytopes are removed using
Chebyshev rather than the lemma, does the trend come back? It is answered per
bit-width, because a trend can only be changed by an exclusion that bites
*differently* at different b — a uniform exclusion rescales the denominator and
leaves the ordering intact.

Restricted to tiles whose stage-2 result is complete: a tile with one failed LP
cannot have (18) applied to it. That is a biased subsample — mostly case-B tiles,
plus the cross-check sample — so the two gammas are reported ON THE SAME TILES,
which is what isolates the effect of the exclusion from the effect of the
selection.

Usage
-----
    python scripts/gamma_measured.py --bits 6 10 16
"""
import argparse, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cheb_dir",   default="results/chebyshev_cnn_gen150")
    ap.add_argument("--stage1_dir", default="results/volumes_v3k_cnn_gen150")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 10, 16])
    ap.add_argument("--zero_tol", type=float, nargs="+",
                    default=[1e-6, 5e-5, 1e-4, 5e-4])
    a = ap.parse_args()

    data = {}
    for b in a.bits:
        tiles = []
        for f in sorted((Path(a.cheb_dir) / f"b{b:02d}").glob("chebyshev_sample*.json")):
            d = json.loads(f.read_text())
            if any(p["status"] == "failed" for p in d["polytopes"]):
                continue                       # (18) cannot be applied to this tile
            tiles.append(d)
        data[b] = tiles

    print("gamma sur les MEMES tuiles (celles dont l'etape 2 est complete)\n")
    hdr = f"{'b':>3} {'tuiles':>7} {'sans exclusion':>15}"
    for zt in a.zero_tol:
        hdr += f"{'zt=' + f'{zt:.0e}':>12}"
    hdr += f"{'% nuls @1e-4':>13}"
    print(hdr); print("-" * len(hdr))

    for b in a.bits:
        T = data[b]
        if not T:
            print(f"{b:>3} {'0':>7}   (aucune tuile complete)"); continue
        num0 = den0 = 0.0
        for d in T:
            c = d["class_c"]
            for p in d["polytopes"]:
                if p["polytope"] == "P2":
                    continue
                w = p["mean_width"] or 0.0
                den0 += w
                if p["k"] == c:
                    num0 += w
        row = f"{b:>3} {len(T):>7} {num0/den0 if den0 else float('nan'):>15.4f}"
        for zt in a.zero_tol:
            num = den = 0.0
            for d in T:
                c = d["class_c"]
                for p in d["polytopes"]:
                    if p["polytope"] == "P2":
                        continue
                    r = p["radius"]
                    if r is None or r <= zt:      # zero volume -> d_0 = 0
                        continue
                    w = p["mean_width"] or 0.0
                    den += w
                    if p["k"] == c:
                        num += w
            row += f"{num/den if den else float('nan'):>12.4f}"
        inc = [p for d in T for p in d["polytopes"]
               if p["polytope"] != "P2" and p["k"] != d["class_c"]]
        frac = np.mean([p["radius"] is not None and p["radius"] <= 1e-4 for p in inc]) if inc else float("nan")
        row += f"{100*frac:>12.1f}%"
        print(row)

    print("\nLecture : la tendance ne peut changer que si l'exclusion mord DIFFEREMMENT"
          "\nselon b. Une exclusion uniforme redimensionne le denominateur sans toucher"
          "\na l'ordre. Comparer la colonne 'sans exclusion' aux colonnes zt=... sur les"
          "\nMEMES tuiles isole l'effet de (18) de celui de la selection.")


if __name__ == "__main__":
    main()
