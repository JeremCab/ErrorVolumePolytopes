#!/usr/bin/env python3
"""
analyse_stage2.py — what the Chebyshev radii actually settle.

Stage 2 answers the question the tolerance could not: are the incorrect
subpolytopes of a tile genuinely of zero volume? It does so with no tolerance on
mean widths at all. But an LP that hits its time limit answers nothing, so the
first thing to establish is how much of the campaign actually converged.

Reports, in order:
  1. convergence, per bit-width and per polytope kind;
  2. among the tiles that fully converged, how many incorrect classes come out
     zero-volume — this is what decides whether the loose or the tight tolerance
     was right on the disputed tiles;
  3. the cross-check: on case-A tiles the lemma PREDICTS zero volume for every
     class but one. If Chebyshev disagrees there, gamma_19_min is wrong and the
     whole lemma argument collapses.

Needs numpy only. On a Jean Zay login node: module load pytorch-gpu/py3/2.8.0

Usage
-----
    python scripts/analyse_stage2.py --bits 4 6 10 16
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cheb_dir",  default="results/chebyshev_cnn_gen150")
    ap.add_argument("--stage1_dir", default="results/volumes_v3k_cnn_gen150")
    ap.add_argument("--task_file", default="data/gen150/chebyshev_tasks.txt")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 10, 16])
    ap.add_argument("--zero_tol", type=float, default=1e-6)
    a = ap.parse_args()

    reason = {}
    for line in Path(a.task_file).read_text().split("\n"):
        p = line.split()
        if len(p) >= 3:
            reason[(int(p[0]), int(p[1]))] = p[2]

    tiles = []
    for b in a.bits:
        for f in sorted((Path(a.cheb_dir) / f"b{b:02d}").glob("chebyshev_sample*.json")):
            d = json.loads(f.read_text())
            aug = d["sample_idx"]
            tiles.append(dict(b=b, aug=aug, c=d["class_c"],
                              why=reason.get((b, aug), "?"), polys=d["polytopes"]))
    if not tiles:
        raise SystemExit(f"[ABORT] no stage-2 result under {a.cheb_dir}")

    # ── 1. convergence ────────────────────────────────────────────────────────
    print(f"{'b':>3} {'tuiles':>7} {'polytopes':>10} {'converges':>10} {'echecs':>8} "
          f"{'tuiles completes':>17}")
    print("-" * 62)
    for b in a.bits:
        T = [t for t in tiles if t["b"] == b]
        if not T: continue
        allp = [p for t in T for p in t["polys"]]
        ok   = [p for p in allp if p["status"] != "failed"]
        full = [t for t in T if all(p["status"] != "failed" for p in t["polys"])]
        print(f"{b:>3} {len(T):>7} {len(allp):>10} {len(ok):>10} {len(allp)-len(ok):>8} "
              f"{len(full):>10} ({100*len(full)/len(T):.0f}%)")
    allp = [p for t in tiles for p in t["polys"]]
    print(f"\ntotal : {len(tiles)} tuiles, {len(allp)} polytopes, "
          f"{sum(1 for p in allp if p['status']!='failed')} converges "
          f"({100*sum(1 for p in allp if p['status']!='failed')/len(allp):.1f}%)")

    # P2 apart: without it a tile says nothing
    p2 = [p for t in tiles for p in t["polys"] if p["polytope"] == "P2"]
    print(f"dont P2 : {sum(1 for p in p2 if p['status']!='failed')}/{len(p2)} converges")

    # ── 2. verdicts on the tiles that fully converged ─────────────────────────
    print(f"\n{'='*62}\nVERDICTS (tuiles entierement convergees)\n{'='*62}")
    for why in ("caseB", "xcheck"):
        T = [t for t in tiles if t["why"] == why
             and all(p["status"] != "failed" for p in t["polys"])]
        if not T:
            print(f"\n{why}: aucune tuile entierement convergee")
            continue
        n_zero_incorrect = n_full_incorrect = 0
        all_incorrect_zero = 0
        for t in T:
            inc = [p for p in t["polys"] if p["polytope"] != "P2" and p["k"] != t["c"]]
            z = [p for p in inc if p["radius"] is not None and p["radius"] <= a.zero_tol]
            n_zero_incorrect += len(z); n_full_incorrect += len(inc) - len(z)
            if inc and len(z) == len(inc):
                all_incorrect_zero += 1
        tot = n_zero_incorrect + n_full_incorrect
        print(f"\n{why}: {len(T)} tuiles completes, {tot} sous-polytopes incorrects")
        print(f"   de volume NUL       : {n_zero_incorrect:>5}  ({100*n_zero_incorrect/tot:.1f}%)"
              if tot else "   (aucun)")
        print(f"   pleinement dimens.  : {n_full_incorrect:>5}  ({100*n_full_incorrect/tot:.1f}%)"
              if tot else "")
        print(f"   tuiles ou TOUS les incorrects sont nuls : {all_incorrect_zero}/{len(T)}"
              f"  ({100*all_incorrect_zero/len(T):.0f}%)")
        if why == "xcheck":
            print("   ^ le lemme predit 100% ici. Tout ecart invalide gamma_19_min.")
        else:
            print("   ^ 100% signifierait que la tolerance lache avait raison ;"
                  "\n     0% que la tolerance serree avait raison.")

    # ── 3. radii, to see whether the verdict is clear-cut or a threshold call ──
    print(f"\n{'='*62}\nRAYONS (sous-polytopes incorrects convergees)\n{'='*62}")
    for why in ("caseB", "xcheck"):
        r = [p["radius"] for t in tiles if t["why"] == why
             for p in t["polys"]
             if p["polytope"] != "P2" and p["k"] != t["c"]
             and p["status"] != "failed" and p["radius"] is not None]
        if not r: continue
        r = np.array(r)
        print(f"\n{why} ({len(r)} rayons) :")
        for q in (0, 10, 50, 90, 100):
            print(f"   percentile {q:>3} : {np.percentile(r, q):.3e}")
        print(f"   <= zero_tol={a.zero_tol:.0e} : {100*np.mean(r <= a.zero_tol):.1f}%")


if __name__ == "__main__":
    main()
