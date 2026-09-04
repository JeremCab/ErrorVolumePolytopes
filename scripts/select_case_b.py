#!/usr/bin/env python3
"""
select_case_b.py — apply the mean-width equality criterion to the stage-1 results,
report both gamma variants, and emit the task table for the Chebyshev stage.

The criterion comes from the lemma (equal mean width => same set): if some class
k satisfies d(P3^k) = d(P2), then P3^k = P2 and every other subpolytope has zero
volume, so (18) is settled without a single Chebyshev LP. Testing it on the
CORRECT class alone is enough — the indicator 1[V3^c = V2] is 0 whether the
equality is carried by another class or by none.

The tolerance is not a free parameter: the same random directions are used for
P2 and for each P3^k, so when the two bodies coincide their widths agree to
solver precision (measured medians 1e-10 CNN, 1e-13 MLP). Any cut between 1e-8
and 1e-4 gives the same verdict; --tol_scan prints the sensitivity so this can
be shown rather than asserted.

Tiles where NO class matches (case B) are the ones that genuinely need the
Chebyshev radius, plus a sample of case-A tiles as a cross-check: there the
lemma predicts rho = 0 for the other classes, and confirming it validates both
criteria against each other.

On a Jean Zay login node this needs numpy, so load the environment first:
    module load pytorch-gpu/py3/2.8.0

Usage
-----
    python scripts/select_case_b.py --results_dir results/volumes_v3k_cnn_gen150 \\
        --bits 4 6 10 16 --tol 1e-8 --n_xcheck 30
"""
import argparse, json, random
from pathlib import Path

import numpy as np


def load_tiles(results_dir: Path, bits):
    """Yield one record per stage-1 JSON."""
    for b in bits:
        for f in sorted((results_dir / f"b{b:02d}").glob("volumes_sample*.json"),
                        key=lambda p: int(p.stem.split("sample")[1])):
            d = json.loads(f.read_text())
            sb, c = str(b), d["class_c"]
            V2 = d.get("widths_correct", {}).get(sb)
            W  = d.get("widths_both", {}).get(sb)
            if V2 is None or W is None or not np.isfinite(V2) or V2 <= 0:
                print(f"  [WARN] {f.name} (b={b}): unusable V2, skipped")
                continue
            W = [0.0 if (x is None or not np.isfinite(x)) else float(x) for x in W]
            yield {"b": b, "aug_idx": int(f.stem.split("sample")[1]), "c": c,
                   "V2": float(V2), "W": W, "gap": abs(W[c] - V2) / V2}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results/volumes_v3k_cnn_gen150")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 10, 16])
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--n_xcheck", type=int, default=30,
                    help="case-A tiles sampled for the lemma/Chebyshev cross-check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/gen150/chebyshev_tasks.txt")
    ap.add_argument("--report", default="results/data_for_jiri/gamma_gen150.json")
    ap.add_argument("--slurm", default="slurms/run_chebyshev_gen150_cnn.slurm")
    ap.add_argument("--tol_scan", action="store_true",
                    help="print how the case-A rate varies with the tolerance")
    a = ap.parse_args()

    tiles = list(load_tiles(Path(a.results_dir), a.bits))
    if not tiles:
        raise SystemExit(f"[ABORT] no stage-1 result under {a.results_dir}")

    # ── gammas and case rates, per b ──────────────────────────────────────────
    print(f"\n{'b':>3} {'tuiles':>7} {'cas A':>7} {'cas B':>7} "
          f"{'gamma_lemme':>12} {'gamma_larg.moy':>15} {'ecart median':>13}")
    print("-" * 72)
    report = {"tol": a.tol, "per_b": {}}
    for b in a.bits:
        T = [t for t in tiles if t["b"] == b]
        if not T:
            continue
        A = [t for t in T if t["gap"] <= a.tol]
        num_l = sum(t["V2"] for t in A);           den_l = sum(t["V2"] for t in T)
        num_m = sum(t["W"][t["c"]] for t in T);    den_m = sum(sum(t["W"]) for t in T)
        g_l = num_l / den_l if den_l else float("nan")
        g_m = num_m / den_m if den_m else float("nan")
        med = float(np.median([t["gap"] for t in T]))
        print(f"{b:>3} {len(T):>7} {len(A):>7} {len(T)-len(A):>7} "
              f"{g_l:>12.4f} {g_m:>15.4f} {med:>13.1e}")
        report["per_b"][str(b)] = {
            "n_tiles": len(T), "n_case_A": len(A), "n_case_B": len(T) - len(A),
            "gamma_lemma": g_l, "gamma_meanwidth_no_exclusion": g_m,
            "median_gap": med}

    print("\ngamma_larg.moy is (19) WITHOUT the zero-volume exclusion — d_0 = d "
          "\neverywhere, since the radii are what stage 2 is about to compute."
          "\ngamma_lemme is a LOWER bound: a split tile (case B) gets no credit.")

    if a.tol_scan:
        print(f"\nsensibilite au seuil (% de cas A) :")
        print("   tol   " + "".join(f"  b={b:<4}" for b in a.bits))
        for t in (1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10):
            row = f"  {t:.0e}  "
            for b in a.bits:
                T = [x for x in tiles if x["b"] == b]
                row += f"  {100*np.mean([x['gap'] <= t for x in T]):5.1f} " if T else "    --  "
            print(row)
        print("   a flat row means the verdict does not depend on where you cut.")

    # ── stage-2 task table ────────────────────────────────────────────────────
    caseB = [t for t in tiles if t["gap"] > a.tol]
    caseA = [t for t in tiles if t["gap"] <= a.tol]
    random.Random(a.seed).shuffle(caseA)
    xcheck = caseA[:a.n_xcheck]
    lines = ([f"{t['b']} {t['aug_idx']} caseB" for t in caseB] +
             [f"{t['b']} {t['aug_idx']} xcheck" for t in xcheck])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    report["n_case_B"] = len(caseB); report["n_xcheck"] = len(xcheck)
    Path(a.report).write_text(json.dumps(report, indent=1))

    print(f"\n{len(caseB)} tuiles en cas B + {len(xcheck)} de controle croise "
          f"= {len(lines)} taches -> {out}")
    print(f"rapport : {a.report}")
    print(f"\nlancer :\n    sbatch --array=0-{len(lines) - 1} {a.slurm}")


if __name__ == "__main__":
    main()
