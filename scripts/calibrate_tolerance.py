#!/usr/bin/env python3
"""
calibrate_tolerance.py — which tolerance in (18) reproduces the actual volume?

The four blue points of Fig. 2 (right) come from the row tol = 1e-06 of the
tolerance scan over the gen150 campaign. Nothing in that campaign designates
1e-06 rather than 1e-08, and the choice decides the shape of the curve: at 1e-06
it increases, at 1e-08 it is flat, at 1e-10 it decreases. This script fixes the
tolerance BY MEASUREMENT instead of by convention.

The idea. For one original point x, the tiles Xi_y of T_x partition Xi-bar_x, so

    sum_y vol(Xi~_y^c) / sum_y vol(Xi_y)  =  vol({z in Xi-bar_x : N~(z) = c}) / vol(Xi-bar_x)

whenever the tiles cover Xi-bar_x. The right-hand side is exactly what
`volume_ratio_hitrun.py --polytope p1` measures, with no mean width and no
threshold. So the directly sampled value IS the quantity gamma_{T_P} estimates,
and comparing the two says which tolerance is right.

Two honest caveats, both reported in the output.

  - The tiles do NOT cover Xi-bar_x perfectly, especially at b=4 where only five
    representatives were kept out of the 500 found. Any residual gap between the
    sampled value and gamma_{T_P} at the best tolerance therefore mixes the
    tolerance with the coverage.
  - gamma_{T_P} is weighted by mean widths, which we know cannot rank tiles.
    Sampling gives per-point ratios and no absolute volumes, so the sampled
    aggregate is an unweighted mean. Both are printed; if they disagree, the
    weighting is doing the work and neither number should be trusted alone.

Usage
-----
    python scripts/calibrate_tolerance.py --bits 4 6 10 16
"""
import argparse, json, re
from pathlib import Path

import numpy as np

TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]


def load_stage1(results_dir, bits):
    """{b: [tile records]} from the gen150 mean-width campaign."""
    out = {}
    for b in bits:
        recs = []
        for f in sorted((Path(results_dir) / f"b{b:02d}").glob("volumes_sample*.json")):
            j = json.loads(f.read_text())
            sb = str(b)
            V2 = j.get("widths_correct", {}).get(sb)
            W = j.get("widths_both", {}).get(sb)
            if V2 is None or W is None or not np.isfinite(V2) or V2 <= 0:
                continue
            W = [0.0 if (x is None or not np.isfinite(x)) else float(x) for x in W]
            recs.append({"c": int(j["class_c"]), "V2": float(V2), "W": W,
                         "aug": int(f.stem.split("sample")[1])})
        out[b] = recs
    return out


def gamma_TP(recs, tol):
    """(19) with (18) settled by the lemma at this tolerance, as in select_case_b."""
    num = den = 0.0
    for t in recs:
        W, V2, c = t["W"], t["V2"], t["c"]
        ks = int(np.argmax(W))
        gap = abs(W[ks] - V2) / V2
        if tol is not None and gap <= tol:
            den += W[ks]
            if ks == c:
                num += W[ks]
        else:
            num += W[c]
            den += sum(W)
    return num / den if den else float("nan")


def load_sampled(vol_dir, logs, bits, max_idx):
    """{b: {orig_idx: sampled fraction}} from the P1 walks over the originals."""
    out = {b: {} for b in bits}
    for lg in logs:
        p = Path(lg)
        if not p.exists():
            continue
        for bk in p.read_text().split("P2 built in")[1:]:
            m = re.search(r"sample (\d+), class c = (\d+)", bk)
            tab = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+\[", bk, re.M)
            if not m or not tab:
                continue
            i = int(m.group(1))
            if i >= max_idx:
                continue
            for b, v in tab:
                if int(b) in out:
                    out[int(b)][i] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1_dir", default="results/volumes_v3k_cnn_gen150")
    ap.add_argument("--vol_dir", default="results/volume_ratio")
    ap.add_argument("--logs", nargs="+",
                    default=["logs/vr_gen150_originals.log", "logs/vr_cnn_s7_P1.log",
                             "logs/vr_cnn_batch_P1.log", "logs/vr_cnn_random_P1.log",
                             "logs/vr_cnn_random2_P1.log"])
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 10, 16])
    ap.add_argument("--n_orig", type=int, default=150)
    a = ap.parse_args()

    s1 = load_stage1(a.stage1_dir, a.bits)
    sm = load_sampled(a.vol_dir, a.logs, a.bits, a.n_orig)

    print(f"\n{'='*78}\nCALIBRATION DE LA TOLERANCE — campagne gen150, CNN\n{'='*78}")
    n_s = len(sm[a.bits[0]])
    print(f"points originaux echantillonnes dans P1 : {n_s}/{a.n_orig}")
    print("tuiles par b :", {b: len(s1[b]) for b in a.bits})
    if n_s < a.n_orig:
        print(f"⚠  incomplet — relancer scripts/run_p1_gen150.sh avant de conclure")

    print(f"\n{'':>22}" + "".join(f"{'b='+str(b):>10}" for b in a.bits))
    print("-" * (22 + 10 * len(a.bits)))
    ideal = {b: (np.mean(list(sm[b].values())) if sm[b] else float("nan")) for b in a.bits}
    se = {b: (np.std(list(sm[b].values())) / np.sqrt(max(len(sm[b]), 1))
              if sm[b] else float("nan")) for b in a.bits}
    print(f"{'VOLUME mesure':>22}" + "".join(f"{ideal[b]:>10.4f}" for b in a.bits))
    print(f"{'  erreur-type':>22}" + "".join(f"{se[b]:>10.4f}" for b in a.bits))
    print("-" * (22 + 10 * len(a.bits)))
    best, best_err = None, np.inf
    for tol in TOLS:
        g = {b: gamma_TP(s1[b], tol) for b in a.bits}
        err = np.nanmean([abs(g[b] - ideal[b]) for b in a.bits])
        if err < best_err:
            best, best_err = tol, err
        print(f"{'gamma_TP tol=' + f'{tol:.0e}':>22}"
              + "".join(f"{g[b]:>10.4f}" for b in a.bits)
              + f"   ecart moyen {err:.4f}")
    g0 = {b: gamma_TP(s1[b], None) for b in a.bits}
    print(f"{'gamma_TP sans (18)':>22}" + "".join(f"{g0[b]:>10.4f}" for b in a.bits))
    print("-" * (22 + 10 * len(a.bits)))
    print(f"\n==> tolerance la plus proche du volume mesure : {best:.0e} "
          f"(ecart moyen {best_err:.4f})")
    print("\n   A lire avec les deux reserves du docstring : la couverture de P1 par"
          "\n   les tuiles est partielle a b=4, et gamma_TP est pondere par des largeurs"
          "\n   moyennes alors que le volume mesure est une moyenne non ponderee.")


if __name__ == "__main__":
    main()
