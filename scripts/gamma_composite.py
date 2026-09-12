#!/usr/bin/env python3
"""
gamma_composite.py — (19) with (18) applied to EVERY tile, per bit-width.

Why this script exists. Neither existing script computes (19) in full:
`select_case_b.py` never looks at a radius, so it leaves the case-B tiles
un-excluded and calls the result a lower bound; `gamma_measured.py` uses the
radii but only on the tiles that went to stage 2, which are the case-B ones, so
it reports a selected subpopulation. Here the two halves are joined:

  case A (some class fills P2)  -> (18) settled by the lemma, no LP needed
  case B (no class fills P2)    -> (18) settled by the measured Chebyshev radii

At b = 6, 10 and 16 this covers EVERY tile — 181+25, 148+10, 139+20 — because
stage 2 converged 100% there. So gamma is no longer a bound and no longer a
subsample: it is (19) evaluated as written, for the first time. At b = 4 the
coverage is partial and the script says so rather than hiding it.

The script also answers the question that decides whether the threshold in (18)
is defensible at all: is the distribution of the radii BIMODAL? If log10(rho)
splits into a mode at zero volume and a mode at positive volume with a wide gap
between them, one cuts in the gap and shows gamma is invariant across two or
three decades — the threshold then stops being a free parameter and becomes an
observation. If it is a continuum, no threshold is defensible, and the problem
is the size measure rather than where one cuts.

Work on the RATIO rho(P3^k)/rho(P2), not on rho itself: rho grows by two orders
of magnitude between b=4 and b=16, so an absolute scale cannot be compared
across bitwidths, whereas each tile normalised by its own P2 can.

Needs numpy only. On a Jean Zay login node: module load pytorch-gpu/py3/2.8.0

Usage
-----
    python scripts/gamma_composite.py --bits 6 10 16
    python scripts/gamma_composite.py --bits 4 6 10 16 --relative
"""
import argparse
import json
from pathlib import Path

import numpy as np


# ── loading ────────────────────────────────────────────────────────────────
def load_stage1(results_dir, bits):
    """{b: {aug_idx: tile}} — mean widths, the single source of truth for d."""
    out = {}
    for b in bits:
        d, root = {}, Path(results_dir) / f"b{b:02d}"
        for f in sorted(root.glob("volumes_sample*.json"),
                        key=lambda p: int(p.stem.split("sample")[1])):
            j = json.loads(f.read_text())
            sb = str(b)
            V2 = j.get("widths_correct", {}).get(sb)
            W = j.get("widths_both", {}).get(sb)
            if V2 is None or W is None or not np.isfinite(V2) or V2 <= 0:
                continue
            W = [0.0 if (x is None or not np.isfinite(x)) else float(x) for x in W]
            aug = int(f.stem.split("sample")[1])
            ks = int(np.argmax(W))
            d[aug] = dict(b=b, aug=aug, c=int(j["class_c"]), V2=float(V2), W=W,
                          ks=ks, gap=abs(W[ks] - V2) / V2)
        out[b] = d
    return out


def load_radii(cheb_dir, bits):
    """{b: {aug_idx: {"r": {"P2"|k: radius}, "failed": bool}}}

    A polytope whose LP did not converge makes the whole tile unusable: (18)
    cannot be applied to it, and pretending a missing radius means zero volume
    would silently drop the class from the denominator of (19).
    """
    out = {b: {} for b in bits}
    for b in bits:
        root = Path(cheb_dir) / f"b{b:02d}"
        if not root.is_dir():
            continue
        for f in sorted(root.glob("chebyshev_sample*.json")):
            j = json.loads(f.read_text())
            rec = {"r": {}, "failed": False}
            for q in j["polytopes"]:
                if q["status"] == "failed" or q["radius"] is None:
                    rec["failed"] = True
                    continue
                key = "P2" if q["polytope"] == "P2" else int(q["k"])
                rec["r"][key] = float(q["radius"])
            out[b][int(j["sample_idx"])] = rec
    return out


def load_rho_p2(rho_dir, bits):
    """{b: {aug_idx: rho(P2)}} — the --p2_only campaign, one LP per tile."""
    out = {b: {} for b in bits}
    for b in bits:
        root = Path(rho_dir) / f"b{b:02d}"
        if not root.is_dir():
            continue
        for f in sorted(root.glob("chebyshev_sample*.json")):
            j = json.loads(f.read_text())
            p2 = next((q for q in j["polytopes"] if q["polytope"] == "P2"), None)
            if p2 and p2["status"] != "failed" and p2["radius"] is not None:
                out[b][int(j["sample_idx"])] = float(p2["radius"])
    return out


# ── the composite gamma ────────────────────────────────────────────────────
def composite(tiles, radii, rho2, tol, zt, relative=False, p2_tol=None):
    """(19) with (18) settled by the lemma on case A and by rho on case B.

    zt is a threshold on rho; with `relative` it is a threshold on
    rho(P3^k)/rho(P2) instead, which is dimensionless and therefore comparable
    across bitwidths. p2_tol, if given, first drops any tile whose own P2 is
    flat: vol(P2) = 0 makes every subpolytope of it zero-volume too, so the
    tile carries no weight in a volume-based accuracy — while carrying a full
    one in the mean-width surrogate.

    Returns a dict; `open_share` is the fraction of the un-excluded denominator
    sitting in tiles that could not be resolved, i.e. how much of the answer is
    still missing.
    """
    num = den = 0.0
    den_all = open_all = 0.0          # denominators WITHOUT any exclusion
    n = dict(lemma=0, cheb=0, open=0, flat=0)

    for t in tiles.values():
        w_all = sum(t["W"])
        den_all += w_all

        if p2_tol is not None:
            r2 = rho2.get(t["aug"])
            if r2 is not None and r2 <= p2_tol:
                n["flat"] += 1                    # whole tile has zero volume
                continue

        if t["gap"] <= tol:
            # Case A: P3^ks = P2, every other subpolytope has empty interior.
            w = t["W"][t["ks"]]
            den += w
            if t["ks"] == t["c"]:
                num += w
            n["lemma"] += 1
            continue

        rec = radii.get(t["aug"])
        if rec is None or rec["failed"]:
            n["open"] += 1
            open_all += w_all
            continue

        r2 = rec["r"].get("P2")
        if relative and (r2 is None or r2 <= 0):
            n["open"] += 1                        # no scale to normalise by
            open_all += w_all
            continue

        n["cheb"] += 1
        for k, w in enumerate(t["W"]):
            if w <= 0:
                continue
            r = rec["r"].get(k)
            if r is None:
                continue                          # class absent from stage 2
            if r <= (zt * r2 if relative else zt):
                continue                          # d_0 = 0
            den += w
            if k == t["c"]:
                num += w

    return dict(gamma=num / den if den else float("nan"),
                open_share=open_all / den_all if den_all else 0.0, **n)


# ── text histogram ─────────────────────────────────────────────────────────
def histogram(vals, label, lo=-9.0, hi=0.5, step=0.5, width=46):
    """log10 histogram. Non-positive values get their own row: a radius <= 0 is
    the solver's numerical zero, and it is exactly what (18) is trying to find."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        print(f"  {label}: (aucune valeur)")
        return
    nonpos = int((v <= 0).sum())
    pos = np.log10(v[v > 0])
    edges = np.arange(lo, hi + step / 2, step)
    counts, _ = np.histogram(np.clip(pos, lo, hi - 1e-9), bins=edges)
    top = max(counts.max(), nonpos, 1)
    print(f"  {label}  (n = {v.size}, dont {nonpos} <= 0)")
    if nonpos:
        print(f"     rho <= 0 | {'#' * int(width * nonpos / top):<{width}} {nonpos}")
    for i, cnt in enumerate(counts):
        if cnt == 0:
            continue
        print(f"    10^{edges[i]:>5.1f}  | {'#' * int(width * cnt / top):<{width}} {cnt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1_dir", default="results/volumes_v3k_cnn_gen150")
    ap.add_argument("--cheb_dir",   default="results/chebyshev_cnn_gen150")
    ap.add_argument("--rho_dir",    default="results/rho_p2_cnn_gen150")
    ap.add_argument("--bits", type=int, nargs="+", default=[6, 10, 16])
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="relative gap on mean widths deciding case A vs case B")
    ap.add_argument("--relative", action="store_true",
                    help="threshold rho(P3^k)/rho(P2) instead of rho itself")
    ap.add_argument("--p2_tol", type=float, default=None,
                    help="also drop tiles whose own rho(P2) is below this")
    a = ap.parse_args()

    tiles = load_stage1(a.stage1_dir, a.bits)
    radii = load_radii(a.cheb_dir, a.bits)
    rho2 = load_rho_p2(a.rho_dir, a.bits)
    if not any(tiles.values()):
        raise SystemExit(f"[ABORT] aucun resultat d'etape 1 sous {a.stage1_dir}")

    ZT = ([1e-3, 1e-2, 1e-1, 3e-1] if a.relative
          else [0.0, 1e-8, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4])
    kind = "rho(P3^k)/rho(P2)" if a.relative else "rho(P3^k)"

    # ── 1. coverage ────────────────────────────────────────────────────────
    print(f"\n{'='*78}\nCOUVERTURE  (cas A regle par le lemme, cas B par Chebyshev)\n{'='*78}")
    print(f"{'b':>3} {'tuiles':>7} {'par lemme':>10} {'par rho':>9} {'non resolues':>13} "
          f"{'% du denominateur non resolu':>30}")
    for b in a.bits:
        if not tiles[b]:
            continue
        r = composite(tiles[b], radii[b], rho2[b], a.tol, ZT[1], a.relative, a.p2_tol)
        print(f"{b:>3} {len(tiles[b]):>7} {r['lemma']:>10} {r['cheb']:>9} "
              f"{r['open']:>13} {100*r['open_share']:>29.1f}%")
    print("   Une couverture de 100% signifie que (19) est evaluee telle qu'ecrite,"
          "\n   sans borne et sans sous-echantillon.")

    # ── 2. gamma vs threshold ──────────────────────────────────────────────
    print(f"\n{'='*78}\nGAMMA COMPOSITE vs seuil sur {kind}\n{'='*78}")
    hdr = f"{'b':>3} " + "".join(
        f"{('aucun' if z == 0 else f'{z:.0e}'):>10}" for z in ZT)
    print(hdr); print("-" * len(hdr))
    for b in a.bits:
        if not tiles[b]:
            continue
        row = f"{b:>3} "
        for z in ZT:
            g = composite(tiles[b], radii[b], rho2[b], a.tol, z,
                          a.relative, a.p2_tol)["gamma"]
            row += f"{g:>10.4f}"
        print(row)
    print("   Lire les COLONNES : chacune est une courbe candidate pour la Fig. 2 (droite)."
          "\n   Une ligne plate sur plusieurs decennies = le seuil n'est plus un parametre"
          "\n   libre mais une observation. C'est la demonstration que Jiri attend.")

    # ── 3. the distributions — is there a gap to cut in? ────────────────────
    print(f"\n{'='*78}\nDISTRIBUTION DES RAYONS  (tuiles de cas B, la ou rho est mesure)\n{'='*78}")
    print("   Les classes correcte et incorrectes sont separees a dessein : si la"
          "\n   bimodalite est 'correcte vs incorrecte', c'est celle qui nous interesse.")
    for b in a.bits:
        abs_c, abs_w, rel_c, rel_w = [], [], [], []
        for t in tiles[b].values():
            rec = radii[b].get(t["aug"])
            if rec is None or rec["failed"]:
                continue
            r2 = rec["r"].get("P2")
            for k, w in enumerate(t["W"]):
                r = rec["r"].get(k)
                if w <= 0 or r is None:
                    continue
                (abs_c if k == t["c"] else abs_w).append(r)
                if r2 and r2 > 0:
                    (rel_c if k == t["c"] else rel_w).append(r / r2)
        print(f"\n--- b = {b} ---")
        histogram(abs_w, "log10 rho(P3^k), classes INCORRECTES")
        histogram(abs_c, "log10 rho(P3^c), classe correcte")
        histogram(rel_w, "log10 rho(P3^k)/rho(P2), classes INCORRECTES", lo=-6.0)
        histogram(rel_c, "log10 rho(P3^c)/rho(P2), classe correcte", lo=-6.0)

    # ── 4. rho(P2) over every tile ─────────────────────────────────────────
    print(f"\n{'='*78}\nRHO(P2) SUR TOUTES LES TUILES  (campagne --p2_only)\n{'='*78}")
    print(f"{'b':>3} {'tuiles':>7} {'rho(P2)<=0':>11} {'p10':>10} {'p50':>10} {'p90':>10}")
    for b in a.bits:
        v = np.array([rho2[b][i] for i in tiles[b] if i in rho2[b]])
        if v.size == 0:
            print(f"{b:>3} {0:>7}   (aucun resultat rho_p2)"); continue
        print(f"{b:>3} {v.size:>7} {int((v <= 0).sum()):>11} "
              f"{np.percentile(v,10):>10.2e} {np.percentile(v,50):>10.2e} "
              f"{np.percentile(v,90):>10.2e}")
    print("   Une tuile dont P2 est plat a un volume nul, donc AUCUN poids dans une"
          "\n   precision volumique — mais un poids plein dans le substitut largeur"
          "\n   moyenne. L'option --p2_tol les retire.")


if __name__ == "__main__":
    main()
