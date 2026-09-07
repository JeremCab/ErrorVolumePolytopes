#!/usr/bin/env python3
"""
analyse_rho_p2.py — what rho(P2) over the whole campaign tells us.

Three questions, in order of importance.

1. HOW MANY TILES ARE DEGENERATE. A tile whose correct polytope P2 is flat has
   zero volume, so it carries no weight in the volume-based accuracy — yet a full
   one in the mean-width surrogate, since a flat polytope keeps a mean width of
   order 1. Counting them says how much of the current gamma rests on tiles that
   should not be in it.

   The criterion is rho(P2)/V2, not rho(P2). rho alone is not comparable across
   bit-widths — it grows from ~0 at b=4 to 1.3e-3 at b=16, so a fixed absolute cut
   would keep nearly everything at b=16 and nearly nothing at b=4, a b-dependent
   selection that manufactures a trend. rho/V2 is dimensionless: it measures how
   fat the polytope is relative to its own extent.

2. IS THE rho(P3^k)/rho(P2) ANOMALY A 0/0 ARTEFACT? Stage 2 found that ratio ≈ 1
   at b=6 and b=10 — incorrect subpolytopes apparently as thick as P2, which the
   partition makes hard to accept. If the tiles where it happens are exactly the
   degenerate ones, the ratio is a badly conditioned 0/0 and the anomaly is
   explained. This joins the two datasets to find out.

3. GAMMA ON THE NON-DEGENERATE TILES ONLY, from the stage-1 mean widths, with the
   retained fraction reported per b — because a filter that keeps 90% at one
   bit-width and 40% at another is itself a trend.

Needs numpy only. On a Jean Zay login node: module load pytorch-gpu/py3/2.8.0

Usage
-----
    python scripts/analyse_rho_p2.py --bits 6 10 16
"""
import argparse, json
from pathlib import Path

import numpy as np


def load_tiles(rho_dir, s1_dir, bits):
    out = []
    for b in bits:
        for f in sorted((Path(rho_dir) / f"b{b:02d}").glob("chebyshev_sample*.json")):
            d = json.loads(f.read_text())
            p2 = next((p for p in d["polytopes"] if p["polytope"] == "P2"), None)
            if p2 is None:
                continue
            aug = d["sample_idx"]
            s1 = Path(s1_dir) / f"b{b:02d}" / f"volumes_sample{aug}.json"
            if not s1.exists():
                continue
            j = json.loads(s1.read_text())
            V2 = j.get("widths_correct", {}).get(str(b))
            W = j.get("widths_both", {}).get(str(b))
            if V2 is None or W is None or not np.isfinite(V2) or V2 <= 0:
                continue
            out.append(dict(b=b, aug=aug, c=j["class_c"], V2=float(V2),
                            W=[0.0 if (x is None or not np.isfinite(x)) else float(x)
                               for x in W],
                            rho=p2["radius"], ok=p2["status"] != "failed"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho_dir", default="results/rho_p2_cnn_gen150")
    ap.add_argument("--s1_dir",  default="results/volumes_v3k_cnn_gen150")
    ap.add_argument("--cheb_dir", default="results/chebyshev_cnn_gen150")
    ap.add_argument("--bits", type=int, nargs="+", default=[6, 10, 16])
    a = ap.parse_args()

    tiles = load_tiles(a.rho_dir, a.s1_dir, a.bits)
    if not tiles:
        raise SystemExit(f"[ABORT] rien sous {a.rho_dir}")

    # ── 1. convergence and the rho(P2)/V2 distribution ────────────────────────
    print(f"{'b':>3} {'tuiles':>7} {'converges':>10} | "
          f"{'rho(P2)  p10/p50/p90':>32} | {'rho/V2  p10/p50/p90':>30}")
    print("-" * 90)
    for b in a.bits:
        T = [t for t in tiles if t["b"] == b]
        ok = [t for t in T if t["ok"] and t["rho"] is not None]
        if not T: continue
        if not ok:
            print(f"{b:>3} {len(T):>7} {0:>10} | aucun LP resolu"); continue
        r = np.array([t["rho"] for t in ok])
        q = np.array([t["rho"] / t["V2"] for t in ok])
        f3 = lambda v: f"{np.percentile(v,10):.2e} {np.percentile(v,50):.2e} {np.percentile(v,90):.2e}"
        print(f"{b:>3} {len(T):>7} {len(ok):>10} | {f3(r):>32} | {f3(q):>30}")

    # ── 2. degeneracy rate, on the dimensionless criterion ────────────────────
    print(f"\n{'='*74}\nTAUX DE TUILES DEGENEREES  (rho(P2)/V2 <= seuil)\n{'='*74}")
    RELS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    print(f"{'seuil':>10}" + "".join(f"{'b=' + str(b):>14}" for b in a.bits))
    for rel in RELS:
        row = f"{rel:>10.0e}"
        for b in a.bits:
            ok = [t for t in tiles if t["b"] == b and t["ok"] and t["rho"] is not None]
            row += (f"{100*np.mean([t['rho']/t['V2'] <= rel for t in ok]):>13.1f}%"
                    if ok else f"{'--':>14}")
        print(row)

    # ── 3. is the stage-2 ratio anomaly a 0/0 ? ───────────────────────────────
    print(f"\n{'='*74}\nL'ANOMALIE rho(P3^k)/rho(P2) ~ 1 EST-ELLE UN 0/0 ?\n{'='*74}")
    print(f"{'b':>3} {'tuiles communes':>17} {'rho/V2 median':>15} "
          f"{'rapport median':>16}   correlation")
    for b in a.bits:
        rows = []
        for t in tiles:
            if t["b"] != b or not t["ok"] or t["rho"] is None or t["rho"] <= 0:
                continue
            f = Path(a.cheb_dir) / f"b{b:02d}" / f"chebyshev_sample{t['aug']}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            inc = [p["radius"] for p in d["polytopes"]
                   if p["polytope"] != "P2" and p["k"] != t["c"]
                   and p["status"] != "failed" and p["radius"] is not None]
            if not inc:
                continue
            rows.append((t["rho"] / t["V2"], float(np.median(inc)) / t["rho"]))
        if len(rows) < 3:
            print(f"{b:>3} {len(rows):>17}   (trop peu de tuiles communes)"); continue
        q = np.array([x[0] for x in rows]); rt = np.array([x[1] for x in rows])
        cc = float(np.corrcoef(np.log10(np.maximum(q, 1e-30)), rt)[0, 1])
        print(f"{b:>3} {len(rows):>17} {np.median(q):>15.2e} {np.median(rt):>16.2f}"
              f"   r(log rho/V2, rapport) = {cc:+.2f}")
    print("   une correlation NEGATIVE forte = les tuiles plates sont celles ou le"
          "\n   rapport vaut 1, donc l'anomalie est bien un 0/0 mal conditionne.")

    # ── 4. gamma once the degenerate tiles are dropped ────────────────────────
    print(f"\n{'='*74}\nGAMMA (largeurs moyennes) APRES FILTRAGE DES TUILES DEGENEREES\n{'='*74}")
    print(f"{'seuil rho/V2':>13}" + "".join(f"{'b=' + str(b):>20}" for b in a.bits))
    print(f"{'':>13}" + "".join(f"{'gamma  (% gardees)':>20}" for _ in a.bits))
    for rel in (0.0,) + RELS:
        row = f"{rel:>13.0e}" if rel else f"{'aucun':>13}"
        for b in a.bits:
            ok = [t for t in tiles if t["b"] == b and t["ok"] and t["rho"] is not None]
            keep = [t for t in ok if t["rho"] / t["V2"] > rel]
            if not keep:
                row += f"{'--':>20}"; continue
            num = sum(t["W"][t["c"]] for t in keep)
            den = sum(sum(t["W"]) for t in keep)
            g = num / den if den else float("nan")
            row += f"{g:>11.4f} ({100*len(keep)/len(ok):>4.0f}%)"
        print(row)
    print("   Lire la fraction gardee AUTANT que gamma : un filtre qui retient 90%"
          "\n   a un b et 40% a un autre est lui-meme une tendance.")


if __name__ == "__main__":
    main()
