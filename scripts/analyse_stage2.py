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
    ap.add_argument("--n_dim", type=int, default=784,
                    help="input dimension, for the volume-faithful rho^n weighting")
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

    # ── 2b. the same verdicts across a range of zero_tol ──────────────────────
    # Jiri (MAIL_23, 5 Sep 2026) argues the small radii are numerical error
    # accumulated over hundreds of thousands of float32 operations while the true
    # radius is zero, and that (18) should be implemented with eps = 5e-5 or even
    # 1e-4. Our default of 1e-6 is a hundred times stricter, and the measured radii
    # sit exactly in the disputed band — so the verdict must be reported as a
    # function of the threshold, not at one value of it.
    print(f"\n{'='*62}\nVERDICTS vs zero_tol  (% de sous-polytopes incorrects NULS)\n{'='*62}")
    TOLS = (1e-8, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4)
    print(f"{'zero_tol':>10} {'caseB':>16} {'xcheck':>16}   (le lemme predit 100% en xcheck)")
    for zt in TOLS:
        row = f"{zt:>10.0e}"
        for why in ("caseB", "xcheck"):
            T = [t for t in tiles if t["why"] == why
                 and all(p["status"] != "failed" for p in t["polys"])]
            inc = [p for t in T for p in t["polys"]
                   if p["polytope"] != "P2" and p["k"] != t["c"]]
            if not inc:
                row += f"{'--':>16}"; continue
            frac = np.mean([p["radius"] is not None and p["radius"] <= zt for p in inc])
            n_all = sum(1 for t in T
                        if (i := [p for p in t["polys"]
                                  if p["polytope"] != "P2" and p["k"] != t["c"]])
                        and all(p["radius"] is not None and p["radius"] <= zt for p in i))
            row += f"{100*frac:>9.1f}% ({n_all:>3})"
        print(row)
    print("   (n) = nombre de tuiles ou TOUS les incorrects sont nuls a ce seuil")

    # ── 2c. four ways of weighting the same subpolytopes ──────────────────────
    # (18) is a binary gate on a continuous quantity: it zeroes a subpolytope of
    # exactly zero volume and keeps one of nearly zero volume at full weight. The
    # continuous alternative is to weight by rho, which goes to 0 with the volume
    # and needs no threshold at all — Jiri's "free parameter" objection disappears.
    #
    # What each weight can actually see, for a body similar to a ball of radius r
    # (d = 2r, rho = r, V ∝ r^n, n = 784):
    #   d          ∝ V^(1/n)   — a volume ratio of 2600 shows up as 1%
    #   d.rho      ∝ V^(2/n)   — the same ratio shows up as 2%
    #   rho        ∝ V^(1/n)   — same blindness as d, but it does vanish with volume
    #   rho^n      ∝ V         — faithful, and numerically explosive: a 1% error on
    #                            rho becomes a factor 2400 on the weight
    # Only the last is volume-faithful, and it is the least usable. That tension is
    # the point of the table, not an implementation detail.
    def _lse(logs):
        logs = np.asarray([l for l in logs if np.isfinite(l)])
        if logs.size == 0:
            return -np.inf
        m = logs.max()
        return m + np.log(np.exp(logs - m).sum())

    print(f"\n{'='*78}\nPONDERATIONS (tuiles completes ; P2 exclu, seuls les P3^k comptent)\n{'='*78}")
    print(f"{'b':>3} {'tuiles':>7} {'gamma[d]':>11} {'gamma[d.rho]':>14} "
          f"{'gamma[rho]':>12} {'gamma[rho^n]':>14} || "
          f"{'gamma_moy[d]':>14} {'gamma_moy[rho]':>16}")
    print("-" * 100)
    for b in a.bits:
        T = [t for t in tiles if t["b"] == b
             and all(p["status"] != "failed" for p in t["polys"])]
        if not T:
            print(f"{b:>3} {0:>7}   (aucune tuile complete)"); continue
        num = {k: 0.0 for k in ("d", "dr", "r")}
        den = {k: 0.0 for k in ("d", "dr", "r")}
        log_c, log_a = [], []
        # Per-tile ratios, averaged unweighted. Dividing a tile's weights by its
        # own total leaves that tile's ratio untouched and removes the weight it
        # carries relative to the others — so the ratio of sums becomes the mean of
        # ratios. Each term is dimensionless and in [0,1], hence comparable across
        # b without rho's scale entering. The case for it: a 1% difference in mean
        # width answers to a factor 2600 in volume, so weighting tiles by these
        # sizes ranks them on a quantity that cannot rank them, which injects noise
        # rather than information. The case against: it is no longer the volume
        # fraction of P1, and it is a change to (19), which Jiri wrote explicitly
        # as a size-weighted average.
        ratio_d, ratio_r = [], []
        for t in T:
            tn_d = td_d = tn_r = td_r = 0.0
            for p in t["polys"]:
                if p["polytope"] == "P2":
                    continue
                w = p["mean_width"] or 0.0
                r = p["radius"]
                r = r if (r is not None and r > 0) else 0.0
                correct = (p["k"] == t["c"])
                for key, val in (("d", w), ("dr", w * r), ("r", r)):
                    den[key] += val
                    if correct:
                        num[key] += val
                td_d += w; td_r += r
                if correct:
                    tn_d += w; tn_r += r
                if r > 0:
                    lv = a.n_dim * np.log(r)
                    log_a.append(lv)
                    if correct:
                        log_c.append(lv)
            if td_d > 0: ratio_d.append(tn_d / td_d)
            if td_r > 0: ratio_r.append(tn_r / td_r)
        g_vol = float(np.exp(_lse(log_c) - _lse(log_a))) if log_a else float("nan")
        f = lambda k: num[k] / den[k] if den[k] else float("nan")
        m = lambda v: float(np.mean(v)) if v else float("nan")
        print(f"{b:>3} {len(T):>7} {f('d'):>11.4f} {f('dr'):>14.4f} "
              f"{f('r'):>12.4f} {g_vol:>14.4f} || "
              f"{m(ratio_d):>14.4f} {m(ratio_r):>16.4f}")
    print("\n   gamma[d]     : la ponderation actuelle, sans exclusion"
          "\n   gamma[d.rho] : ponderation continue — remplace la porte binaire de (18)"
          "\n                  sans aucun seuil, mais ne voit le volume qu'en V^(2/n)"
          "\n   gamma[rho]   : rho comme mesure de taille a la place de la largeur"
          "\n   gamma[rho^n] : fidele au volume (logsumexp), et donc instable — une"
          "\n                  erreur de 1%% sur rho y devient un facteur 2400"
          "\n   gamma_moy[.] : MOYENNE des rapports par tuile, au lieu du rapport des"
          "\n                  sommes. Chaque tuile pese pareil ; les termes sont sans"
          "\n                  dimension et dans [0,1], donc comparables entre b sans que"
          "\n                  l'echelle de rho intervienne."
          "\n\n   Si gamma_moy diverge de gamma, c'est la PONDERATION PAR LA TAILLE qui"
          "\n   porte le resultat — or on sait qu'elle classe les tuiles sur une quantite"
          "\n   incapable de les classer (1%% de largeur = facteur 2600 en volume)."
          "\n   Reserve valable pour les deux : l'echantillon de tuiles n'est deja pas"
          "\n   proportionnel au volume (un representant par classe, puis 5 choisis par"
          "\n   diversite), donc aucune ponderation posterieure ne rend la vraie fraction"
          "\n   volumique de P1.")

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
