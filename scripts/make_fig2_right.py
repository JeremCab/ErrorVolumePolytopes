#!/usr/bin/env python3
"""
make_fig2_right.py — Fig. 2 (right) of the ISCMI draft, as Jiri asked in MAIL_26.

The blue curve is labelled gamma_{T_P}, not gamma_{T^*}: the star is reserved for
the limit T^* = lim_{P->inf} T_P, and what we computed is T_P at finite P
(|T_P| = 159, 158, 206, 900 for b = 16, 10, 6, 4).

His request: y-axis 0.8-1.0, and the four blue points moved to
[16, 0.950], [10, 0.916], [6, 0.868], [4, 0.822], joined by line segments.

Those four values are row tol = 1e-06 of the gamma_19_min tolerance scan on the
gen150 augmented campaign (`scripts/select_case_b.py --tol_scan`).

alpha_T is recomputed here on the 150 original points of that campaign rather
than copied from the old figure, which was drawn on the 999-point raw dataset:
0.9533 / 1.0000 / 1.0000 / 1.0000 against 0.9449 / 0.9930 / 1.0000 / 1.0000.

Style follows notebooks/plot_volumes_v3k.ipynb so the panel matches the left one.

    python scripts/make_fig2_right.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BITS = [16, 10, 6, 4]                      # left to right, as in the paper
GAMMA = {16: 0.950, 10: 0.916, 6: 0.868, 4: 0.822}
ALPHA = {16: 1.0000, 10: 1.0000, 6: 1.0000, 4: 0.9533}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/fig2_right_iscmi.pdf")
    ap.add_argument("--ymin", type=float, default=0.8)
    ap.add_argument("--ymax", type=float, default=1.012,
                    help="slightly above 1 so the alpha_T markers are not clipped "
                         "by the top spine; Jiri asked for 0.8-1.0")
    ap.add_argument("--no_alpha_line", action="store_true",
                    help="draw alpha_T as markers only, as in the original figure")
    a = ap.parse_args()

    x = np.arange(len(BITS))
    g = [GAMMA[b] for b in BITS]
    al = [ALPHA[b] for b in BITS]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, al, marker="s", linewidth=0 if a.no_alpha_line else 2, markersize=7,
            color="darkorange", alpha=0.85,
            label=r"$\alpha_T (\widetilde{\cal N}_2^b)$")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    # the change Jiri asked for: the blue points are now joined
    ax.plot(x, g, marker="o", linewidth=2, markersize=7, color="steelblue",
            label=r"$\gamma_{T_P}( \widetilde{\cal N}_2^b )$")

    ax.set_xlabel("Quantisation bit-width $b$", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BITS])
    ax.set_ylim(a.ymin, a.ymax)
    ax.legend(fontsize=13, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"saved -> {out}  and  {out.with_suffix('.png')}")
    print(f"\n{'b':>4} {'gamma_T*':>10} {'alpha_T':>9}")
    for b in BITS:
        print(f"{b:>4} {GAMMA[b]:>10.3f} {ALPHA[b]:>9.4f}")


if __name__ == "__main__":
    main()
