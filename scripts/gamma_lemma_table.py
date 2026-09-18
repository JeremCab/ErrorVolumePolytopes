#!/usr/bin/env python3
"""
gamma_lemma_table.py — Table 1 of the Jiri mail, on the NON-augmented datasets.

Same quantity and same layout as `select_case_b.py --tol_scan` produces for the
augmented campaign, but read from the local CSVs of the raw dataset T (999 CNN
and 989 MLP points) and over all six bitwidths instead of four. The question it
answers: does the tolerance at which the Lemma is applied set the sign of the
trend here too, and is there a tolerance at which the CNN curve comes out the
way the theory predicts?

Columns
-------
no excl.  (19) with d_0 = d: every non-empty subpolytope is counted. The only
          column with no free parameter; the exclusion (18) is simply not
          applied.
tol       (19) with (18) applied wherever the Lemma proves the volume is zero.
          For each point, k* = argmax_k d(P3^k) is the only class that can fill
          P2, since the P3^k partition it. If |d(P3^k*) - d(P2)|/d(P2) <= tol
          then P3^k* = P2 by the Lemma, every other subpolytope has empty
          interior, and the point contributes d(P3^k*) to the denominator and
          the same to the numerator iff k* = c. Otherwise the Lemma is silent
          and nothing is excluded for that point.

CAVEAT worth keeping in mind when reading the result: on the raw T the
reference domain changes with b, since P2 depends on the approximated network.
That is the comparability problem the augmentation was introduced to solve, so
these numbers reproduce Fig. 2 rather than replace it.

Usage
-----
    python scripts/gamma_lemma_table.py
    python scripts/gamma_lemma_table.py --transpose
"""
import argparse

import numpy as np
import pandas as pd

BITS = [4, 6, 8, 10, 12, 16]
TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
K = 10


def rows_for(df, b):
    """(c, V2, W) per point, NaN read as 0."""
    c = df["class_c"].to_numpy(dtype=int)
    V2 = np.nan_to_num(df[f"V2_b{b}"].to_numpy(dtype=float))
    W = np.nan_to_num(np.column_stack(
        [df[f"V3_b{b}_k{k}"].to_numpy(dtype=float) for k in range(K)]))
    return c, V2, W


def gammas(df, b, tols):
    """Return no-exclusion gamma, {tol: gamma}, {tol: case-A rate}, ACC."""
    c, V2, W = rows_for(df, b)
    ok = V2 > 0
    c, V2, W = c[ok], V2[ok], W[ok]
    ks = W.argmax(axis=1)
    idx = np.arange(len(c))
    Wc, Wks = W[idx, c], W[idx, ks]
    Wsum = W.sum(axis=1)

    g_none = Wc.sum() / Wsum.sum() if Wsum.sum() else np.nan
    acc = float((ks == c).mean())

    out, rate = {}, {}
    gap = np.abs(Wks - V2) / V2
    for t in tols:
        A = gap <= t                       # the Lemma settles (18) here
        num = np.where(A, np.where(ks == c, Wks, 0.0), Wc).sum()
        den = np.where(A, Wks, Wsum).sum()
        out[t] = num / den if den else np.nan
        rate[t] = float(A.mean())
    return g_none, out, rate, acc, len(c)


def trend(vals, eps=2e-3):
    """Monotone in b? Steps below eps are read as ties, not as reversals — at
    solver precision a difference of 1e-4 in gamma is not a change of direction,
    and treating it as one made the MLP, which is flat at 0.998, look erratic."""
    d = np.diff(vals)
    d = d[np.abs(d) > eps]
    if d.size == 0:
        return "flat"
    if (d > 0).all():
        return "incr."
    if (d < 0).all():
        return "decr."
    return "neither"


def table(df, name, tols, transpose):
    res = {b: gammas(df, b, tols) for b in BITS}
    n = res[BITS[0]][4]
    print(f"\n{'='*84}\n{name}  (n = {n} points, non-augmented T)\n{'='*84}")

    head = ["no excl."] + [f"{t:.0e}" for t in tols]
    cols = [[res[b][0] for b in BITS]] + [[res[b][1][t] for b in BITS] for t in tols]

    if transpose:                                   # b in columns, tol in rows
        print(f"{'':>10}" + "".join(f"{'b=' + str(b):>9}" for b in BITS))
        print("-" * (10 + 9 * len(BITS)))
        for lab, col in zip(head, cols):
            print(f"{lab:>10}" + "".join(f"{v:>9.4f}" for v in col))
    else:                                           # b in rows, tol in columns
        print(f"{'b':>3} {'no excl.':>10}" + "".join(f"{h:>9}" for h in head[1:]))
        print("-" * (14 + 9 * len(tols)))
        for i, b in enumerate(BITS):
            print(f"{b:>3}" + "".join(f"{col[i]:>10.4f}" if j == 0 else
                                      f"{col[i]:>9.4f}" for j, col in enumerate(cols)))
        print("-" * (14 + 9 * len(tols)))
        print(f"{'trend':>3}" + "".join(f"{trend(col):>10}" if j == 0 else
                                        f"{trend(col):>9}" for j, col in enumerate(cols)))

    print(f"\n{'b':>3} {'ACC':>8}   % of points where the Lemma applies")
    print(f"{'':>12}" + "".join(f"{f'{t:.0e}':>9}" for t in tols))
    for b in BITS:
        print(f"{b:>3} {res[b][3]:>8.4f}" + "".join(f"{100*res[b][2][t]:>8.1f}%"
                                                    for t in tols))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnn", default="results/data_for_jiri/results_df_cnn.csv")
    ap.add_argument("--mlp", default="results/data_for_jiri/results_df_mlp.csv")
    ap.add_argument("--tols", type=float, nargs="+", default=TOLS)
    ap.add_argument("--transpose", action="store_true",
                    help="b in columns and tolerance in rows")
    a = ap.parse_args()
    for name, f in (("MLP  (N_1)", a.mlp), ("CNN  (N_2)", a.cnn)):
        table(pd.read_csv(f), name, a.tols, a.transpose)
    print("\n'trend' reads DOWN the column, i.e. from b=4 to b=16; steps below 0.002"
          "\nare read as ties. The expected behaviour is 'incr.': the finer the"
          "\nquantisation, the higher the GACC.")


if __name__ == "__main__":
    main()
