#!/usr/bin/env python3
"""
list_failed_cheb.py — the Chebyshev LPs of the campaign that did not converge.

Writes one "<bits> <aug_idx>" line per tile containing at least one polytope with
status 'failed', for the retry experiment. The campaign ran with the default
solver (dual simplex, `highs`) and a 3600 s cap; 315 of the 589 b=4 LPs failed.

Two things are being separated by the retry, and they must not be confused:

  arm A, same settings   are the failures REPRODUCIBLE, or is there
                         non-determinism? (Sample 683 once failed after 252 s
                         and then solved in 172 s with no code change, which is
                         unexplained and rests on a single observation.)
  arm B, --lp_method auto   does INTERIOR POINT recover them? This one has a
                         measured basis: on CNN sample 0 at b=16, `highs` hit a
                         1800 s cap without converging while `highs-ipm` solved
                         the same LP in 472 s.

Caveat for either arm: at b=4 the polytope itself is not uniquely defined, since
10 neurons of 11 856 flip activation between float32 and float64. Some failures
are therefore genuine and no solver will change them.

    python scripts/list_failed_cheb.py --bits 4 --limit 20
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cheb_dir", default="results/chebyshev_cnn_gen150")
    ap.add_argument("--bits", type=int, nargs="+", default=[4])
    ap.add_argument("--limit", type=int, default=20,
                    help="how many tiles to retry (0 = all)")
    ap.add_argument("--out", default="data/gen150/retry_cheb_tasks.txt")
    a = ap.parse_args()

    rows, n_tiles, n_failed_poly = [], 0, 0
    for b in a.bits:
        for f in sorted((Path(a.cheb_dir) / f"b{b:02d}").glob("chebyshev_sample*.json"),
                        key=lambda p: int(p.stem.split("sample")[1])):
            d = json.loads(f.read_text())
            n_tiles += 1
            bad = [p for p in d["polytopes"] if p["status"] == "failed"]
            if bad:
                n_failed_poly += len(bad)
                rows.append((b, int(d["sample_idx"]), len(bad), len(d["polytopes"])))

    print(f"{n_tiles} tuiles examinees, {len(rows)} contiennent au moins un LP echoue, "
          f"{n_failed_poly} polytopes echoues au total")
    sel = rows[:a.limit] if a.limit else rows
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{b} {i}" for b, i, _, _ in sel) + "\n")
    print(f"{len(sel)} tuiles ecrites -> {out}")
    print(f"\nlancer :\n    sbatch --array=0-{len(sel)-1} --export=ALL,METHOD=highs,TAG=armA "
          f"slurms/retry_cheb.slurm"
          f"\n    sbatch --array=0-{len(sel)-1} --export=ALL,METHOD=auto,TAG=armB "
          f"slurms/retry_cheb.slurm")


if __name__ == "__main__":
    main()
