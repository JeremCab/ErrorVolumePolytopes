#!/usr/bin/env python3
"""
list_b04_tiles.py — the b=4 tiles to recompute with interior point.

The campaign ran the Chebyshev LPs with the dual simplex and got 274 of 589
polytopes at b=4. The retry experiment showed those failures are reproducible
with the same solver (1 tile of 20 recovered) and almost all recoverable with
interior point (61/63, i.e. 97%). This lists the tiles to redo.

By default it lists only the tiles that contain at least one failed polytope,
since the converged ones need nothing; pass --all to redo the whole bitwidth,
which is the cleaner option if the result is to go into the paper, so that every
radius comes from the same solver.

    python scripts/list_b04_tiles.py            # only the incomplete tiles
    python scripts/list_b04_tiles.py --all      # all 186
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cheb_dir", default="results/chebyshev_cnn_gen150")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="data/gen150/b04_ipm_tasks.txt")
    a = ap.parse_args()

    rows, npoly, nok = [], 0, 0
    for f in sorted((Path(a.cheb_dir) / f"b{a.bits:02d}").glob("chebyshev_sample*.json"),
                    key=lambda p: int(p.stem.split("sample")[1])):
        j = json.loads(f.read_text())
        bad = sum(1 for q in j["polytopes"] if q["status"] == "failed")
        npoly += len(j["polytopes"]); nok += len(j["polytopes"]) - bad
        if a.all or bad:
            rows.append((a.bits, int(j["sample_idx"])))

    print(f"campagne b={a.bits} : {npoly} polytopes, {nok} converges "
          f"({100*nok/npoly:.0f}%)")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{b} {i}" for b, i in rows) + "\n")
    print(f"{len(rows)} tuiles a recalculer -> {out}")
    print(f"\n    sbatch --array=0-{len(rows)-1} slurms/complete_b04_ipm.slurm")


if __name__ == "__main__":
    main()
