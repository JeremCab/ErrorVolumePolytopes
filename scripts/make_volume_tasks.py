#!/usr/bin/env python3
"""
make_volume_tasks.py — build the (bits, aug_idx) task table for the volume array.

The number of tiles is not known until the generation has run: b=4 yields
~6 points per original while b>=10 yields 1. Hard-coding cumulative ranges in
the SLURM script (as the sample-760 one did) breaks the moment a count changes.
This reads the merged provenance files instead and writes one line per task,

    <bits> <aug_idx>

so the SLURM script only has to pick line SLURM_ARRAY_TASK_ID+1. It also prints
the exact sbatch command, array bounds included, so the size cannot be mistyped.

Usage
-----
    python scripts/make_volume_tasks.py --bits 4 6 10 16
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 10, 16])
    ap.add_argument("--prefix",
                    default="data/fashionMNIST_augmented_cnn_seed42_walk_gen150")
    ap.add_argument("--out", default="data/gen150/volume_tasks.txt")
    ap.add_argument("--slurm", default="slurms/run_volumes_gen150_cnn.slurm")
    a = ap.parse_args()

    lines, summary = [], []
    for b in a.bits:
        prov = Path(f"{a.prefix}_b{b}_provenance.json")
        if not prov.exists():
            raise SystemExit(f"[ABORT] {prov} missing — run merge_aug_shards.py first.")
        meta = json.loads(prov.read_text())
        if meta["missing_orig_idx"]:
            raise SystemExit(f"[ABORT] b={b}: the merged dataset is incomplete "
                             f"({len(meta['missing_orig_idx'])} original points missing). "
                             f"Relaunch the generation array before computing volumes.")
        n = meta["n_points"]
        lines += [f"{b} {i}" for i in range(n)]
        summary.append((b, meta["n_orig_merged"], n, meta["avg_reps"]))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    print(f"{'b':>3} {'originaux':>10} {'tuiles':>8} {'reps moy':>9}")
    print("-" * 34)
    for b, n_orig, n, avg in summary:
        print(f"{b:>3} {n_orig:>10} {n:>8} {avg:>9.2f}")
    print(f"{'':>3} {'TOTAL':>10} {len(lines):>8}")
    print(f"\ntable écrite : {out}")
    print(f"\nlancer :\n    sbatch --array=0-{len(lines) - 1} {a.slurm}")


if __name__ == "__main__":
    main()
