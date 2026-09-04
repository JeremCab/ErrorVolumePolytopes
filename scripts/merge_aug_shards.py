#!/usr/bin/env python3
"""
merge_aug_shards.py — merge the per-task shards produced by gen_aug_150_cnn.slurm
into one augmented dataset per bit-width.

The generation array writes one .pt + one _log.json per (original point, b) into
data/gen150/b{BB}/. The volume stage, in contrast, wants a single file per b that
it can index with --sample_idx. This script does that concatenation, in ascending
original-point order, and writes the provenance alongside so that every augmented
point can be traced back to the original it came from.

It also VALIDATES, rather than assuming:
  - that every expected shard is present (missing ones are listed, not silently
    skipped — a partially failed array must not produce a plausible-looking file);
  - that each shard's first point really is the original it claims (the generator
    always puts x_0 first; if that ever changed, the provenance would be wrong).

On a Jean Zay login node this needs torch, so load the environment first:
    module load pytorch-gpu/py3/2.8.0

Usage
-----
    python scripts/merge_aug_shards.py --shard_dir data/gen150 --n_orig 150 \
        --bits 4 6 10 16 --out_prefix data/fashionMNIST_augmented_cnn_seed42_walk_gen150
"""
import argparse, hashlib, json, sys
from pathlib import Path

import torch


def _md5(t):
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()


def merge_one(shard_dir: Path, bits: int, n_orig: int, orig_ds, out_pt: Path,
              tag_prefix: str, strict: bool):
    sub = shard_dir / f"b{bits:02d}"
    points, prov, reps, missing, mismatched = [], [], [], [], []

    for i in range(n_orig):
        f = sub / f"{tag_prefix}_b{bits}_i{i}.pt"
        if not f.exists():
            missing.append(i)
            continue
        shard = torch.load(f, map_location="cpu", weights_only=False)
        if _md5(shard[0][0]) != _md5(orig_ds[i][0]):
            mismatched.append(i)
        reps.append(len(shard) - 1)
        for j, (x, c) in enumerate(shard):
            prov.append({"aug_idx": len(points), "orig_idx": i, "is_original": j == 0})
            points.append((x, int(c)))

    if missing:
        head = ", ".join(map(str, missing[:10])) + (" ..." if len(missing) > 10 else "")
        msg = f"b={bits}: {len(missing)}/{n_orig} shards MISSING (indices {head})"
        if strict:
            raise SystemExit(f"[ABORT] {msg}\n        Relaunch the array (it is idempotent), "
                             f"or pass --allow_missing to merge anyway.")
        print(f"  [WARN] {msg}")
    if mismatched:
        raise SystemExit(f"[ABORT] b={bits}: shard(s) {mismatched[:10]} do not start with "
                         f"their original point — provenance would be wrong.")

    torch.save(points, out_pt)
    meta = {
        "bits": bits, "n_orig_expected": n_orig, "n_orig_merged": n_orig - len(missing),
        "missing_orig_idx": missing, "n_points": len(points),
        "reps_per_original": reps,
        "avg_reps": round(sum(reps) / len(reps), 3) if reps else 0.0,
        "orig_dataset": "data/fashionMNIST_correct_cnn.pt",
        "points": prov,
    }
    out_pt.with_name(out_pt.stem + "_provenance.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard_dir",  default="data/gen150")
    ap.add_argument("--n_orig",     type=int, default=150)
    ap.add_argument("--bits",       type=int, nargs="+", default=[4, 6, 10, 16])
    ap.add_argument("--orig_data",  default="data/fashionMNIST_correct_cnn.pt")
    ap.add_argument("--tag_prefix", default="fashionMNIST_augmented_cnn_seed42_walk_gen150")
    ap.add_argument("--out_prefix", default="data/fashionMNIST_augmented_cnn_seed42_walk_gen150")
    ap.add_argument("--allow_missing", action="store_true",
                    help="Merge even if some shards are absent. Off by default: a partial "
                         "array would otherwise yield a file that looks complete.")
    a = ap.parse_args()

    orig_ds = torch.load(a.orig_data, map_location="cpu", weights_only=False)
    print(f"{'b':>3} {'originaux':>10} {'points':>8} {'reps moy':>9}  fichier")
    print("-" * 78)
    for b in a.bits:
        out = Path(f"{a.out_prefix}_b{b}.pt")
        m = merge_one(Path(a.shard_dir), b, a.n_orig, orig_ds, out,
                      a.tag_prefix, strict=not a.allow_missing)
        print(f"{b:>3} {m['n_orig_merged']:>10} {m['n_points']:>8} {m['avg_reps']:>9.2f}  {out}")
    print("\nprovenance : *_provenance.json à côté de chaque .pt")


if __name__ == "__main__":
    main()
