"""
build_aug_diversity.py

For each b-bit q-model (b ∈ {4, 6, 8, 10, 12, 16}), identify which augmented
points from the MCMC walk dataset lie in a DIFFERENT P2 cell (i.e., different
q-model activation pattern) from their original x0.

The augmented dataset was generated with b=4, so by construction all augmented
points are in different P2 cells under the 4-bit model.  For higher bit-widths
(coarser quantisation, fewer distinct patterns) some augmented points may fall
back into x0's P2 cell — this script quantifies that.

Output: data/fashionMNIST_aug_diversity_{model_type}.json  containing:
  {
    "orig_to_aug":    {"0": [0,...,35], "1": [36,...,71], ...},
    "diverse_per_bit": {"4": [...], "6": [...], ..., "16": [...]}
  }

Usage (from project root):
    python scripts/build_aug_diversity.py --model_type mlp
    python scripts/build_aug_diversity.py --model_type cnn
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.networks       import FashionMLP_Large, FashionCNN_Small
from src.quantization.quantize import quantize_model
from src.optim.mcmc_augment    import activation_pattern

BITS_GRID = [4, 6, 8, 10, 12, 16]


def prep_input(x: torch.Tensor, model_type: str, device: torch.device) -> torch.Tensor:
    """Return x reshaped to the batch input expected by the model."""
    if model_type == "mlp":
        return x.flatten().unsqueeze(0).to(device)   # (1, 784)
    else:
        if x.dim() == 2:                              # (28,28) edge case
            x = x.unsqueeze(0)                        # (1,28,28)
        return x.unsqueeze(0).to(device)              # (1,1,28,28)


def build_orig_to_aug(log_path: Path) -> dict[int, list[int]]:
    """Read reps_per_sample from the log and derive orig_idx → [aug_idx, ...] mapping."""
    with open(log_path) as f:
        log = json.load(f)
    orig_to_aug: dict[int, list[int]] = {}
    cursor = 0
    for orig_idx, n_reps in enumerate(log["reps_per_sample"]):
        orig_to_aug[orig_idx] = list(range(cursor, cursor + n_reps))
        cursor += n_reps
    return orig_to_aug


def main():
    parser = argparse.ArgumentParser(
        description="Build per-bit-width diversity index for the augmented dataset."
    )
    parser.add_argument("--model_type", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--model_path", default=None,
                        help="FP model checkpoint (default: checkpoints/fashion_{type}_best.pth)")
    parser.add_argument("--orig_data",  default=None,
                        help="Original correct dataset (default: data/fashionMNIST_correct_{type}.pt)")
    parser.add_argument("--aug_data",   default=None,
                        help="Augmented dataset (default: data/fashionMNIST_augmented_{type}_seed42_walk_aug200.pt)")
    parser.add_argument("--aug_log",    default=None,
                        help="Augmented dataset log (default: same stem as aug_data + _log.json)")
    parser.add_argument("--output_dir", default="data",
                        help="Directory for output JSON (default: data/)")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    TAG  = "seed42_walk_aug200"

    if args.model_path is None:
        args.model_path = f"checkpoints/fashion_{args.model_type}_best.pth"
    if args.orig_data is None:
        args.orig_data  = f"data/fashionMNIST_correct_{args.model_type}.pt"
    if args.aug_data is None:
        args.aug_data   = f"data/fashionMNIST_augmented_{args.model_type}_{TAG}.pt"
    if args.aug_log is None:
        args.aug_log    = f"data/fashionMNIST_augmented_{args.model_type}_{TAG}_log.json"

    device = torch.device("cpu")

    # ── orig → aug index mapping (from log, no regeneration needed) ───────────
    print(f"Building orig→aug mapping from {args.aug_log} ...")
    orig_to_aug = build_orig_to_aug(ROOT / args.aug_log)
    n_aug_total = sum(len(v) for v in orig_to_aug.values())
    print(f"  {len(orig_to_aug)} originals, {n_aug_total} augmented points")

    # ── reverse map: aug_idx → orig_idx ───────────────────────────────────────
    aug_to_orig: dict[int, int] = {}
    for orig_idx, aug_indices in orig_to_aug.items():
        for ai in aug_indices:
            aug_to_orig[ai] = orig_idx

    # ── load FP model + q-models ───────────────────────────────────────────────
    print(f"\nLoading {args.model_type.upper()} model ...")
    model_cls = FashionCNN_Small if args.model_type == "cnn" else FashionMLP_Large
    fp_model  = model_cls()
    fp_model.load_state_dict(
        torch.load(ROOT / args.model_path, weights_only=True, map_location=device)
    )
    fp_model.eval()
    qmodels = {b: quantize_model(fp_model, bits=b).eval() for b in BITS_GRID}
    print(f"  Built q-models for b = {BITS_GRID}")

    # ── load datasets ──────────────────────────────────────────────────────────
    print(f"\nLoading datasets ...")
    dataset_orig = torch.load(ROOT / args.orig_data, weights_only=False)
    dataset_aug  = torch.load(ROOT / args.aug_data,  weights_only=False)
    print(f"  Original  : {len(dataset_orig)} samples")
    print(f"  Augmented : {len(dataset_aug)} samples")

    # ── precompute activation patterns for the 200 originals (all bits) ───────
    print(f"\nComputing patterns for original samples ...")
    # orig_pat[b][orig_idx] = activation-pattern tuple under b-bit q-model
    orig_pat: dict[int, dict[int, tuple]] = {b: {} for b in BITS_GRID}
    for orig_idx in tqdm(list(orig_to_aug.keys()), desc="originals"):
        x0, _ = dataset_orig[orig_idx]
        x0_in = prep_input(x0, args.model_type, device)
        for b in BITS_GRID:
            orig_pat[b][orig_idx] = tuple(activation_pattern(x0_in, qmodels[b]))

    # ── scan augmented points once; tag diverse indices per bit-width ─────────
    print(f"\nScanning {n_aug_total} augmented points ...")
    diverse_per_bit: dict[int, list[int]] = {b: [] for b in BITS_GRID}

    for ai in tqdm(range(n_aug_total), desc="augmented"):
        x_aug, _ = dataset_aug[ai]
        x_aug_in  = prep_input(x_aug, args.model_type, device)
        orig_idx  = aug_to_orig[ai]
        for b in BITS_GRID:
            aug_pat = tuple(activation_pattern(x_aug_in, qmodels[b]))
            if aug_pat != orig_pat[b][orig_idx]:
                diverse_per_bit[b].append(ai)

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\nDiversity summary — {args.model_type.upper()}  ({n_aug_total} augmented points):")
    print(f"  {'b':>4}  {'diverse':>10}  {'%':>8}")
    for b in BITS_GRID:
        n = len(diverse_per_bit[b])
        print(f"  {b:4d}  {n:10d}  {100 * n / n_aug_total:7.1f}%")

    # ── save ───────────────────────────────────────────────────────────────────
    output = {
        "model_type":      args.model_type,
        "bits_grid":       BITS_GRID,
        "n_originals":     len(orig_to_aug),
        "n_augmented":     n_aug_total,
        "orig_to_aug":     {str(k): v for k, v in orig_to_aug.items()},
        "diverse_per_bit": {str(b): diverse_per_bit[b] for b in BITS_GRID},
    }
    out_path = ROOT / args.output_dir / f"fashionMNIST_aug_diversity_{args.model_type}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
