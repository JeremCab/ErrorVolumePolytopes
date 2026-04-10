"""
run_volumes.py

For each sample, estimates the mean width of three nested polytopes:

  A_base    : model activation + model classification
              (model-only, independent of quantization)
  A_correct : A_base + qmodel activation          (per bit-width)
  A_both    : A_correct + qmodel classification   (per bit-width)

A_base ⊇ A_correct ⊇ A_both  →  width_base ≥ width_correct ≥ width_both

All polytopes share the same N=200 random directions.
Each worker task = one direction → solves 2 + 6×4 = 26 LP calls.

Output: mean widths of all three polytopes per bit-width saved to JSON.
Error metrics should be derived from the saved widths in post-processing.

Output: one JSON file per sample in --output_dir.
Already-computed samples are skipped automatically (safe to resubmit).

Usage (from project root):

  MLP:
    python scripts/run_volumes.py \\
        --model_type    mlp \\
        --sample_idx    0 \\
        --model_path    checkpoints/fashion_mlp_best.pth \\
        --data_path     data/fashionMNIST_correct_mlp.pt \\
        --n_directions  200 \\
        --output_dir    results/volumes_mlp

  CNN:
    python scripts/run_volumes.py \\
        --model_type    cnn \\
        --sample_idx    0 \\
        --model_path    checkpoints/fashion_cnn_best.pth \\
        --data_path     data/fashionMNIST_correct_cnn.pt \\
        --n_directions  200 \\
        --output_dir    results/volumes_cnn
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linprog
import torch
from tqdm import tqdm

from src.models.networks import FashionMLP_Large, FashionCNN_Small
from src.optim.build_polytopes import build_all_polytopes
from src.optim.build_polytopes_cnn import build_cnn_all_polytopes
from src.quantization.quantize import quantize_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BITS_GRID = [4, 6, 8, 10, 12, 16]


# ---------------------------------------------------------------------------
# Worker state — initialised once per worker via _init_worker.
# Stores A_base, per-bit (A_correct, A_both) polytopes, pre-sampled
# directions, and bounds. Each task is a single direction index.
# ---------------------------------------------------------------------------

_WORKER_A_BASE     = None
_WORKER_B_BASE     = None
_WORKER_POLYTOPES  = None   # {bits: (A_correct_np, b_correct_np, A_both_np, b_both_np)}
_WORKER_DIRECTIONS = None   # (n_directions, d)
_WORKER_BOUNDS     = None


def _init_worker(A_base, b_base, polytopes, directions, bounds):
    global _WORKER_A_BASE, _WORKER_B_BASE
    global _WORKER_POLYTOPES, _WORKER_DIRECTIONS, _WORKER_BOUNDS
    _WORKER_A_BASE     = A_base
    _WORKER_B_BASE     = b_base
    _WORKER_POLYTOPES  = polytopes
    _WORKER_DIRECTIONS = directions
    _WORKER_BOUNDS     = bounds


def _run_direction(k):
    """Solve all LPs for direction k.

    Returns (k, w_base, {bits: (w_correct, w_both)}) on success.
    w_base is None if the A_base LP fails (direction discarded entirely).
    Per-bit entries are None if that bit-width's LP fails (isolated failure).
    """
    u = _WORKER_DIRECTIONS[k]

    # A_base (model-only)
    res_max_base = linprog(c=-u, A_ub=_WORKER_A_BASE, b_ub=_WORKER_B_BASE,
                           bounds=_WORKER_BOUNDS, method="highs")
    res_min_base = linprog(c= u, A_ub=_WORKER_A_BASE, b_ub=_WORKER_B_BASE,
                           bounds=_WORKER_BOUNDS, method="highs")
    if not (res_max_base.success and res_min_base.success):
        return k, None, None

    w_base = (-res_max_base.fun) - res_min_base.fun

    # Per-bit: A_correct and A_both (failures are isolated per bit-width)
    w_bits = {}
    for bits, (A_c, b_c, A_b, b_b) in _WORKER_POLYTOPES.items():
        res_max_c = linprog(c=-u, A_ub=A_c, b_ub=b_c,
                            bounds=_WORKER_BOUNDS, method="highs")
        res_min_c = linprog(c= u, A_ub=A_c, b_ub=b_c,
                            bounds=_WORKER_BOUNDS, method="highs")
        res_max_b = linprog(c=-u, A_ub=A_b, b_ub=b_b,
                            bounds=_WORKER_BOUNDS, method="highs")
        res_min_b = linprog(c= u, A_ub=A_b, b_ub=b_b,
                            bounds=_WORKER_BOUNDS, method="highs")
        if not all(r.success for r in [res_max_c, res_min_c, res_max_b, res_min_b]):
            w_bits[bits] = None
        else:
            w_bits[bits] = ((-res_max_c.fun) - res_min_c.fun,
                            (-res_max_b.fun) - res_min_b.fun)

    return k, w_base, w_bits


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_model(model_path, model_type):
    if model_type == "cnn":
        model = FashionCNN_Small()
    else:
        model = FashionMLP_Large()
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_sample(data_path, sample_idx):
    dataset = torch.load(data_path, weights_only=False)
    x, c = dataset[sample_idx]
    return x, int(c)


# ---------------------------------------------------------------------------
# Main volume computation (parallel over directions)
# ---------------------------------------------------------------------------

def run_volumes(A_base, b_base, polytopes_dict, bounds, n_directions, n_workers):
    """
    Estimate the mean width of A_base and, per bit-width, of A_correct and
    A_both, parallelizing over the N random directions.

    Parameters
    ----------
    A_base, b_base : model-only polytope (Ax + b <= 0)
    polytopes_dict : {bits: (A_correct, b_correct, A_both, b_both)}
    bounds : list of (lo, hi) pairs
    n_directions : int
    n_workers : int

    Returns
    -------
    dict with "width_base", "widths_correct", "widths_both", "n_directions_used"
    """

    def to_numpy(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    from src.optim.compute_volumes import _prep_lp
    A_base_np, b_base_np = _prep_lp(to_numpy(A_base), to_numpy(b_base))

    polytopes_np = {
        bits: (
            _prep_lp(to_numpy(A_c), to_numpy(b_c)) +
            _prep_lp(to_numpy(A_b), to_numpy(b_b))
        )
        for bits, (A_c, b_c, A_b, b_b) in polytopes_dict.items()
    }
    # polytopes_np[bits] = (A_c_np, b_c_np, A_b_np, b_b_np)

    # Sample all directions in the main process (no randomness in workers)
    d = A_base_np.shape[1]
    directions = np.random.randn(n_directions, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    widths_base    = {}
    widths_correct = {bits: {} for bits in polytopes_np}
    widths_both    = {bits: {} for bits in polytopes_np}
    n_failed = 0

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(A_base_np, b_base_np, polytopes_np, directions, bounds),
    ) as executor:
        for k, w_base, w_bits in tqdm(
            executor.map(_run_direction, list(range(n_directions))),
            total=n_directions,
            desc="Directions",
        ):
            if w_base is None:
                n_failed += 1
                continue
            widths_base[k] = w_base
            for bits, val in w_bits.items():
                if val is not None:
                    w_c, w_b = val
                    widths_correct[bits][k] = w_c
                    widths_both[bits][k]    = w_b

    elapsed = time.perf_counter() - t0
    log.info(f"Elapsed: {elapsed:.1f}s  |  Failed directions: {n_failed}/{n_directions}")

    valid_keys = sorted(widths_base.keys())
    n_used     = len(valid_keys)
    width_base = float(np.mean([widths_base[k] for k in valid_keys]))
    log.info(f"  width_base={width_base:.4f}")

    out_correct = {}
    out_both    = {}
    for bits in polytopes_np:
        wc_vals = [widths_correct[bits][k] for k in valid_keys if k in widths_correct[bits]]
        wb_vals = [widths_both[bits][k]    for k in valid_keys if k in widths_both[bits]]
        wc = float(np.mean(wc_vals)) if wc_vals else 0.0
        wb = float(np.mean(wb_vals)) if wb_vals else 0.0
        out_correct[bits] = wc
        out_both[bits]    = wb
        log.info(f"  bits={bits:2d}  width_correct={wc:.4f}  width_both={wb:.4f}")

    return {
        "width_base":        width_base,
        "widths_correct":    out_correct,
        "widths_both":       out_both,
        "n_directions_used": n_used,
    }


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def _default_n_workers():
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None:
        return int(slurm)
    return min(os.cpu_count() or 1, 8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Volume experiment: mean-width of correct and b-approximated polytopes."
    )
    parser.add_argument("--model_type",   type=str, default="mlp",
                        choices=["mlp", "cnn"],
                        help="Model architecture: mlp or cnn (default: mlp)")
    parser.add_argument("--sample_idx",   type=int, required=True)
    parser.add_argument("--model_path",   type=str, default=None)
    parser.add_argument("--data_path",    type=str, default=None)
    parser.add_argument("--n_directions", type=int, default=200)
    parser.add_argument("--n_workers",    type=int, default=_default_n_workers())
    parser.add_argument("--output_dir",   type=str, default=None)
    args = parser.parse_args()

    # Fill defaults that depend on model_type
    if args.model_path is None:
        args.model_path = f"checkpoints/fashion_{args.model_type}_best.pth"
    if args.data_path is None:
        args.data_path = f"data/fashionMNIST_correct_{args.model_type}.pt"
    if args.output_dir is None:
        args.output_dir = f"results/volumes_{args.model_type}"

    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"volumes_sample{args.sample_idx}.json"

    # Skip if already computed (safe to resubmit)
    if output_file.exists():
        log.info(f"Sample {args.sample_idx} already computed. Skipping.")
        sys.exit(0)

    log.info(f"Model type   : {args.model_type}")
    log.info(f"Sample idx   : {args.sample_idx}")
    log.info(f"Model        : {args.model_path}")
    log.info(f"N directions : {args.n_directions}")
    log.info(f"Bits grid    : {BITS_GRID}")
    log.info(f"Workers      : {args.n_workers}")

    # Load model
    log.info("\nLoading model...")
    model = load_model(args.model_path, args.model_type)

    # Load sample
    log.info("Loading sample...")
    x, c = load_sample(args.data_path, args.sample_idx)
    log.info(f"x shape: {x.shape}  label: {c}")

    # Prepare input batch (shape depends on model type)
    if args.model_type == "cnn":
        x_batch = x.unsqueeze(0)          # (1, 1, 28, 28)
    else:
        x_batch = x.flatten().unsqueeze(0)  # (1, 784)

    # Build polytopes
    log.info("Building polytopes...")
    t0 = time.perf_counter()

    # Quantized models: one per bit-width
    qmodels_dict = {}
    for bits in BITS_GRID:
        qmodel = quantize_model(model, bits=bits)
        qmodel.eval()
        qmodels_dict[bits] = qmodel

    # Model shortcuts computed once; one (A_correct, A_both) pair per bit-width
    if args.model_type == "cnn":
        A_base, b_base, polytopes_dict = build_cnn_all_polytopes(
            model, qmodels_dict, x_batch, c
        )
    else:
        A_base, b_base, polytopes_dict = build_all_polytopes(
            model, qmodels_dict, x_batch, c
        )
    log.info(f"A_base (model-only): {tuple(A_base.shape)}")
    for bits, (A_correct, _, A_both, _) in polytopes_dict.items():
        log.info(f"  bits={bits:2d}  A_correct={tuple(A_correct.shape)}  A_both={tuple(A_both.shape)}")

    log.info(f"Polytopes built in {time.perf_counter() - t0:.2f}s")

    bounds = [(-1., 1.)] * x.flatten().shape[0]

    # Run experiment
    log.info("\nRunning volume experiment...")
    result = run_volumes(A_base, b_base, polytopes_dict, bounds,
                         args.n_directions, args.n_workers)

    # Save
    output = {
        "model_type":        args.model_type,
        "sample_idx":        args.sample_idx,
        "model_path":        args.model_path,
        "data_path":         args.data_path,
        "n_directions":      args.n_directions,
        "n_directions_used": result["n_directions_used"],
        "bits_grid":         BITS_GRID,
        "width_base":        result["width_base"],
        "widths_correct":    {str(k): v for k, v in result["widths_correct"].items()},
        "widths_both":       {str(k): v for k, v in result["widths_both"].items()},
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
