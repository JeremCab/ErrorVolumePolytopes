"""
run_volumes.py

For each sample, estimates the mean width of:
  - the correct polytope  (full-precision model only, computed ONCE)
  - one b-approximated polytope per bit-width in BITS_GRID

All polytopes share the same N=200 random directions.
Each worker task = one direction → solves 2 + 6×2 = 14 LP calls.

The correct polytope is built from the full-precision model alone and does
NOT depend on the quantization bit-width. It is therefore a superset of
every b-approximated polytope, ensuring width_both_b <= width_correct.

Errors (1 - width_both_b / width_correct) are NOT computed here and should
be derived from the saved widths afterwards.

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
# Stores the correct polytope, all b-approximated polytopes, pre-sampled
# directions, and bounds. Each task is a single direction index.
# ---------------------------------------------------------------------------

_WORKER_A_CORRECT    = None
_WORKER_B_UB_CORRECT = None
_WORKER_POLYTOPES    = None   # {bits: (A_both_np, b_ub_both_np)}
_WORKER_DIRECTIONS   = None   # (n_directions, d)
_WORKER_BOUNDS       = None


def _init_worker(A_correct, b_ub_correct, polytopes, directions, bounds):
    global _WORKER_A_CORRECT, _WORKER_B_UB_CORRECT
    global _WORKER_POLYTOPES, _WORKER_DIRECTIONS, _WORKER_BOUNDS
    _WORKER_A_CORRECT    = A_correct
    _WORKER_B_UB_CORRECT = b_ub_correct
    _WORKER_POLYTOPES    = polytopes
    _WORKER_DIRECTIONS   = directions
    _WORKER_BOUNDS       = bounds


def _run_direction(k):
    """Solve all LPs for direction k.

    Returns (k, w_correct, {bits: w_both}) on success,
    or       (k, None,      None)           if any LP fails.
    """
    u = _WORKER_DIRECTIONS[k]

    # Correct polytope (full-precision model only)
    res_max_c = linprog(c=-u, A_ub=_WORKER_A_CORRECT, b_ub=_WORKER_B_UB_CORRECT,
                        bounds=_WORKER_BOUNDS, method="highs")
    res_min_c = linprog(c= u, A_ub=_WORKER_A_CORRECT, b_ub=_WORKER_B_UB_CORRECT,
                        bounds=_WORKER_BOUNDS, method="highs")

    if not (res_max_c.success and res_min_c.success):
        return k, None, None

    w_correct = (-res_max_c.fun) - res_min_c.fun

    # b-approximated polytopes
    w_bits = {}
    for bits, (A_b, b_ub_b) in _WORKER_POLYTOPES.items():
        res_max_b = linprog(c=-u, A_ub=A_b, b_ub=b_ub_b,
                            bounds=_WORKER_BOUNDS, method="highs")
        res_min_b = linprog(c= u, A_ub=A_b, b_ub=b_ub_b,
                            bounds=_WORKER_BOUNDS, method="highs")
        if not (res_max_b.success and res_min_b.success):
            return k, None, None
        w_bits[bits] = (-res_max_b.fun) - res_min_b.fun

    return k, w_correct, w_bits


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

def run_volumes(A_correct, b_correct, polytopes_dict, bounds, n_directions, n_workers):
    """
    Estimate the mean width of the correct polytope and all b-approximated
    polytopes, parallelizing over the N random directions.

    Parameters
    ----------
    A_correct, b_correct : correct polytope (model-only, Ax + b <= 0)
    polytopes_dict : {bits: (A_both, b_both)}
    bounds : list of (lo, hi) pairs
    n_directions : int
    n_workers : int

    Returns
    -------
    dict with "width_correct", "n_directions_used",
    "widths_both": {bits: float}
    """

    def to_numpy(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    from src.optim.compute_volumes import _prep_lp
    A_np, b_ub_correct = _prep_lp(to_numpy(A_correct), to_numpy(b_correct))

    polytopes_np = {
        bits: _prep_lp(to_numpy(A_b), to_numpy(b_b))
        for bits, (A_b, b_b) in polytopes_dict.items()
    }

    # Sample all directions in the main process (no randomness in workers)
    d = A_np.shape[1]
    directions = np.random.randn(n_directions, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    widths_correct = {}
    widths_bits    = {bits: {} for bits in polytopes_np}
    n_failed = 0

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(A_np, b_ub_correct, polytopes_np, directions, bounds),
    ) as executor:
        for k, w_correct, w_bits in tqdm(
            executor.map(_run_direction, list(range(n_directions))),
            total=n_directions,
            desc="Directions",
        ):
            if w_correct is None:
                n_failed += 1
                continue
            widths_correct[k] = w_correct
            for bits, w in w_bits.items():
                widths_bits[bits][k] = w

    elapsed = time.perf_counter() - t0
    log.info(f"Elapsed: {elapsed:.1f}s  |  Failed directions: {n_failed}/{n_directions}")

    valid_keys    = sorted(widths_correct.keys())
    n_used        = len(valid_keys)
    width_correct = float(np.mean([widths_correct[k] for k in valid_keys]))

    widths_both = {}
    for bits in polytopes_np:
        arr = np.array([widths_bits[bits][k] for k in valid_keys])
        widths_both[bits] = float(arr.mean())
        log.info(f"  bits={bits:2d}  width_both={widths_both[bits]:.4f}")

    log.info(f"  width_correct={width_correct:.4f}")

    return {
        "width_correct":     width_correct,
        "n_directions_used": n_used,
        "widths_both":       widths_both,
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

    # Model shortcuts computed once; one b-approximated polytope per bit-width
    if args.model_type == "cnn":
        A_correct, b_correct, polytopes_dict = build_cnn_all_polytopes(
            model, qmodels_dict, x_batch, c
        )
    else:
        A_correct, b_correct, polytopes_dict = build_all_polytopes(
            model, qmodels_dict, x_batch, c
        )
    log.info(f"A_correct (model-only): {tuple(A_correct.shape)}")
    for bits, (A_both, _) in polytopes_dict.items():
        log.info(f"A_both (bits={bits:2d}): {tuple(A_both.shape)}")

    log.info(f"Polytopes built in {time.perf_counter() - t0:.2f}s")

    bounds = [(-1., 1.)] * x.flatten().shape[0]

    # Run experiment
    log.info("\nRunning volume experiment...")
    result = run_volumes(A_correct, b_correct, polytopes_dict, bounds,
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
        "width_correct":     result["width_correct"],
        "widths_both":       {str(k): v for k, v in result["widths_both"].items()},
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
