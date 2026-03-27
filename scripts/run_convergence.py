"""
run_convergence.py

Tests convergence of the mean-width estimator for the correct polytope
as the number of random directions grows.

For each N in a fixed grid, runs R independent replications of
estimate_polytope_width (non-paired) and saves raw results to a JSON file
for later plotting in notebooks/test_convergence.ipynb.

Usage (from project root):
    python scripts/run_convergence.py \
        --sample_idx 0 \
        --model_path checkpoints/fashion_mlp_best.pth \
        --data_path  data/fashionMNIST_correct_mlp.pt \
        --bits 4 \
        --n_replications 20 \
        --output_dir results
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from src.models.networks import FashionMLP_Large
from src.optim.build_polytopes import build_two_class_polytopes
from src.optim.compute_volumes import estimate_polytope_width
from src.quantization.quantize import quantize_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# N_DIRECTIONS_GRID = [10, 20, 30, 50, 75, 100, 150, 200]
N_DIRECTIONS_GRID = [3, 5, 10]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> FashionMLP_Large:
    model = FashionMLP_Large()
    # Checkpoint is a plain state_dict (no wrapper key)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_sample(data_path: str, sample_idx: int):
    # weights_only=False: file contains a torch.utils.data.Subset object
    dataset = torch.load(data_path, weights_only=False)
    x, c = dataset[sample_idx]
    return x, int(c)


# ---------------------------------------------------------------------------
# Polytope construction
# ---------------------------------------------------------------------------

def build_correct_polytope(model, qmodel, x, c):
    x_batch = x.flatten().unsqueeze(0)   # (1, 784)
    A_correct, b_correct, _, _ = build_two_class_polytopes(model, qmodel, x_batch, c)
    return A_correct, b_correct


# ---------------------------------------------------------------------------
# Convergence experiment
# ---------------------------------------------------------------------------

def run_convergence(A_correct, b_correct, bounds, n_replications):
    """
    For each N in N_DIRECTIONS_GRID, run n_replications independent
    estimates of the mean width of A_correct.

    Returns
    -------
    dict: {N (int) -> {"widths": list, "mean": float, "std": float, "elapsed_s": float}}
    """
    results = {}

    for N in N_DIRECTIONS_GRID:
        log.info(f"  N={N:4d} | {n_replications} replications...")
        t0 = time.perf_counter()

        widths = []
        for _ in tqdm(range(n_replications), desc=f"    N={N}", leave=False):
            out = estimate_polytope_width(
                A_correct, b_correct, bounds,
                n_directions=N,
            )
            widths.append(out["mean_width_correct"])

        elapsed = time.perf_counter() - t0
        widths_arr = np.array(widths)

        results[N] = {
            "widths":    widths_arr.tolist(),
            "mean":      float(widths_arr.mean()),
            "std":       float(widths_arr.std()),
            "elapsed_s": round(elapsed, 2),
        }
        log.info(
            f"         mean={results[N]['mean']:.4f}  "
            f"std={results[N]['std']:.4f}  "
            f"({elapsed:.1f}s)"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convergence test for the mean-width estimator."
    )
    parser.add_argument("--sample_idx",     type=int, required=True,
                        help="Index of the sample in the dataset.")
    parser.add_argument("--model_path",     type=str,
                        default="checkpoints/fashion_mlp_best.pth")
    parser.add_argument("--data_path",      type=str,
                        default="data/fashionMNIST_correct_mlp.pt")
    parser.add_argument("--bits",           type=int, default=4,
                        help="Quantization bits for qmodel (needed to build A_correct).")
    parser.add_argument("--n_replications", type=int, default=20)
    parser.add_argument("--output_dir",     type=str, default="results")
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"convergence_sample{args.sample_idx}_bits{args.bits}.json"

    log.info(f"Model        : {args.model_path}")
    log.info(f"Data         : {args.data_path}")
    log.info(f"Sample idx   : {args.sample_idx}")
    log.info(f"Bits         : {args.bits}")
    log.info(f"Replications : {args.n_replications}")
    log.info(f"N grid       : {N_DIRECTIONS_GRID}")

    # --- Load model and qmodel ---
    log.info("\nLoading model...")
    model  = load_model(args.model_path)
    qmodel = quantize_model(model, bits=args.bits)
    qmodel.eval()

    # --- Load sample ---
    log.info("Loading sample...")
    x, c = load_sample(args.data_path, args.sample_idx)
    log.info(f"x shape: {x.shape}  label: {c}")

    # --- Build polytope once (expensive) ---
    log.info("Building polytope...")
    t0 = time.perf_counter()
    A_correct, b_correct = build_correct_polytope(model, qmodel, x, c)
    log.info(f"A_correct: {tuple(A_correct.shape)}  ({time.perf_counter() - t0:.2f}s)")

    dim    = x.flatten().shape[0]
    bounds = [(-1., 1.)] * dim

    # --- Run experiment ---
    log.info(f"\nRunning convergence experiment...")
    results = run_convergence(A_correct, b_correct, bounds, args.n_replications)

    # --- Save ---
    # Note: integer keys are serialised as strings in JSON; the notebook
    # reads them back with int(k).
    output = {
        "model_path":        args.model_path,
        "data_path":         args.data_path,
        "sample_idx":        args.sample_idx,
        "bits":              args.bits,
        "n_replications":    args.n_replications,
        "n_directions_grid": N_DIRECTIONS_GRID,
        "results":           {str(k): v for k, v in results.items()},
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
