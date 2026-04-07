"""
run_convergence.py

Tests convergence of the mean-width estimator for the correct polytope
as the number of random directions grows.

All (N, replication) pairs are submitted as independent tasks to a
ProcessPoolExecutor. On Jean Zay (40 CPUs, 50 samples via Slurm array),
the full experiment (9 N values × 20 replications) takes ~2h per sample,
with all 50 samples running in parallel.

Usage (from project root):

  MLP:
    python scripts/run_convergence.py \
        --model_type mlp \
        --sample_idx 42 \
        --model_path checkpoints/fashion_mlp_best.pth \
        --data_path  data/fashionMNIST_correct_mlp.pt \
        --bits 4

  CNN:
    python scripts/run_convergence.py \
        --model_type cnn \
        --sample_idx 42 \
        --model_path checkpoints/fashion_cnn_best.pth \
        --data_path  data/fashionMNIST_correct_cnn.pt
        # --bits is ignored for CNN (correct polytope uses full-precision model only)
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Ensure project root is on sys.path when running as a script
# (also executed in worker processes on spawn, which is correct)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.networks import FashionCNN_Small, FashionMLP_Large
from src.optim.build_polytopes import build_two_class_polytopes
from src.optim.build_polytopes_cnn import build_cnn_correct_polytope
from src.optim.compute_volumes import estimate_polytope_width
from src.quantization.quantize import quantize_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

N_DIRECTIONS_GRID = [10, 20, 30, 50, 75, 100, 150, 200, 300]


# ---------------------------------------------------------------------------
# Worker state — initialised once per worker process via _init_worker.
# Using an initializer avoids pickling A_np / b_np for every task (160×).
# Instead they are pickled only once per worker (n_workers×).
# ---------------------------------------------------------------------------

_WORKER_A      = None
_WORKER_B      = None
_WORKER_BOUNDS = None


def _init_worker(A_np, b_np, bounds):
    global _WORKER_A, _WORKER_B, _WORKER_BOUNDS
    _WORKER_A      = A_np
    _WORKER_B      = b_np
    _WORKER_BOUNDS = bounds


def _run_single_task(N: int):
    """One task = one call to estimate_polytope_width for a given N.
    Returns (N, mean_width) so results can be grouped by N afterwards.
    """
    # Re-seed from OS entropy before sampling directions.
    # On Linux, forked workers inherit the parent's numpy random state,
    # so without this all replications would use identical directions → std ≈ 0.
    np.random.seed()
    out = estimate_polytope_width(
        _WORKER_A, _WORKER_B, _WORKER_BOUNDS,
        n_directions=N,
    )
    return N, out["mean_width_correct"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str, model_type: str) -> nn.Module:
    model = FashionCNN_Small() if model_type == "cnn" else FashionMLP_Large()
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
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

def build_correct_polytope(model, qmodel, x, c, model_type: str):
    if model_type == "cnn":
        # build_cnn_correct_polytope needs only the full-precision model
        x_in = x.unsqueeze(0) if x.dim() == 3 else x   # (1, C, H, W)
        return build_cnn_correct_polytope(model, x_in, c)
    else:
        x_batch = x.flatten().unsqueeze(0)              # (1, 784)
        A_correct, b_correct, _, _ = build_two_class_polytopes(model, qmodel, x_batch, c)
        return A_correct, b_correct


# ---------------------------------------------------------------------------
# Convergence experiment
# ---------------------------------------------------------------------------

def run_convergence(A_correct, b_correct, bounds, n_replications, n_workers, grid=None):
    """
    Submit all (N, replication) pairs as independent tasks to a
    ProcessPoolExecutor. Workers share A_correct / b_correct via
    an initializer (one pickle per worker, not per task).

    Returns
    -------
    dict: {N (int) -> {"widths": list, "mean": float, "std": float}}
    """

    # Convert to numpy once in the main process
    if hasattr(A_correct, "detach"):
        A_np = A_correct.detach().cpu().numpy()
        b_np = b_correct.detach().cpu().numpy()
    else:
        A_np = np.asarray(A_correct)
        b_np = np.asarray(b_correct)

    if grid is None:
        grid = N_DIRECTIONS_GRID

    # Flat task list: one entry per (N, replication) pair
    tasks = [N for N in grid for _ in range(n_replications)]
    n_tasks = len(tasks)
    log.info(f"Total tasks  : {n_tasks}  ({len(grid)} N values × {n_replications} replications)")
    log.info(f"Workers      : {n_workers}")

    raw = {N: [] for N in grid}

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(A_np, b_np, bounds),
    ) as executor:
        for N, width in tqdm(
            executor.map(_run_single_task, tasks),
            total=n_tasks,
            desc="Tasks completed",
        ):
            raw[N].append(width)

    elapsed_total = time.perf_counter() - t0
    log.info(f"Total elapsed: {elapsed_total:.1f}s")

    # Aggregate per N
    results = {}
    for N in grid:
        arr = np.array(raw[N])
        results[N] = {
            "widths": arr.tolist(),
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
        }
        log.info(f"  N={N:4d}  mean={results[N]['mean']:.4f}  std={results[N]['std']:.4f}")

    return results, round(elapsed_total, 2)


# ---------------------------------------------------------------------------
# CPU detection (Jean Zay vs laptop)
# ---------------------------------------------------------------------------

def _default_n_workers() -> int:
    """Use SLURM_CPUS_PER_TASK on Jean Zay, fallback to local cpu_count."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None:
        return int(slurm)
    return min(len(N_DIRECTIONS_GRID), os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convergence test for the mean-width estimator."
    )
    parser.add_argument("--model_type",     type=str, default="mlp",
                        choices=["mlp", "cnn"],
                        help="Architecture type: 'mlp' (default) or 'cnn'.")
    parser.add_argument("--sample_idx",     type=int, required=True,
                        help="Index of the sample in the dataset.")
    parser.add_argument("--model_path",     type=str,
                        default=None,
                        help="Path to model checkpoint. "
                             "Defaults to checkpoints/fashion_{model_type}_best.pth.")
    parser.add_argument("--data_path",      type=str,
                        default=None,
                        help="Path to dataset. "
                             "Defaults to data/fashionMNIST_correct_{model_type}.pt.")
    parser.add_argument("--bits",           type=int, default=4,
                        help="Quantization bits for qmodel (MLP only; ignored for CNN).")
    parser.add_argument("--n_directions_grid", type=int, nargs="+", default=None,
                        help="Override N_DIRECTIONS_GRID (e.g. --n_directions_grid 2 5 10). "
                             "Useful for smoke tests.")
    parser.add_argument("--n_replications", type=int, default=20)
    parser.add_argument("--n_workers",      type=int,
                        default=_default_n_workers(),
                        help="Number of parallel workers. Defaults to SLURM_CPUS_PER_TASK "
                             "if set (Jean Zay), otherwise min(N_grid size, cpu_count) locally.")
    parser.add_argument("--output_dir",     type=str, default="results")
    args = parser.parse_args()

    # Fill in default paths based on model_type
    if args.model_path is None:
        args.model_path = f"checkpoints/fashion_{args.model_type}_best.pth"
    if args.data_path is None:
        args.data_path = f"data/fashionMNIST_correct_{args.model_type}.pt"

    # Override N_DIRECTIONS_GRID if requested (e.g. for smoke tests)
    grid = sorted(args.n_directions_grid) if args.n_directions_grid else N_DIRECTIONS_GRID

    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bits_tag    = f"_bits{args.bits}" if args.model_type == "mlp" else ""
    output_file = output_dir / f"convergence_{args.model_type}_sample{args.sample_idx}{bits_tag}.json"

    log.info(f"Model type   : {args.model_type}")
    log.info(f"Model        : {args.model_path}")
    log.info(f"Data         : {args.data_path}")
    log.info(f"Sample idx   : {args.sample_idx}")
    if args.model_type == "mlp":
        log.info(f"Bits         : {args.bits}")
    log.info(f"Replications : {args.n_replications}")
    log.info(f"N grid       : {grid}")

    # --- Load model (and qmodel for MLP only) ---
    log.info("\nLoading model...")
    model  = load_model(args.model_path, args.model_type)
    qmodel = None
    if args.model_type == "mlp":
        qmodel = quantize_model(model, bits=args.bits)
        qmodel.eval()

    # --- Load sample ---
    log.info("Loading sample...")
    x, c = load_sample(args.data_path, args.sample_idx)
    log.info(f"x shape: {x.shape}  label: {c}")

    # --- Build polytope once (expensive) ---
    log.info("Building polytope...")
    t0 = time.perf_counter()
    A_correct, b_correct = build_correct_polytope(model, qmodel, x, c, args.model_type)
    log.info(f"A_correct: {tuple(A_correct.shape)}  ({time.perf_counter() - t0:.2f}s)")

    dim    = x.flatten().shape[0]
    bounds = [(-1., 1.)] * dim

    # --- Run experiment ---
    log.info(f"\nRunning convergence experiment...")
    results, elapsed_total = run_convergence(
        A_correct, b_correct, bounds, args.n_replications, args.n_workers, grid=grid
    )

    # --- Save (same format as before — notebook works unchanged) ---
    output = {
        "model_path":        args.model_path,
        "data_path":         args.data_path,
        "sample_idx":        args.sample_idx,
        "bits":              args.bits,
        "n_replications":    args.n_replications,
        "n_workers":         args.n_workers,
        "elapsed_total_s":   elapsed_total,
        "n_directions_grid": grid,
        "results":           {str(k): v for k, v in results.items()},
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
