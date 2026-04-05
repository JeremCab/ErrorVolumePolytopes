"""
fix_nan_volumes.py

Post-processing utility to repair JSON result files produced by run_volumes.py
that contain NaN values.

NaNs arise when the b-approximated polytope is empty (LP infeasible), which
happens when the quantized model predicts a different class than the
full-precision model on the reference point x_0. In that case:

  - width_correct  : recomputed independently — should be a valid positive number.
  - widths_both[b] : recomputed independently per bit-width.
                     If all LPs fail (empty polytope) → set to 0.0.

Parallelism: directions are distributed across workers (same as run_volumes.py).
Unlike run_volumes.py, a LP failure for bits=b does NOT discard the direction
for other bit-widths — each bits accumulates its own list of successful widths.

The files are updated in-place. A list of modified sample indices is returned.

Usage (from a notebook or script):
    from src.optim.fix_nan_volumes import fix_nan_volumes

    modified = fix_nan_volumes(
        results_dir  = "results/volumes",
        model_path   = "checkpoints/fashion_mlp_best.pth",
        data_path    = "data/fashionMNIST_correct_mlp.pt",
        n_directions = 200,
    )
    print("Modified samples:", modified)
"""

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import torch

from src.models.networks import FashionMLP_Large
from src.optim.build_polytopes import build_all_polytopes
from src.quantization.quantize import quantize_model


# ---------------------------------------------------------------------------
# Worker state — initialised once per worker via _init_worker
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

    Unlike run_volumes._run_direction, a failure for bits=b does NOT discard
    the direction for other bit-widths. Each bits accumulates independently.

    Returns
    -------
    k : int
    w_correct : float or None
        None if the correct-polytope LP failed (direction discarded entirely).
    w_bits : dict {bits: float or None}
        None for a given bits if its LP failed.
    """
    u = _WORKER_DIRECTIONS[k]

    # --- Correct polytope ---
    res_max_c = linprog(c=-u, A_ub=_WORKER_A_CORRECT, b_ub=_WORKER_B_UB_CORRECT,
                        bounds=_WORKER_BOUNDS, method="highs")
    res_min_c = linprog(c= u, A_ub=_WORKER_A_CORRECT, b_ub=_WORKER_B_UB_CORRECT,
                        bounds=_WORKER_BOUNDS, method="highs")

    if not (res_max_c.success and res_min_c.success):
        return k, None, {}

    w_correct = (-res_max_c.fun) - res_min_c.fun

    # --- b-approximated polytopes: independent per bits ---
    w_bits = {}
    for bits, (A_b, b_ub_b) in _WORKER_POLYTOPES.items():
        res_max_b = linprog(c=-u, A_ub=A_b, b_ub=b_ub_b,
                            bounds=_WORKER_BOUNDS, method="highs")
        res_min_b = linprog(c= u, A_ub=A_b, b_ub=b_ub_b,
                            bounds=_WORKER_BOUNDS, method="highs")
        if res_max_b.success and res_min_b.success:
            w_bits[bits] = (-res_max_b.fun) - res_min_b.fun
        else:
            w_bits[bits] = None   # LP failed for this bits

    return k, w_correct, w_bits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path):
    model = FashionMLP_Large()
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _has_nan(record):
    if math.isnan(record["width_correct"]):
        return True
    return any(math.isnan(v) for v in record["widths_both"].values())


def _default_n_workers():
    return min(os.cpu_count() or 1, 8)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def fix_nan_volumes(results_dir, model_path, data_path,
                    n_directions=200, n_workers=None, verbose=True):
    """
    Scan result files for NaNs, recompute volumes for affected samples,
    and update the JSON files in-place.

    Directions are shared across the correct polytope and all b-approximated
    polytopes (paired estimation, consistent with run_volumes.py).
    A LP failure for bits=b does not discard the direction for other bit-widths.

    Parameters
    ----------
    results_dir : str or Path
    model_path : str or Path
    data_path : str or Path
    n_directions : int
    n_workers : int or None
        Number of parallel workers. Defaults to min(cpu_count, 8).
    verbose : bool

    Returns
    -------
    modified_indices : list of int
        Sample indices whose JSON files were updated, sorted ascending.
    """

    if n_workers is None:
        n_workers = _default_n_workers()

    results_dir = Path(results_dir)
    files       = sorted(results_dir.glob("volumes_sample*.json"))

    # ------------------------------------------------------------------ #
    # Step 1 — Identify files with NaNs                                   #
    # ------------------------------------------------------------------ #

    nan_files = []
    for f in files:
        with open(f) as fh:
            record = json.load(fh)
        if _has_nan(record):
            nan_files.append((f, record))

    if verbose:
        print(f"Found {len(nan_files)} NaN files out of {len(files)} total.")

    if not nan_files:
        return []

    # ------------------------------------------------------------------ #
    # Step 2 — Load model and dataset once                                #
    # ------------------------------------------------------------------ #

    if verbose:
        print(f"Loading model and dataset...  (workers={n_workers})")

    model   = _load_model(model_path)
    dataset = torch.load(data_path, weights_only=False)

    # Infer bits_grid and bounds from the first NaN record
    bits_grid = sorted(int(b) for b in nan_files[0][1]["bits_grid"])
    x0, _     = dataset[nan_files[0][1]["sample_idx"]]
    dim       = x0.flatten().shape[0]
    bounds    = [(-1., 1.)] * dim

    if verbose:
        print(f"Bits grid : {bits_grid}")
        print(f"Bounds    : [(-1, 1)] × {dim}\n")

    # ------------------------------------------------------------------ #
    # Step 3 — Recompute and update each NaN file (samples sequential,    #
    #           directions parallel within each sample)                   #
    # ------------------------------------------------------------------ #

    def to_numpy(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    modified_indices = []

    for f, record in nan_files:
        sample_idx = record["sample_idx"]
        if verbose:
            print(f"--- Sample {sample_idx} ---")

        x, c    = dataset[sample_idx]
        x_batch = x.flatten().unsqueeze(0)

        # Build all polytopes (model shortcuts computed once)
        qmodels_dict = {b: quantize_model(model, bits=b) for b in bits_grid}
        A_correct, b_correct, polytopes_dict = build_all_polytopes(
            model, qmodels_dict, x_batch, c
        )

        # Convert to numpy
        A_np         = to_numpy(A_correct)
        b_ub_correct = -to_numpy(b_correct)
        polytopes_np = {
            bits: (to_numpy(A_b), -to_numpy(b_b))
            for bits, (A_b, b_b) in polytopes_dict.items()
        }

        # Sample directions once in main process (shared across all polytopes)
        directions = np.random.randn(n_directions, dim)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        # Accumulators: indexed by direction k
        widths_correct = {}          # {k: float}
        widths_bits    = {b: {} for b in bits_grid}  # {b: {k: float}}

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(A_np, b_ub_correct, polytopes_np, directions, bounds),
        ) as executor:
            for k, w_correct, w_bits in executor.map(
                _run_direction, range(n_directions)
            ):
                if w_correct is None:
                    continue   # correct polytope LP failed — discard direction
                widths_correct[k] = w_correct
                for bits, w in w_bits.items():
                    if w is not None:
                        widths_bits[bits][k] = w

        # Aggregate
        n_used        = len(widths_correct)
        width_correct = float(np.mean(list(widths_correct.values()))) if n_used > 0 else 0.0

        widths_both = {}
        for b in bits_grid:
            vals = list(widths_bits[b].values())
            if vals:
                widths_both[str(b)] = float(np.mean(vals))
            else:
                widths_both[str(b)] = 0.0   # empty polytope

        if verbose:
            print(f"  width_correct = {width_correct:.4f}  (n_used={n_used})")
            for b in bits_grid:
                n_b = len(widths_bits[b])
                print(f"  bits={b:2d}: width_both = {widths_both[str(b)]:.4f}"
                      f"  (n_used={n_b})")

        # Update record and write back
        record["width_correct"]     = width_correct
        record["widths_both"]       = widths_both
        record["n_directions_used"] = n_used

        with open(f, "w") as fh:
            json.dump(record, fh, indent=2)

        modified_indices.append(sample_idx)

    modified_indices.sort()

    if verbose:
        print(f"\nDone. Updated {len(modified_indices)} files.")
        print("Modified sample indices:", modified_indices)

    return modified_indices
