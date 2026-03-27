from unittest import result

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp

import numpy as np
from scipy.optimize import linprog

from src.shortcuts.shortcut_weights import compute_shortcut_weights
from src.optim.build_polytopes import build_two_class_polytopes



def estimate_polytope_width(A_correct, b_correct, bounds,
                             A_both=None, b_both=None,
                             n_directions=100, verbose=False):
    """
    Estimate mean width of the correct polytope, and optionally the error
    (eq. 27) by also estimating the width of the 'both' polytope.

    Convention: Ax + b <= 0  (throughout this codebase)

    Parameters
    ----------
    A_correct : (m, d) array or torch.Tensor
    b_correct : (m,) array or torch.Tensor
    bounds : list of (l, u) pairs passed to linprog
    A_both : (m + k, d), optional
        If provided together with b_both, activates paired mode:
        widths of both polytopes are estimated with the same directions
        and the error (eq. 27) is returned.
    b_both : (m + k,), optional
    n_directions : int
    verbose : bool

    Returns
    -------
    dict with keys:
        "mean_width_correct", "std_width_correct"
        "n_directions_used"
        (paired mode only) "mean_width_both", "std_width_both", "error"
    """

    if (A_both is None) != (b_both is None):
        raise ValueError("A_both and b_both must both be provided or both be None.")

    paired = A_both is not None

    # --- Convert to numpy once ---
    if hasattr(A_correct, "detach"):
        A_correct = A_correct.detach().cpu().numpy()
        b_correct = b_correct.detach().cpu().numpy()

    if paired:
        if hasattr(A_both, "detach"):
            A_both = A_both.detach().cpu().numpy()
            b_both  = b_both.detach().cpu().numpy()

    d = A_correct.shape[1]

    # --- Precompute b_ub outside the loop ---
    b_ub_correct = -b_correct
    if paired:
        b_ub_both = -b_both

    # --- Sample directions once, shared across both polytopes ---
    directions = np.random.randn(n_directions, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    widths_correct = []
    widths_both    = []

    for k, u in enumerate(directions):

        # Width of A_correct
        res_max_c = linprog(c=-u, A_ub=A_correct, b_ub=b_ub_correct, bounds=bounds, method="highs")
        res_min_c = linprog(c= u, A_ub=A_correct, b_ub=b_ub_correct, bounds=bounds, method="highs")

        if not (res_max_c.success and res_min_c.success):
            if verbose:
                print(f"LP failed for A_correct at direction {k}")
            continue

        w_correct = (-res_max_c.fun) - res_min_c.fun

        if paired:
            # Width of A_both — skip this direction entirely if it fails,
            # to keep widths_correct and widths_both strictly aligned.
            res_max_b = linprog(c=-u, A_ub=A_both, b_ub=b_ub_both, bounds=bounds, method="highs")
            res_min_b = linprog(c= u, A_ub=A_both, b_ub=b_ub_both, bounds=bounds, method="highs")

            if not (res_max_b.success and res_min_b.success):
                if verbose:
                    print(f"LP failed for A_both at direction {k}, skipping direction.")
                continue

            w_both = (-res_max_b.fun) - res_min_b.fun
            widths_both.append(w_both)

            if verbose:
                print(f"Direction {k}: w_correct={w_correct:.4f}, w_both={w_both:.4f}")
        else:
            if verbose:
                print(f"Direction {k}: w_correct={w_correct:.4f}")

        widths_correct.append(w_correct)

    widths_correct = np.array(widths_correct)

    result = {
        "mean_width_correct": float(widths_correct.mean()),
        "std_width_correct":  float(widths_correct.std()),
        "n_directions_used":  len(widths_correct),
    }

    if paired:
        widths_both  = np.array(widths_both)
        mean_correct = result["mean_width_correct"]
        mean_both    = float(widths_both.mean())
        error = (1.0 - mean_both / mean_correct) if mean_correct > 0 else float("nan")
        result["mean_width_both"] = mean_both
        result["std_width_both"]  = float(widths_both.std())
        result["error"]           = error

    return result




# ================================================= #
# Example usage: 'Play' or run from root directory: #
# >>> python -m src.optim.compute_volumes           #
# ================================================= #

if __name__ == "__main__":

    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    import torch
    from torch.utils.data import Subset

    from data.mnist_data import load_mnist_datasets
    from src.models.networks import SmallMLP
    from src.quantization.quantize import quantize_model
    from src.shortcuts.shortcut_weights import *

    import matplotlib.pyplot as plt

    # ============================= #
    # Compute polytope-based error  #
    # ============================= #

    # -------------------- #
    # Load models and data #
    # -------------------- #
    print("\n\n*** Compute polytope-based errors with LPs... ***\n\n")

    model_type = "smallmlp"
    nb_epochs = 10
    bits = 2
    p = 0.85
    device = torch.device("cpu")

    # Model loading
    model_name = f"{model_type}_{nb_epochs}.pth"
    model_path = os.path.join("./checkpoints", model_name)

    # Model
    model = SmallMLP()
    model.load_state_dict(torch.load(model_path)['model_state'])
    model.to(device).eval()

    # qModel
    qmodel = quantize_model(model, bits=bits)
    qmodel.to(device).eval()

    # Dataset (and sample)
    _, test_dataset = load_mnist_datasets()
    x_0, c = test_dataset[123]
    print("Models and dataset have been loaded.")


    # -------------------------------------------- #
    # Evaluation on a single sample of the dataset #
    # -------------------------------------------- #
    print("\n\n*** Evaluation on a single sample of the dataset... ***\n\n")
    
    dim = 784
    nb_directions = 100

    bounds = [(-1., 1.)] * dim
    
    A_correct, b_correct, A_both, b_both = build_two_class_polytopes(model, qmodel, x_0, c)

    print("LP method...")
    results = estimate_polytope_width(A_correct, b_correct, bounds,
                                      A_both=A_both, b_both=b_both,
                                      n_directions=nb_directions, verbose=True)
    print("Mean width (correct):", results["mean_width_correct"])
    print("Std  width (correct):", results["std_width_correct"])
    print("Mean width (both)   :", results["mean_width_both"])
    print("Error               :", results["error"])


