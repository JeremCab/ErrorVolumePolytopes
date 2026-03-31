import numpy as np
from scipy.optimize import linprog
from src.optim.build_polytopes import build_model_correct_polytope, build_two_class_polytopes


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




def estimate_multi_bit_widths(A_correct, b_correct, bounds, polytopes_dict,
                               n_directions=200, verbose=False):
    """
    Estimate the mean width of one correct polytope and multiple b-approximated
    polytopes in a single pass, sharing the same random directions.

    This is the core estimator for the volume experiment (as opposed to the
    convergence experiment). There are no repetitions: one set of N directions
    is sampled and the mean over those directions is the volume estimate.

    Convention: Ax + b <= 0  (throughout this codebase)

    Parameters
    ----------
    A_correct : (m, d) array or torch.Tensor
    b_correct : (m,) array or torch.Tensor
    bounds : list of (lo, hi) pairs passed to linprog
    polytopes_dict : dict {bits (int): (A_both, b_both)}
        One entry per quantization bit-width. Each A_both / b_both follows
        the same Ax + b <= 0 convention.
    n_directions : int
        Number of random directions (default 200).
    verbose : bool

    Returns
    -------
    dict with keys:
        "width_correct"     : float  — mean-width estimate of the correct polytope
        "n_directions_used" : int    — directions on which all LPs succeeded
        "bits"              : dict   — {bits: {"width_both": float}}

    Notes
    -----
    Directions are shared across the correct polytope and all b-approximated
    polytopes (paired estimation). If any LP fails for a given direction, that
    direction is skipped entirely so all arrays remain aligned.

    The "mean" here is the mean over random directions, which defines the
    mean-width of the polytope. There is no averaging over repetitions
    (unlike estimate_polytope_width used in the convergence experiment).

    Errors (1 - width_both / width_correct) are NOT computed here — compute
    them from the returned widths to avoid ambiguity about what the reference
    correct polytope is.
    """

    # --- Convert to numpy once ---
    if hasattr(A_correct, "detach"):
        A_correct = A_correct.detach().cpu().numpy()
        b_correct = b_correct.detach().cpu().numpy()

    polytopes_np = {}
    for bits, (A_b, b_b) in polytopes_dict.items():
        if hasattr(A_b, "detach"):
            A_b = A_b.detach().cpu().numpy()
            b_b = b_b.detach().cpu().numpy()
        polytopes_np[bits] = (A_b, b_b)

    d = A_correct.shape[1]

    # --- Precompute b_ub = -b outside the loop ---
    b_ub_correct = -b_correct
    b_ub_dict = {bits: -b_b for bits, (_, b_b) in polytopes_np.items()}

    # --- Sample directions once, shared across all polytopes ---
    directions = np.random.randn(n_directions, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    widths_correct = []
    widths_both = {bits: [] for bits in polytopes_np}

    for k, u in enumerate(directions):

        # --- Correct polytope ---
        res_max_c = linprog(c=-u, A_ub=A_correct, b_ub=b_ub_correct, bounds=bounds, method="highs")
        res_min_c = linprog(c= u, A_ub=A_correct, b_ub=b_ub_correct, bounds=bounds, method="highs")

        if not (res_max_c.success and res_min_c.success):
            if verbose:
                print(f"Direction {k}: LP failed for correct polytope, skipping.")
            continue

        w_correct = (-res_max_c.fun) - res_min_c.fun

        # --- All b-approximated polytopes ---
        w_bits = {}
        failed = False
        for bits, (A_b, _) in polytopes_np.items():
            res_max_b = linprog(c=-u, A_ub=A_b, b_ub=b_ub_dict[bits], bounds=bounds, method="highs")
            res_min_b = linprog(c= u, A_ub=A_b, b_ub=b_ub_dict[bits], bounds=bounds, method="highs")
            if not (res_max_b.success and res_min_b.success):
                if verbose:
                    print(f"Direction {k}: LP failed for bits={bits}, skipping direction.")
                failed = True
                break
            w_bits[bits] = (-res_max_b.fun) - res_min_b.fun

        if failed:
            continue

        widths_correct.append(w_correct)
        for bits, w in w_bits.items():
            widths_both[bits].append(w)

        if verbose:
            bits_str = "  ".join(f"b{bits}={w:.4f}" for bits, w in w_bits.items())
            print(f"Direction {k}: w_correct={w_correct:.4f}  {bits_str}")

    widths_correct = np.array(widths_correct)
    width_correct = float(widths_correct.mean())
    n_used = len(widths_correct)

    bits_results = {}
    for bits in polytopes_np:
        arr = np.array(widths_both[bits])
        bits_results[bits] = {"width_both": float(arr.mean())}

    return {
        "width_correct":     width_correct,
        "n_directions_used": n_used,
        "bits":              bits_results,
    }


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


    # --------------------------------------------------------------------------- #
    # Evaluation of `build_two_class_polytopes` on a single sample of the dataset #
    # --------------------------------------------------------------------------- #
    print("\n\n*** Evaluation of `build_two_class_polytopes`\
           on a single sample of the dataset... ***\n\n")
    
    dim = 784
    nb_directions = 30

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



    # --------------------------------------------------------------------------- #
    # Evaluation of `estimate_multi_bit_widths` on a single sample of the dataset #
    # --------------------------------------------------------------------------- #
    print("\n\n*** Evaluation of `estimate_multi_bit_widths`\
           on a single sample of the dataset... ***\n\n")
    
    dim = 784
    nb_directions = 30

    bounds = [(-1., 1.)] * dim
    
    # Correct polytope: built ONCE outside the loop
    A_correct, b_correct = build_model_correct_polytope(model, x_0.flatten().unsqueeze(0), c)

    polytope_dict = {}
    for b in [4, 8]:
        qmodel_b = quantize_model(model, bits=b)
        qmodel_b.to(device).eval()
        _, _, A_both, b_both = build_two_class_polytopes(model, qmodel_b, x_0, c)
        polytope_dict[b] = (A_both, b_both)

    print("LP method...")
    results = estimate_multi_bit_widths(A_correct, b_correct, bounds, polytope_dict,
                                       n_directions=nb_directions, verbose=True)
    print("Mean width (correct):", results["width_correct"])
    for b in polytope_dict:
        print(f"Mean width (b{b})   :", results["bits"][b]["width_both"])


