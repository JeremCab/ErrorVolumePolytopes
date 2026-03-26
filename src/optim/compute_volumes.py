from unittest import result

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp

import numpy as np
from scipy.optimize import linprog

from src.shortcuts.shortcut_weights import compute_shortcut_weights
from src.optim.build_polytopes import build_two_class_polytopes



# ---------------------------------------------------------------- #
# OPTIMIZATION 1: compute widths in parallel using multiprocessing #
# ---------------------------------------------------------------- #


# def sample_unit_sphere_batch(n, dim):

#     U = np.random.normal(size=(n, dim))
#     U /= np.linalg.norm(U, axis=1, keepdims=True)

#     return U


# def build_activation_constraints(W):

#     A = W[:, 1:]
#     b = -W[:, 0]

#     return A, b


# def add_classification_constraints(A, b, W, c, nb_outputs):

#     y = W[-nb_outputs:]

#     constraints = y - y[c]

#     mask = np.arange(nb_outputs) != c
#     constraints = constraints[mask]

#     A_class = constraints[:, 1:]
#     b_class = -constraints[:, 0]

#     A_new = np.vstack((A, A_class))
#     b_new = np.concatenate((b, b_class))

#     return A_new, b_new


# def estimate_mean_width(A, b, bounds, directions):

#     nb_directions = directions.shape[0]

#     # Build all objective vectors
#     objectives = np.vstack((directions, -directions))

#     values = []

#     for c in objectives:

#         res = linprog(
#             c=c,
#             A_ub=A,
#             b_ub=b,
#             bounds=bounds,
#             method="highs"
#         )

#         if not res.success:
#             values.append(np.nan)
#         else:
#             values.append(res.fun)

#     values = np.array(values)

#     # Recover max and min projections
#     max_vals = -values[:nb_directions]
#     min_vals = values[nb_directions:]

#     widths = max_vals - min_vals

#     return float(np.nanmean(widths))


# def evaluate_sample(W_model, W_qmodel, c, bounds, directions, nb_outputs):

#     # activation region
#     A, b = build_activation_constraints(W_model)

#     # reference polytope
#     A_ref, b_ref = add_classification_constraints(
#         A, b, W_model, c, nb_outputs
#     )

#     # quantized polytope
#     A_q, b_q = add_classification_constraints(
#         A_ref, b_ref, W_qmodel, c, nb_outputs
#     )

#     width_ref = estimate_mean_width(A_ref, b_ref, bounds, directions)

#     if np.isnan(width_ref):
#         return np.nan, np.nan, np.nan

#     width_q = estimate_mean_width(A_q, b_q, bounds, directions)

#     if np.isnan(width_q):
#         return np.nan, np.nan, np.nan
    
#     error = 1.0 - width_q / width_ref

#     return width_ref, width_q, error


# def process_sample(idx, model, qmodel, dataset, bounds, directions):

#     x, c = dataset[idx]

#     nb_outputs = model(x.unsqueeze(0)).shape[-1]

#     # shortcut weights
#     sat_mask, unsat_mask = get_saturation_masks_including_output(
#         model, x, include_output=True
#     )

#     W_model = compute_shortcut_weights(model.net, sat_mask)

#     unsat_mask = torch.cat([v.flatten() for v in unsat_mask.values()])
#     unsat_mask = unsat_mask[:-nb_outputs].numpy()

#     rows = np.where(unsat_mask)[0]
#     W_model[rows] *= -1

#     qsat_mask, _ = get_saturation_masks_including_output(
#         qmodel, x, include_output=True
#     )

#     W_qmodel = compute_shortcut_weights(qmodel.net, qsat_mask)

#     return evaluate_sample(
#         W_model,
#         W_qmodel,
#         c,
#         bounds,
#         directions,
#         nb_outputs
#     )


# def evaluate_dataset(model, qmodel, dataset, bounds, directions, n_workers=4):

#     with ProcessPoolExecutor(max_workers=n_workers) as executor:

#         futures = {
#             executor.submit(
#                 process_sample,
#                 i,
#                 model,
#                 qmodel,
#                 dataset,
#                 bounds,
#                 directions
#             ): i
#             for i in range(len(dataset))
#         }

#         results = [None] * len(dataset)

#         for future in tqdm(as_completed(futures), total=len(futures)):
#             i = futures[future]
#             results[i] = future.result()

#     return results


# # ------------------------------------------------------------------ #
# # OPTIMIZATION 2: send models and dataset to the worker once for all # 
# # ------------------------------------------------------------------ #


# MODEL = None
# QMODEL = None
# DATASET = None
# BOUNDS = None
# DIRECTIONS = None
# NB_OUTPUTS = None


# def init_worker(model_path, bits, dataset, bounds, directions):

#     global MODEL, QMODEL, DATASET, BOUNDS, DIRECTIONS, NB_OUTPUTS

#     from src.models.networks import SmallMLP
#     from src.quantization.quantize import quantize_model

#     MODEL = SmallMLP()
#     MODEL.load_state_dict(torch.load(model_path)['model_state'])
#     MODEL.eval()

#     QMODEL = quantize_model(MODEL, bits=bits)
#     QMODEL.eval()

#     DATASET = dataset
#     BOUNDS = bounds
#     DIRECTIONS = directions

#     x0, _ = dataset[0]
#     NB_OUTPUTS = MODEL(x0.unsqueeze(0)).shape[-1]


# def process_sample_worker(i):

#     global MODEL, QMODEL, DATASET, BOUNDS, DIRECTIONS, NB_OUTPUTS

#     x_0, c = DATASET[i]

#     sat_mask, unsat_mask = get_saturation_masks_including_output(
#         MODEL, x_0, include_output=True
#     )

#     W_model = compute_shortcut_weights(MODEL.net, sat_mask)

#     unsat_mask = torch.cat([v.flatten() for v in unsat_mask.values()])
#     unsat_mask = unsat_mask[:-NB_OUTPUTS].numpy()

#     W_model[:-NB_OUTPUTS][unsat_mask] *= -1

#     qsat_mask, _ = get_saturation_masks_including_output(
#         QMODEL, x_0, include_output=True
#     )

#     W_qmodel = compute_shortcut_weights(QMODEL.net, qsat_mask)

#     return evaluate_sample(
#         W_model,
#         W_qmodel,
#         c,
#         BOUNDS,
#         DIRECTIONS,
#         NB_OUTPUTS
#     )


# def evaluate_dataset_worker(dataset, model_path, bits, bounds, directions, n_workers=8):

#     results = []

#     with ProcessPoolExecutor(
#         max_workers=n_workers,
#         initializer=init_worker,
#         initargs=(model_path, bits, dataset, bounds, directions)
#     ) as executor:

#         futures = {
#         executor.submit(process_sample_worker, i): i
#         for i in range(len(dataset))
#         }

#         results = [None] * len(dataset)

#         for future in tqdm(as_completed(futures), total=len(futures)):
#             i = futures[future]
#             try:
#                 results[i] = future.result()
#             except Exception as e:
#                 print(f"Sample {i} failed:", e)
#                 results[i] = (np.nan, np.nan, np.nan)

#     return results




def estimate_polytope_width(A, b, bounds, n_directions=100, 
                            tol=1e-9, verbose=False):
    """
    Estimate polytope volume surrogate via random directional widths.

    Parameters
    ----------
    A : (m, d) array
    b : (m,) array
    bounds : list of (l, u)
    n_directions : int
    tol : float

    Returns
    -------
    mean_width : float
    widths : list
    """

    # Convert to numpy if torch
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
        b = b.detach().cpu().numpy()

    m, d = A.shape

    widths = []

    # Sample directions
    directions = np.random.randn(n_directions, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    for k, u in enumerate(directions):

        # --- MAX ---
        res_max = linprog(
            c=-u,
            A_ub=A,
            b_ub=-b,
            bounds=bounds,
            method="highs"
        )

        # --- MIN ---
        res_min = linprog(
            c=u,
            A_ub=A,
            b_ub=-b,
            bounds=bounds,
            method="highs"
        )

        if not (res_max.success and res_min.success):
            if verbose:
                print(f"LP failed at direction {k}")
            continue

        width = (-res_max.fun) - (res_min.fun)
        widths.append(width)
        if verbose:
            print("LP success at direction", k, "width:", width)

    return np.mean(widths), np.std(widths)



# ---------------------------------------------- #
# Hit & Run : Fast approcimation of LP solutions #
# ---------------------------------------------- #

def add_box_constraints(A, b, lower=-1.0, upper=1.0):
    """
    Add box constraints: lower <= x_i <= upper

    Converts them into:
        x_i <= upper
        -x_i <= -lower

    Returns augmented (A, b)
    """

    d = A.shape[1]
    I = np.eye(d)

    A_box = np.vstack([I, -I])
    b_box = np.hstack([-upper * np.ones(d), lower * np.ones(d)])

    A_new = np.vstack([A, A_box])
    b_new = np.hstack([b, b_box])

    return A_new, b_new


def chord_length_from_point(A, b, x0, u, tol=1e-12):
    """
    Compute chord length of polytope along direction u,
    passing through point x0.

    Polytope:
        A x + b <= 0

    Returns:
        length = t_max - t_min
    """

    v = u / np.linalg.norm(u)

    Av = A @ v
    Ax = A @ x0

    t_min = -np.inf
    t_max = np.inf

    for i in range(len(b)): # XXX ask why???

        if abs(Av[i]) < tol:
            # Parallel → no restriction
            continue

        # A(x + t v) + b <= 0
        # ⇒ Ax + t Av + b <= 0
        # ⇒ t <= (-b - Ax)/Av  or ≥ depending on sign
        t = (-b[i] - Ax[i]) / Av[i]

        if Av[i] > 0:
            t_max = min(t_max, t)
        else:
            t_min = max(t_min, t)

    if t_min > t_max + tol:
        raise ValueError("Empty intersection (x0 may not be feasible)")

    return t_max - t_min



def hybrid_chord_width(
    A,
    b,
    x0,
    n_directions=100,
    add_bounds=True,
    lower=-1.0,
    upper=1.0,
    seed=None,
):
    """
    Fast approximation of polytope width using chord estimator.

    Parameters
    ----------
    A, b : define polytope A x + b <= 0
    x0 : feasible point (numpy array shape (d,))
    n_directions : number of random directions
    add_bounds : whether to include box constraints [-1,1]^d
    seed : reproducibility

    Returns
    -------
    mean_width, std_width, widths
    """

    # --- Convert to numpy if needed ---
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
        b = b.detach().cpu().numpy()

    if hasattr(x0, "detach"):
        x0 = x0.detach().cpu().numpy()

    x0 = x0.reshape(-1)

    # --- Add box constraints ---
    if add_bounds:
        A, b = add_box_constraints(A, b, lower=lower, upper=upper)

    # --- Check feasibility ---
    if not np.all(A @ x0 + b <= 1e-5):
        print(f"{np.max(A @ x0 + b)} : WARNING: x0 slightly infeasible (numerical)")

    # --- Random generator ---
    rng = np.random.default_rng(seed)

    d = x0.shape[0]
    widths = []

    for _ in range(n_directions):

        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)

        w = chord_length_from_point(A, b, x0, u)
        widths.append(w)

    widths = np.array(widths)

    return widths.mean(), widths.std(), widths



# XXX XXX XXX #
# XXX XXX XXX #
# XXX XXX XXX #



def hit_and_run_step(x, A, b, tol=1e-12):
    """
    Perform one Hit-and-Run step inside the polytope:

        A x + b <= 0

    Parameters
    ----------
    x : (d,) current point (must be feasible)
    A : (m, d)
    b : (m,)
    tol : float (numerical tolerance)

    Returns
    -------
    x_new : (d,) next sampled point
    """


    # Convert to numpy if torch
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
        b = b.detach().cpu().numpy()

    d = x.shape[0]

    # --- 1. Sample random direction on unit sphere ---
    v = np.random.randn(d)
    v /= np.linalg.norm(v)

    # --- 2. Precompute ---
    Av = A @ v           # shape (m,)
    Ax = A @ x           # shape (m,)

    # --- 3. Compute feasible interval [t_min, t_max] ---
    t_min = -np.inf
    t_max = np.inf

    for i in range(len(b)):

        if abs(Av[i]) < tol:
            # Direction is parallel to constraint → ignore
            continue

        # A x + b <= 0  ⇒  A x <= -b
        t = (-b[i] - Ax[i]) / Av[i]

        if Av[i] > 0:
            # upper bound
            t_max = min(t_max, t)
        else:
            # lower bound
            t_min = max(t_min, t)

    # Safety check (should not happen if x is inside)
    if t_min > t_max:
        raise ValueError("Empty interval in Hit-and-Run step")

    # --- 4. Sample uniformly ---
    t_sample = np.random.uniform(t_min, t_max)

    return x + t_sample * v


def sample_polytope(A, b, x0, n_samples=1000, burnin=100):
    """
    Generate samples approximately uniform in the polytope
    using Hit-and-Run.

    Parameters
    ----------
    x0 : initial feasible point
    burnin : number of steps before collecting samples
    """

    x = x0.copy()
    samples = []

    for i in range(n_samples + burnin):
        x = hit_and_run_step(x, A, b)

        if i >= burnin:
            samples.append(x.copy())

    return np.array(samples)


def hit_and_run_estimate_width(A, b, x0,
                              n_directions=100,
                              n_samples=2000,
                              burnin=200):

    print("Sampling points with Hit-and-Run...")

    A, b = add_box_constraints(A, b) # ensure boundedness for Hit-and-Run!

    samples = sample_polytope(A, b, x0,
                             n_samples=n_samples,
                             burnin=burnin)

    d = samples.shape[1]
    widths = []

    for _ in range(n_directions):
        u = np.random.randn(d)
        u /= np.linalg.norm(u)

        projections = samples @ u  # shape (n_samples,)
        width = projections.max() - projections.min()

        widths.append(width)

    widths = np.array(widths)

    return widths.mean(), widths.std(), widths




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
    bits = 4
    p = 0.85
    device = torch.device("cpu")

    # Model loading
    model_name = f"{model_type}_{nb_epochs}.pth"
    model_path = os.path.join("./src/models/checkpoints", model_name)

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
    nb_directions = 20

    bounds = [(-1., 1.)] * dim
    
    A_correct, b_correct, A_both, b_both = build_two_class_polytopes(model, qmodel, x_0, c)

    print("LP method...")
    results = estimate_polytope_width(A_correct, b_correct, bounds, 
                                      n_directions=nb_directions,
                                      tol=1e-9, verbose=True)
    print("Mean polytope width:", results[0])
    print("Std polytope width:", results[1])


    print("\nHit & Run method...")
    x_0 = x_0.view(-1).cpu().numpy() # XXX important: convert to numpy and flatten

    # mean_width, std_width, widths = hit_and_run_estimate_width(
    #     A_correct, b_correct, x_0,
    #     n_directions=nb_directions,
    #     n_samples=1000,
    #     burnin=200
    # )
    mean_width, std_width, widths = hybrid_chord_width(
        A_correct, b_correct, x_0,
        n_directions=nb_directions,
        add_bounds=True,
        lower=-1.0,
        upper=1.0,
        seed=42
    )
    print("Hit-and-Run mean width:", mean_width)
    print("Hit-and-Run std width:", std_width)
