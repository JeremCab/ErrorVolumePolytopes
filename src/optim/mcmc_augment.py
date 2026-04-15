"""
mcmc_augment.py

Hit-and-run MCMC augmentation inside Polytope #1.

Given a reference point x0 and the Polytope #1 constraints (Ax + b <= 0),
finds a new point x' inside the same polytope where the quantized model has
a *different* ReLU activation pattern than at x0.

Convention: Ax + b <= 0  (same as the rest of this codebase).

The chord intersection is computed analytically — no LP solver needed.

Works for both MLP and CNN inputs:  x0 can be any shape; it is flattened
internally for the chord computation and reshaped back before the model call.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1.  Chord intersection
# ---------------------------------------------------------------------------

def chord_interval(x0_flat: np.ndarray,
                   A: np.ndarray,
                   b: np.ndarray,
                   d: np.ndarray) -> tuple[float, float]:
    """
    Given a point x0 inside {x : Ax + b <= 0} and a direction d, return
    (t_min, t_max) such that x0 + t*d is inside the polytope for t in
    [t_min, t_max].

    Parameters
    ----------
    x0_flat : (n,) numpy array — reference point (must be interior)
    A       : (m, n) numpy array — constraint LHS
    b       : (m,) numpy array  — constraint RHS  (Ax + b <= 0 convention)
    d       : (n,) numpy array  — direction (need not be unit)

    Returns
    -------
    t_min, t_max : floats  (t_min < 0 < t_max when x0 is strictly interior)
    """
    c = A @ d                              # (m,)
    s = -(A @ x0_flat + b) + 1e-6         # (m,), >= 0; eps absorbs floating-point violations at boundary

    pos = c > 0
    neg = c < 0

    t_max = float(np.min(s[pos] / c[pos])) if pos.any() else  np.inf
    t_min = float(np.max(s[neg] / c[neg])) if neg.any() else -np.inf

    return t_min, t_max


# ---------------------------------------------------------------------------
# 2.  Activation pattern
# ---------------------------------------------------------------------------

def activation_pattern(x: torch.Tensor, model: nn.Module) -> np.ndarray:
    """
    Returns the ReLU activation pattern of *model* at input *x* as a 1-D
    boolean numpy array (True = neuron active, i.e. pre-activation > 0).

    Hooks the output of every nn.Linear and nn.Conv2d layer, skipping the
    last one (the classification head).  This captures the pre-activation
    value for each hidden neuron and works for both architectures:

      - FashionMLP_Large uses nn.ReLU modules in an nn.Sequential, so the
        output of each hidden Linear is the value passed to ReLU.
      - FashionCNN_Small uses F.relu (functional), so there are no nn.ReLU
        modules to hook; instead we hook the preceding Conv2d / Linear layer.

    Parameters
    ----------
    x     : torch.Tensor — input in the model's native shape
    model : nn.Module    — should be in eval() mode (Dropout disabled)

    Returns
    -------
    pattern : (total_hidden_neurons,) bool numpy array
    """
    model.eval()
    captured = []
    hooks = []

    def _hook(module, inp, out):
        # out is the pre-activation value (before ReLU / F.relu)
        captured.append((out > 0).detach().cpu().numpy().flatten())

    # Collect all Linear and Conv2d layers, skip the last (output layer)
    linear_conv = [m for m in model.modules()
                   if isinstance(m, (nn.Linear, nn.Conv2d))]
    for m in linear_conv[:-1]:
        hooks.append(m.register_forward_hook(_hook))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return np.concatenate(captured)


# ---------------------------------------------------------------------------
# 3.  Hit-and-run augmentation
# ---------------------------------------------------------------------------

def find_augmented_point(
    x0: torch.Tensor,
    A: np.ndarray,
    b: np.ndarray,
    qmodel: nn.Module,
    max_tries: int = 200,
    rng: np.random.Generator | None = None,
) -> torch.Tensor | None:
    """
    Hit-and-run MCMC: find x' inside Polytope #1 where qmodel has a
    different activation pattern than at x0.

    Starting from x0, at each step:
      1. Sample a random unit direction d.
      2. Compute the chord [t_min, t_max] inside the polytope analytically.
      3. Sample t ~ Uniform(t_min, t_max) and set x' = x0 + t*d.
      4. Accept x' if its qmodel activation pattern differs from x0's.

    Parameters
    ----------
    x0       : torch.Tensor — reference point, any shape (flattened internally)
    A        : (m, n) numpy array — Polytope #1 LHS  (Ax + b <= 0)
    b        : (m,) numpy array  — Polytope #1 RHS
    qmodel   : nn.Module         — quantized model (eval mode recommended)
    max_tries: int               — number of attempts before returning None
    rng      : numpy Generator   — for reproducibility; defaults to new RNG

    Returns
    -------
    x' : torch.Tensor (same shape as x0), or None if not found within max_tries
    """
    if rng is None:
        rng = np.random.default_rng()

    qmodel.eval()

    x0_flat = x0.detach().cpu().numpy().flatten()   # (n,)
    n = x0_flat.shape[0]
    ref_pattern = activation_pattern(x0, qmodel)

    for _ in range(max_tries):
        # --- random unit direction ---
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)

        # --- chord endpoints ---
        t_min, t_max = chord_interval(x0_flat, A, b, d)
        if not (np.isfinite(t_min) and np.isfinite(t_max) and t_min < t_max):
            continue

        # --- sample a point on the chord ---
        t = rng.uniform(t_min, t_max)
        x_prime_flat = x0_flat + t * d

        # --- check activation pattern ---
        x_prime = torch.tensor(
            x_prime_flat.reshape(x0.shape),
            dtype=x0.dtype,
            device=x0.device,
        )
        pattern = activation_pattern(x_prime, qmodel)
        if not np.array_equal(pattern, ref_pattern):
            return x_prime

    return None
