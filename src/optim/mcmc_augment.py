"""
mcmc_augment.py

Hit-and-run MCMC augmentation inside Polytope #1.

Three strategies are provided:

  find_augmented_point        (Strategy A — activation pattern)
      Finds x' inside Polytope #1 where the q-model has a *different* ReLU
      activation pattern than at x0.  Fast acceptance check (no model
      forward pass beyond the pattern hook), but the resulting x' may still
      have V3(x') ≈ V2(x') because it does not target the q-model's
      classification margin.  Returns at most ONE point (or None).

  find_augmented_point_margin (Strategy B — classification margin)
      Evaluates both chord endpoints for every random direction and keeps the
      x' (across all max_tries directions) that *minimises* the q-model's
      classification margin for the correct class c:

          margin(x') = logit_c(x') − max_{k ≠ c} logit_k(x')

      A smaller (or negative) margin means the q-model is near (or past) its
      decision boundary → V3(x') << V2(x').  No extra LP calls.
      Always returns exactly ONE tensor.

  find_augmented_points_walk  (Strategy C — full MCMC walk, paper algorithm)
      Implements the full hit-and-run Markov chain described in Šíma &
      Cabessa (ICONIP 2026).  Runs N steps of a TRUE Markov chain inside P1
      (x_k = x_{k-1} + λ_k u_k, always moving), collects one representative
      per distinct q-model activation-pattern equivalence class encountered.
      Returns a LIST of tensors (one per new class found), potentially empty.

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
    pixel_lo: float = -1.0,
    pixel_hi: float =  1.0,
) -> torch.Tensor | None:
    """
    Hit-and-run MCMC: find x' inside Polytope #1 where qmodel has a
    different activation pattern than at x0.

    Starting from x0, at each step:
      1. Sample a random unit direction d.
      2. Compute the chord [t_min, t_max] inside the polytope analytically.
      3. Sample t ~ Uniform(t_min, t_max) and set x' = x0 + t*d.
      4. Clip x' to [pixel_lo, pixel_hi]^n  (projected hit-and-run).
      5. Accept x' if its qmodel activation pattern differs from x0's.

    Parameters
    ----------
    x0       : torch.Tensor — reference point, any shape (flattened internally)
    A        : (m, n) numpy array — Polytope #1 LHS  (Ax + b <= 0)
    b        : (m,) numpy array  — Polytope #1 RHS
    qmodel   : nn.Module         — quantized model (eval mode recommended)
    max_tries: int               — number of attempts before returning None
    rng      : numpy Generator   — for reproducibility; defaults to new RNG
    pixel_lo : float             — lower pixel bound after clipping (default −1)
    pixel_hi : float             — upper pixel bound after clipping (default +1)

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

        # --- sample a point on the chord, then project onto pixel box ---
        t = rng.uniform(t_min, t_max)
        x_prime_flat = np.clip(x0_flat + t * d, pixel_lo, pixel_hi)

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


# ---------------------------------------------------------------------------
# 4.  q-model classification margin helper
# ---------------------------------------------------------------------------

def _qmodel_margin(x: torch.Tensor, qmodel: nn.Module, c: int) -> float:
    """
    Return the q-model's classification margin for class c at input x:

        margin = logit_c(x) − max_{k ≠ c} logit_k(x)

    Positive  → q-model predicts c (correctly).
    Zero      → q-model is on the decision boundary.
    Negative  → q-model misclassifies x.
    """
    with torch.no_grad():
        logits = qmodel(x).squeeze(0).cpu().float().numpy()   # (n_classes,)
    other = np.delete(logits, c)
    return float(logits[c] - other.max())


# ---------------------------------------------------------------------------
# 5.  Hit-and-run augmentation — Strategy B (margin minimisation)
# ---------------------------------------------------------------------------

def find_augmented_point_margin(
    x0: torch.Tensor,
    A: np.ndarray,
    b: np.ndarray,
    qmodel: nn.Module,
    c: int,
    max_tries: int = 200,
    rng: np.random.Generator | None = None,
    pixel_lo: float = -1.0,
    pixel_hi: float =  1.0,
) -> torch.Tensor:
    """
    Hit-and-run augmentation that targets the q-model's classification
    boundary instead of its activation-pattern boundary.

    For each of max_tries random directions:
      1. Compute the chord [t_min, t_max] inside Polytope #1 analytically.
      2. Evaluate the q-model at both chord endpoints (with a tiny inward
         step for numerical safety).
      3. Track the endpoint with the smallest classification margin for
         class c across all directions.

    Returns the x' with the globally smallest margin.  If that margin is
    negative the q-model misclassifies x' (V3 = 0); if it is small but
    positive the q-model is near its boundary (V3 << V2).

    Unlike find_augmented_point this function always returns a tensor
    (never None): if no valid chord is found in max_tries attempts the
    fallback is x0 itself.

    Parameters
    ----------
    x0       : torch.Tensor — reference point, any shape (flattened internally)
    A        : (m, n) numpy array — Polytope #1 LHS  (Ax + b <= 0)
    b        : (m,) numpy array  — Polytope #1 RHS
    qmodel   : nn.Module         — quantized model (eval mode recommended)
    c        : int               — correct class index
    max_tries: int               — number of random directions to try
    rng      : numpy Generator   — for reproducibility; defaults to new RNG
    pixel_lo : float             — lower pixel bound after clipping (default −1)
    pixel_hi : float             — upper pixel bound after clipping (default +1)

    Returns
    -------
    x' : torch.Tensor (same shape as x0)
    """
    if rng is None:
        rng = np.random.default_rng()

    qmodel.eval()

    x0_flat = x0.detach().cpu().numpy().flatten()   # (n,)
    n = x0_flat.shape[0]

    best_x_flat = x0_flat          # fallback: return x0 if no chord found
    best_margin = _qmodel_margin(x0, qmodel, c)

    for _ in range(max_tries):
        # --- random unit direction ---
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)

        # --- chord endpoints ---
        t_min, t_max = chord_interval(x0_flat, A, b, d)
        if not (np.isfinite(t_min) and np.isfinite(t_max) and t_min < t_max):
            continue

        # small inward step so we stay strictly inside the polytope
        eps_t = 1e-4 * (t_max - t_min)

        for t in (t_min + eps_t, t_max - eps_t):
            # project onto pixel box (projected hit-and-run)
            x_cand_flat = np.clip(x0_flat + t * d, pixel_lo, pixel_hi)
            x_cand = torch.tensor(
                x_cand_flat.reshape(x0.shape),
                dtype=x0.dtype,
                device=x0.device,
            )
            margin = _qmodel_margin(x_cand, qmodel, c)
            if margin < best_margin:
                best_margin = margin
                best_x_flat = x_cand_flat

    return torch.tensor(
        best_x_flat.reshape(x0.shape),
        dtype=x0.dtype,
        device=x0.device,
    )


# ---------------------------------------------------------------------------
# 6.  Full hit-and-run MCMC walk — Strategy C (paper algorithm)
# ---------------------------------------------------------------------------

def find_augmented_points_walk(
    x0: torch.Tensor,
    A: np.ndarray,
    b: np.ndarray,
    qmodel: nn.Module,
    nb_aug_points: int = 10,
    max_steps: int = 2000,
    rng: np.random.Generator | None = None,
    pixel_lo: float = -1.0,
    pixel_hi: float =  1.0,
    mode: str = "projected",
    p1_filter_tol: float | None = None,
) -> list[torch.Tensor]:
    """
    Full hit-and-run MCMC walk inside Polytope #1.

    Two modes are available, selected via the `mode` parameter:

    mode = "projected"  (default — practical, pixel-safe)
        Classical hit-and-run inside A (P1 constraints only), then the step
        is clipped onto [pixel_lo, pixel_hi]^n:

            x_k = clip( x_{k-1} + λ_k u_k, pixel_lo, pixel_hi )

        The clip ("projected hit-and-run") is necessary because real images
        have many pixels at exactly ±1 after normalisation; adding pixel-
        bound rows to A collapses all chords to length 0 for such starting
        points.  The projected point may slightly violate A, but this is
        absorbed by the 1e-6 tolerance in chord_interval.
        Walk mixes freely — diverse augmented points.

    mode = "pixel_bounds"  (paper-exact, degenerate for real images)
        Pixel-bound constraints (±x_j ≤ 1, i.e. x_j ∈ [pixel_lo, pixel_hi])
        are added as explicit rows to A before every chord computation.
        This is exactly the polytope Ξ̄_x from the paper (eqs. 9–10, 12).
        For real images with pixels at exactly ±1, the starting point sits on
        ~355 faces of the box simultaneously → every chord has length ≈ 0 →
        walk is stuck at x0 → all augmented points ≈ x0.

    In both modes the chain ALWAYS moves (every step accepted). One
    representative is kept per distinct q-model activation pattern Ã(x_k),
    excluding x0's own pattern.

    Parameters
    ----------
    x0            : torch.Tensor   — starting point x ∈ T, any shape
    A             : (m, n) array   — P1 constraint LHS  (Ax + b ≤ 0)
    b             : (m,) array     — P1 constraint RHS
    qmodel        : nn.Module      — quantized model (eval mode recommended)
    nb_aug_points : int            — target number of new representatives
    max_steps     : int            — hard cap on walk steps
    rng           : numpy Generator — for reproducibility; defaults to new RNG
    pixel_lo      : float          — lower pixel bound (default −1)
    pixel_hi      : float          — upper pixel bound (default +1)
    mode          : str            — "projected" (default) or "pixel_bounds"
    p1_filter_tol : float | None   — if not None, drop representatives that
                                     violate any P1 constraint by more than
                                     this tolerance (max(A·x + b) > tol).
                                     Only meaningful for mode="projected", where
                                     clipping may slightly violate A.
                                     Pixel bounds are always satisfied by
                                     construction in projected mode (via clip).
                                     Default: None (no filtering).

    Returns
    -------
    representatives : list of torch.Tensor
        One representative per new equivalence class found.  Same shape as x0.
        len(representatives) <= nb_aug_points.  May be empty.
        If p1_filter_tol is set, only strict P1 members are returned.
    """
    if mode not in ("projected", "pixel_bounds"):
        raise ValueError(f"mode must be 'projected' or 'pixel_bounds', got {mode!r}")
    if rng is None:
        rng = np.random.default_rng()

    qmodel.eval()

    x0_flat = x0.detach().cpu().numpy().flatten()   # (n,)
    n = x0_flat.shape[0]

    # In pixel_bounds mode, append box constraints once (they are fixed)
    if mode == "pixel_bounds":
        I = np.eye(n, dtype=A.dtype)
        A_walk = np.vstack([A,  I, -I])
        b_walk = np.concatenate([b, -np.full(n, pixel_hi), np.full(n, pixel_lo)])
    else:
        A_walk = A
        b_walk = b

    ref_pattern_tuple: tuple = tuple(activation_pattern(x0, qmodel))
    seen_patterns: set[tuple] = {ref_pattern_tuple}
    representatives: list[torch.Tensor] = []

    x_cur_flat = x0_flat.copy()

    for _ in range(max_steps):
        # --- random unit direction ---
        d = rng.standard_normal(n)
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            continue
        d /= norm

        # --- chord inside walk polytope from current position ---
        t_min, t_max = chord_interval(x_cur_flat, A_walk, b_walk, d)
        if not (np.isfinite(t_min) and np.isfinite(t_max) and t_min < t_max):
            continue

        # --- uniform sample on chord ---
        t = rng.uniform(t_min, t_max)
        x_next_flat = x_cur_flat + t * d

        # In projected mode: clip onto pixel box after the step
        if mode == "projected":
            x_next_flat = np.clip(x_next_flat, pixel_lo, pixel_hi)

        x_next = torch.tensor(
            x_next_flat.reshape(x0.shape),
            dtype=x0.dtype,
            device=x0.device,
        )

        # --- update Markov chain state ---
        x_cur_flat = x_next_flat

        # --- check if new equivalence class ---
        pattern_tuple = tuple(activation_pattern(x_next, qmodel))
        if pattern_tuple not in seen_patterns:
            seen_patterns.add(pattern_tuple)
            representatives.append(x_next.detach().cpu())
            if len(representatives) >= nb_aug_points:
                break   # target reached — stop early

    # Optional post-walk filter: keep only strict P1 members.
    # Pixel bounds are satisfied by construction in projected mode (via clip),
    # so only the A·x + b ≤ 0 constraints need to be checked here.
    if p1_filter_tol is not None:
        filtered = []
        for rep in representatives:
            rep_flat = rep.detach().cpu().numpy().flatten()
            max_violation = float((A @ rep_flat + b).max())
            if max_violation <= p1_filter_tol:
                filtered.append(rep)
        representatives = filtered

    return representatives


# ---------------------------------------------------------------------------
# 7.  Greedy diversity selection (farthest-point / Gonzalez algorithm)
# ---------------------------------------------------------------------------

def select_diverse_representatives(
    reps: list[torch.Tensor],
    x0: torch.Tensor,
    k: int,
) -> list[torch.Tensor]:
    """
    Select the k most spatially diverse representatives from *reps* using
    greedy farthest-point (max-min) sampling (Gonzalez 1985):

        1. Initialise the selected set S = {x0}  (anchor on the original point).
        2. For each of k iterations, add to S the remaining point in *reps*
           that maximises its minimum L2 distance to any point already in S.

    This is a 2-approximation of the optimal k-center problem: the minimum
    pairwise distance in the returned set is at most 2× the optimum.

    If len(reps) <= k, returns all of *reps* unchanged.

    Parameters
    ----------
    reps : list of torch.Tensor  — candidates (output of find_augmented_points_walk)
    x0   : torch.Tensor          — original data point (anchor)
    k    : int                   — number of representatives to select

    Returns
    -------
    selected : list of torch.Tensor  (len == min(k, len(reps)))
    """
    if len(reps) <= k:
        return list(reps)

    x0_np = x0.detach().cpu().numpy().flatten()                       # (d,)
    R     = np.stack([r.detach().cpu().numpy().flatten() for r in reps])  # (n, d)
    n     = len(reps)

    # Minimum distance from each candidate to the selected set.
    # Initialise by anchoring on x0.
    min_dists = np.linalg.norm(R - x0_np, axis=1)   # (n,)
    available = np.ones(n, dtype=bool)
    selected  : list[int] = []

    for _ in range(k):
        avail_idx = np.where(available)[0]
        best      = int(avail_idx[np.argmax(min_dists[avail_idx])])
        selected.append(best)
        available[best] = False
        if not available.any():
            break
        # Update min-dists: take element-wise minimum with distance to new point
        rest     = np.where(available)[0]
        new_d    = np.linalg.norm(R[rest] - R[best], axis=1)
        min_dists[rest] = np.minimum(min_dists[rest], new_d)

    return [reps[i] for i in selected]
