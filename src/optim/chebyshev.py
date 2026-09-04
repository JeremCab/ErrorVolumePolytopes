"""
chebyshev.py

Largest-inscribed-ball (Chebyshev) radius of a polytope, used to decide whether a
polytope is full-dimensional (positive volume) or lower-dimensional (zero volume).
Independent of the mean-width machinery — a direct volume-emptiness test.

Convention
----------
Polytopes are given in the codebase's native `A x + b <= 0` form (same as
build_polytopes* / compute_volumes). We reuse `_prep_lp` to convert to the standard
`A_ub x <= b_ub` form (b_ub = -b) and prune zero rows, then solve the Chebyshev LP:

    max_{x, r}  r
    s.t.        a_i^T x + ||a_i|| * r <= (b_ub)_i   for every constraint i
                                                    (including input-box rows)
                r >= 0

Interpretation of the optimum r*:
    r* > 0  (above tol)  -> full-dimensional  -> POSITIVE volume
    r* = 0  (within tol) -> lower-dimensional -> ZERO volume
    LP infeasible        -> EMPTY polytope

Note that `linprog` reports success=False for several distinct reasons, and only
status 2 (infeasible) means the polytope is empty.  A solver failure (iteration
limit, unbounded, numerical difficulties) is reported as 'failed' and must never
be read as 'empty': on ill-conditioned CNN polytopes that would silently drop a
P3(k) from the GACC denominator.

The `||a_i||` normalisation makes r the true Euclidean inradius. Reference:
Boyd & Vandenberghe, Convex Optimization, Section 4.3.1 (Chebyshev centre).
"""

import numpy as np
from scipy.optimize import linprog

from src.optim.compute_volumes import _prep_lp


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def chebyshev_radius(A, b, box=(-1.0, 1.0), eps=0.0,
                     zero_tol=1e-6, method="highs", scale_rows=True):
    """
    Chebyshev (largest inscribed ball) radius of  {x : A x + b <= 0} ∩ box.

    Parameters
    ----------
    A, b : constraint arrays/tensors in the native `A x + b <= 0` convention.
    box  : (lo, hi) input-domain bounds applied to every coordinate, added as
           explicit rows so the inscribed ball respects them. Pass None to skip.
           Default (-1, 1) matches the bounds used by the mean-width pipeline.
    eps  : slack forwarded to `_prep_lp`. Default 0.0 gives the exact polytope;
           pass 1e-6 to reproduce the slightly-inflated polytope the mean-width
           code actually solved over.
    zero_tol : radius at/below which the polytope is declared zero-volume.
    method   : linprog solver (default 'highs').
    scale_rows : divide each constraint by ||a_j|| before solving (default True).
           Dividing an inequality by a positive number leaves its solution set
           unchanged, so this is the SAME LP — but the rho column becomes all-ones
           instead of holding the row norms, and every row has unit norm.
           Measured on sample 760 (CNN): at b=16 both forms return the same radius
           (relative gap 6e-6) with the scaled one 1.1x faster; at b=4 the UNSCALED
           form fails to converge altogether (2029 s, no solution) while the scaled
           one solves in 1628 s. Pass False only to reproduce the older unscaled
           results.

    Returns
    -------
    dict:
      'radius'        : float r*  (np.nan if 'empty' or 'failed')
      'status'        : 'full_dim' | 'zero_volume' | 'empty' | 'failed'
                        'empty'  -> LP infeasible (linprog status 2): the
                                    polytope really is empty.
                        'failed' -> solver did not converge (status 1/3/4):
                                    the result is unusable and says NOTHING
                                    about the volume.
      'success'       : bool  (LP solved to optimality)
      'n_constraints' : rows used (after pruning, incl. box rows)
      'lp_status'     : int   — scipy linprog status code
      'lp_message'    : str   — scipy linprog message
    """
    A = _to_numpy(A).astype(float)
    b = _to_numpy(b).astype(float)

    # 1. Native (A x + b <= 0)  ->  standard (A_ub x <= b_ub); prune zero rows.
    A_ub, b_ub = _prep_lp(A, b, eps=eps)
    n = A_ub.shape[1]

    # 2. Append box constraints as explicit rows (each has Euclidean norm 1):
    #      x_i <=  hi   -> row  e_i,  rhs  hi
    #     -x_i <= -lo   -> row -e_i,  rhs -lo
    if box is not None:
        lo, hi = box
        I = np.eye(n)
        A_all = np.vstack([A_ub, I, -I])
        b_all = np.concatenate([b_ub, np.full(n, hi), np.full(n, -lo)])
    else:
        A_all, b_all = A_ub, b_ub

    # 3. Row norms = coefficient of r in each Chebyshev constraint.
    #    (_prep_lp has pruned rows of norm <= 1e-10, and the box rows have norm 1,
    #     so every norm here is strictly positive.)
    norms = np.linalg.norm(A_all, axis=1)
    if not np.all(norms > 0):
        raise ValueError("zero row norm survived _prep_lp; cannot scale")

    # 4. Chebyshev LP: variables [x (n), r (1)]; maximise r.
    if scale_rows:
        # a_j.y + ||a_j||.r <= b_j   divided by ||a_j|| > 0  ->  same solution set,
        # unit-norm rows and an all-ones rho column.
        A_cheb = np.hstack([A_all / norms[:, None], np.ones((A_all.shape[0], 1))])
        b_cheb = b_all / norms
    else:
        A_cheb = np.hstack([A_all, norms[:, None]])
        b_cheb = b_all
    c = np.zeros(n + 1)
    c[-1] = -1.0                                   # max r  ==  min -r
    bounds = [(None, None)] * n + [(0.0, None)]    # x free, r >= 0

    res = linprog(c, A_ub=A_cheb, b_ub=b_cheb, bounds=bounds, method=method)

    # Only linprog status 2 (infeasible) proves the polytope is empty: with r >= 0
    # allowed, the LP is feasible iff the polytope is non-empty. Statuses 1
    # (iteration limit), 3 (unbounded) and 4 (numerical difficulties) are solver
    # failures — reporting them as 'empty' would silently exclude a polytope from
    # the GACC denominator on nothing but a solver hiccup.
    if not res.success:
        return {"radius": np.nan,
                "status": "empty" if res.status == 2 else "failed",
                "success": False, "n_constraints": A_all.shape[0],
                "lp_status": int(res.status), "lp_message": str(res.message)}

    r_star = float(res.x[-1])
    status = "zero_volume" if r_star <= zero_tol else "full_dim"
    return {"radius": r_star, "status": status,
            "success": True, "n_constraints": A_all.shape[0],
            "lp_status": int(res.status), "lp_message": str(res.message)}
