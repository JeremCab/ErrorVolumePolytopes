"""
Worker module for sanity_check_V3k.ipynb.

Defines a pool initializer and worker function that can be pickled by
the macOS `spawn` multiprocessing backend.
"""
import numpy as np
from scipy.optimize import linprog

# Module-level globals set by init_worker
_A_c_np    = None
_b_c_np    = None
_A3_np     = None   # dict {k: (A_np, b_np)}
_BOUNDS    = None
_N_CLASSES = None


def init_worker(A_c_np, b_c_np, A3_np, bounds, n_classes):
    global _A_c_np, _b_c_np, _A3_np, _BOUNDS, _N_CLASSES
    _A_c_np    = A_c_np
    _b_c_np    = b_c_np
    _A3_np     = A3_np
    _BOUNDS    = bounds
    _N_CLASSES = n_classes


def width_in_direction(v):
    """
    Compute mean width of P2 and each P3(k) along direction v.
    Returns (w_correct, {k: w_both_k}) or None if P2 LP fails.
    """
    res_max = linprog(-v, A_ub=_A_c_np, b_ub=-_b_c_np, bounds=_BOUNDS, method="highs")
    res_min = linprog( v, A_ub=_A_c_np, b_ub=-_b_c_np, bounds=_BOUNDS, method="highs")
    if not (res_max.success and res_min.success):
        return None
    w_correct = (-res_max.fun) - res_min.fun
    if w_correct <= 0:
        return None

    w_both = {}
    for k in range(_N_CLASSES):
        Ak, bk = _A3_np[k]
        r_max = linprog(-v, A_ub=Ak, b_ub=-bk, bounds=_BOUNDS, method="highs")
        r_min = linprog( v, A_ub=Ak, b_ub=-bk, bounds=_BOUNDS, method="highs")
        if r_max.success and r_min.success:
            w_both[k] = max(0.0, (-r_max.fun) - r_min.fun)
        else:
            w_both[k] = 0.0
    return w_correct, w_both
