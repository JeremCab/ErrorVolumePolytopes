"""
run_volumes_v3k.py

Like run_volumes.py, but estimates the mean width of P3(k) for ALL classes k,
not just the true class c.

Three polytopes (shared across bit-widths):
  A_base    : model activation + model classification        (model-only)

Per bit-width:
  A_correct : A_base + qmodel activation                    (P2)
  A_k       : A_correct + "qmodel predicts k" constraints   (P3(k) for each k)

Output JSON format (note: widths_both is now a list of n_classes floats):
  {
    "model_type": "mlp",
    "sample_idx": 5,
    "class_c": 3,
    "n_directions": 200,
    "n_directions_used": 195,
    "bits_grid": [4, 6, 8, 10, 12, 16],
    "width_base": 40.185,
    "widths_correct": {"4": 38.23, ...},
    "widths_both": {"4": [w0, w1, ..., w9], ...}
  }

widths_both[b][k] = mean width of P3(k) at bit-width b.
class_c records the true class for this sample.

Usage (from project root):

  MLP:
    python scripts/run_volumes_v3k.py \\
        --model_type    mlp \\
        --sample_idx    0 \\
        --model_path    checkpoints/fashion_mlp_best.pth \\
        --data_path     data/fashionMNIST_correct_mlp.pt \\
        --n_directions  200 \\
        --output_dir    results/volumes_v3k_mlp

  CNN:
    python scripts/run_volumes_v3k.py \\
        --model_type    cnn \\
        --sample_idx    0 \\
        --model_path    checkpoints/fashion_cnn_best.pth \\
        --data_path     data/fashionMNIST_correct_cnn.pt \\
        --n_directions  200 \\
        --output_dir    results/volumes_v3k_cnn
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linprog
import torch
from tqdm import tqdm

from src.models.networks import FashionMLP_Large, FashionCNN_Small
from src.optim.build_polytopes import build_all_polytopes_per_class
from src.optim.build_polytopes_cnn import build_cnn_all_polytopes_per_class
from src.quantization.quantize import quantize_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BITS_GRID = [4, 6, 8, 10, 12, 16]


# ---------------------------------------------------------------------------
# Worker state — one direction = one task.
#
# Memory-efficient design: we do NOT store the full A_k (3860-row) matrix for
# each class k. Instead we store only the small per-class classification delta
# (9 rows × 784 cols) and stack it onto A_correct inside the worker on-the-fly.
#
# _WORKER_POLYTOPES_V3K[bits] = (A_c_np, b_c_np, {k: (A_cls_k_np, b_cls_k_np)})
#   where A_cls_k_np has only 9 rows (the "qmodel predicts k" constraints).
# ---------------------------------------------------------------------------

_WORKER_A_BASE          = None
_WORKER_B_BASE          = None
_WORKER_POLYTOPES_V3K   = None   # {bits: (A_c_np, b_c_np, {k: (A_cls_k_np, b_cls_k_np)})}
_WORKER_DIRECTIONS      = None   # (n_directions, d)
_WORKER_BOUNDS          = None


def _init_worker_v3k(A_base, b_base, polytopes_v3k, directions, bounds, skip_base=False,
                     lp_method="highs"):
    global _WORKER_A_BASE, _WORKER_B_BASE, _WORKER_SKIP_BASE, _WORKER_LP_METHOD
    global _WORKER_POLYTOPES_V3K, _WORKER_DIRECTIONS, _WORKER_BOUNDS
    _WORKER_SKIP_BASE     = skip_base
    _WORKER_LP_METHOD     = lp_method
    _WORKER_A_BASE        = A_base
    _WORKER_B_BASE        = b_base
    _WORKER_POLYTOPES_V3K = polytopes_v3k
    _WORKER_DIRECTIONS    = directions
    _WORKER_BOUNDS        = bounds


def _run_direction_v3k(k):
    """Solve all LPs for direction k.

    Returns (k, w_base, w_bits) where:
      w_base  : float  — width of P1 along u
                None   — the P1 LPs failed, so the direction is discarded
                nan    — P1 was not computed (skip_base); validity then comes
                         from P2, i.e. from w_bits
      w_bits  : {bits: (w_correct, {cls_k: w_k})} or None
    """
    u = _WORKER_DIRECTIONS[k]

    # A_base (P1) LP — skipped entirely under skip_base: P1 appears nowhere in
    # the generalized accuracy (19), which only involves P2 and its subpolytopes.
    # It costs 2 LPs per direction, i.e. a full polytope's worth of work.
    if _WORKER_SKIP_BASE:
        w_base = float("nan")
    else:
        res_max_base = linprog(c=-u, A_ub=_WORKER_A_BASE, b_ub=_WORKER_B_BASE,
                               bounds=_WORKER_BOUNDS, method=_WORKER_LP_METHOD)
        res_min_base = linprog(c= u, A_ub=_WORKER_A_BASE, b_ub=_WORKER_B_BASE,
                               bounds=_WORKER_BOUNDS, method=_WORKER_LP_METHOD)
        if not (res_max_base.success and res_min_base.success):
            return k, None, None

        w_base = (-res_max_base.fun) - res_min_base.fun

    # Per-bit: A_correct + per-class P3(k)
    w_bits = {}
    for bits, (A_c, b_c, per_class_deltas) in _WORKER_POLYTOPES_V3K.items():
        # A_correct width
        res_max_c = linprog(c=-u, A_ub=A_c, b_ub=b_c,
                            bounds=_WORKER_BOUNDS, method=_WORKER_LP_METHOD)
        res_min_c = linprog(c= u, A_ub=A_c, b_ub=b_c,
                            bounds=_WORKER_BOUNDS, method=_WORKER_LP_METHOD)
        if not (res_max_c.success and res_min_c.success):
            w_bits[bits] = None
            continue

        w_correct = (-res_max_c.fun) - res_min_c.fun

        # Per-class P3(k) widths — reconstruct A_k = stack(A_correct, A_cls_k)
        # Failures are isolated per class.
        w_per_class = {}
        for cls_k, (A_cls_k, b_cls_k) in per_class_deltas.items():
            A_k = np.vstack([A_c, A_cls_k])
            b_k = np.concatenate([b_c, b_cls_k])
            res_max_k = linprog(c=-u, A_ub=A_k, b_ub=b_k,
                                bounds=_WORKER_BOUNDS, method=_WORKER_LP_METHOD)
            res_min_k = linprog(c= u, A_ub=A_k, b_ub=b_k,
                                bounds=_WORKER_BOUNDS, method=_WORKER_LP_METHOD)
            if res_max_k.success and res_min_k.success:
                w_per_class[cls_k] = (-res_max_k.fun) - res_min_k.fun
            else:
                w_per_class[cls_k] = None

        w_bits[bits] = (w_correct, w_per_class)

    return k, w_base, w_bits


# ---------------------------------------------------------------------------
# Loading helpers (same as run_volumes.py)
# ---------------------------------------------------------------------------

def load_model(model_path, model_type):
    if model_type == "cnn":
        model = FashionCNN_Small()
    else:
        model = FashionMLP_Large()
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_sample(data_path, sample_idx):
    dataset = torch.load(data_path, weights_only=False)
    x, c = dataset[sample_idx]
    return x, int(c)


# ---------------------------------------------------------------------------
# Main volume computation (parallel over directions)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pre-screen workers (one feasibility LP per (bits, class))
# ---------------------------------------------------------------------------

_PS_POLYTOPES = None   # {bits: (A_c, b_c, {k: (A_cls_k, b_cls_k)})}
_PS_BOUNDS    = None
_PS_TL        = None   # seconds allowed to each emptiness LP


def _init_worker_prescreen(polytopes_np, bounds, time_limit):
    global _PS_POLYTOPES, _PS_BOUNDS, _PS_TL
    _PS_POLYTOPES = polytopes_np
    _PS_BOUNDS    = bounds
    _PS_TL        = time_limit


def _run_prescreen(task):
    """Feasibility LP for one (bits, class): is P3(k) non-empty?

    Same test as the former serial loop — zero objective, `success` as the
    verdict — so the outcome is bit-for-bit identical; only the scheduling
    changes. Returns the linprog status too, so a solver failure can be told
    apart from a genuine infeasibility downstream.
    """
    bits, k = task
    A_c, b_c, deltas = _PS_POLYTOPES[bits]
    A_cls_k, b_cls_k = deltas[k]
    A_k = np.vstack([A_c, A_cls_k])
    b_k = np.concatenate([b_c, b_cls_k])
    c0   = np.zeros(A_k.shape[1])
    opts = {} if _PS_TL is None else {"time_limit": float(_PS_TL)}

    # Interior point FIRST. These are feasibility LPs (zero objective), the hardest
    # kind for a simplex, which has to certify infeasibility with nothing to guide
    # it. Measured on the CNN at b=4: IPM settles them in ~33-37 s uniformly, the
    # simplex takes 194-600 s and can run away entirely — a smoke-test task hit the
    # 4 h wall inside this loop. Both agree on every verdict; only the time differs.
    feas = linprog(c0, A_ub=A_k, b_ub=b_k, bounds=_PS_BOUNDS,
                   method="highs-ipm", options=opts)

    retried = False
    if not feas.success and feas.status != 2:
        # Second opinion from the simplex. On the MLP it was the other way round
        # (simplex returned status 4 "HiGHS Status 0: Not Set" where IPM decided),
        # so neither method dominates and it is worth asking both before giving up.
        retried = True
        feas = linprog(c0, A_ub=A_k, b_ub=b_k, bounds=_PS_BOUNDS,
                       method="highs", options=opts)
    return bits, k, bool(feas.success), int(feas.status), retried


def run_volumes_v3k(A_base, b_base, polytopes_dict, bounds, n_directions, n_workers, n_classes,
                    skip_base=False, prescreen_time_limit=600.0, lp_method="highs"):
    """
    Estimate mean widths of A_base, A_correct (P2), and P3(k) for all k.

    Parameters
    ----------
    A_base, b_base : model-only polytope (Ax + b <= 0)
    polytopes_dict : {bits: (A_correct, b_correct, {k: (A_k, b_k)})}
    bounds : list of (lo, hi) pairs
    n_directions : int
    n_workers : int
    n_classes : int

    Returns
    -------
    dict with "width_base", "widths_correct", "widths_both" (per-class list),
    "n_directions_used".  Under skip_base, "width_base" is None: P1 is not
    computed, and a direction counts as valid iff the P2 LPs succeeded.
    """

    def to_numpy(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    from src.optim.compute_volumes import _prep_lp
    A_base_np, b_base_np = _prep_lp(to_numpy(A_base), to_numpy(b_base))

    # Memory-efficient: store only the per-class classification delta (9 rows),
    # NOT the full A_k matrices (3860 rows each).  Workers reconstruct A_k on-the-fly.
    polytopes_np = {}
    for bits, (A_c, b_c, per_class) in polytopes_dict.items():
        A_c_np, b_c_np = _prep_lp(to_numpy(A_c), to_numpy(b_c))
        n_c = A_c.shape[0]   # number of rows in A_correct
        per_class_deltas = {}
        for k, (A_k, b_k) in per_class.items():
            # A_k = cat([A_correct, A_cls_k]) — extract only the delta rows
            A_cls_k = to_numpy(A_k)[n_c:]
            b_cls_k = to_numpy(b_k)[n_c:]
            A_cls_k_np, b_cls_k_np = _prep_lp(A_cls_k, b_cls_k)
            per_class_deltas[k] = (A_cls_k_np, b_cls_k_np)
        polytopes_np[bits] = (A_c_np, b_c_np, per_class_deltas)

    # Sample all directions in main process
    d = A_base_np.shape[1]

    # ── Pre-screen empty P3(k) polytopes (1 feasibility LP each) ────────────
    # The feasibility LP (zero objective) asks: "does ANY point x satisfy all
    # constraints of P3(k)?"  This is direction-independent — unlike the per-
    # direction LPs which optimise u·x for a specific u.  If HiGHS reports the
    # system infeasible, the polytope is provably empty (no feasible point
    # exists at all), so every one of the 200×2 direction LPs would also fail.
    # Skipping them is safe: absent classes produce wk_vals=[] → wk=0.0 in the
    # aggregation below, which is the correct result for an empty polytope.
    # Cost: 6×10 = 60 small LPs upfront.  Saving: up to 200×2×(skipped classes)
    # LPs — typically ~8/10 classes are empty, giving ~3–5× fewer LP solves.
    # These LPs are the same size as the direction LPs, so run them in a pool:
    # serially they dominated the task (10 LPs x ~460 s measured on the CNN at
    # b=16, i.e. ~77 min that no amount of extra cores could shorten).
    ps_tasks = [(bits, k) for bits in polytopes_np for k in polytopes_np[bits][2]]
    ps_res   = {}
    t_ps = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(n_workers, max(1, len(ps_tasks))),
        initializer=_init_worker_prescreen,
        initargs=(polytopes_np, bounds, prescreen_time_limit),
    ) as ps_exec:
        n_retried = 0
        for bits, k, ok, status, retried in ps_exec.map(_run_prescreen, ps_tasks):
            ps_res[(bits, k)] = (ok, status)
            n_retried += int(retried)

    # A class is discarded ONLY on a proven infeasibility (linprog status 2).
    # Any other failure (iteration limit, numerical trouble) means the LP could
    # not decide, so the class is KEPT and the direction LPs settle it: they all
    # fail — giving width 0 — if the polytope really is empty, and give the true
    # width otherwise. This is the pre-2026-05-10 behaviour, kept as a safety net.
    # Without it, the pre-screen concentrates into a single LP a risk that the
    # direction LPs spread over 200: one bad solve would silently zero a whole
    # class and remove it from the denominator of (19), inflating gamma.
    # It costs 2*n_directions extra LPs, but only when a failure actually occurs.
    n_prescreened      = 0
    n_kept_on_failure  = 0
    for bits in list(polytopes_np.keys()):
        A_c_np, b_c_np, per_class_deltas = polytopes_np[bits]
        non_empty = {}
        for k, (A_cls_k_np, b_cls_k_np) in per_class_deltas.items():
            ok, status = ps_res[(bits, k)]
            if ok or status != 2:
                non_empty[k] = (A_cls_k_np, b_cls_k_np)
                if not ok:
                    n_kept_on_failure += 1
            else:
                n_prescreened += 1
        polytopes_np[bits] = (A_c_np, b_c_np, non_empty)
    n_total = len(polytopes_np) * n_classes
    log.info(f"Pre-screening: {n_prescreened}/{n_total} P3(k) polytopes provably empty "
             f"→ skipped ({time.perf_counter() - t_ps:.1f}s)")
    if n_retried:
        log.info(f"Pre-screening: {n_retried} LP(s) retried with the simplex after "
                 f"interior point failed to decide.")
    if n_kept_on_failure:
        log.warning(f"Pre-screening: {n_kept_on_failure} polytope(s) KEPT because their "
                    f"feasibility LP failed without proving infeasibility (status != 2). "
                    f"They are being computed in full rather than silently zeroed.")

    directions = np.random.randn(n_directions, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    widths_base    = {}
    widths_correct = {bits: {} for bits in polytopes_np}
    # widths_per_class[bits][k][direction_k] = float
    widths_per_class = {bits: {k: {} for k in range(n_classes)} for bits in polytopes_np}
    n_failed = 0

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker_v3k,
        initargs=(A_base_np, b_base_np, polytopes_np, directions, bounds, skip_base,
                  lp_method),
    ) as executor:
        for k, w_base, w_bits in tqdm(
            executor.map(_run_direction_v3k, list(range(n_directions))),
            total=n_directions,
            desc="Directions",
        ):
            if w_base is None:              # P1 LPs failed -> discard direction
                n_failed += 1
                continue
            if not np.isnan(w_base):            # nan = skip_base, nothing to store
                widths_base[k] = w_base
            if w_bits is None:
                n_failed += 1
                continue
            for bits, val in w_bits.items():
                if val is None:
                    continue
                w_correct, w_per_class = val
                widths_correct[bits][k] = w_correct
                for cls_k, w_k in w_per_class.items():
                    if w_k is not None:
                        widths_per_class[bits][cls_k][k] = w_k

    elapsed = time.perf_counter() - t0
    log.info(f"Elapsed: {elapsed:.1f}s  |  Failed directions: {n_failed}/{n_directions}")

    if skip_base:
        # P1 was not computed: a direction is valid iff P2 was solved for at
        # least one bit-width. This replaces P1 as the direction filter.
        valid_set = set()
        for bits in polytopes_np:
            valid_set |= set(widths_correct[bits].keys())
        valid_keys = sorted(valid_set)
        width_base = None
        log.info("  width_base=skipped (--skip_base); direction validity taken from P2")
    else:
        valid_keys = sorted(widths_base.keys())
        width_base = float(np.mean([widths_base[k] for k in valid_keys]))
        log.info(f"  width_base={width_base:.4f}")
    n_used = len(valid_keys)

    out_correct   = {}
    out_both      = {}   # {bits: [w_0, w_1, ..., w_{n_classes-1}]}

    for bits in polytopes_np:
        wc_vals = [widths_correct[bits][k] for k in valid_keys if k in widths_correct[bits]]
        wc = float(np.mean(wc_vals)) if wc_vals else 0.0
        out_correct[bits] = wc

        per_class_means = []
        for cls_k in range(n_classes):
            wk_vals = [widths_per_class[bits][cls_k][k]
                       for k in valid_keys if k in widths_per_class[bits][cls_k]]
            wk = float(np.mean(wk_vals)) if wk_vals else 0.0
            per_class_means.append(wk)
        out_both[bits] = per_class_means

        log.info(f"  bits={bits:2d}  width_correct={wc:.4f}  "
                 f"widths_both={[f'{v:.4f}' for v in per_class_means]}")

    return {
        "width_base":        width_base,
        "widths_correct":    out_correct,
        "widths_both":       out_both,
        "n_directions_used": n_used,
    }


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def _default_n_workers():
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None:
        return int(slurm)
    return min(os.cpu_count() or 1, 8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V3k volume experiment: mean width of P3(k) for all classes k."
    )
    parser.add_argument("--model_type",   type=str, default="mlp",
                        choices=["mlp", "cnn"],
                        help="Model architecture: mlp or cnn (default: mlp)")
    parser.add_argument("--sample_idx",   type=int, required=True)
    parser.add_argument("--model_path",   type=str, default=None)
    parser.add_argument("--data_path",    type=str, default=None)
    parser.add_argument("--n_directions", type=int, default=200)
    parser.add_argument("--n_workers",    type=int, default=_default_n_workers())
    parser.add_argument("--output_dir",   type=str, default=None)
    parser.add_argument("--bits_grid",    type=int, nargs="+", default=None,
                        help="Bit-widths to evaluate (default: 4 6 8 10 12 16). "
                             "Example: --bits_grid 4")
    parser.add_argument("--lp_method", type=str, default="highs",
                        choices=["highs", "highs-ds", "highs-ipm"],
                        help="Solver for the DIRECTION LPs (default: highs). These "
                             "produce the numbers the paper reports, so the default is "
                             "left unchanged until an end-to-end comparison says "
                             "otherwise. On one CNN b=4 direction, highs-ipm was 3-5x "
                             "faster with an identical w(P2) to 15 decimals; that is "
                             "one direction on one tile, not a validation.")
    parser.add_argument("--prescreen_time_limit", type=float, default=600.0,
                        help="Seconds allowed to each emptiness LP (default: 600). Very "
                             "wide against the ~35 s interior point needs; it exists so "
                             "that a runaway solve can never consume a whole task, as "
                             "one did on a 4 h cluster job. A capped LP is treated as "
                             "undecided, never as an empty polytope.")
    parser.add_argument("--skip_base",    action="store_true",
                        help="Do not compute the model-only polytope P1. P1 does not "
                             "appear in the generalized accuracy (19), which involves "
                             "only P2 and its subpolytopes, yet it costs 2 LPs per "
                             "direction (~21-27%% of a task). Direction validity is "
                             "then taken from P2 instead. 'width_base' is null in the "
                             "output; run without this flag on a subsample if the "
                             "V2/V1 ratio is needed.")
    args = parser.parse_args()

    bits_grid = args.bits_grid if args.bits_grid is not None else BITS_GRID

    if args.model_path is None:
        args.model_path = f"checkpoints/fashion_{args.model_type}_best.pth"
    if args.data_path is None:
        args.data_path = f"data/fashionMNIST_correct_{args.model_type}.pt"
    if args.output_dir is None:
        args.output_dir = f"results/volumes_v3k_{args.model_type}"

    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"volumes_sample{args.sample_idx}.json"

    if output_file.exists():
        log.info(f"Sample {args.sample_idx} already computed. Skipping.")
        sys.exit(0)

    log.info(f"Model type   : {args.model_type}")
    log.info(f"Sample idx   : {args.sample_idx}")
    log.info(f"Model        : {args.model_path}")
    log.info(f"N directions : {args.n_directions}")
    log.info(f"Bits grid    : {bits_grid}")
    log.info(f"Workers      : {args.n_workers}")

    log.info("\nLoading model...")
    model = load_model(args.model_path, args.model_type)

    log.info("Loading sample...")
    x, c = load_sample(args.data_path, args.sample_idx)
    log.info(f"x shape: {x.shape}  label: {c}")

    if args.model_type == "cnn":
        x_batch = x.unsqueeze(0)
    else:
        x_batch = x.flatten().unsqueeze(0)

    log.info("Building polytopes...")
    t0 = time.perf_counter()

    qmodels_dict = {}
    for bits in bits_grid:
        qmodel = quantize_model(model, bits=bits)
        qmodel.eval()
        qmodels_dict[bits] = qmodel

    if args.model_type == "cnn":
        A_base, b_base, polytopes_dict = build_cnn_all_polytopes_per_class(
            model, qmodels_dict, x_batch, c
        )
    else:
        A_base, b_base, polytopes_dict = build_all_polytopes_per_class(
            model, qmodels_dict, x_batch, c
        )

    # Infer n_classes from the per_class dict of the first bit-width
    first_bits  = bits_grid[0]
    n_classes   = len(polytopes_dict[first_bits][2])

    log.info(f"A_base (model-only): {tuple(A_base.shape)}")
    for bits, (A_correct, _, per_class) in polytopes_dict.items():
        log.info(f"  bits={bits:2d}  A_correct={tuple(A_correct.shape)}  "
                 f"n_classes={len(per_class)}")
    log.info(f"Polytopes built in {time.perf_counter() - t0:.2f}s")

    bounds = [(-1., 1.)] * x.flatten().shape[0]

    log.info("\nRunning V3k volume experiment...")
    result = run_volumes_v3k(A_base, b_base, polytopes_dict, bounds,
                             args.n_directions, args.n_workers, n_classes,
                             skip_base=args.skip_base,
                             prescreen_time_limit=args.prescreen_time_limit,
                             lp_method=args.lp_method)

    output = {
        "model_type":        args.model_type,
        "sample_idx":        args.sample_idx,
        "class_c":           c,
        "model_path":        args.model_path,
        "data_path":         args.data_path,
        "n_directions":      args.n_directions,
        "n_directions_used": result["n_directions_used"],
        "bits_grid":         bits_grid,
        "width_base":        result["width_base"],
        "widths_correct":    {str(k): v for k, v in result["widths_correct"].items()},
        "widths_both":       {str(k): v for k, v in result["widths_both"].items()},
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
