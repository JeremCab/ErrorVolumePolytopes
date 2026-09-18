#!/usr/bin/env python3
"""
volume_ratio_hitrun.py — vol(P3^k)/vol(P2) by direct Monte-Carlo, no surrogate.

What (19) wants per data point is the fraction of the correct polytope that the
approximated network still classifies correctly, vol(Xi~_x^c)/vol(Xi_x). The
pipeline estimates it through the mean width, which in n = 784 cannot see how
flat a body is. This script measures it instead: sample points uniformly in
Xi_x by Hit-and-Run, evaluate N~ at each, and count. The fraction predicting
class k is an unbiased estimate of vol(Xi~_x^k)/vol(Xi_x), with a binomial
error ~1/sqrt(M) that does NOT depend on the dimension. No mean width, no
Chebyshev threshold, no zero-volume gate, and no LP except the one that
provides a starting point.

Why this is a NEW sampler and not `mcmc_augment.find_augmented_points_walk`
--------------------------------------------------------------------------
That walk samples the chord inside A alone and then CLIPS the result onto the
pixel box. Clipping is a non-invertible projection: it maps a whole region onto
the faces of the box, so the chain's stationary law is not uniform — it piles
mass on the boundary. For finding distinct activation patterns that is a fair
heuristic; for estimating a volume ratio it is fatal, since the estimate is
exactly the stationary measure of the region. Here the box is handled
ANALYTICALLY inside the chord computation, so no projection ever occurs and the
chain is a genuine uniform sampler on Xi_x.

The starting point is the Chebyshev CENTRE, not x0. Normalised images have a
few hundred pixels at exactly +-1, so x0 lies on hundreds of faces of the box
at once and every chord through it collapses to zero length — the reason the
original walk had to clip in the first place. The centre is the deepest
interior point available and costs the one LP we already know how to solve.

That same LP answers, for free, the open question of why rho(P3^k)/rho(P2) is
about 1 for several disjoint subpolytopes at b = 6 and b = 10: the script
reports which class N~ predicts AT the centre, i.e. which subpolytope actually
owns P2's largest inscribed ball. Exactly one can.

Usage
-----
    python scripts/volume_ratio_hitrun.py --selftest
    python scripts/volume_ratio_hitrun.py --model_type cnn --sample_idx 0 --bits 16
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.networks import FashionMLP_Large, FashionCNN_Small      # noqa: E402
from src.optim.build_polytopes import build_all_polytopes_per_class      # noqa: E402
from src.optim.build_polytopes_cnn import build_cnn_all_polytopes_per_class  # noqa: E402
from src.optim.chebyshev import chebyshev_radius                        # noqa: E402
from src.quantization.quantize import quantize_model                    # noqa: E402


# ── the sampler ────────────────────────────────────────────────────────────
def hit_and_run(A, b, start, lo, hi, n_samples, thin, burn_in, rng):
    """Uniform Hit-and-Run on {y : A y + b <= 0} ∩ [lo, hi]^n.

    The box enters the chord bounds analytically, exactly like any other pair of
    half-spaces, so the sampled point is inside by construction and is never
    projected. Returns the samples and a diagnostics dict.
    """
    x = np.asarray(start, dtype=np.float64).copy()
    n = x.size
    out = np.empty((n_samples, n), dtype=np.float64)
    kept = 0
    lens, degenerate, max_viol = [], 0, -np.inf

    total = burn_in + n_samples * thin
    for step in range(total):
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)

        # half-space bounds
        c = A @ d
        s = -(A @ x + b)                       # slack, >= 0 inside
        t_hi, t_lo = np.inf, -np.inf
        pos, neg = c > 0, c < 0
        if pos.any():
            t_hi = min(t_hi, float(np.min(s[pos] / c[pos])))
        if neg.any():
            t_lo = max(t_lo, float(np.max(s[neg] / c[neg])))
        # box bounds, same treatment
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(d > 0, (hi - x) / d, np.inf)
            dn = np.where(d < 0, (lo - x) / d, np.inf)
            lo_up = np.where(d > 0, (lo - x) / d, -np.inf)
            lo_dn = np.where(d < 0, (hi - x) / d, -np.inf)
        t_hi = min(t_hi, float(np.min(np.minimum(up, dn))))
        t_lo = max(t_lo, float(np.max(np.maximum(lo_up, lo_dn))))

        if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_hi <= t_lo:
            degenerate += 1
            continue                            # keep the state, redraw
        lens.append(t_hi - t_lo)

        x = x + rng.uniform(t_lo, t_hi) * d
        np.clip(x, lo, hi, out=x)               # rounding-level only, not a projection
        max_viol = max(max_viol, float((A @ x + b).max()))

        if step >= burn_in and (step - burn_in) % thin == 0 and kept < n_samples:
            out[kept] = x
            kept += 1

    return out[:kept], {"degenerate_steps": degenerate,
                        "chord_len_p50": float(np.median(lens)) if lens else float("nan"),
                        "chord_len_p10": float(np.percentile(lens, 10)) if lens else float("nan"),
                        "max_constraint_violation": max_viol}


def coord_hit_and_run(A, b, start, lo, hi, n_samples, thin, burn_in, rng,
                      refresh=20000):
    """Coordinate (Gibbs) Hit-and-Run: move along ONE axis at a time.

    Why it is the default. A radial step needs A @ d, i.e. m*n flops — 32 MFLOP
    for the CNN. A coordinate step needs one COLUMN of A, i.e. m flops, and the
    running value of A y + b is updated incrementally, so it is some 400x
    cheaper. Against that, one coordinate moves per step, so an independent
    sample costs about n steps; the net gain on the real polytopes is still an
    order of magnitude. It is also the better-behaved choice when the pixel box
    dominates the geometry, which is our case: on a box it is exact after one
    sweep, whereas radial Hit-and-Run has to mix.

    `refresh` recomputes A y + b from scratch periodically, so the incremental
    update cannot drift.
    """
    x = np.asarray(start, dtype=np.float64).copy()
    n = x.size
    Ax_b = A @ x + b
    out = np.empty((n_samples, n), dtype=np.float64)
    kept, degenerate, max_viol = 0, 0, -np.inf

    total = burn_in + n_samples * thin
    for step in range(total):
        j = int(rng.integers(n))
        c = A[:, j]
        s = -Ax_b
        t_hi, t_lo = hi - x[j], lo - x[j]
        pos, neg = c > 0, c < 0
        if pos.any():
            t_hi = min(t_hi, float(np.min(s[pos] / c[pos])))
        if neg.any():
            t_lo = max(t_lo, float(np.max(s[neg] / c[neg])))

        if t_hi > t_lo:
            t = rng.uniform(t_lo, t_hi)
            x[j] = min(max(x[j] + t, lo), hi)
            Ax_b += t * c
        else:
            degenerate += 1

        if (step + 1) % refresh == 0:
            Ax_b = A @ x + b
            max_viol = max(max_viol, float(Ax_b.max()))

        if step >= burn_in and (step - burn_in) % thin == 0 and kept < n_samples:
            out[kept] = x
            kept += 1

    max_viol = max(max_viol, float((A @ x + b).max()))
    return out[:kept], {"degenerate_steps": degenerate,
                        "chord_len_p50": float("nan"),
                        "chord_len_p10": float("nan"),
                        "max_constraint_violation": max_viol}


def binom_ci(k, m):
    """Wald interval, good enough at these sample sizes; returns (lo, hi)."""
    if m == 0:
        return float("nan"), float("nan")
    p = k / m
    h = 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / m)
    return max(0.0, p - h), min(1.0, p + h)


# ── self-test: a body whose answer is known in closed form ─────────────────
def selftest(rng, n=50, m=4000):
    """Bodies whose answer is known in closed form.

    Thinning is n steps per kept sample, which is the point: a coordinate step
    moves one axis, so consecutive states share n-1 coordinates and a marginal
    measured on undecimated samples looks badly biased even when the sampler is
    exact. The first version of this test used thin = 1 and reported FAIL on a
    correct sampler.
    """
    lo, hi = -1.0, 1.0
    tol = 4 / np.sqrt(m)
    for name, fn in (("coord ", coord_hit_and_run), ("radial", hit_and_run)):
        print(f"\nSELF-TEST {name}  (n = {n}, {m} samples, thin = {n}, "
              f"tolerance {tol:.4f})\n" + "-" * 66)
        A0 = np.zeros((1, n)); b0 = np.array([-1.0])       # vacuous row
        S, _ = fn(A0, b0, np.zeros(n), lo, hi, m, n, 10 * n, rng)
        for a in (-0.5, 0.0, 0.5):
            got, exp = float((S[:, 0] <= a).mean()), (a + 1) / 2
            print(f"  box,  P(y_0 <= {a:+.1f})          = {got:.4f}  expected "
                  f"{exp:.4f}   {'OK' if abs(got-exp) < tol else 'FAIL'}")
        A1 = np.zeros((1, n)); A1[0, 0] = 1.0; b1 = np.array([-0.5])
        S, d = fn(A1, b1, np.zeros(n), lo, hi, m, n, 10 * n, rng)
        got, exp = float((S[:, 0] <= 0.0).mean()), 1.0 / 1.5
        print(f"  slab, P(y_0 <= 0 | y_0 <= 0.5) = {got:.4f}  expected {exp:.4f}"
              f"   {'OK' if abs(got-exp) < tol else 'FAIL'}")
        print(f"  max A y + b over the run       = "
              f"{d['max_constraint_violation']:.2e}  (must be <= 0)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model_type", default="cnn", choices=["mlp", "cnn"])
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--polytope", default="p2", choices=["p2", "p1"],
                    help="p2 = Xi_x, the correct polytope (needs one walk per b). "
                         "p1 = Xi-bar_x, the EXTENDED correct polytope of (9),(10),"
                         "(12), which does not involve N~ at all: one walk serves "
                         "every bitwidth, and the fraction of samples that N~^b "
                         "classifies as c is the ideal volume GACC of that point, "
                         "comparable across b by construction.")
    ap.add_argument("--bits_grid", type=int, nargs="+", default=None,
                    help="with --polytope p1: evaluate these bitwidths on the SAME "
                         "samples, so the curve across b is paired")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--csv", default=None,
                    help="results_df_{cnn,mlp}.csv, to print the mean-width ratio "
                         "of the same tile next to the sampled one")
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=784,
                    help="steps per kept sample; for the coordinate sampler "
                         "this should be ~n so that every axis is refreshed")
    ap.add_argument("--burn_in", type=int, default=7840)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sampler", default="coord", choices=["coord", "radial"],
                    help="coord = one axis per step, ~400x cheaper per step on "
                         "the real polytopes; radial = classical Hit-and-Run")
    ap.add_argument("--lp_method", default="auto", choices=["highs", "highs-ipm", "auto"])
    ap.add_argument("--time_limit", type=float, default=1800.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.selftest:
        selftest(rng)
        return

    dev = torch.device("cpu")
    mp = a.model_path or f"checkpoints/fashion_{a.model_type}_best.pth"
    dp = a.data_path or f"data/fashionMNIST_correct_{a.model_type}.pt"
    Net = FashionCNN_Small if a.model_type == "cnn" else FashionMLP_Large
    fp = Net(); fp.load_state_dict(torch.load(mp, map_location=dev, weights_only=True))
    fp.eval()
    ds = torch.load(dp, map_location=dev, weights_only=False)
    x0, c = ds[a.sample_idx]
    c = int(c)
    q = quantize_model(fp, bits=a.bits).eval()

    t0 = time.perf_counter()
    if a.model_type == "cnn":
        A_base, b_base, poly = build_cnn_all_polytopes_per_class(
            fp, {a.bits: q}, x0.unsqueeze(0), c)
        shape = x0.shape
    else:
        A_base, b_base, poly = build_all_polytopes_per_class(
            fp, {a.bits: q}, x0.flatten().unsqueeze(0), c)
        shape = (x0.numel(),)
    A_t, b_t, _ = poly[a.bits]
    if a.polytope == "p1":
        # Xi-bar_x: the original network alone, (9), (10), (12). It does not
        # depend on b, so ONE walk serves every bitwidth and the comparison
        # across b is paired — the estimate uses common random numbers.
        A_t, b_t = A_base, b_base
    A = A_t.detach().cpu().numpy().astype(np.float64)
    b = b_t.detach().cpu().numpy().astype(np.float64)
    print(f"P2 built in {time.perf_counter()-t0:.1f}s — {A.shape[0]} rows, "
          f"{A.shape[1]} vars, sample {a.sample_idx}, class c = {c}, b = {a.bits}")

    # ── the one LP: an interior starting point, and the centre diagnostic ──
    t0 = time.perf_counter()
    ch = chebyshev_radius(A, b, box=(-1.0, 1.0), method=a.lp_method,
                          time_limit=a.time_limit)
    print(f"Chebyshev: rho = {ch['radius']:.6e}  status = {ch['status']}  "
          f"({time.perf_counter()-t0:.1f}s)")
    if ch["centre"] is None or not np.isfinite(ch["radius"]) or ch["radius"] <= 0:
        raise SystemExit("[ABORT] no usable interior point: P2 measures flat or the "
                         "LP did not converge. Hit-and-Run needs a strictly "
                         "interior start; x0 will not do, it lies on the box faces.")

    def predict(X):
        cls = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                t = torch.tensor(X[i:i+256].reshape(-1, *shape), dtype=x0.dtype)
                cls.append(q(t).argmax(1).cpu().numpy())
        return np.concatenate(cls)

    k_centre = int(predict(ch["centre"][None, :])[0])
    print(f"class predicted by N~ at the Chebyshev CENTRE: {k_centre}"
          f"{'  (= c)' if k_centre == c else f'  (NOT c, c = {c})'}"
          "\n   -> this is the single subpolytope that owns P2's largest inscribed"
          "\n      ball; no other P3^k can have rho(P3^k) = rho(P2).")

    # ── sample ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sampler = coord_hit_and_run if a.sampler == "coord" else hit_and_run
    S, diag = sampler(A, b, ch["centre"], -1.0, 1.0,
                      a.n_samples, a.thin, a.burn_in, rng)
    cls = predict(S)
    sec = time.perf_counter() - t0
    m = len(S)
    print(f"\n{m} samples in {sec:.1f}s  (thin {a.thin}, burn-in {a.burn_in}); "
          f"chord length p10/p50 = {diag['chord_len_p10']:.2e}/{diag['chord_len_p50']:.2e}; "
          f"degenerate steps {diag['degenerate_steps']}; "
          f"max A y + b = {diag['max_constraint_violation']:.2e}")
    switch = float((cls[1:] != cls[:-1]).mean()) if m > 1 else float("nan")
    print(f"class changes between consecutive kept samples: {100*switch:.1f}% "
          f"(a low value means poor mixing — read the ratios with care)")

    if a.polytope == "p1" and a.bits_grid:
        print(f"\n{'='*66}\nIDEAL VOLUME GACC OF THIS POINT, on the SAME {m} samples of P1"
              f"\n{'='*66}")
        print(f"{'b':>4} {'vol correct / vol(P1)':>23} {'95% CI':>20}")
        for bb in a.bits_grid:
            qb = quantize_model(fp, bits=bb).eval()
            with torch.no_grad():
                pr = []
                for i in range(0, m, 256):
                    t = torch.tensor(S[i:i+256].reshape(-1, *shape), dtype=x0.dtype)
                    pr.append(qb(t).argmax(1).cpu().numpy())
            pr = np.concatenate(pr)
            k = int((pr == c).sum())
            loi, hii = binom_ci(k, m)
            print(f"{bb:>4} {k/m:>23.4f} {f'[{loi:.4f}, {hii:.4f}]':>20}")
        print("   No tiling, no mean width, no lemma, no Chebyshev threshold."
              "\n   P1 is the same body for every b, so these numbers ARE comparable."
              "\n   This is what the whole augmentation machinery approximates.")

    counts = np.bincount(cls, minlength=10)
    body = "P2" if a.polytope == "p2" else "P1"
    print(f"\n{'k':>3} {'samples':>9} {f'vol / vol({body})':>19} {'95% CI':>18}")
    print("-" * 54)
    for k in range(10):
        if counts[k] == 0:
            continue
        loi, hii = binom_ci(counts[k], m)
        print(f"{k:>3}{'*' if k == c else ' ':<1}{counts[k]:>8} "
              f"{counts[k]/m:>19.4f} {f'[{loi:.4f}, {hii:.4f}]':>18}")
    print("   * = the correct class c")

    # ── the comparison that matters ────────────────────────────────────────
    csv = a.csv or f"results/data_for_jiri/results_df_{a.model_type}.csv"
    if Path(csv).exists():
        import pandas as pd
        df = pd.read_csv(csv)
        row = df[df["sample_idx"] == a.sample_idx]
        if len(row):
            r = row.iloc[0]
            W = np.array([float(r[f"V3_b{a.bits}_k{k}"]) for k in range(10)])
            W = np.nan_to_num(W)
            mw = W[c] / W.sum() if W.sum() else float("nan")
            print(f"\n{'='*60}\nTHE COMPARISON\n{'='*60}")
            print(f"  mean-width ratio   d(P3^c) / sum_k d(P3^k) = {mw:.4f}"
                  f"   (over P2)")
            print(f"  sampled volume ratio  vol(correct) / vol({body})"
                  f"{'':>3} = {counts[c]/m:.4f}")
            print("  The first is what (19) uses; the second is what (19) means.")
            if body == "P1":
                print("  NOTE: the walk was in P1, so the denominators differ. They"
                      "\n  coincide only where P2 fills P1, which holds at large b"
                      "\n  (V2/V1 = 0.9998 at b=16) and NOT at small b.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"model_type": a.model_type, "sample_idx": a.sample_idx, "bits": a.bits,
             "class_c": c, "rho_P2": ch["radius"], "class_at_centre": k_centre,
             "n_samples": m, "counts": counts.tolist(),
             "ratio_c": counts[c] / m if m else None,
             "switch_rate": switch, **diag}, indent=2))
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
