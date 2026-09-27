#!/usr/bin/env python3
"""
neighbourhood_size.py — how large must a neighbourhood be before GACC says
something that pointwise accuracy does not?

The measurements of 2026-09-20 left one question standing. Measured exactly, the
gap between the generalized accuracy and plain accuracy is 0.66 points for the
MLP and ~0 for the CNN — tiny in both cases. The suspected reason is structural:
P1 is the region where the ORIGINAL network is still linear, which for a net with
~12 000 hidden units is a minuscule neighbourhood. If that is the cause, the gap
should open as the neighbourhood grows, and this measures exactly that.

Method, and it needs no polytope, no LP and no MCMC — only forward passes:

  for each data point x0 of class c, and each radius r:
      draw M points uniformly in the L-infinity ball of radius r around x0,
      clipped to the input domain;
      KEEP those the full-precision N still classifies as c — that is the
      "correct neighbourhood" at scale r, the natural widening of P1;
      among those, count the fraction the approximated N~^b also calls c.

At r -> 0 every sample collapses onto x0 and the average is exactly ACC. As r
grows the two can part company, and the rate at which they do is the answer.

Two further columns make the result readable rather than merely suggestive:

  kept    the share of samples N still classifies correctly. Once this falls,
          the neighbourhood is leaving N's own correct region and the quantity
          stops being comparable — it bounds how far the question makes sense.
  in P1   the share still inside P1 proper, i.e. carrying x0's activation
          pattern under N. This is what ties the experiment back to the paper:
          it shows at which radius one leaves the linearity cell, and therefore
          how small P1 really is. Costs one hooked pass; skip with --no_p1.

Usage
-----
    python scripts/neighbourhood_size.py --model_type mlp --n_points 74
    python scripts/neighbourhood_size.py --model_type cnn --n_points 74 --no_p1
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.networks import FashionCNN_Small, FashionMLP_Large   # noqa: E402
from src.quantization.quantize import quantize_model                 # noqa: E402

BITS = [4, 6, 8, 10, 12, 16]
RADII = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def batched_pred(model, X, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs]).argmax(1).cpu().numpy())
    return np.concatenate(out)


def batched_pattern(model, X, bs=256):
    """Sign pattern of every hidden pre-activation, as a packed bool array."""
    layers = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))][:-1]
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            cap = []
            hs = [m.register_forward_hook(
                lambda mod, inp, out: cap.append((out > 0).flatten(1))) for m in layers]
            model(X[i:i + bs])
            for h in hs:
                h.remove()
            chunks.append(torch.cat(cap, 1).cpu().numpy())
    return np.concatenate(chunks, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_type", default="mlp", choices=["mlp", "cnn"])
    ap.add_argument("--n_points", type=int, default=74)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--radii", type=float, nargs="+", default=RADII)
    ap.add_argument("--bits_grid", type=int, nargs="+", default=BITS)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no_p1", action="store_true", help="skip the P1-membership column")
    ap.add_argument("--points_file", default=None,
                    help="whitespace-separated indices; default = the same random "
                         "points used by the direct-sampling campaign")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device("cpu")
    Net = FashionCNN_Small if a.model_type == "cnn" else FashionMLP_Large
    fp = Net(); fp.load_state_dict(torch.load(
        f"checkpoints/fashion_{a.model_type}_best.pth", map_location=dev,
        weights_only=True)); fp.eval()
    ds = torch.load(f"data/fashionMNIST_correct_{a.model_type}.pt",
                    map_location=dev, weights_only=False)

    pf = a.points_file or (f"results/volume_ratio/"
                           f"{'mlp_points' if a.model_type == 'mlp' else 'random_points'}.txt")
    if Path(pf).exists():
        pts = [int(v) for v in Path(pf).read_text().split()][:a.n_points]
    else:
        pts = list(range(a.n_points))
    q = {b: quantize_model(fp, bits=b).eval() for b in a.bits_grid}
    rng = np.random.default_rng(a.seed)

    # accumulate per (radius, b): sum of per-point fractions, and the counts
    gacc = {r: {b: [] for b in a.bits_grid} for r in a.radii}
    kept = {r: [] for r in a.radii}
    inp1 = {r: [] for r in a.radii}
    ok0 = {b: [] for b in a.bits_grid}
    t0 = time.perf_counter()

    for n, i in enumerate(pts):
        x0, c = ds[i]; c = int(c)
        shape = x0.shape if a.model_type == "cnn" else (x0.numel(),)
        x0f = x0.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if not a.no_p1:
            p0 = batched_pattern(fp, torch.tensor(x0f.reshape(1, *shape),
                                                  dtype=x0.dtype))[0]
        for r in a.radii:
            Y = x0f[None, :] + rng.uniform(-r, r, size=(a.n_samples, x0f.size)).astype(np.float32)
            np.clip(Y, -1.0, 1.0, out=Y)
            T = torch.tensor(Y.reshape(-1, *shape), dtype=x0.dtype)
            ok = batched_pred(fp, T) == c                 # N still correct
            kept[r].append(float(ok.mean()))
            if not a.no_p1:
                pat = batched_pattern(fp, T)
                inp1[r].append(float((ok & (pat == p0).all(1)).mean()))
            if ok.sum() == 0:
                for b in a.bits_grid:
                    gacc[r][b].append(float("nan"))
                continue
            Tk = T[torch.tensor(ok)]
            for b in a.bits_grid:
                gacc[r][b].append(float((batched_pred(q[b], Tk) == c).mean()))
        # is x0 itself correctly classified by N~^b ? Needed to decompose the
        # average: a misclassified x0 whose neighbourhood is partly correct pushes
        # GACC UP, a correct x0 whose neighbourhood is partly wrong pushes it DOWN,
        # and the two can cancel — which is exactly what hid the effect on the CNN.
        x0t = torch.tensor(x0f.reshape(1, *shape), dtype=x0.dtype)
        for b in a.bits_grid:
            ok0[b].append(bool(batched_pred(q[b], x0t)[0] == c))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(pts)} points  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ── the table ─────────────────────────────────────────────────────────
    accs = {}
    for b in a.bits_grid:
        X0 = torch.stack([ds[i][0] for i in pts])
        if a.model_type == "mlp":
            X0 = X0.flatten(1)
        C0 = np.array([int(ds[i][1]) for i in pts])
        accs[b] = float((batched_pred(q[b], X0) == C0).mean())

    print(f"\n{'='*92}")
    print(f"{a.model_type.upper()} — {len(pts)} points, {a.n_samples} tirages par rayon, "
          f"boule L-infini\n{'='*92}")
    print(f"{'rayon':>8} {'gardes':>8} {'dans P1':>9} |" +
          "".join(f"{'b='+str(b):>9}" for b in a.bits_grid))
    print(f"{'(ACC)':>8} {'':>8} {'':>9} |" + "".join(f"{accs[b]:>9.4f}" for b in a.bits_grid))
    print("-" * (28 + 9 * len(a.bits_grid)))
    rows = []
    for r in a.radii:
        m = {b: float(np.nanmean(gacc[r][b])) for b in a.bits_grid}
        k = float(np.mean(kept[r]))
        p = float(np.mean(inp1[r])) if not a.no_p1 else float("nan")
        print(f"{r:>8.3f} {k:>8.3f} {p:>9.3f} |" +
              "".join(f"{m[b]:>9.4f}" for b in a.bits_grid))
        rows.append({"radius": r, "kept": k, "in_P1": p, "gacc": m})
    print("-" * (28 + 9 * len(a.bits_grid)))
    print(f"{'ecart a ACC':>27} |" + "".join(
        f"{float(np.nanmean(gacc[a.radii[-1]][b]))-accs[b]:>+9.4f}" for b in a.bits_grid))
    # ── decomposition: the cancellation, made visible ─────────────────────
    for b in a.bits_grid:
        m0 = np.array(ok0[b], dtype=bool)
        if m0.all():
            continue                       # nothing to decompose
        print(f"\n{'-'*70}\nDECOMPOSITION b={b} : "
              f"{int(m0.sum())} points ou N~ classe x0 CORRECTEMENT, "
              f"{int((~m0).sum())} ou il se trompe\n{'-'*70}")
        print(f"{'rayon':>8} {'x0 correct':>12} {'x0 faux':>10} {'moyenne':>10}")
        for r in a.radii:
            v = np.array(gacc[r][b], dtype=float)
            print(f"{r:>8.3f} {np.nanmean(v[m0]):>12.4f} "
                  f"{np.nanmean(v[~m0]):>10.4f} {np.nanmean(v):>10.4f}")
        print("   x0 correct -> part du voisinage que N~ rate   (tire vers le BAS)"
              "\n   x0 faux    -> part du voisinage que N~ reussit (tire vers le HAUT)")

    print("\n  gardes  : part des tirages que N classe encore correctement. Quand elle"
          "\n            chute, on quitte la region correcte de N et la quantite cesse"
          "\n            d'etre comparable."
          "\n  dans P1 : part encore dans la cellule de linearite de N. Montre a quel"
          "\n            rayon on sort de P1, donc a quel point P1 est petit.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"model_type": a.model_type, "n_points": len(pts),
             "n_samples": a.n_samples, "acc": accs, "rows": rows,
             "x0_correct": {str(b): ok0[b] for b in a.bits_grid},
             "points": pts,
             "per_point": {str(r): {str(b): gacc[r][b] for b in a.bits_grid}
                           for r in a.radii}}, indent=1))
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
