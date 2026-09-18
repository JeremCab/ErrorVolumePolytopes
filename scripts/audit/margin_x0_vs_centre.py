#!/usr/bin/env python3
"""
margin_x0_vs_centre.py — is the thinness of P2 an artefact of building it at x0?

For each data point it reports, WITHOUT ever launching the Chebyshev LP on P2
(the one that fails), how far the reference point sits from its own polytope's
faces, at x0 and at the Chebyshev centre of P1. The largest slack max(A y + b)
bounds rho from above once divided by the row norm, so it separates the two
populations at a thousandth of the cost.

Resumable by design: one JSON per point, and a point whose file exists is
skipped. Kill it and relaunch, it picks up where it stopped.

    python scripts/audit/margin_x0_vs_centre.py --n 20
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.networks import FashionCNN_Small, FashionMLP_Large        # noqa: E402
from src.optim.build_polytopes_cnn import build_cnn_all_polytopes_per_class  # noqa: E402
from src.optim.build_polytopes import build_all_polytopes_per_class       # noqa: E402
from src.optim.chebyshev import chebyshev_radius                          # noqa: E402
from src.quantization.quantize import quantize_model                      # noqa: E402


def stats(A, b, y):
    """(rows kept, violated, tight, max slack, max slack / row norm)."""
    A = A.detach().cpu().numpy().astype(float)
    b = b.detach().cpu().numpy().astype(float)
    nrm = np.linalg.norm(A, axis=1)
    keep = nrm > 1e-12
    s = (A[keep] @ np.asarray(y).ravel() + b[keep])
    return dict(rows=int(keep.sum()), violated=int((s > 0).sum()),
                tight=int((np.abs(s) < 1e-6).sum()), max_slack=float(s.max()),
                max_slack_normalised=float((s / nrm[keep]).max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_type", default="cnn", choices=["cnn", "mlp"])
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=600.0)
    ap.add_argument("--out_dir", default="results/audit/margins")
    a = ap.parse_args()

    dev = torch.device("cpu")
    Net = FashionCNN_Small if a.model_type == "cnn" else FashionMLP_Large
    fp = Net(); fp.load_state_dict(torch.load(
        f"checkpoints/fashion_{a.model_type}_best.pth", map_location=dev,
        weights_only=True)); fp.eval()
    ds = torch.load(f"data/fashionMNIST_correct_{a.model_type}.pt",
                    map_location=dev, weights_only=False)
    q = quantize_model(fp, bits=a.bits).eval()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    for i in range(a.start, a.start + a.n):
        f = out / f"{a.model_type}_b{a.bits:02d}_s{i}.json"
        if f.exists():
            print(f"[{i}] deja fait — saute"); continue
        t0 = time.perf_counter()
        x0, c = ds[i]; c = int(c)
        xin = x0.unsqueeze(0) if a.model_type == "cnn" else x0.flatten().unsqueeze(0)
        build = (build_cnn_all_polytopes_per_class if a.model_type == "cnn"
                 else build_all_polytopes_per_class)
        A1, b1, poly = build(fp, {a.bits: q}, xin, c)
        A2, b2, _ = poly[a.bits]
        x0f = x0.detach().cpu().numpy().ravel()
        rec = {"sample_idx": i, "class_c": c, "bits": a.bits,
               "x0": {"P1": stats(A1, b1, x0f), "P2": stats(A2, b2, x0f)}}

        ch = chebyshev_radius(A1.detach().cpu().numpy().astype(float),
                              b1.detach().cpu().numpy().astype(float),
                              box=(-1.0, 1.0), method="auto",
                              time_limit=a.time_limit)
        rec["rho_P1"] = ch["radius"] if np.isfinite(ch["radius"]) else None
        rec["rho_P1_status"] = ch["status"]
        if ch["centre"] is not None:
            y = ch["centre"]
            yt = torch.tensor(y.reshape(x0.shape if a.model_type == "cnn"
                                        else (-1,)), dtype=x0.dtype)
            yin = yt.unsqueeze(0) if a.model_type == "cnn" else yt.unsqueeze(0)
            A1c, b1c, polyc = build(fp, {a.bits: q}, yin, c)
            A2c, b2c, _ = polyc[a.bits]
            rec["centre"] = {"P1": stats(A1c, b1c, y), "P2": stats(A2c, b2c, y),
                             "dist_to_x0": float(np.linalg.norm(y - x0f)),
                             "norm_x0": float(np.linalg.norm(x0f))}
        rec["sec"] = time.perf_counter() - t0
        f.write_text(json.dumps(rec, indent=1))
        s0, sc = rec["x0"]["P2"], rec.get("centre", {}).get("P2", {})
        print(f"[{i}] rho(P1)={rec['rho_P1']} | P2 en x0 : {s0['violated']} violees, "
              f"{s0['tight']} tendues, marge {s0['max_slack_normalised']:+.2e}"
              f" | au centre : {sc.get('violated','-')} / {sc.get('tight','-')}, "
              f"marge {sc.get('max_slack_normalised', float('nan')):+.2e}"
              f"  ({rec['sec']:.0f}s)")


if __name__ == "__main__":
    main()
