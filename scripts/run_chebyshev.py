#!/usr/bin/env python3
"""
run_chebyshev.py — Chebyshev radius of P2 and of every non-empty P3(k) for one
augmented point at one bit-width. Stage 2 of the augmented-GACC campaign.

This is what evaluates the condition of paper eq. (18): vol(Xi) = 0 iff the
Chebyshev radius rho(Xi) = 0. It is deliberately a SEPARATE stage from the mean
widths, for two reasons measured beforehand: a Chebyshev LP costs ~17-19 times a
mean-width LP, and its cost is unpredictable (at b=4 the unscaled formulation did
not converge at all). Mixing the two would put the predictable, primary
computation at the mercy of the unpredictable one.

Which classes to solve for is READ from the stage-1 JSON rather than
re-determined: a class with zero mean width is empty, and an empty polytope has
no radius to compute. This also keeps the two stages consistent by construction.

Every LP is capped by --time_limit. A capped LP comes back as 'failed', never as
'empty' — the distinction matters: 'empty' would zero the polytope in (19),
'failed' says the result is unusable and must be reported as such.

Usage
-----
    python scripts/run_chebyshev.py --sample_idx 12 --bits 4 \\
        --stage1_dir results/volumes_v3k_cnn_gen150 \\
        --data_path data/..._gen150_b4.pt --output_dir results/chebyshev_cnn_gen150/b04
"""
import argparse, json, logging, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.networks import FashionMLP_Large, FashionCNN_Small          # noqa: E402
from src.optim.build_polytopes import build_all_polytopes_per_class          # noqa: E402
from src.optim.build_polytopes_cnn import build_cnn_all_polytopes_per_class  # noqa: E402
from src.optim.chebyshev import chebyshev_radius                            # noqa: E402
from src.quantization.quantize import quantize_model                        # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_W = {}   # per-worker polytope store


def _init(polys, tl):
    global _W
    _W = {"polys": polys, "tl": tl}


def _solve(name):
    A, b = _W["polys"][name]
    t = time.perf_counter()
    r = chebyshev_radius(A, b, time_limit=_W["tl"])
    return name, r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_type", default="cnn", choices=["mlp", "cnn"])
    ap.add_argument("--sample_idx", type=int, required=True)
    ap.add_argument("--bits",       type=int, required=True)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--data_path",  required=True)
    ap.add_argument("--stage1_dir", required=True,
                    help="stage-1 results root; b{BB}/volumes_sample{I}.json is read "
                         "to learn which classes are non-empty")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--p2_only", action="store_true",
                    help="Compute the radius of P2 alone, not of the P3(k). One LP per "
                         "tile instead of ~3, which is what makes it affordable over the "
                         "whole campaign. rho(P2) says whether the TILE is degenerate — a "
                         "flat P2 has zero volume, so it carries zero weight in the ideal "
                         "volume GACC while carrying a full one in the mean-width "
                         "surrogate. That is a different question from whether a given "
                         "P3(k) is zero-volume, which still needs its own radius.")
    ap.add_argument("--time_limit", type=float, default=3600.0)
    ap.add_argument("--n_workers",  type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()

    out_file = Path(a.output_dir) / f"chebyshev_sample{a.sample_idx}.json"
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        log.info(f"sample {a.sample_idx} (b={a.bits}) already done — skipping.")
        return

    s1 = Path(a.stage1_dir) / f"b{a.bits:02d}" / f"volumes_sample{a.sample_idx}.json"
    if not s1.exists():
        raise SystemExit(f"[ABORT] stage-1 result {s1} missing.")
    d1 = json.loads(s1.read_text())
    c  = d1["class_c"]
    W  = d1["widths_both"][str(a.bits)]
    non_empty = [k for k, w in enumerate(W) if w is not None and w > 0]
    log.info(f"b={a.bits} sample={a.sample_idx} c={c} non-empty classes: {non_empty}")

    if a.model_path is None:
        a.model_path = f"checkpoints/fashion_{a.model_type}_best.pth"
    dev = torch.device("cpu")
    Net = FashionCNN_Small if a.model_type == "cnn" else FashionMLP_Large
    fp = Net(); fp.load_state_dict(torch.load(a.model_path, map_location=dev,
                                              weights_only=True)); fp.eval()
    ds = torch.load(a.data_path, map_location=dev, weights_only=False)
    x, cc = ds[a.sample_idx]
    if int(cc) != c:
        raise SystemExit(f"[ABORT] class mismatch: dataset {int(cc)} vs stage-1 {c}")
    q = quantize_model(fp, bits=a.bits).eval()

    t0 = time.perf_counter()
    if a.model_type == "cnn":
        _, _, poly = build_cnn_all_polytopes_per_class(fp, {a.bits: q}, x.unsqueeze(0), c)
    else:
        _, _, poly = build_all_polytopes_per_class(fp, {a.bits: q},
                                                   x.flatten().unsqueeze(0), c)
    A_c, b_c, per = poly[a.bits]
    log.info(f"  polytopes built in {time.perf_counter() - t0:.1f}s")

    tn = lambda t: t.detach().cpu().numpy()
    polys = {"P2": (tn(A_c), tn(b_c))}
    if not a.p2_only:
        for k in non_empty:
            polys[f"P3_k{k}"] = (tn(per[k][0]), tn(per[k][1]))

    records = []
    with ProcessPoolExecutor(max_workers=min(a.n_workers, len(polys)),
                             initializer=_init,
                             initargs=(polys, a.time_limit)) as ex:
        for name, r, sec in ex.map(_solve, list(polys)):
            k = -1 if name == "P2" else int(name.split("k")[1])
            log.info(f"  {name:<9}: {sec:8.1f}s  r={r['radius']:.4e}  {r['status']}")
            # NaN is not valid JSON for anything but Python's own parser; a failed
            # LP has no radius, so write null rather than a value that would break
            # any other reader.
            if r["radius"] is not None and np.isnan(r["radius"]):
                r = {**r, "radius": None}
            records.append({"polytope": name, "k": k,
                            "mean_width": (d1["widths_correct"][str(a.bits)]
                                           if name == "P2" else W[k]),
                            **{kk: r[kk] for kk in ("radius", "status", "lp_status")},
                            "sec": sec})

    n_failed = sum(1 for r in records if r["status"] == "failed")
    if n_failed:
        log.warning(f"  {n_failed}/{len(records)} LP(s) did NOT converge "
                    f"(time_limit={a.time_limit}s) — reported as 'failed', not 'empty'.")

    out_file.write_text(json.dumps(
        {"model_type": a.model_type, "sample_idx": a.sample_idx, "bits": a.bits,
         "class_c": c, "time_limit": a.time_limit, "n_failed": n_failed,
         "polytopes": records}, indent=2))
    log.info(f"saved -> {out_file}")


if __name__ == "__main__":
    main()
