"""
float_precision.py — does float32 propagation define the polytope unambiguously?

Jiri (MAIL_23) suggested the small Chebyshev radii are rounding error accumulated
over hundreds of thousands of float32 operations, the true radius being zero. This
tests that directly: rebuild the same polytopes in float64 and measure how far each
face moves. A face a.y <= b sits at distance b/||a|| from the origin, so the
displacement is (|db| + ||da||.||y||)/||a|| with ||y|| <= sqrt(784).

CRITICAL: quantise in float32 as production does, THEN cast the quantised model to
float64. Quantising an already-float64 model is a different experiment — at b=4 a
weight sitting on a rounding boundary jumps a whole level — and it inflated the first
measurement tenfold.

It also compares the ACTIVATION PATTERNS between precisions, which is where the real
problem turned out to be.

Result on record (2026-09-11), CNN sample 760:
    b=16  face displacement  median 1.96e-06  max 1.08e-05   0 neurons flip
    b=4   face displacement  median 2.2e-06   max 2.6e+02   10 neurons flip (quantised only)
So at b=16 the displacement is below the smallest radius Jiri attributes to noise
(1.6e-5) — his mechanism cannot explain them. At b=4 the polytope is not defined
unambiguously at float32: 4-bit weights take 16 values, pre-activations pile up on the
ReLU threshold, and 10 neurons land on opposite sides depending on the arithmetic.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from src.models.networks import FashionCNN_Small
from src.optim.build_polytopes_cnn import build_cnn_all_polytopes_per_class
from src.optim.mcmc_augment import activation_pattern
from src.quantization.quantize import quantize_model


def load32():
    m = FashionCNN_Small()
    m.load_state_dict(torch.load("checkpoints/fashion_cnn_best.pth",
                                 map_location="cpu", weights_only=True))
    return m.eval()


def main(bits_list=(16, 4), sample=760):
    for bits in bits_list:
        ds = torch.load(f"data/fashionMNIST_augmented_cnn_seed42_walk_sample{sample}_b{bits}.pt",
                        map_location="cpu", weights_only=False)
        x, c = ds[0]; c = int(c)
        print(f"\n===== CNN b={bits}, sample {sample} =====", flush=True)

        out = {}
        for tag, dt in (("f32", torch.float32), ("f64", torch.float64)):
            fp = load32().to(dt).eval()
            q = quantize_model(load32(), bits=bits).to(dt).eval()   # quantise in f32 first
            _, _, poly = build_cnn_all_polytopes_per_class(fp, {bits: q},
                                                           x.to(dt).unsqueeze(0), c)
            A, b, _ = poly[bits]
            out[tag] = (A.numpy().astype(np.float64), b.numpy().astype(np.float64))

        (A32, b32), (A64, b64) = out["f32"], out["f64"]
        if A32.shape != A64.shape:
            print("  shapes differ -> activation patterns differ outright", flush=True); continue
        nrm = np.linalg.norm(A64, axis=1); keep = nrm > 1e-10
        R = np.sqrt(A64.shape[1])
        dA = np.linalg.norm(A32 - A64, axis=1)[keep]
        db = np.abs(b32 - b64)[keep]
        shift = (db + dA * R) / nrm[keep]
        print(f"  constraints kept          : {keep.sum()}", flush=True)
        print(f"  coefficient rel. error    : median {np.median(dA/nrm[keep]):.2e}  "
              f"max {np.max(dA/nrm[keep]):.2e}", flush=True)
        print(f"  FACE DISPLACEMENT         : median {np.median(shift):.2e}  "
              f"p99 {np.percentile(shift,99):.2e}  max {np.max(shift):.2e}", flush=True)

        for lbl, mk in (("FP model  ", lambda dt: load32().to(dt).eval()),
                        ("quantised ", lambda dt: quantize_model(load32(), bits=bits).to(dt).eval())):
            p32 = activation_pattern(x.unsqueeze(0).to(torch.float32), mk(torch.float32))
            p64 = activation_pattern(x.unsqueeze(0).to(torch.float64), mk(torch.float64))
            d = int((p32 != p64).sum())
            print(f"  {lbl}: {d} neuron(s) flip of {len(p32)} ({100*d/len(p32):.3f} %)",
                  flush=True)


if __name__ == "__main__":
    main()
