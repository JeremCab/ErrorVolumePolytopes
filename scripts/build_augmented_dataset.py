"""
build_augmented_dataset.py

For each sample x_i in a "correct" dataset, finds an augmented point x'_i
inside Polytope #1 where the 4-bit q-model has a different activation pattern
(hit-and-run MCMC, see src/optim/mcmc_augment.py).

Falls back to the original x_i when no augmented point is found within
max_tries.  Failed sample indices and the seed are saved to a JSON log so
the experiment is fully reproducible and failures can be inspected later.

Changing --seed produces a different augmented dataset from the same originals.

Usage (from project root):
    # Full run
    python scripts/build_augmented_dataset.py --model_type mlp --seed 42
    python scripts/build_augmented_dataset.py --model_type cnn --seed 42

    # Smoke test (5 samples — verify correctness and estimate per-sample time)
    python scripts/build_augmented_dataset.py --model_type mlp --seed 42 --n_samples 5
    python scripts/build_augmented_dataset.py --model_type cnn --seed 42 --n_samples 5

Output:
    data/fashionMNIST_augmented_{model_type}_seed{seed}.pt
    data/fashionMNIST_augmented_{model_type}_seed{seed}_log.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from src.models.networks           import FashionMLP_Large, FashionCNN_Small
from src.quantization.quantize     import quantize_model
from src.optim.build_polytopes     import build_all_polytopes
from src.optim.build_polytopes_cnn import build_cnn_all_polytopes
from src.optim.mcmc_augment        import find_augmented_point

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str, model_type: str, device: torch.device):
    model = FashionCNN_Small() if model_type == "cnn" else FashionMLP_Large()
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build MCMC-augmented dataset inside Polytope #1."
    )
    parser.add_argument("--model_type", type=str, default="mlp",
                        choices=["mlp", "cnn"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (default: checkpoints/fashion_{type}_best.pth)")
    parser.add_argument("--data_path",  type=str, default=None,
                        help="Path to input dataset  (default: data/fashionMNIST_correct_{type}.pt)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory for output files (default: data/)")
    parser.add_argument("--bits",       type=int, default=4,
                        help="Bit-width of q-model used to search for x' (default: 4)")
    parser.add_argument("--max_tries",  type=int, default=200,
                        help="MCMC attempts per sample before falling back (default: 200)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="RNG seed — change to generate a different augmented dataset")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Process only the first N samples (default: all). Use for smoke tests.")
    args = parser.parse_args()

    # Fill path defaults
    if args.model_path is None:
        args.model_path = f"checkpoints/fashion_{args.model_type}_best.pth"
    if args.data_path is None:
        args.data_path = f"data/fashionMNIST_correct_{args.model_type}.pt"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem       = f"fashionMNIST_augmented_{args.model_type}_seed{args.seed}"
    output_pt  = output_dir / f"{stem}.pt"
    output_log = output_dir / f"{stem}_log.json"

    device = get_device()

    log.info("=" * 55)
    log.info(f"  Model type   : {args.model_type.upper()}")
    log.info(f"  Device       : {device}")
    log.info(f"  Model path   : {args.model_path}")
    log.info(f"  Data path    : {args.data_path}")
    log.info(f"  Q-model bits : {args.bits}")
    log.info(f"  Max tries    : {args.max_tries}")
    log.info(f"  Seed         : {args.seed}")
    log.info(f"  N samples    : {args.n_samples if args.n_samples else 'all'}")
    log.info(f"  Output       : {output_pt}")
    log.info("=" * 55)

    # Load full-precision model and quantized model
    log.info("\nLoading model...")
    model  = load_model(args.model_path, args.model_type, device)
    qmodel = quantize_model(model, bits=args.bits)
    qmodel.eval()
    log.info(f"  {args.model_type.upper()} loaded on {device}, {args.bits}-bit q-model built.")

    # Load dataset
    log.info("\nLoading dataset...")
    dataset = torch.load(args.data_path, weights_only=False)
    n_total = len(dataset)
    n       = args.n_samples if args.n_samples is not None else n_total
    if n > n_total:
        raise ValueError(f"--n_samples {n} exceeds dataset size {n_total}")
    log.info(f"  {n_total} samples in dataset, processing {n}.")

    # Augmentation loop
    rng            = np.random.default_rng(args.seed)
    augmented      = []       # list of (x', c) — same format as original dataset
    failed_indices = []       # indices where fallback to original was used

    log.info("\nAugmenting...")
    t_total = time.perf_counter()

    for idx in tqdm(range(n), desc="samples"):
        x_i, c_i = dataset[idx]
        c_i = int(c_i)

        # Build Polytope #1 (A_base, b_base) using the 4-bit q-model.
        # We pass {bits: qmodel} so the function runs; only A_base is used here.
        # Both x_batch and x0 are moved to device so they match the model.
        if args.model_type == "cnn":
            x_batch = x_i.unsqueeze(0).to(device)              # (1, 1, 28, 28)
            x0      = x_i.unsqueeze(0).to(device)
            A, b, _ = build_cnn_all_polytopes(model, {args.bits: qmodel}, x_batch, c_i)
        else:
            x_batch = x_i.flatten().unsqueeze(0).to(device)    # (1, 784)
            x0      = x_i.flatten().unsqueeze(0).to(device)
            A, b, _ = build_all_polytopes(model, {args.bits: qmodel}, x_batch, c_i)

        A_np = A.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()

        x_prime = find_augmented_point(
            x0, A_np, b_np, qmodel,
            max_tries=args.max_tries,
            rng=rng,
        )

        if x_prime is None:
            failed_indices.append(idx)
            augmented.append((x_i, c_i))           # fallback: keep original
        else:
            # Remove batch dim — same shape as the original x_i
            augmented.append((x_prime.squeeze(0).detach().cpu(), c_i))

    elapsed  = time.perf_counter() - t_total
    n_failed = len(failed_indices)

    log.info(f"\nFinished in {elapsed:.1f}s  ({elapsed/n:.2f}s/sample)"
             f"  →  full {n_total} samples would take ~{elapsed/n*n_total/60:.0f} min"
             if n < n_total else
             f"\nFinished in {elapsed:.1f}s  ({elapsed/n:.2f}s/sample)")
    log.info(f"Success rate : {n - n_failed}/{n}  ({100*(n - n_failed)/n:.1f}%)")
    log.info(f"Fallbacks    : {n_failed}  (original kept)")

    # Save augmented dataset (same format as input — list of (x, c) tuples)
    torch.save(augmented, output_pt)
    log.info(f"\nSaved dataset : {output_pt}")

    # Save log with seed and failed indices
    log_data = {
        "seed":           args.seed,
        "model_type":     args.model_type,
        "model_path":     args.model_path,
        "data_path":      args.data_path,
        "bits":           args.bits,
        "max_tries":      args.max_tries,
        "n_samples":      n_total,
        "n_processed":    n,
        "n_failed":       n_failed,
        "success_rate":   round((n - n_failed) / n, 4),
        "elapsed_s":      round(elapsed, 1),
        "failed_indices": failed_indices,
    }
    with open(output_log, "w") as f:
        json.dump(log_data, f, indent=2)
    log.info(f"Saved log     : {output_log}")


if __name__ == "__main__":
    main()
