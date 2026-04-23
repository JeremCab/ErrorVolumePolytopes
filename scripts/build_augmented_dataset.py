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
from src.optim.mcmc_augment        import (find_augmented_point,
                                            find_augmented_point_margin,
                                            find_augmented_points_walk,
                                            select_diverse_representatives)

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
                        help="MCMC attempts per sample before falling back — strategies A and B (default: 200)")
    parser.add_argument("--nb_aug_points", type=int, default=100,
                        help="Target number of new representatives per original point — strategy C / walk (default: 100)")
    parser.add_argument("--max_steps",  type=int, default=5000,
                        help="Hard cap on walk steps per sample — strategy C / walk (default: 5000)")
    parser.add_argument("--nb_diverse",  type=int, default=None,
                        help="After the walk, apply greedy farthest-point diversity selection and keep "
                             "only nb_diverse representatives per original point. "
                             "None = no selection, keep all found (default: None).")
    parser.add_argument("--seed",       type=int, default=42,
                        help="RNG seed — change to generate a different augmented dataset")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Process only the first N samples (default: all). Use for smoke tests.")
    parser.add_argument("--strategy",  type=str, default="activation",
                        choices=["activation", "margin", "walk"],
                        help="Augmentation strategy: "
                             "'activation' (Strategy A — first x' with different q-model activation pattern), "
                             "'margin'     (Strategy B — x' minimising q-model classification margin), "
                             "'walk'       (Strategy C — full MCMC walk, nb_aug_points representatives per sample). "
                             "Default: activation.")
    parser.add_argument("--p1_filter_tol", type=float, default=None,
                        help="If set, drop walk representatives that violate any P1 constraint "
                             "by more than this tolerance (max(A·x+b) > tol). "
                             "Only meaningful for --walk_mode projected. "
                             "Default: None (keep all).")
    parser.add_argument("--walk_mode", type=str, default="projected",
                        choices=["projected", "pixel_bounds"],
                        help="Walk mode for strategy C: "
                             "'projected'    — hit-and-run inside P1, then clip to [-1,+1] (default). "
                             "                 Walk mixes freely; all augmented points are valid images. "
                             "'pixel_bounds' — pixel-bound constraints added as explicit rows to P1 "
                             "                 (paper-exact Ξ̄_x). For real images the walk gets stuck "
                             "                 (zero-length chords) because x0 sits on the pixel-box "
                             "                 boundary; use this mode to demonstrate the degeneracy.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag appended to the output filename stem, e.g. 'qcorrect' "
                             "or 'qincorrect', to avoid overwriting outputs from different input "
                             "datasets. Default: '' (no tag).")
    args = parser.parse_args()

    # Fill path defaults
    if args.model_path is None:
        args.model_path = f"checkpoints/fashion_{args.model_type}_best.pth"
    if args.data_path is None:
        args.data_path = f"data/fashionMNIST_correct_{args.model_type}.pt"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = f"_{args.walk_mode}" if args.strategy == "walk" and args.walk_mode != "projected" else ""
    strategy_suffix = "" if args.strategy == "activation" else f"_{args.strategy}"
    tag_suffix  = f"_{args.tag}" if args.tag else ""
    stem        = f"fashionMNIST_augmented_{args.model_type}_seed{args.seed}{strategy_suffix}{mode_suffix}{tag_suffix}"
    output_pt  = output_dir / f"{stem}.pt"
    output_log = output_dir / f"{stem}_log.json"

    device = get_device()

    log.info("=" * 55)
    log.info(f"  Model type   : {args.model_type.upper()}")
    log.info(f"  Device       : {device}")
    log.info(f"  Model path   : {args.model_path}")
    log.info(f"  Data path    : {args.data_path}")
    log.info(f"  Q-model bits : {args.bits}")
    log.info(f"  Strategy     : {args.strategy}")
    if args.strategy == "walk":
        log.info(f"  Walk mode    : {args.walk_mode}")
        log.info(f"  Nb aug pts   : {args.nb_aug_points}  (collect target per sample)")
        log.info(f"  Max steps    : {args.max_steps}   (hard cap)")
        if args.nb_diverse is not None:
            log.info(f"  Nb diverse   : {args.nb_diverse}  (kept after greedy farthest-point selection)")
        else:
            log.info(f"  Nb diverse   : all  (no diversity selection)")
        if args.p1_filter_tol is not None:
            log.info(f"  P1 filter    : tol={args.p1_filter_tol}  (drop reps outside P1)")
        else:
            log.info(f"  P1 filter    : off  (keep all reps)")
    else:
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
    failed_indices = []       # indices where fallback to original was used (activation strategy)
    reps_per_sample = []      # number of representatives per sample (walk strategy)

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

        if args.strategy == "walk":
            # Strategy C: full MCMC walk — collect nb_aug_points equivalence-class representatives
            reps = find_augmented_points_walk(
                x0, A_np, b_np, qmodel,
                nb_aug_points=args.nb_aug_points,
                max_steps=args.max_steps,
                rng=rng,
                mode=args.walk_mode,
                p1_filter_tol=args.p1_filter_tol,
            )
            # Optional greedy farthest-point diversity selection
            if args.nb_diverse is not None and len(reps) > args.nb_diverse:
                reps = select_diverse_representatives(reps, x0, k=args.nb_diverse)
            n_reps = len(reps)
            reps_per_sample.append(n_reps)
            if n_reps == 0:
                failed_indices.append(idx)
            for rep in reps:
                # Remove batch dim — same shape as the original x_i
                augmented.append((rep.squeeze(0), c_i))

        elif args.strategy == "margin":
            # Strategy B: minimise q-model classification margin
            x_prime = find_augmented_point_margin(
                x0, A_np, b_np, qmodel, c_i,
                max_tries=args.max_tries,
                rng=rng,
            )
            # margin strategy always returns a tensor (never None)
            augmented.append((x_prime.squeeze(0).detach().cpu(), c_i))

        else:
            # Strategy A: accept first x' with different q-model activation pattern
            x_prime = find_augmented_point(
                x0, A_np, b_np, qmodel,
                max_tries=args.max_tries,
                rng=rng,
            )
            if x_prime is None:
                failed_indices.append(idx)
                augmented.append((x_i, c_i))       # fallback: keep original
            else:
                # Remove batch dim — same shape as the original x_i
                augmented.append((x_prime.squeeze(0).detach().cpu(), c_i))

    elapsed  = time.perf_counter() - t_total
    n_failed = len(failed_indices)
    n_aug    = len(augmented)

    log.info(f"\nFinished in {elapsed:.1f}s  ({elapsed/n:.2f}s/sample)"
             f"  →  full {n_total} samples would take ~{elapsed/n*n_total/60:.0f} min"
             if n < n_total else
             f"\nFinished in {elapsed:.1f}s  ({elapsed/n:.2f}s/sample)")

    if args.strategy == "walk":
        avg_reps = np.mean(reps_per_sample) if reps_per_sample else 0.0
        max_reps = max(reps_per_sample) if reps_per_sample else 0
        log.info(f"Total augmented points : {n_aug}")
        log.info(f"Reps/sample  avg={avg_reps:.2f}  max={max_reps}")
        log.info(f"Samples with 0 reps    : {n_failed}/{n}  ({100*n_failed/n:.1f}%)")
    else:
        log.info(f"Success rate : {n - n_failed}/{n}  ({100*(n - n_failed)/n:.1f}%)")
        log.info(f"Fallbacks    : {n_failed}  (original kept)")

    # Save augmented dataset (same format as input — list of (x, c) tuples)
    torch.save(augmented, output_pt)
    log.info(f"\nSaved dataset : {output_pt}  ({n_aug} points)")

    # Save log with seed and indices/counts
    log_data: dict = {
        "seed":           args.seed,
        "model_type":     args.model_type,
        "model_path":     args.model_path,
        "data_path":      args.data_path,
        "bits":           args.bits,
        "max_tries":      args.max_tries,
        "nb_aug_points":  args.nb_aug_points,
        "max_steps":      args.max_steps,
        "nb_diverse":     args.nb_diverse,
        "strategy":       args.strategy,
        "tag":            args.tag if args.tag else None,
        "walk_mode":      args.walk_mode if args.strategy == "walk" else None,
        "p1_filter_tol":  args.p1_filter_tol if args.strategy == "walk" else None,
        "n_samples":      n_total,
        "n_processed":    n,
        "n_augmented":    n_aug,
        "elapsed_s":      round(elapsed, 1),
    }
    if args.strategy == "walk":
        log_data["reps_per_sample"]     = reps_per_sample
        log_data["avg_reps_per_sample"] = round(float(np.mean(reps_per_sample)), 4) if reps_per_sample else 0.0
        log_data["max_reps_per_sample"] = int(max(reps_per_sample)) if reps_per_sample else 0
        log_data["n_zero_reps"]         = n_failed
        log_data["zero_rep_indices"]    = failed_indices
    else:
        log_data["n_failed"]       = n_failed
        log_data["success_rate"]   = round((n - n_failed) / n, 4)
        log_data["failed_indices"] = failed_indices

    with open(output_log, "w") as f:
        json.dump(log_data, f, indent=2)
    log.info(f"Saved log     : {output_log}")


if __name__ == "__main__":
    main()
