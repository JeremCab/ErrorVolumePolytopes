"""
find_qmodel_samples.py

Loads a "correct" dataset (already filtered for the full-precision model),
evaluates the 4-bit q-model on it, and saves two lists of sample indices:
  - q-correct  : q-model predicts the right class c
  - q-incorrect: q-model predicts a wrong class

Up to --nb_points indices are saved from each category (drawn randomly with
--seed for reproducibility).  The full counts are always reported regardless
of --nb_points.

Usage (from project root):
    python scripts/find_qmodel_samples.py --model_type mlp --bits 4 --nb_points 50
    python scripts/find_qmodel_samples.py --model_type cnn --bits 4 --nb_points 50

Output:
    data/qmodel_samples_{model_type}_b{bits}.json   — indices + summary stats
    data/qmodel_qcorrect_{model_type}_b{bits}.pt    — subset of (x, c) tuples (q-correct)
    data/qmodel_qincorrect_{model_type}_b{bits}.pt  — subset of (x, c) tuples (q-incorrect)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.models.networks       import FashionMLP_Large, FashionCNN_Small
from src.quantization.quantize import quantize_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Split a correct dataset by q-model classification outcome."
    )
    parser.add_argument("--model_type", type=str, default="mlp",
                        choices=["mlp", "cnn"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (default: checkpoints/fashion_{type}_best.pth)")
    parser.add_argument("--data_path",  type=str, default=None,
                        help="Path to input dataset  (default: data/fashionMNIST_correct_{type}.pt)")
    parser.add_argument("--bits",       type=int, default=4,
                        help="Bit-width for q-model evaluation (default: 4)")
    parser.add_argument("--nb_points",  type=int, default=50,
                        help="Number of indices to save from each category (default: 50). "
                             "Use 0 to save all.")
    parser.add_argument("--seed",       type=int, default=42,
                        help="RNG seed for random selection of nb_points (default: 42)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory for output files (default: data/)")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"checkpoints/fashion_{args.model_type}_best.pth"
    if args.data_path is None:
        args.data_path = f"data/fashionMNIST_correct_{args.model_type}.pt"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"qmodel_samples_{args.model_type}_b{args.bits}"

    device = get_device()

    log.info("=" * 55)
    log.info(f"  Model type   : {args.model_type.upper()}")
    log.info(f"  Device       : {device}")
    log.info(f"  Model path   : {args.model_path}")
    log.info(f"  Data path    : {args.data_path}")
    log.info(f"  Q-model bits : {args.bits}")
    log.info(f"  Nb points    : {args.nb_points if args.nb_points > 0 else 'all'} per category")
    log.info(f"  Seed         : {args.seed}")
    log.info("=" * 55)

    # ── Load model and quantize ──────────────────────────────────────────────
    if args.model_type == "cnn":
        model = FashionCNN_Small()
    else:
        model = FashionMLP_Large()
    state_dict = torch.load(args.model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    qmodel = quantize_model(model, bits=args.bits)
    qmodel.eval()
    log.info(f"\n{args.model_type.upper()} loaded, {args.bits}-bit q-model built.")

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset = torch.load(args.data_path, weights_only=False)
    n_total = len(dataset)
    log.info(f"Dataset loaded: {n_total} samples.")

    # ── Evaluate q-model ─────────────────────────────────────────────────────
    q_correct_idx   = []   # indices where q-model predicts correctly
    q_incorrect_idx = []   # indices where q-model predicts incorrectly

    log.info("\nEvaluating q-model...")
    with torch.no_grad():
        for idx in range(n_total):
            x, c = dataset[idx]
            c = int(c)

            if args.model_type == "cnn":
                x_in = x.unsqueeze(0).to(device)
            else:
                x_in = x.flatten().unsqueeze(0).to(device)

            logits = qmodel(x_in)           # (1, n_classes)
            pred   = int(logits.argmax(dim=1).item())

            if pred == c:
                q_correct_idx.append(idx)
            else:
                q_incorrect_idx.append(idx)

    n_correct   = len(q_correct_idx)
    n_incorrect = len(q_incorrect_idx)
    acc = n_correct / n_total

    log.info(f"\nQ-model accuracy on correct subset:")
    log.info(f"  Correct   : {n_correct}/{n_total}  ({100*acc:.2f}%)")
    log.info(f"  Incorrect : {n_incorrect}/{n_total}  ({100*(1-acc):.2f}%)")

    # ── Select nb_points from each category ──────────────────────────────────
    rng = np.random.default_rng(args.seed)

    def select(indices, k):
        if k <= 0 or k >= len(indices):
            return list(indices)
        chosen = rng.choice(len(indices), size=k, replace=False)
        return [indices[i] for i in sorted(chosen)]

    nb = args.nb_points
    saved_correct   = select(q_correct_idx,   nb)
    saved_incorrect = select(q_incorrect_idx, nb)

    log.info(f"\nSelected for saving:")
    log.info(f"  Q-correct   : {len(saved_correct)} indices")
    log.info(f"  Q-incorrect : {len(saved_incorrect)} indices")

    # ── Save .pt subsets ─────────────────────────────────────────────────────
    def save_subset(indices, suffix):
        subset = [dataset[i] for i in indices]
        path   = output_dir / f"qmodel_{suffix}_{args.model_type}_b{args.bits}.pt"
        torch.save(subset, path)
        log.info(f"  Saved {len(subset):4d} samples → {path}")
        return str(path)

    log.info("\nSaving subsets...")
    path_correct   = save_subset(saved_correct,   "qcorrect")
    path_incorrect = save_subset(saved_incorrect, "qincorrect")

    # ── Save JSON log ─────────────────────────────────────────────────────────
    log_data = {
        "model_type":          args.model_type,
        "model_path":          args.model_path,
        "data_path":           args.data_path,
        "bits":                args.bits,
        "seed":                args.seed,
        "n_total":             n_total,
        "n_qcorrect_total":    n_correct,
        "n_qincorrect_total":  n_incorrect,
        "qmodel_accuracy":     round(acc, 6),
        "nb_points_requested": nb,
        "n_qcorrect_saved":    len(saved_correct),
        "n_qincorrect_saved":  len(saved_incorrect),
        "qcorrect_indices":    saved_correct,
        "qincorrect_indices":  saved_incorrect,
        "output_qcorrect":     path_correct,
        "output_qincorrect":   path_incorrect,
    }
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=2)
    log.info(f"  Saved log         → {json_path}")


if __name__ == "__main__":
    main()
