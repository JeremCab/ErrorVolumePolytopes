"""
smoke_test_qmodel_samples.py

Verifies that the datasets produced by find_qmodel_samples.py are consistent:
  - Every sample in qmodel_qcorrect_*  is correctly   classified by the q-model
  - Every sample in qmodel_qincorrect_* is incorrectly classified by the q-model

Run from project root:
    python scripts/smoke_test_qmodel_samples.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.networks       import FashionMLP_Large, FashionCNN_Small
from src.quantization.quantize import quantize_model

BITS = 4

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_dataset(dataset, qmodel, model_type, expected_correct, device):
    """Returns (n_checked, n_failures)."""
    n_failures = 0
    with torch.no_grad():
        for x, c in dataset:
            c = int(c)
            if model_type == "cnn":
                x_in = x.unsqueeze(0).to(device)
            else:
                x_in = x.flatten().unsqueeze(0).to(device)
            pred = int(qmodel(x_in).argmax(dim=1).item())
            classified_correctly = (pred == c)
            if classified_correctly != expected_correct:
                n_failures += 1
    return len(dataset), n_failures


def run(model_type, device):
    print(f"\n{'='*50}")
    print(f"  {model_type.upper()}  —  {BITS}-bit q-model")
    print(f"{'='*50}")

    # Load model
    model = FashionCNN_Small() if model_type == "cnn" else FashionMLP_Large()
    ckpt  = f"checkpoints/fashion_{model_type}_best.pth"
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval().to(device)
    qmodel = quantize_model(model, bits=BITS)
    qmodel.eval()

    all_pass = True
    for expected_correct, tag in [(True, "qcorrect"), (False, "qincorrect")]:
        path = Path(f"data/qmodel_{tag}_{model_type}_b{BITS}.pt")
        dataset = torch.load(path, weights_only=False)
        n, n_fail = check_dataset(dataset, qmodel, model_type, expected_correct, device)
        status = "PASS" if n_fail == 0 else f"FAIL  ({n_fail}/{n} wrong)"
        print(f"  {tag:<22} ({n:>4} samples)  →  {status}")
        if n_fail > 0:
            all_pass = False

    return all_pass


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")

    results = {}
    for mt in ["mlp", "cnn"]:
        results[mt] = run(mt, device)

    print(f"\n{'='*50}")
    overall = all(results.values())
    print(f"  Overall : {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    print(f"{'='*50}")
    sys.exit(0 if overall else 1)
