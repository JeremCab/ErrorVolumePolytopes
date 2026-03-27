"""
dummy_parallel.py

Validates CPU parallelization, dataset loading, and result writing on Jean Zay.

Dataset format assumed: each sample is a (image_tensor, label) tuple,
e.g. from torch.utils.data.Subset of FashionMNIST.
"""

import argparse
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(data_path: str):
    """Load a .pt dataset (Subset or list of (tensor, label) tuples)."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    # weights_only=False is required because the file contains a
    # torch.utils.data.Subset object (not a plain tensor). Only load
    # files from trusted sources.
    data = torch.load(path, weights_only=False)
    # Materialise into a plain list to allow pickling across workers
    samples = [(x.clone(), int(y)) for x, y in data]
    if len(samples) == 0:
        raise ValueError(f"Dataset at '{data_path}' is empty.")
    return samples


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_into_chunks(dataset, n_chunks: int):
    """Split dataset into n_chunks roughly equal parts."""
    n = len(dataset)
    n_chunks = min(n_chunks, n)  # never more chunks than samples
    chunk_size = math.ceil(n / n_chunks)
    return [dataset[i:i + chunk_size] for i in range(0, n, chunk_size)]


# ---------------------------------------------------------------------------
# Dummy CPU-bound function
# ---------------------------------------------------------------------------

def _dummy(x: torch.Tensor, label: int) -> float:
    """Non-trivial but cheap CPU-bound computation on a (x, label) sample."""
    flat = x.float().flatten()
    return float(flat.norm()) * math.sin(float(flat.mean()) + label + 1.0)


def process_chunk(indexed_samples):
    """
    Process a list of (global_index, (x, label)) pairs.
    Returns a list of {"sample_id": int, "result": float} dicts.
    Workers never write to disk.
    """
    results = []
    for idx, (x, label) in indexed_samples:
        results.append({"sample_id": idx, "result": _dummy(x, label)})
    return results


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def get_num_cpus() -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None:
        n = int(slurm)
        if n < 1:
            raise ValueError(f"SLURM_CPUS_PER_TASK must be >= 1, got {n}")
        return n
    return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dummy parallel experiment for Jean Zay validation."
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to a .pt dataset file (list of (tensor, label) tuples)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory where results.jsonl will be written."
    )
    args = parser.parse_args()

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.jsonl"

    if output_file.exists():
        log.warning(f"Output file already exists and will be overwritten: {output_file}")

    n_cpus = get_num_cpus()
    log.info(f"CPUs available : {n_cpus}")

    # --- Load ---
    dataset = load_dataset(args.data_path)
    n = len(dataset)
    log.info(f"Dataset size   : {n} samples")

    # --- Chunk (attach global indices for traceability) ---
    indexed = list(enumerate(dataset))
    chunks = split_into_chunks(indexed, n_cpus)
    log.info(f"Chunks         : {len(chunks)} (≈{math.ceil(n / len(chunks))} samples each)")

    # --- Parallel processing ---
    t0 = time.perf_counter()
    all_results = []
    try:
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            chunk_results_list = list(tqdm(
                executor.map(process_chunk, chunks),
                total=len(chunks),
                desc="Processing chunks",
            ))
    except Exception as e:
        log.error(f"Worker failure during parallel processing: {e}")
        raise

    for chunk_results in chunk_results_list:
        all_results.extend(chunk_results)

    elapsed = time.perf_counter() - t0

    # Sort by sample_id for deterministic output
    all_results.sort(key=lambda r: r["sample_id"])

    # --- Write (main process only) ---
    with open(output_file, "w") as f:
        for record in all_results:
            f.write(json.dumps(record) + "\n")

    log.info(f"Processed {len(all_results)} samples in {elapsed:.2f}s")
    log.info(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
