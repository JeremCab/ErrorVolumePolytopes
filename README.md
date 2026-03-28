# ErrorVolumePolytopes

Code accompanying the paper **"Quantization Error via Polytope Volumes"** (2025, Cabessa & Svozil).

The central idea is to measure how much weight quantization shrinks the region of input space where a neural network classifies correctly. This region is a polytope (defined by ReLU activation constraints and classification constraints), and its size is estimated via a Monte Carlo mean-width estimator.

---

## Project structure

```
ErrorVolumePolytopes/
├── checkpoints/          # Trained model weights (.pth)
├── data/                 # Preprocessed datasets (.pt)
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── results/              # JSON output files from experiments
├── scripts/              # Runnable Python scripts
│   ├── run_convergence.py      # Convergence experiment (mean-width estimator)
│   └── dummy_parallel.py       # HPC validation script
├── slurms/               # Slurm job submission files
│   ├── run_convergence.slurm
│   └── run_dummy_parallel.slurm
└── src/                  # Library code
    ├── models/           # Network architectures and training
    ├── quantization/     # Weight quantization (post-training)
    ├── shortcuts/        # Shortcut (effective) weight computation
    └── optim/            # Polytope construction, pruning, volume estimation
```

---

## Setup

### Local (laptop / workstation)

Requires Python >= 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jeremiecabessa/ErrorVolumePolytopes.git
cd ErrorVolumePolytopes
uv sync
source .venv/bin/activate
```

### Jean Zay (IDRIS supercomputer)

```bash
# 1. Clone the repo into $WORK
git clone https://github.com/jeremiecabessa/ErrorVolumePolytopes.git $WORK/ErrorVolumePolytopes
cd $WORK/ErrorVolumePolytopes

# 2. Load the PyTorch module (provides torch, numpy, scipy)
module load pytorch-gpu/py3/2.8.0

# 3. Install the remaining dependencies (cvxpy, tqdm, matplotlib, ...)
export PYTHONUSERBASE=$WORK/.local_python
pip install --user --no-cache-dir -e .

# 4. Create runtime directories
mkdir -p logs results

# 5. Copy data and checkpoint files from your laptop
#    (run these on your laptop, not on Jean Zay)
#    scp data/fashionMNIST_correct_mlp.pt  jz:$WORK/ErrorVolumePolytopes/data/
#    scp checkpoints/fashion_mlp_best.pth  jz:$WORK/ErrorVolumePolytopes/checkpoints/
```

---

## Convergence experiment

Tests how quickly the mean-width estimator converges as the number of random
directions grows. Runs over a grid `N ∈ {10, 20, 30, 50, 75, 100, 150, 200}`
with `R = 20` independent replications per value of N.

### Locally (single sample)

```bash
python scripts/run_convergence.py \
    --sample_idx     0 \
    --model_path     checkpoints/fashion_mlp_best.pth \
    --data_path      data/fashionMNIST_correct_mlp.pt \
    --bits           4 \
    --n_replications 20 \
    --output_dir     results
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--sample_idx` | required | Index of the sample in the dataset |
| `--bits` | `4` | Quantization bit-width |
| `--n_replications` | `20` | Replications per N value |
| `--n_workers` | auto | Workers for `ProcessPoolExecutor` (defaults to `SLURM_CPUS_PER_TASK` or local cpu count) |
| `--output_dir` | `results` | Directory where the JSON result is saved |

Output: `results/convergence_sample{idx}_bits{bits}.json`

### On Jean Zay (50 samples in parallel)

The Slurm script launches one job per sample as a job array (50 jobs × 40 CPUs each).

```bash
cd $WORK/ErrorVolumePolytopes
sbatch slurms/run_convergence.slurm
```

Results are written to `$WORK/ErrorVolumePolytopes/results/` once each job completes.
To change the number of samples, edit the `--array` line in `slurms/run_convergence.slurm`:

```bash
#SBATCH --array=0-49   # runs sample indices 0 through 49
```

---

## Visualising results

Open `notebooks/test_convergence.ipynb`, set `RESULTS_PATH` to the JSON file
produced by the experiment, and run all cells. The notebook plots:

1. Mean width ± 1 std vs N
2. Std vs N on a log-log scale (expected slope: −½)
3. Coefficient of variation (%) vs N

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `test_shortcut_weights.ipynb` | Verify that shortcut (effective) weight computation is correct |
| `test_polytopes.ipynb` | Build polytopes and check sample membership |
| `test_pruning_constraints.ipynb` | Test redundant constraint removal (Clarkson / Ray-Tracing) |
| `test_volumes.ipynb` | Manually run the mean-width estimator on a single sample |
| `test_convergence.ipynb` | Visualise convergence experiment results |

---

## Key concepts

**Shortcut weights.** A ReLU network is locally linear in any fixed activation region. Given a sample `x`, we compute the effective affine map from input to each layer's pre-activations (`src/shortcuts/shortcut_weights.py`).

**Polytopes.** The activation region around `x` is a polytope defined by the shortcut weights. We build two polytopes (`src/optim/build_polytopes.py`):
- `correct_polytope`: region where the full-precision model classifies as class `c`
- `both_polytope`: sub-region where the quantized model also classifies as `c`

**Mean-width estimator.** We estimate the width of each polytope along `N` random directions via linear programming (`src/optim/compute_volumes.py`). The quantization error is:

```
error = 1 - mean_width(both) / mean_width(correct)
```

**Constraint pruning.** Redundant constraints can be removed before volume estimation (`src/optim/prune_constraints.py`) using Clarkson's sequential LP method or a faster Ray-Tracing + Clarkson hybrid.
