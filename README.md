# ErrorVolumePolytopes

Code accompanying the paper **"Quantization Error via Polytope Volumes"** (2025, Cabessa & Svozil).

The central idea is to measure how much weight quantization shrinks the region of input space where a neural network classifies correctly. This region is a polytope (defined by ReLU activation constraints and classification constraints), and its size is estimated via a Monte Carlo mean-width estimator.

---

## Project structure

```
ErrorVolumePolytopes/
├── checkpoints/          # Trained model weights (.pth)
├── data/                 # Preprocessed datasets (.pt)
│   ├── fashionMNIST_correct_{mlp,cnn}.pt          # correctly-classified samples
│   └── fashionMNIST_augmented_{type}_seed{s}_walk_{tag}.pt  # MCMC augmented
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── results/              # JSON output files from experiments
│   ├── volumes_v3k_{mlp,cnn}/       # per-sample JSON from run_volumes_v3k.py
│   └── volumes_v3k_{mlp,cnn}_aug/   # per-augmented-point JSON (new GACC)
├── scripts/              # Runnable Python scripts
│   ├── run_convergence.py           # Convergence experiment (mean-width estimator)
│   ├── run_volumes_v3k.py           # V3k volumes: P1/P2/P3(k) for all bit-widths
│   ├── build_augmented_dataset.py   # MCMC Hit-and-Run augmented dataset generation
│   └── dummy_parallel.py            # HPC validation script
├── slurms/               # Slurm job submission files
│   ├── run_convergence.slurm
│   ├── run_volumes_v3k.slurm        # MLP: array over correct dataset
│   ├── run_volumes_v3k_cnn.slurm   # CNN: array over correct dataset
│   ├── run_volumes_v3k_aug_mlp.slurm  # MLP: array over augmented dataset
│   ├── run_volumes_v3k_aug_cnn.slurm  # CNN: array over augmented dataset
│   └── run_dummy_parallel.slurm
└── src/                  # Library code
    ├── models/           # Network architectures and training
    ├── quantization/     # Weight quantization (post-training)
    ├── shortcuts/        # Shortcut (effective) weight computation
    └── optim/            # Polytope construction, pruning, volume estimation
                          #   mcmc_augment.py  — Hit-and-Run MCMC inside P1
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

### Other HPC clusters (e.g., Prague)

The slurm scripts use `module load pytorch-gpu/py3/2.8.0` and `$WORK` — adjust these
to match the cluster's module system and scratch directory convention.
Required files to copy to the cluster:
- `checkpoints/fashion_mlp_best.pth` and/or `fashion_cnn_best.pth`
- `data/fashionMNIST_correct_{mlp,cnn}.pt` (original correct datasets)
- `data/fashionMNIST_augmented_*.pt` (augmented datasets, if running new-GACC pipeline)

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

## V3k volume experiment (P1 / P2 / P3(k) for all bit-widths)

Estimates mean widths of three nested polytopes for every sample and every bit-width
`b ∈ {4, 6, 8, 10, 12, 16}` in a single run. This is the main experiment for the
generalised accuracy (GACC) metric.

### Locally (single sample)

```bash
python scripts/run_volumes_v3k.py \
    --model_type    mlp \
    --sample_idx    0 \
    --model_path    checkpoints/fashion_mlp_best.pth \
    --data_path     data/fashionMNIST_correct_mlp.pt \
    --n_directions  200 \
    --output_dir    results/volumes_v3k_mlp
```

Output: `results/volumes_v3k_mlp/volumes_sample0.json`

### On HPC (array over all samples)

```bash
sbatch slurms/run_volumes_v3k.slurm      # MLP, samples 0-999
sbatch slurms/run_volumes_v3k_cnn.slurm  # CNN, samples 0-999
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--sample_idx` | required | Index of the sample in the dataset |
| `--data_path` | `data/fashionMNIST_correct_{type}.pt` | Input dataset |
| `--n_directions` | `200` | Monte Carlo directions for mean-width estimator |
| `--bits_grid` | `4 6 8 10 12 16` | Bit-widths to evaluate |
| `--output_dir` | `results/volumes_v3k_{type}` | Output directory |

---

## MCMC Hit-and-Run augmented dataset (new GACC pipeline)

Generates augmented data points inside Polytope #1 (the FP-model linear region) by
running a Hit-and-Run Markov chain. Each collected point lands in a *different*
q-model activation region (a distinct P2 polytope), so the union of augmented points
covers P1 more completely. The new GACC averages over all these augmented points,
making it comparable across bit-widths (since P1 is b-independent).

### Step 1 — Generate augmented datasets (run locally, ~5–10 min)

```bash
# MLP: 200 original samples × up to 50 reps each ≈ 10 000 augmented points
python scripts/build_augmented_dataset.py \
    --model_type     mlp \
    --strategy       walk \
    --walk_mode      projected \
    --nb_aug_points  50 \
    --max_steps      10000 \
    --p1_filter_tol  1e-4 \
    --n_samples      200 \
    --seed           42 \
    --tag            aug200

# CNN: same parameters
python scripts/build_augmented_dataset.py \
    --model_type     cnn \
    --strategy       walk \
    --walk_mode      projected \
    --nb_aug_points  50 \
    --max_steps      10000 \
    --p1_filter_tol  1e-4 \
    --n_samples      200 \
    --seed           42 \
    --tag            aug200
```

Outputs:
- `data/fashionMNIST_augmented_mlp_seed42_walk_aug200.pt`
- `data/fashionMNIST_augmented_cnn_seed42_walk_aug200.pt`

Key options for `build_augmented_dataset.py`:

| Argument | Default | Description |
|---|---|---|
| `--strategy` | `activation` | Use `walk` (Strategy C) for Jiri's experiment |
| `--walk_mode` | `projected` | Always use `projected` — real images have pixels at ±1 so `pixel_bounds` mode gets stuck |
| `--nb_aug_points` | `100` | Target reps per original sample |
| `--max_steps` | `5000` | Hard cap on walk steps (increase for harder samples) |
| `--p1_filter_tol` | `None` | Drop reps with max(A·x+b) > tol — use `1e-4` in production |
| `--nb_diverse` | `None` | Keep all reps (do NOT set for Jiri's experiment — we want all distinct P2s) |
| `--n_samples` | all | Process only first N samples (use for smoke tests or bounded runs) |
| `--tag` | `""` | Appended to output filename stem to avoid collisions |

### Step 2 — Compute V3k volumes on augmented points (HPC)

```bash
sbatch slurms/run_volumes_v3k_aug_mlp.slurm  # one task per augmented point
sbatch slurms/run_volumes_v3k_aug_cnn.slurm
```

The augmented `.pt` file is a flat list of `(x', c)` pairs, directly readable by
`run_volumes_v3k.py` via `--data_path` and `--sample_idx`. Set `--array` to
`0-<len(augmented_dataset)-1>` in the slurm script.

Output: `results/volumes_v3k_mlp_aug/volumes_sample{i}.json` for each augmented point.

### Step 3 — Compute new GACC

```
new_GACC(b) = Σ_y V3_c(y, b) / Σ_y Σ_k V3_k(y, b)
```

where the sum runs over all augmented points `y`. Use `notebooks/plot_volumes_v3k.ipynb`
(point it to `results/volumes_v3k_mlp_aug/`) to visualise convergence and GACC vs b.

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
| `plot_volumes_v3k.ipynb` | Plot V1/V2/V3k widths and GACC vs bit-width (MLP and CNN) |
| `test_data_augmentation_mcmc.ipynb` | Unit tests for Hit-and-Run MCMC walk (chord interval, walk, modes) |
| `verify_augmented_datasets.ipynb` | Verify augmented .pt files: pixel bounds, P1 membership, visual grids |
| `data_for_jiri.ipynb` | Generate CSV tables of V1/V2/V3k for 10 random samples (MLP and CNN) |

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
