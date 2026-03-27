# My Project

## Environment

Activate the virtual environment with
```bash
uv sync
source .venv/bin/activate
```

## Train model

```bash
python -m src.models.train --model mlp --epochs 3
```


## Run convergence experiment

From project root, run
```bash
python scripts/run_convergence.py \
    --sample_idx 0 \
    --model_path checkpoints/fashion_mlp_best.pth \
    --data_path  data/fashionMNIST_correct_mlp.pt \
    --bits 4 \
    --n_replications 20 \
    --output_dir results
```
Then open notebooks/test_convergence.ipynb, set RESULTS_PATH to match your run, and execute the cells.


## Run experiments

```bash
sbatch slurms/<slurm_name>.slurm
```
