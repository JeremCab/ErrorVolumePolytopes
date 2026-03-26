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

## Run experiments

```bash
python -m scripts.run_experiment scripts/config.yaml
```
