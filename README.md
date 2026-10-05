# Explaining the in-context learning capacity of language models for classification under the cross-entropy loss: theory, mechanism and interpretation
Code for reproducing the experiments and figures in the paper *Explaining the in-context learning capacity of language models for classification under the cross-entropy loss: theory, mechanism and interpretation*.

## Setup

We use [`uv`](https://github.com/astral-sh/uv) to manage dependencies:

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies into .venv
uv sync
```

You can run commands directly with `uv run python main.py` without activating the virtual environment.

Requirements:
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12+ (tested on an RTX A6000 48GB and RTX 4500 PRO 32GB)

---

## Reproducing paper figures

To plot the figures directly from the saved logs in `results/runs/`:

### Data generating process (DGP)

```bash
uv run python main.py --dgp
```

### Paper plots (combined)

```bash
# Basic asymptotic analysis
uv run python main.py --run 2026_09_23_20_10_18 2026_09_23_20_07_26 --plot asymptotic --target_entropy 0.26

# ICL scaling analysis
uv run python main.py --run 2026_07_23_17_24_35 2026_06_20_00_31_10 --xlim 1 100 --plot scaling --target_entropy 0.69
```

### Individual plots

```bash
uv run python main.py --run 2026_09_23_20_10_18 --plot asymptotic 
uv run python main.py --run 2026_09_23_20_07_26 --plot asymptotic 
uv run python main.py --run 2026_07_23_17_24_35 --plot scaling --xlim 1 100
uv run python main.py --run 2026_06_20_00_31_10 --plot scaling --xlim 1 100 
```

Plots are saved as PDFs under `results/figures/run_<timestamp>/` for single-run plots, and directly under `results/figures/` for combined and DGP figures.

---

## Running experiments from scratch

### Re-running paper experiments

Here are the exact commands to reproduce the evaluations for each paper run from scratch:

```bash
# run_2026_09_23_20_10_18: 7B model entropy scaling
uv run python main.py --models 7B --max_train 300 --num_train 64 --num_contexts 256 --num_val 64 --num_entropy_levels 10

# run_2026_09_23_20_07_26: Model scaling at H=0.69
uv run python main.py --max_train 300 --num_train 64 --num_contexts 256 --num_val 64 --entropies 0.69

# run_2026_07_23_17_24_35: 7B model entropy scaling for bound analysis
uv run python main.py --models 7B --max_train 1000 --num_train 500 --num_contexts 100 --num_val 100 --old_geom --entropies 0.36 0.4 0.44 0.47 0.51 0.55 0.58 0.62 0.66 0.69

# run_2026_06_20_00_31_10: Model scaling at H=0.69 for bound analysis
uv run python main.py --max_train 1000 --num_train 500 --num_contexts 100 --num_val 100 --old_geom --entropies 0.69
```

### Custom runs

To run an experiment with custom settings, pass whatever flags you want to override:
```bash
uv run python main.py --dim 2 --models 0.5B 7B --max_train 500 --num_train 32 --num_contexts 128 --num_val 128
```

Or run with default settings:
```bash
uv run python main.py
```

---

## Repository structure

```
├── configs/
├── src/
│   ├── theory/
│   ├── evaluation/
│   ├── analysis/
│   └── utils/
├── results/
│   ├── .cache/
│   ├── runs/
│   └── figures/
└── main.py
```

## License

This repository is licensed under the [MIT License](LICENSE).
