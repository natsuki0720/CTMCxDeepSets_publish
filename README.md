# CTMCxDeepSets_publish

This repository covers the full workflow from CTMC synthetic data generation (true Q and MLE-estimated Q') to DeepSets-based surrogate model training.

## Setup

The original `environment.yml` was a Linux-exported, OS-specific snapshot, so the environment definitions were reorganized for cross-platform setup.

### Linux + NVIDIA GPU

```bash
conda env create -f environment.linux.gpu.yml
conda activate ctmc
```

### Linux CPU-only

```bash
conda env create -f environment.base.yml
conda activate ctmc
```

### Windows + NVIDIA GPU

```bash
conda env create -f environment.windows.gpu.yml
conda activate ctmc
```

### macOS / Windows CPU-only

```bash
conda env create -f environment.base.yml
conda activate ctmc
```

### Backward compatibility with existing command

`environment.yml` is kept as a compatibility alias and creates the same CPU-oriented environment as `environment.base.yml`.

```bash
conda env create -f environment.yml
conda activate ctmc
```

If needed, install Playwright browser binaries after environment setup.

```bash
python -m playwright install
```

---

## Quick Start

### 1) Check `entrypoint_gen_with_MLE.py` options

`scripts/data_generation/entrypoint_gen_with_MLE.py` generates multiple CTMC datasets and saves them as sequential CSV files such as `dataset_0000.csv`. Key arguments are listed below.

- `--count`: Number of CSV files to generate (>= 1)
- `--out-dir`: Output directory
- `--states`: Number of states `N` (>= 3)
- `--lifespan`: Upper bound of transition time
- `--min-n`, `--max-n`: Sample-size range for each dataset (`min-n <= max-n`)
- `--base-seed`: Random seed
- `--init-r`: Initial value for MLE. **Specify exactly `states-1` values in comma-separated format** (for `states=4`, provide 3 values)
- `--run-parallel`, `--workers`: Parallel generation options

Because `--init-r` values often start with `-`, use the `=` form such as `--init-r=-0.5,-1,-1.5` to avoid shell parsing issues.

Also, for multi-line shell commands, do not add trailing spaces after a line-continuation backslash `\` (trailing spaces break continuation).

### 2) Generate about 1,000 datasets for testing

Linux/macOS (bash/zsh):

```bash
python scripts/data_generation/entrypoint_gen_with_MLE.py \
  --count 1000 \
  --out-dir ./data/test_1k \
  --states 4 \
  --lifespan 100.0 \
  --min-n 500 \
  --max-n 5000 \
  --base-seed 20250924 \
  --init-r=-0.5,-1,-1.5 \
  --run-parallel \
  --workers 8
```

Windows (PowerShell / Command Prompt, single line):

```powershell
python scripts/data_generation/entrypoint_gen_with_MLE.py --count 1000 --out-dir .\data\test_1k --states 4 --lifespan 100.0 --min-n 500 --max-n 5000 --base-seed 20250924 --init-r=-0.5,-1,-1.5 --run-parallel --workers 8
```

### 3) Generate about 200,000 datasets for training (parallel)

Linux/macOS (bash/zsh):

```bash
python scripts/data_generation/entrypoint_gen_with_MLE.py \
  --count 200000 \
  --out-dir ./data/train_200k \
  --states 4 \
  --lifespan 100.0 \
  --min-n 500 \
  --max-n 5000 \
  --base-seed 20250924 \
  --init-r=-0.5,-1,-1.5 \
  --run-parallel \
  --workers 8
```

Windows (PowerShell / Command Prompt, single line):

```powershell
python scripts/data_generation/entrypoint_gen_with_MLE.py --count 200000 --out-dir .\data\train_200k --states 4 --lifespan 100.0 --min-n 500 --max-n 5000 --base-seed 20250924 --init-r=-0.5,-1,-1.5 --run-parallel --workers 8
```

### 4) Run the training entry point

During training, `--n` samples are randomly drawn after data screening.

Linux/macOS (bash/zsh):

```bash
python scripts/train_entrypoint.py \
  --data-dir ./data/train_200k \
  --n 50000 \
  --out-dir ./out \
  --recursive \
  --val-ratio 0.1 \
  --epochs 1000 \
  --batch-size 128 \
  --lr 1e-3 \
  --patience 10 \
  --num-workers 8 \
  --state-index-base auto
```

Windows (PowerShell / Command Prompt, single line):

```powershell
python scripts/train_entrypoint.py --data-dir .\data\train_200k --n 50000 --out-dir .\out --recursive --val-ratio 0.1 --epochs 1000 --batch-size 128 --lr 1e-3 --patience 10 --num-workers 8 --state-index-base auto
```

> Adjust `--n` based on your available compute resources and runtime budget.

---

## Run Evaluation in Notebook

You can evaluate a pretrained model with `notebook/pretrained_model_eval.ipynb`.

1. Launch Jupyter:
   ```bash
   jupyter lab
   ```
2. Open `notebook/pretrained_model_eval.ipynb`.
3. In the path/config cells near the beginning of the notebook, update paths as needed:
   - Evaluation dataset directory
   - Model weights file (for example, `out/run_*/weights/best_model.pt`)
4. Run cells from top to bottom and review metrics/plots.

---

## Troubleshooting

- `EnvironmentNameNotFound: Could not find conda environment: ctmc`
  - Environment creation has not completed successfully. Run one of the setup commands (`environment.base.yml`, `environment.linux.gpu.yml`, or `environment.windows.gpu.yml`) first.
- Using a GPU environment file (`environment.linux.gpu.yml` or `environment.windows.gpu.yml`) on a machine without a CUDA-capable NVIDIA setup can fail during dependency resolution or runtime.
- On macOS, use `environment.base.yml` (CPU-only).
