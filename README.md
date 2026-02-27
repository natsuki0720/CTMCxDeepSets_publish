# CTMCxDeepSets_publish

This repository covers the full workflow from CTMC synthetic data generation (true Q and MLE-estimated Q') to DeepSets-based surrogate model training.

## Setup

```bash
conda env create -f environment.yml
conda activate ctmc
```

---

## Quick Start

### 1) Generate ~1,000 datasets for testing

Use `scripts/data_generation/entrypoint_gen_with_MLE.py` to generate a lightweight dataset for smoke tests and pipeline validation.

```bash
python scripts/data_generation/entrypoint_gen_with_MLE.py \
  --count 1000 \
  --out-dir ./data/test_1k \
  --states 4 \
  --lifespan 100.0 \
  --min-n 5000 \
  --max-n 5000 \
  --base-seed 20250924 \
  --init-r "-0.5,-1,-1.5"
```

### 2) Generate ~200,000 datasets for training

Use the same entrypoint to generate large-scale training data (parallel execution is recommended).

```bash
python scripts/data_generation/entrypoint_gen_with_MLE.py \
  --count 200000 \
  --out-dir ./data/train_200k \
  --states 4 \
  --lifespan 100.0 \
  --min-n 5000 \
  --max-n 5000 \
  --base-seed 20250924 \
  --init-r "-0.5,-1,-1.5" \
  --run-parallel \
  --workers 8
```

### 3) Run the training entrypoint

During training, the script randomly samples `--n` datasets after screening.

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

> Adjust `--n` based on your hardware and execution time budget.

---

## Run evaluation in Notebook

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
