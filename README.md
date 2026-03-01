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

### 1) `entrypoint_gen_with_MLE.py` の仕様確認

`scripts/data_generation/entrypoint_gen_with_MLE.py` は、CTMC データセットを複数生成して `dataset_0000.csv` のような連番 CSV として保存します。主要引数の仕様は以下です。

- `--count` : 生成する CSV ファイル数（1 以上）
- `--out-dir` : 出力先ディレクトリ
- `--states` : 状態数 `N`（3 以上）
- `--lifespan` : 遷移時間上限
- `--min-n`, `--max-n` : 各データセットのサンプル数レンジ（`min-n <= max-n`）
- `--base-seed` : 乱数シード
- `--init-r` : MLE 初期値。**カンマ区切りで `states-1` 個**指定（`states=4` なら 3 要素）
- `--run-parallel`, `--workers` : 並列生成オプション

`--init-r` は値先頭が `-` になるため、`--init-r=-0.5,-1,-1.5` のように `=` 形式で渡すとシェル解釈の揺れを避けやすく安全です。

また、複数行コマンドでは行末バックスラッシュ `\` の後ろに空白を入れないでください（`\` の後ろに空白があると改行継続が壊れます）。

### 2) テスト用に約 1,000 データセットを生成

```bash
python scripts/data_generation/entrypoint_gen_with_MLE.py \
  --count 1000 \
  --out-dir ./data/test_1k \
  --states 4 \
  --lifespan 100.0 \
  --min-n 500 \
  --max-n 5000 \
  --base-seed 20250924 \
  --init-r=-0.5,-1,-1.5
```

### 3) 学習用に約 200,000 データセットを生成（並列）

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

### 4) 学習エントリポイントを実行

学習時は、スクリーニング後のデータから `--n` 件がランダムに抽出されます。

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

> `--n` は計算資源と実行時間に合わせて調整してください。

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

---

## Troubleshooting

- `EnvironmentNameNotFound: Could not find conda environment: ctmc`
  - Environment creation has not completed successfully. Run one of the setup commands (`environment.base.yml`, `environment.linux.gpu.yml`, or `environment.windows.gpu.yml`) first.
- Using a GPU environment file (`environment.linux.gpu.yml` or `environment.windows.gpu.yml`) on a machine without a CUDA-capable NVIDIA setup can fail during dependency resolution or runtime.
- On macOS, use `environment.base.yml` (CPU-only).
