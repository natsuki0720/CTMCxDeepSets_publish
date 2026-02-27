#!/usr/bin/env python3
"""Entrypoint to train the CTMC surrogate model from generated CSV datasets.

Usage:
  python train_entrypoint.py --data-dir ./data --n 200 --out-dir ./runs

Key options:
  --data-dir: Directory containing generated CSV datasets used for training
  --n: Number of datasets randomly sampled for training after screening
  --out-dir: Output location for training artifacts. By default, a run subdirectory is created automatically
  --no-auto-run-dir: Save artifacts directly under out-dir (advanced usage)
  --run-name: Explicit run directory name when auto-creation is enabled (timestamp if omitted)
  --recursive: Recursively collect CSV files under data-dir
  --val-ratio: Validation split ratio (0.0 to 1.0)
  --min-lambda: Lower bound of acceptable λ during screening
  --max-lambda: Upper bound of acceptable λ during screening
  --no-structure-check: Skip structural data checks
  --no-naninf-check: Skip NaN/Inf checks
  --epochs: Number of training epochs
  --batch-size: Mini-batch size
  --lr: Learning rate used by the optimizer
  --patience: Patience for early stopping
  --num-workers: Number of DataLoader worker processes
  --device: Device used for training (e.g., cuda, cpu)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ctmc_surrogate.data.collate import ctmc_collate_fn
from ctmc_surrogate.data.dataset import CTMCSurrogateDataset
from ctmc_surrogate.data.dataset_csv_loader import ParsedCTMCDataset, load_dir
from ctmc_surrogate.data.dataset_screening import ScreeningConfig, extract_lambdas_from_Q, screen_datasets
from ctmc_surrogate.models import build_model
from ctmc_surrogate.train import EarlyStoppingConfig, TrainLoopConfig, fit, save_run_artifacts, CustomLoss


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSV loading, screening, split, and training in one flow.")

    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with generated CSV files")
    parser.add_argument("--n", type=int, required=True, help="Number of datasets used after screening")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for training artifacts (parent run directory)")
    parser.add_argument(
        "--no-auto-run-dir",
        action="store_true",
        help="Save artifacts directly under out-dir (default creates a run subdirectory)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name when auto-run-dir is enabled. Uses timestamp if omitted",
    )

    parser.add_argument("--recursive", action="store_true", help="Recursively search data-dir")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and split seed")

    parser.add_argument("--min-lambda", type=float, default=1e-2, help="Screening lower bound")
    parser.add_argument("--max-lambda", type=float, default=1, help="Screening upper bound")
    parser.add_argument("--no-structure-check", action="store_true", help="Disable structure checks")
    parser.add_argument("--no-naninf-check", action="store_true", help="Disable NaN/Inf checks")

    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of DataLoader workers")
    parser.add_argument(
        "--state-index-base",
        type=str,
        choices=["auto", "zero", "one"],
        default="auto",
        help="State index base. auto infers zero-based or one-based from CSV values (errors if ambiguous)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help='Training device (e.g., "cuda", "cpu")',
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n must be >= 1.")
    if not (0.0 <= float(args.val_ratio) <= 1.0):
        raise ValueError("--val-ratio must be in [0.0, 1.0].")
    if args.epochs <= 0:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if bool(args.no_auto_run_dir) and args.run_name is not None:
        raise ValueError("--run-name cannot be used together with --no-auto-run-dir.")


def _resolve_run_dir(out_dir: Path, no_auto_run_dir: bool, run_name: str | None) -> Path:
    """Resolve and return the final output directory.

    By default, this creates a run subdirectory under `out_dir`.
    Artifacts are written directly under `out_dir` only when `--no-auto-run-dir` is specified.

    Args:
        out_dir: Output directory specified via CLI.
        no_auto_run_dir: Flag to enable direct output under out_dir.
        run_name: Run directory name. Uses timestamp when omitted.

    Returns:
        Run directory where artifacts are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if no_auto_run_dir:
        return out_dir

    base_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = out_dir / base_name
    suffix = 1
    while candidate.exists():
        candidate = out_dir / f"{base_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _validate_state_range(ds: ParsedCTMCDataset, n_state: int, state_index_base: str) -> None:
    state_min = int(np.min(ds.samples[:, :2]))
    state_max = int(np.max(ds.samples[:, :2]))
    if state_index_base == "one":
        if state_min < 1 or state_max > n_state:
            raise ValueError(
                "State ID is out of range: "
                f"path={ds.path}, base=one, allowed=[1,{n_state}], actual=[{state_min},{state_max}]"
            )
        return

    if state_index_base == "zero":
        if state_min < 0 or state_max > (n_state - 1):
            raise ValueError(
                "State ID is out of range: "
                f"path={ds.path}, base=zero, allowed=[0,{n_state - 1}], actual=[{state_min},{state_max}]"
            )
        return

    raise ValueError(f"Unsupported state_index_base: {state_index_base}")


def _infer_state_index_base(datasets: list[ParsedCTMCDataset]) -> str:
    is_zero_based_all = True
    is_one_based_all = True
    for ds in datasets:
        n_state = int(ds.q.shape[0])
        state_min = int(np.min(ds.samples[:, :2]))
        state_max = int(np.max(ds.samples[:, :2]))
        if not (state_min >= 0 and state_max <= (n_state - 1)):
            is_zero_based_all = False
        if not (state_min >= 1 and state_max <= n_state):
            is_one_based_all = False

    if is_one_based_all and not is_zero_based_all:
        return "one"
    if is_zero_based_all and not is_one_based_all:
        return "zero"

    raise ValueError(
        "Could not infer state index base automatically."
        " Please explicitly set --state-index-base to zero or one."
    )


def _build_dataset_from_filewise(datasets: list[ParsedCTMCDataset], state_index_base: str) -> CTMCSurrogateDataset:
    state_list: list[torch.Tensor] = []
    delta_t_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []

    expected_n: int | None = None

    for ds in datasets:
        if ds.samples.ndim != 2 or ds.samples.shape[1] != 3:
            raise ValueError(f"Invalid samples shape: path={ds.path}, shape={ds.samples.shape}")

        n_state = int(ds.q.shape[0])
        if ds.q_mle is None:
            raise ValueError(f"q_mle is None for dataset: {ds.path}")
        if ds.q.shape != (n_state, n_state) or ds.q_mle.shape != (n_state, n_state):
            raise ValueError(f"Invalid Q / Q' shape: path={ds.path}, q={ds.q.shape}, q_mle={ds.q_mle.shape}")

        if expected_n is None:
            expected_n = n_state
        elif expected_n != n_state:
            raise ValueError(f"Mixed state counts (N) detected: expected={expected_n}, got={n_state}, path={ds.path}")

        if ds.samples.shape[0] < 1:
            raise ValueError(f"Datasets with zero-length sequences cannot be used for training: path={ds.path}")

        _validate_state_range(ds, n_state=n_state, state_index_base=state_index_base)

        state = torch.as_tensor(ds.samples[:, :2].T, dtype=torch.long)
        delta_t = torch.as_tensor(ds.samples[:, 2], dtype=torch.float32)
        if not np.isfinite(ds.q_mle).all():
            raise ValueError(f"q_mle has NaN/Inf for dataset: {ds.path}")
        target = torch.as_tensor(extract_lambdas_from_Q(ds.q_mle), dtype=torch.float32)

        state_list.append(state)
        delta_t_list.append(delta_t)
        target_list.append(target)

    return CTMCSurrogateDataset(state_list=state_list, delta_t_list=delta_t_list, target_list=target_list)


def _build_screening_report_payload(screening_report: list[dict]) -> dict:
    reason_counter = Counter(item.get("reason", "unknown") for item in screening_report)
    return {
        "dropped": screening_report,
        "reason_counts": dict(reason_counter),
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    run_dir = _resolve_run_dir(args.out_dir, no_auto_run_dir=bool(args.no_auto_run_dir), run_name=args.run_name)

    datasets = load_dir(args.data_dir, recursive=bool(args.recursive))

    cfg = ScreeningConfig(
        min_lambda=float(args.min_lambda),
        max_lambda=float(args.max_lambda),
        check_nan_inf=not bool(args.no_naninf_check),
        require_structure=not bool(args.no_structure_check),
    )
    screening = screen_datasets(datasets, cfg)
    kept = screening.kept

    if len(kept) < int(args.n):
        raise ValueError(f"Not enough datasets passed screening: kept={len(kept)}, requested={args.n}")

    rng = np.random.default_rng(int(args.seed))
    selected_ids = rng.choice(len(kept), size=int(args.n), replace=False)
    selected = [kept[int(i)] for i in selected_ids]

    if float(args.val_ratio) == 0.0:
        val_size = 0
    else:
        val_size = max(1, int(round(int(args.n) * float(args.val_ratio))))

    perm = rng.permutation(int(args.n))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    train_sets = [selected[int(i)] for i in train_idx]
    val_sets = [selected[int(i)] for i in val_idx]

    if len(train_sets) == 0:
        raise ValueError("Training dataset is empty. Revisit --n or --val-ratio.")

    if len(val_sets) == 0:
        val_sets = train_sets

    resolved_state_index_base = (
        _infer_state_index_base(selected) if str(args.state_index_base) == "auto" else str(args.state_index_base)
    )

    train_dataset = _build_dataset_from_filewise(train_sets, state_index_base=resolved_state_index_base)
    val_dataset = _build_dataset_from_filewise(val_sets, state_index_base=resolved_state_index_base)

    pin_memory = str(args.device).startswith("cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=ctmc_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=ctmc_collate_fn,
    )

    first_n = int(train_sets[0].q.shape[0])
    model_config = {
        "num_categories": first_n,
        "embedding_dim": 16,
        "output_dim": first_n - 1,
        "input_is_one_based": (resolved_state_index_base == "one"),
    }
    model = build_model(model_config)

    loop_cfg = TrainLoopConfig(
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        device=str(args.device),
        early_stopping=EarlyStoppingConfig(patience=int(args.patience)),
    )

    train_result = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        config=loop_cfg,
        loss_fn=CustomLoss(),
    )

    save_run_artifacts(
        run_dir=run_dir,
        model=model,
        model_config=model_config,
        metrics=train_result,
    )

    run_config_payload = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "resolved_run_dir": str(run_dir),
        "resolved_state_index_base": resolved_state_index_base,
        "screening_config": asdict(cfg),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    screening_payload = {
        "total_loaded": len(datasets),
        "kept_count": len(screening.kept),
        "dropped_count": len(screening.dropped),
        "selected_count": len(selected),
        "selected_paths": [d.path for d in selected],
        **_build_screening_report_payload(screening.report),
    }
    (run_dir / "screening_report.json").write_text(
        json.dumps(screening_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
