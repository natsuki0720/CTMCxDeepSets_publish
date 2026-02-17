#!/usr/bin/env python3
"""合成CSVからCTMC surrogate学習を実行するエントリーポイント。

使い方:
  python train_entrypoint.py --data-dir ./data --n 200 --out-dir ./runs

主要オプション:
  --data-dir: 学習対象の合成CSV群が格納されたディレクトリ
  --n: スクリーニング通過後にランダム抽出して学習に使う件数
  --out-dir: 学習成果物の保存先。既定では run サブディレクトリを自動作成
  --no-auto-run-dir: out-dir 直下に直接保存（上級者向け）
  --run-name: 自動作成時の run ディレクトリ名を明示指定（未指定時はタイムスタンプ）
  --recursive: data-dir配下を再帰的に探索してCSVを収集
  --val-ratio: 検証データに割り当てる割合（0.0〜1.0）
  --min-lambda: スクリーニング時に許容するλの下限値
  --max-lambda: スクリーニング時に許容するλの上限値
  --no-structure-check: データ構造に関する検証をスキップ
  --no-naninf-check: NaN/Infに関する検証をスキップ
  --epochs: 学習エポック数
  --batch-size: ミニバッチサイズ
  --lr: 最適化で使用する学習率
  --patience: Early Stoppingの打ち切り待機エポック数
  --num-workers: DataLoaderで使う並列ワーカ数
  --device: 学習に使うデバイス（例: cuda, cpu）
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
    parser = argparse.ArgumentParser(description="合成CSVの読み込み〜スクリーニング〜分割〜学習を実行します。")

    parser.add_argument("--data-dir", type=Path, required=True, help="合成CSVのディレクトリ")
    parser.add_argument("--n", type=int, required=True, help="スクリーニング通過後に使用するデータセット数")
    parser.add_argument("--out-dir", type=Path, required=True, help="学習成果物の保存先（run親ディレクトリ）")
    parser.add_argument(
        "--no-auto-run-dir",
        action="store_true",
        help="out-dir直下に直接成果物を保存する（既定ではrunサブディレクトリを自動作成）",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="auto-run-dir有効時のrunディレクトリ名。未指定時はタイムスタンプ名を利用",
    )

    parser.add_argument("--recursive", action="store_true", help="data-dir配下を再帰探索する")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="validation比率")
    parser.add_argument("--seed", type=int, default=42, help="抽出・分割シード")

    parser.add_argument("--min-lambda", type=float, default=1e-8, help="スクリーニング下限")
    parser.add_argument("--max-lambda", type=float, default=1e4, help="スクリーニング上限")
    parser.add_argument("--no-structure-check", action="store_true", help="構造チェックを無効化")
    parser.add_argument("--no-naninf-check", action="store_true", help="NaN/Infチェックを無効化")

    parser.add_argument("--epochs", type=int, default=1000, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=128, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--patience", type=int, default=10, help="Early Stoppingのpatience")
    parser.add_argument("--num-workers", type=int, default=32, help="DataLoaderのワーカ数")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help='学習デバイス（例: "cuda", "cpu"）',
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n は1以上で指定してください。")
    if not (0.0 <= float(args.val_ratio) <= 1.0):
        raise ValueError("--val-ratio は 0.0〜1.0 の範囲で指定してください。")
    if args.epochs <= 0:
        raise ValueError("--epochs は1以上で指定してください。")
    if args.batch_size <= 0:
        raise ValueError("--batch-size は1以上で指定してください。")
    if args.lr <= 0:
        raise ValueError("--lr は正の値で指定してください。")
    if args.patience <= 0:
        raise ValueError("--patience は1以上で指定してください。")
    if args.num_workers < 0:
        raise ValueError("--num-workers は0以上で指定してください。")
    if bool(args.no_auto_run_dir) and args.run_name is not None:
        raise ValueError("--no-auto-run-dir 指定時は --run-name を併用できません。")


def _resolve_run_dir(out_dir: Path, no_auto_run_dir: bool, run_name: str | None) -> Path:
    """出力先ディレクトリを解決して返す。

    既定では `out_dir` 配下に run サブディレクトリを自動作成する。
    `--no-auto-run-dir` 指定時のみ `out_dir` 直下へ成果物を保存する。

    Args:
        out_dir: CLIで指定された出力先。
        no_auto_run_dir: 直下保存を有効化するフラグ。
        run_name: run ディレクトリ名。未指定時はタイムスタンプを使用。

    Returns:
        実際に成果物を書き込む run ディレクトリ。
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


def _build_dataset_from_filewise(datasets: list[ParsedCTMCDataset]) -> CTMCSurrogateDataset:
    state_list: list[torch.Tensor] = []
    delta_t_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []

    expected_n: int | None = None

    for ds in datasets:
        if ds.samples.ndim != 2 or ds.samples.shape[1] != 3:
            raise ValueError(f"samples 形状が不正です: path={ds.path}, shape={ds.samples.shape}")

        n_state = int(ds.q.shape[0])
        if ds.q_mle is None:
            raise ValueError(f"q_mle is None for dataset: {ds.path}")
        if ds.q.shape != (n_state, n_state) or ds.q_mle.shape != (n_state, n_state):
            raise ValueError(f"Q / Q' 形状が不正です: path={ds.path}, q={ds.q.shape}, q_mle={ds.q_mle.shape}")

        if expected_n is None:
            expected_n = n_state
        elif expected_n != n_state:
            raise ValueError(f"状態数Nが混在しています: expected={expected_n}, got={n_state}, path={ds.path}")

        if ds.samples.shape[0] < 1:
            raise ValueError(f"系列長0のデータセットは学習に使えません: path={ds.path}")

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
        raise ValueError(f"スクリーニング通過件数が不足しています: kept={len(kept)}, requested={args.n}")

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
        raise ValueError("trainデータセットが0件です。--n か --val-ratio を見直してください。")

    if len(val_sets) == 0:
        val_sets = train_sets

    train_dataset = _build_dataset_from_filewise(train_sets)
    val_dataset = _build_dataset_from_filewise(val_sets)

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
        "input_is_one_based": False,
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
