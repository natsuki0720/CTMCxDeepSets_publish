from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset_csv_loader import ParsedCTMCDataset


@dataclass
class ScreeningConfig:
    """Q' の異常値スクリーニング設定。"""

    min_lambda: float = 1e-8
    max_lambda: float = 1e6
    check_nan_inf: bool = True
    require_structure: bool = True
    max_abs_diag_diff: float = 1e-8


@dataclass
class ScreeningResult:
    """スクリーニング結果（保持・除外データセットとレポート）。"""

    kept: list[ParsedCTMCDataset] = field(default_factory=list)
    dropped: list[ParsedCTMCDataset] = field(default_factory=list)
    report: list[dict[str, Any]] = field(default_factory=list)


def extract_lambdas_from_Q(Q: np.ndarray) -> np.ndarray:
    """pure birth 直列型の Q から λ_i=Q[i, i+1] を抽出する。"""

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q は正方行列である必要があります: shape={Q.shape}")

    n = Q.shape[0]
    if n <= 1:
        return np.empty((0,), dtype=np.float64)
    return Q[np.arange(n - 1), np.arange(1, n)]


def validate_Q_structure(Q: np.ndarray, tol: float) -> None | str:
    """pure birth 構造の整合性を検査し、問題理由を返す。"""

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        return f"Q が正方行列ではありません: shape={Q.shape}"

    n = Q.shape[0]
    if n == 0:
        return "Q が空行列です"

    # 最終行は吸収状態のためほぼ 0 を期待する。
    last_row_abs_max = float(np.max(np.abs(Q[n - 1, :])))
    if last_row_abs_max >= tol:
        return f"最終行が 0 に近くありません: max_abs={last_row_abs_max:.3e}, tol={tol:.3e}"

    for i in range(n - 1):
        lam = float(Q[i, i + 1])
        diag = float(Q[i, i])
        diff = abs(diag + lam)
        if diff >= tol:
            return (
                f"対角と上隣接要素が不整合です: i={i}, Qii={diag:.6e}, "
                f"Q(i,i+1)={lam:.6e}, |Qii+Q(i,i+1)|={diff:.3e}, tol={tol:.3e}"
            )
        if lam < 0:
            return f"上隣接要素が負です: i={i}, lambda={lam:.6e}"

    for i in range(n):
        for j in range(n):
            if j == i or j == i + 1:
                continue
            v = float(Q[i, j])
            if abs(v) >= tol:
                return (
                    f"pure birth 以外の要素が 0 に近くありません: "
                    f"index=({i},{j}), value={v:.6e}, tol={tol:.3e}"
                )

    return None


def has_nan_inf(mat: np.ndarray) -> bool:
    """行列に NaN/Inf が含まれるかを返す。"""

    return not np.isfinite(mat).all()


def screen_datasets(
    datasets: list[ParsedCTMCDataset],
    cfg: ScreeningConfig,
) -> ScreeningResult:
    """Q' の異常値を基準にデータセットを保持/除外する。"""

    result = ScreeningResult()

    for dataset in datasets:
        q_mle = dataset.q_mle
        report_base: dict[str, Any] = {"path": dataset.path}

        if cfg.check_nan_inf and has_nan_inf(q_mle):
            result.dropped.append(dataset)
            result.report.append({**report_base, "reason": "nan_or_inf"})
            continue

        if cfg.require_structure:
            structure_err = validate_Q_structure(q_mle, tol=cfg.max_abs_diag_diff)
            if structure_err is not None:
                result.dropped.append(dataset)
                result.report.append(
                    {
                        **report_base,
                        "reason": "invalid_structure",
                        "detail": structure_err,
                    }
                )
                continue

        lambdas = extract_lambdas_from_Q(q_mle)

        negative_idx = np.where(lambdas < 0)[0]
        if negative_idx.size > 0:
            idx = int(negative_idx[0])
            result.dropped.append(dataset)
            result.report.append(
                {
                    **report_base,
                    "reason": "lambda_negative",
                    "index": idx,
                    "lambda": float(lambdas[idx]),
                }
            )
            continue

        too_small_idx = np.where(lambdas < cfg.min_lambda)[0]
        if too_small_idx.size > 0:
            idx = int(too_small_idx[0])
            result.dropped.append(dataset)
            result.report.append(
                {
                    **report_base,
                    "reason": "lambda_too_small",
                    "index": idx,
                    "lambda": float(lambdas[idx]),
                    "min_lambda": cfg.min_lambda,
                    "max_lambda": cfg.max_lambda,
                }
            )
            continue

        too_large_idx = np.where(lambdas > cfg.max_lambda)[0]
        if too_large_idx.size > 0:
            idx = int(too_large_idx[0])
            result.dropped.append(dataset)
            result.report.append(
                {
                    **report_base,
                    "reason": "lambda_too_large",
                    "index": idx,
                    "lambda": float(lambdas[idx]),
                    "min_lambda": cfg.min_lambda,
                    "max_lambda": cfg.max_lambda,
                }
            )
            continue

        result.kept.append(dataset)

    return result
