from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ParsedCTMCDataset:
    """CTMC 合成データ 1 ファイル分のパース結果。"""

    path: str
    q: np.ndarray
    q_mle: np.ndarray
    samples: np.ndarray


def _to_float(value: str, path: Path, row_idx: int, col_idx: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"CSV値の float 変換に失敗しました: path={path}, row={row_idx}, col={col_idx}, value={value!r}"
        ) from exc


def _to_int_like(value: str, path: Path, row_idx: int, col_idx: int) -> int:
    x = _to_float(value, path, row_idx, col_idx)
    if not np.isfinite(x):
        raise ValueError(
            f"有限な整数値ではありません: path={path}, row={row_idx}, col={col_idx}, value={value!r}"
        )
    xi = int(round(x))
    if not np.isclose(x, xi):
        raise ValueError(
            f"整数として解釈できません: path={path}, row={row_idx}, col={col_idx}, value={value!r}"
        )
    return xi


def parse_ctmc_csv(path: str | Path) -> ParsedCTMCDataset:
    """1 つの CTMC CSV を読み込み、Q / Q' / samples に分解する。"""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVファイルが存在しません: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"指定パスはファイルではありません: {csv_path}")

    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"CSVが空です: {csv_path}")

    n = len(rows[0])
    if n <= 0:
        raise ValueError(f"1行目から列数 N を推定できません: {csv_path}")

    for i, row in enumerate(rows):
        if len(row) != n:
            raise ValueError(
                f"列数が不一致です: path={csv_path}, row={i}, expected_n={n}, actual={len(row)}"
            )

    if len(rows) < 2 * n:
        raise ValueError(
            f"行数不足です (rows < 2N): path={csv_path}, rows={len(rows)}, N={n}"
        )

    q_rows = rows[:n]
    q_mle_rows = rows[n : 2 * n]
    sample_rows = rows[2 * n :]

    q = np.empty((n, n), dtype=np.float64)
    q_mle = np.empty((n, n), dtype=np.float64)

    for r in range(n):
        for c in range(n):
            q[r, c] = _to_float(q_rows[r][c], csv_path, r, c)
            q_mle[r, c] = _to_float(q_mle_rows[r][c], csv_path, n + r, c)

    if sample_rows:
        pre = np.empty(len(sample_rows), dtype=np.int64)
        post = np.empty(len(sample_rows), dtype=np.int64)
        dt = np.empty(len(sample_rows), dtype=np.float64)
        for i, row in enumerate(sample_rows):
            row_idx = 2 * n + i
            pre[i] = _to_int_like(row[0], csv_path, row_idx, 0)
            post[i] = _to_int_like(row[1], csv_path, row_idx, 1)
            dt[i] = _to_float(row[2], csv_path, row_idx, 2)
        samples = np.column_stack((pre, post, dt)).astype(np.float64, copy=False)
    else:
        samples = np.empty((0, 3), dtype=np.float64)

    return ParsedCTMCDataset(
        path=str(csv_path),
        q=q,
        q_mle=q_mle,
        samples=samples,
    )


def load_dir(dir_path: str | Path, recursive: bool = False) -> list[ParsedCTMCDataset]:
    """ディレクトリ配下の CSV を探索し、全件をパースする。"""

    base = Path(dir_path)
    if not base.exists():
        raise FileNotFoundError(f"ディレクトリが存在しません: {base}")
    if not base.is_dir():
        raise ValueError(f"指定パスはディレクトリではありません: {base}")

    iterator = base.rglob("*.csv") if recursive else base.glob("*.csv")
    csv_paths = sorted(
        p
        for p in iterator
        if p.is_file() and not p.name.startswith(".") and p.stat().st_size > 0
    )

    return [parse_ctmc_csv(p) for p in csv_paths]


def as_filewise(
    datasets: list[ParsedCTMCDataset],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """ファイル単位で Q / Q' / samples を取り出す。"""

    q_list = [d.q for d in datasets]
    q_mle_list = [d.q_mle for d in datasets]
    samples_list = [d.samples for d in datasets]
    return q_list, q_mle_list, samples_list


def as_samplewise(
    datasets: list[ParsedCTMCDataset],
    keep_source_index: bool = True,
) -> dict[str, Any]:
    """全ファイルの samples を縦結合し、学習向けの形で返す。"""

    if not datasets:
        result: dict[str, Any] = {"samples": np.empty((0, 3), dtype=np.float64)}
        if keep_source_index:
            result["source_file_index"] = np.empty((0,), dtype=np.int64)
            result["source_row_index"] = np.empty((0,), dtype=np.int64)
        return result

    samples_chunks: list[np.ndarray] = []
    source_file_chunks: list[np.ndarray] = []
    source_row_chunks: list[np.ndarray] = []

    for file_idx, d in enumerate(datasets):
        m = d.samples.shape[0]
        if d.samples.ndim != 2 or d.samples.shape[1] != 3:
            raise ValueError(
                f"samples の形状が不正です: path={d.path}, expected=(M,3), actual={d.samples.shape}"
            )
        samples_chunks.append(d.samples)
        if keep_source_index:
            source_file_chunks.append(np.full((m,), file_idx, dtype=np.int64))
            source_row_chunks.append(np.arange(m, dtype=np.int64))

    result = {"samples": np.vstack(samples_chunks) if samples_chunks else np.empty((0, 3), dtype=np.float64)}

    if keep_source_index:
        result["source_file_index"] = (
            np.concatenate(source_file_chunks) if source_file_chunks else np.empty((0,), dtype=np.int64)
        )
        result["source_row_index"] = (
            np.concatenate(source_row_chunks) if source_row_chunks else np.empty((0,), dtype=np.int64)
        )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CTMC CSV ローダ簡易確認")
    parser.add_argument("dir_path", type=str, help="CSV を含むディレクトリ")
    parser.add_argument("--recursive", action="store_true", help="再帰探索を有効化")
    args = parser.parse_args()

    datasets = load_dir(args.dir_path, recursive=args.recursive)
    merged = as_samplewise(datasets, keep_source_index=True)

    print(f"files={len(datasets)}")
    if datasets:
        first_n = datasets[0].q.shape[0]
        print(f"first_N={first_n}")
    print(f"total_samples={merged['samples'].shape[0]}")
