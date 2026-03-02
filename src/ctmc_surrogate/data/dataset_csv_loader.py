from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ParsedCTMCDataset:
    """Parsing result for one CTMC synthetic-data file."""

    path: str
    q: np.ndarray
    q_mle: np.ndarray
    samples: np.ndarray


def _to_float(value: str, path: Path, row_idx: int, col_idx: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Failed to convert CSV value to float: path={path}, row={row_idx}, col={col_idx}, value={value!r}"
        ) from exc


def _to_int_like(value: str, path: Path, row_idx: int, col_idx: int) -> int:
    x = _to_float(value, path, row_idx, col_idx)
    if not np.isfinite(x):
        raise ValueError(
            f"Value is not a finite integer: path={path}, row={row_idx}, col={col_idx}, value={value!r}"
        )
    xi = int(round(x))
    if not np.isclose(x, xi):
        raise ValueError(
            f"Value cannot be interpreted as an integer: path={path}, row={row_idx}, col={col_idx}, value={value!r}"
        )
    return xi


def parse_ctmc_csv(path: str | Path) -> ParsedCTMCDataset:
    """Load one CTMC CSV and split it into Q / Q' / samples."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Specified path is not a file: {csv_path}")

    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    n = len(rows[0])
    if n <= 0:
        raise ValueError(f"Cannot infer column count N from the first row: {csv_path}")

    for i, row in enumerate(rows):
        if len(row) != n:
            raise ValueError(
                f"Column count mismatch: path={csv_path}, row={i}, expected_n={n}, actual={len(row)}"
            )

    if len(rows) < 2 * n:
        raise ValueError(
            f"Insufficient number of rows (rows < 2N): path={csv_path}, rows={len(rows)}, N={n}"
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


def parse_ctmc_csv_header(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight loading of only Q / Q' from one CTMC CSV."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Specified path is not a file: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            first_row = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {csv_path}") from exc

        if not first_row:
            raise ValueError(f"Cannot infer column count N from the first row: {csv_path}")

        n = len(first_row)
        if n <= 0:
            raise ValueError(f"Cannot infer column count N from the first row: {csv_path}")

        q = np.empty((n, n), dtype=np.float64)
        q_mle = np.empty((n, n), dtype=np.float64)

        if len(first_row) != n:
            raise ValueError(
                f"Column count mismatch: path={csv_path}, row=0, expected_n={n}, actual={len(first_row)}"
            )
        for c in range(n):
            q[0, c] = _to_float(first_row[c], csv_path, 0, c)

        for r in range(1, 2 * n):
            try:
                row = next(reader)
            except StopIteration as exc:
                raise ValueError(
                    f"Insufficient number of rows (rows < 2N): path={csv_path}, rows={r}, N={n}"
                ) from exc

            if not row:
                raise ValueError(
                    f"Column count mismatch: path={csv_path}, row={r}, expected_n={n}, actual=0"
                )
            if len(row) != n:
                raise ValueError(
                    f"Column count mismatch: path={csv_path}, row={r}, expected_n={n}, actual={len(row)}"
                )

            if r < n:
                for c in range(n):
                    q[r, c] = _to_float(row[c], csv_path, r, c)
            else:
                rr = r - n
                for c in range(n):
                    q_mle[rr, c] = _to_float(row[c], csv_path, r, c)

    return q, q_mle


def load_dir(dir_path: str | Path, recursive: bool = False) -> list[ParsedCTMCDataset]:
    """Search for CSVs under a directory and parse all files."""

    base = Path(dir_path)
    if not base.exists():
        raise FileNotFoundError(f"Directory does not exist: {base}")
    if not base.is_dir():
        raise ValueError(f"Specified path is not a directory: {base}")

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
    """Retrieve Q / Q' / samples on a per-file basis."""

    q_list = [d.q for d in datasets]
    q_mle_list = [d.q_mle for d in datasets]
    samples_list = [d.samples for d in datasets]
    return q_list, q_mle_list, samples_list


def as_samplewise(
    datasets: list[ParsedCTMCDataset],
    keep_source_index: bool = True,
) -> dict[str, Any]:
    """Vertically concatenate samples from all files and return training-ready tensors."""

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
                f"Invalid samples shape: path={d.path}, expected=(M,3), actual={d.samples.shape}"
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

    parser = argparse.ArgumentParser(description="Quick check for CTMC CSV loader")
    parser.add_argument("dir_path", type=str, help="Directory containing CSV files")
    parser.add_argument("--recursive", action="store_true", help="Enable recursive search")
    args = parser.parse_args()

    datasets = load_dir(args.dir_path, recursive=args.recursive)
    merged = as_samplewise(datasets, keep_source_index=True)

    print(f"files={len(datasets)}")
    if datasets:
        first_n = datasets[0].q.shape[0]
        print(f"first_N={first_n}")
    print(f"total_samples={merged['samples'].shape[0]}")
