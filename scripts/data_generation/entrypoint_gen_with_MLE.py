#!/usr/bin/env python3
"""CLI to generate CTMC datasets and save each dataset as a CSV file.

Usage:
    python scripts/data_generation/entrypoint_gen_with_MLE.py \
        --count 10 \
        --out-dir ./artifacts/generated_csv

Key options:
    # Specify state count, lifetime upper bound, and sample count range
    python scripts/data_generation/entrypoint_gen_with_MLE.py \
        --count 5 \
        --out-dir ./artifacts/generated_csv \
        --states 4 \
        --lifespan 100.0 \
        --min-n 4000 \
        --max-n 6000 \
        --base-seed 20250924 \
        --init-r "-0.5,-1,-1.5"

    # Run in parallel (enabled only when --run-parallel is set)
    python scripts/data_generation/entrypoint_gen_with_MLE.py \
        --count 20 \
        --out-dir ./artifacts/generated_csv \
        --run-parallel \
        --workers 8

Output format:
    - Writes files such as `dataset_0000.csv`, `dataset_0001.csv`, ...
    - Each CSV is a vertical concatenation in the following order:
      1) True Q matrix (N rows x N cols)
      2) MLE-estimated Q' matrix (N rows x N cols)
      3) Sample rows (N cols per row)
         `[state_pre, state_post, delta_t] + zero padding as needed`
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ctmc_surrogate.data_generation.config import DatasetGenerationConfig, TransitionRateConfig
from ctmc_surrogate.data_generation.generator import DataGenerator, GeneratedDataset


_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple CTMC datasets and export them as CSV files.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--lifespan", type=float, default=100.0)
    parser.add_argument("--min-n", type=int, default=5000)
    parser.add_argument("--max-n", type=int, default=5000)
    parser.add_argument("--base-seed", type=int, default=20250924)
    parser.add_argument("--init-r", type=str, default="-0.5,-1,-1.5")
    parser.add_argument("--run-parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def _parse_init_r(raw: str) -> list[float]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("--init-r must not be empty.")
    return [float(v) for v in values]


def _child_seed(base_seed: int, dataset_id: int) -> int:
    ss = np.random.SeedSequence([base_seed, dataset_id])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def _generate_one(
    dataset_id: int,
    base_seed: int,
    states: int,
    lifespan: float,
    min_n: int,
    max_n: int,
    init_r: list[float],
) -> tuple[int, GeneratedDataset]:
    seed = _child_seed(base_seed, dataset_id)
    rng = np.random.default_rng(seed)
    n_samples = int(rng.integers(min_n, max_n + 1))

    config = DatasetGenerationConfig(
        num_samples=n_samples,
        transition_rate=TransitionRateConfig(num_states=states, lifetime_upper=lifespan),
        enable_mle=True,
        mle_init_r=init_r,
    )
    generator = DataGenerator(config)
    generated = generator.generate_matrix(rng)
    return dataset_id, generated


def _write_dataset_csv(path: Path, dataset: GeneratedDataset) -> None:
    q = dataset.q_matrix
    q_mle = dataset.q_mle
    if q_mle is None:
        raise ValueError("Q' (q_mle) is None.")

    n = len(q)
    if n < 3:
        raise ValueError("State count N must be at least 3.")

    for row in q:
        if len(row) != n:
            raise ValueError("The number of Q columns must match state count N.")
    for row in q_mle:
        if len(row) != n:
            raise ValueError("The number of Q' columns must match state count N.")

    pad_zeros = n - 3

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in q:
            writer.writerow(row)
        for row in q_mle:
            writer.writerow(row)
        for sample in dataset.samples:
            row: list[float | int] = [sample.start_state, sample.next_state, float(sample.delta_t)]
            if pad_zeros > 0:
                row.extend([0] * pad_zeros)
            writer.writerow(row)


def _validate_args(args: argparse.Namespace) -> list[float]:
    if args.count <= 0:
        raise ValueError("--count must be >= 1.")
    if args.states <= 0:
        raise ValueError("--states must be >= 1.")
    if args.states < 3:
        raise ValueError("--states must be >= 3.")
    if args.min_n <= 0 or args.max_n <= 0:
        raise ValueError("--min-n and --max-n must be >= 1.")
    if args.min_n > args.max_n:
        raise ValueError("--min-n must be <= --max-n.")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be >= 1.")

    init_r = _parse_init_r(args.init_r)
    expected_len = args.states - 1
    if len(init_r) != expected_len:
        raise ValueError(f"--init-r must contain states-1={expected_len} values.")
    return init_r


def _set_single_thread_env() -> None:
    for key in _THREAD_ENV_VARS:
        os.environ[key] = "1"


def main() -> None:
    args = _parse_args()
    init_r = _validate_args(args)

    _set_single_thread_env()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_parallel:
        max_workers = args.workers if args.workers is not None else os.cpu_count()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _generate_one,
                    dataset_id,
                    args.base_seed,
                    args.states,
                    args.lifespan,
                    args.min_n,
                    args.max_n,
                    init_r,
                )
                for dataset_id in range(args.count)
            ]
            for future in futures:
                dataset_id, dataset = future.result()
                out_path = args.out_dir / f"dataset_{dataset_id:04d}.csv"
                _write_dataset_csv(out_path, dataset)
    else:
        for dataset_id in range(args.count):
            _, dataset = _generate_one(
                dataset_id,
                args.base_seed,
                args.states,
                args.lifespan,
                args.min_n,
                args.max_n,
                init_r,
            )
            out_path = args.out_dir / f"dataset_{dataset_id:04d}.csv"
            _write_dataset_csv(out_path, dataset)


if __name__ == "__main__":
    main()
