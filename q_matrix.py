"""直列鎖 pure birth CTMC の Q 行列生成。"""

from __future__ import annotations

import numpy as np


def generate_q(
    rng: np.random.Generator, n_state: int, lifespan_upper: float
) -> tuple[np.ndarray, np.ndarray]:
    """仕様に従い Q と遷移率 lambda を生成する。"""
    if n_state < 2:
        raise ValueError("n_state は 2 以上である必要があります。")
    if lifespan_upper <= 1.0:
        raise ValueError("lifespan_upper は 1 より大きい必要があります。")

    nus = rng.uniform(1.0, lifespan_upper, size=n_state - 1)
    lambdas = 1.0 / nus

    q = np.zeros((n_state, n_state), dtype=float)
    idx = np.arange(n_state - 1)
    q[idx, idx] = -lambdas
    q[idx, idx + 1] = lambdas
    # 最終行は吸収状態のためゼロ行のまま。
    return q, lambdas
