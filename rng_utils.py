"""乱数生成ユーティリティ。"""

from __future__ import annotations

import numpy as np


def make_rng(base_seed: int, dataset_id: int) -> np.random.Generator:
    """仕様どおりの子シードから NumPy Generator を作成する。"""
    child_seed = (base_seed * 1_000_003 + dataset_id) % (2**63 - 1)
    return np.random.default_rng(child_seed)


def dirichlet_ones(rng: np.random.Generator, k: int) -> np.ndarray:
    """-log(U) 正規化で Dirichlet(1,...,1) を生成する。"""
    if k <= 0:
        raise ValueError("k は正の整数である必要があります。")
    u = rng.uniform(0.0, 1.0, size=k)
    # log(0) 回避のため機械イプシロンで下限を設ける。
    u = np.clip(u, np.finfo(float).tiny, 1.0)
    r = -np.log(u)
    s = r.sum()
    if s <= 0.0:
        return np.full(k, 1.0 / k, dtype=float)
    return r / s
