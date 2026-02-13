"""Δt の候補点＋重み混合生成。"""

from __future__ import annotations

import numpy as np

from rng_utils import dirichlet_ones


def make_dt_mixture(
    rng: np.random.Generator,
    k_min: int,
    k_max: int,
    cand_low: float,
    cand_high: float,
    round_digits: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """データセット固定の Δt 混合分布を作成する。"""
    if k_min <= 0 or k_max < k_min:
        raise ValueError("k_min, k_max の範囲が不正です。")
    if cand_high <= cand_low:
        raise ValueError("cand_low < cand_high が必要です。")

    k = int(rng.integers(k_min, k_max + 1))
    candidates = rng.uniform(cand_low, cand_high, size=k)
    weights = dirichlet_ones(rng, k)
    return candidates, weights, round_digits


def sample_dt(
    rng: np.random.Generator,
    candidates: np.ndarray,
    weights: np.ndarray,
    round_digits: int,
) -> float:
    """混合分布から Δt を1つサンプルし、指定桁で丸める。"""
    k = len(candidates)
    j = int(rng.choice(k, p=weights))
    return float(np.round(candidates[j], round_digits))
