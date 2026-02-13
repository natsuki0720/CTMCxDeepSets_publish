"""discrete DelT生成モジュール。"""

from __future__ import annotations

import numpy as np

from .config import DeltaTSamplingConfig


class DirichletDeltaT:
    """Dirichlet混合により離散候補点からDelTをサンプリングする。"""

    def __init__(self, config: DeltaTSamplingConfig, rng: np.random.Generator) -> None:
        if config.k_min < 1 or config.k_max < config.k_min:
            raise ValueError("k_min, k_max の設定が不正です。")
        if config.candidate_high <= config.candidate_low:
            raise ValueError("candidate_high は candidate_low より大きくする必要があります。")

        self._config = config
        self._rng = rng

        self._num_candidates = int(rng.integers(config.k_min, config.k_max + 1))
        self._candidates = rng.uniform(config.candidate_low, config.candidate_high, size=self._num_candidates)

        uniform = rng.random(self._num_candidates)
        raw = -np.log(uniform)
        self._weights = raw / raw.sum()

    @property
    def candidates(self) -> np.ndarray:
        """候補時刻を返す。"""
        return self._candidates.copy()

    @property
    def weights(self) -> np.ndarray:
        """候補点の出現確率を返す。"""
        return self._weights.copy()

    def sample(self) -> float:
        """1つのDelTをサンプリングする。"""
        idx = int(self._rng.choice(self._num_candidates, p=self._weights))
        value = self._candidates[idx]
        return float(np.round(value, self._config.round_digits))
