"""discrete DelT生成モジュール。"""

from __future__ import annotations

import torch

from .config import DeltaTSamplingConfig


class DirichletDeltaT:
    """Dirichlet混合により離散候補点からDelTをサンプリングする。"""

    def __init__(self, config: DeltaTSamplingConfig, rng: torch.Generator) -> None:
        if config.k_min < 1 or config.k_max < config.k_min:
            raise ValueError("k_min, k_max の設定が不正です。")
        if config.candidate_high <= config.candidate_low:
            raise ValueError("candidate_high は candidate_low より大きくする必要があります。")

        self._config = config
        self._rng = rng

        self._num_candidates = int(torch.randint(config.k_min, config.k_max + 1, (1,), generator=rng).item())
        span = config.candidate_high - config.candidate_low
        self._candidates = config.candidate_low + span * torch.rand(self._num_candidates, generator=rng, dtype=torch.float64)

        uniform = torch.rand(self._num_candidates, generator=rng, dtype=torch.float64)
        raw = -torch.log(uniform.clamp_min(1e-12))
        self._weights = raw / raw.sum()

    @property
    def candidates(self) -> torch.Tensor:
        """候補時刻を返す。"""
        return self._candidates.clone()

    @property
    def weights(self) -> torch.Tensor:
        """候補点の出現確率を返す。"""
        return self._weights.clone()

    def sample(self) -> float:
        """1つのDelTをサンプリングする。"""
        idx = int(torch.multinomial(self._weights, 1, replacement=True, generator=self._rng).item())
        value = self._candidates[idx]
        scale = 10**self._config.round_digits
        return float(torch.round(value * scale).item() / scale)
