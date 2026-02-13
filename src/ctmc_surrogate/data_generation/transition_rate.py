"""推移率行列を生成するモジュール。"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import TransitionRateConfig


class DiagonalTransitionRateMatrixGenerator:
    """劣化方向のみ遷移する上三角隣接CTMCの推移率行列を生成する。"""

    def __init__(self, config: TransitionRateConfig) -> None:
        if config.num_states < 2:
            raise ValueError("num_states は2以上である必要があります。")
        if config.lifetime_upper <= 1.0:
            raise ValueError("lifetime_upper は1より大きい必要があります。")
        self._config = config

    def generate(self, rng: torch.Generator) -> Tensor:
        """推移率行列Qを生成する。"""
        n = self._config.num_states
        q = torch.zeros((n, n), dtype=torch.float64)

        nus = 1.0 + (self._config.lifetime_upper - 1.0) * torch.rand(n - 1, generator=rng, dtype=torch.float64)
        lambdas = 1.0 / nus

        row_idx = torch.arange(n - 1)
        q[row_idx, row_idx] = -lambdas
        q[row_idx, row_idx + 1] = lambdas

        return q
