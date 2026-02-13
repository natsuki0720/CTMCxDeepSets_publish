"""推移率行列を生成するモジュール。"""

from __future__ import annotations

import numpy as np
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

    def generate(self, rng: np.random.Generator) -> tuple[Tensor, np.ndarray]:
        """推移率行列Qと遷移率lambdaを生成する。"""
        n = self._config.num_states

        nus = rng.uniform(1.0, self._config.lifetime_upper, size=n - 1)
        lambdas = 1.0 / nus

        q = np.zeros((n, n), dtype=np.float64)
        idx = np.arange(n - 1)
        q[idx, idx] = -lambdas
        q[idx, idx + 1] = lambdas

        return torch.from_numpy(q).to(dtype=torch.float64), lambdas
