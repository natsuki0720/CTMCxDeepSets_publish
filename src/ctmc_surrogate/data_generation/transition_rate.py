"""Module for generating transition-rate matrices."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from .config import TransitionRateConfig


class DiagonalTransitionRateMatrixGenerator:
    """Generate transition-rate matrices for upper-triangular adjacent CTMCs with degradation-only transitions."""

    def __init__(self, config: TransitionRateConfig) -> None:
        if config.num_states < 2:
            raise ValueError("num_states must be at least 2.")
        if config.lifetime_upper <= 1.0:
            raise ValueError("lifetime_upper must be greater than 1.")
        self._config = config

    def generate(self, rng: Generator) -> np.ndarray:
        """Generate the transition-rate matrix Q."""
        n = self._config.num_states
        q = np.zeros((n, n), dtype=np.float64)

        nus = rng.uniform(1.0, self._config.lifetime_upper, size=n - 1)
        lambdas = 1.0 / nus

        row_idx = np.arange(n - 1)
        q[row_idx, row_idx] = -lambdas
        q[row_idx, row_idx + 1] = lambdas

        return q
