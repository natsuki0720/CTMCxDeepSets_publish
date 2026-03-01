"""Module for discrete DelT generation."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from .config import DeltaTSamplingConfig


class DirichletDeltaT:
    """Sample DelT from discrete candidate points using a Dirichlet mixture."""

    def __init__(self, config: DeltaTSamplingConfig, rng: Generator) -> None:
        if config.k_min < 1 or config.k_max < config.k_min:
            raise ValueError("Invalid k_min or k_max setting.")
        if config.candidate_high <= config.candidate_low:
            raise ValueError("candidate_high must be greater than candidate_low.")

        self._config = config
        self._rng = rng

        self._num_candidates = int(rng.integers(config.k_min, config.k_max + 1))
        self._candidates = rng.uniform(config.candidate_low, config.candidate_high, size=self._num_candidates)
        self._weights = _dirichlet_ones(self._num_candidates, self._rng)

    @property
    def candidates(self) -> np.ndarray:
        """Return candidate times."""
        return self._candidates.copy()

    @property
    def weights(self) -> np.ndarray:
        """Return occurrence probabilities of candidate points."""
        return self._weights.copy()

    def sample(self) -> float:
        """Sample a single DelT value."""
        idx = int(self._rng.choice(self._num_candidates, p=self._weights))
        return round(float(self._candidates[idx]), self._config.round_digits)


def _dirichlet_ones(size: int, rng: Generator) -> np.ndarray:
    u = rng.random(size)
    r = -np.log(u)
    return r / r.sum()
