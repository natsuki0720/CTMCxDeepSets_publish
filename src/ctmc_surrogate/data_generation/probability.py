"""CTMC transition probability matrix computation module."""

from __future__ import annotations

import numpy as np


class CalcProbmatrix:
    """Compute P(Δt) using the analytical formula for a pure birth process."""

    def __init__(self, eps: float = 1e-12) -> None:
        self._eps = eps

    def __call__(self, q_matrix: np.ndarray, delta_t: float) -> np.ndarray:
        if q_matrix.ndim != 2 or q_matrix.shape[0] != q_matrix.shape[1]:
            raise ValueError("q_matrix must be a square matrix.")
        if delta_t < 0:
            raise ValueError("delta_t must be non-negative.")

        n = q_matrix.shape[0]
        lambdas = -np.diag(q_matrix)[: n - 1]
        prob = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            prob[i] = transition_row(lambdas=lambdas, start_index=i, delta_t=delta_t, eps=self._eps)
        return prob


def transition_row(lambdas: np.ndarray, start_index: int, delta_t: float, eps: float = 1e-12) -> np.ndarray:
    """Return one-row transition probabilities analytically for a pure birth process."""
    n = lambdas.shape[0] + 1
    if not (0 <= start_index < n):
        raise ValueError("start_index is out of range for number of states.")

    lambda_ext = np.zeros(n, dtype=np.float64)
    lambda_ext[: n - 1] = lambdas
    lambda_ext[n - 1] = 0.0

    for a in range(start_index, n):
        for b in range(a + 1, n):
            if abs(lambda_ext[a] - lambda_ext[b]) < eps:
                lambda_ext[b] += eps * (b + 1)

    p = np.zeros(n, dtype=np.float64)
    i = start_index
    p[i] = np.exp(-lambda_ext[i] * delta_t)

    for j in range(i + 1, n):
        prefactor = float(np.prod(lambda_ext[i:j]))
        s = 0.0
        for k in range(i, j + 1):
            denom = 1.0
            for r in range(i, j + 1):
                if r == k:
                    continue
                denom *= (lambda_ext[r] - lambda_ext[k])
            s += np.exp(-lambda_ext[k] * delta_t) / denom
        p[j] = prefactor * s

    p = np.clip(p, a_min=0.0, a_max=None)
    row_sum = float(p.sum())
    if row_sum <= 0:
        p[:] = 0.0
        p[i] = 1.0
        return p
    return p / row_sum
