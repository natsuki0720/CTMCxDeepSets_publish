"""Maximum-likelihood estimation for diagonal-exponential CTMC."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .probability import transition_row


class LikelihoodDiagonalExp:
    """Minimize the negative log-likelihood of a diagonal-exponential Q matrix."""

    def __init__(self, data: np.ndarray, num_state: int = 4) -> None:
        if num_state < 2:
            raise ValueError("num_state must be at least 2.")
        self._data = data
        self._num_state = num_state

    def generate_q_from_r(self, r_vec: np.ndarray) -> np.ndarray:
        if len(r_vec) != self._num_state - 1:
            raise ValueError("Length of r_vec must equal num_state - 1.")

        q_matrix = np.zeros((self._num_state, self._num_state), dtype=float)
        rates = np.exp(r_vec)
        for i in range(self._num_state - 1):
            q_matrix[i, i + 1] = rates[i]
            q_matrix[i, i] = -rates[i]
        return q_matrix

    def log_likelihood(self, r_vec: np.ndarray) -> float:
        likelihood = 0.0
        q_matrix = self.generate_q_from_r(r_vec)
        for sample in self._data:
            start = int(sample[0])
            nxt = int(sample[1])
            delta_t = float(sample[2])
            prob_m = expm(q_matrix * delta_t)
            likelihood += np.log(prob_m[start - 1, nxt - 1] + 1e-12)
        return -likelihood

    def optimize(self, vec: np.ndarray) -> np.ndarray:
        result = minimize(
            self.log_likelihood,
            vec,
            method="BFGS",
            options={"gtol": 1e-5, "maxiter": 1000, "disp": False},
        )
        # if not result.success:
        #     raise RuntimeError(f"MLE optimization failed: {result.message}")
        return self.generate_q_from_r(result.x)


class LikelihoodDiagonalExpFast(LikelihoodDiagonalExp):
    """Drop-in faster MLE that aggregates samples and uses the analytic row formula.

    The base class calls :func:`scipy.linalg.expm` once per *sample*. The serial
    pure-birth structure lets us instead (1) aggregate the data into the unique
    ``(start, target, delta_t)`` cells and (2) evaluate each transition
    probability with the closed-form
    :func:`ctmc_surrogate.data_generation.probability.transition_row`. The result
    is identical up to floating point but orders of magnitude cheaper for large
    ``K`` (NPE design note, section 8) — useful for a robust speed comparison
    against the surrogate.
    """

    def __init__(self, data: np.ndarray, num_state: int = 4, eps: float = 1e-12) -> None:
        super().__init__(data, num_state=num_state)
        self._eps = float(eps)
        self._cells = self._aggregate(np.asarray(data, dtype=np.float64))

    def _aggregate(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Data uses 1-based state ids; convert to 0-based indices for transition_row.
        start = np.rint(data[:, 0]).astype(np.int64) - 1
        target = np.rint(data[:, 1]).astype(np.int64) - 1
        dt = data[:, 2].astype(np.float64)
        keys = np.stack([start, target, dt], axis=1)
        uniq, counts = np.unique(keys, axis=0, return_counts=True)
        return (
            uniq[:, 0].astype(np.int64),
            uniq[:, 1].astype(np.int64),
            uniq[:, 2].astype(np.float64),
            counts.astype(np.float64),
        )

    def log_likelihood(self, r_vec: np.ndarray) -> float:
        if len(r_vec) != self._num_state - 1:
            raise ValueError("Length of r_vec must equal num_state - 1.")
        rates = np.exp(np.asarray(r_vec, dtype=np.float64))
        starts, targets, dts, counts = self._cells
        row_cache: dict[tuple[int, float], np.ndarray] = {}
        total = 0.0
        for start, target, dt, count in zip(starts, targets, dts, counts):
            key = (int(start), float(dt))
            row = row_cache.get(key)
            if row is None:
                row = transition_row(lambdas=rates, start_index=int(start), delta_t=float(dt))
                row_cache[key] = row
            total += count * float(np.log(row[int(target)] + self._eps))
        return -total

    def optimize(self, vec: np.ndarray) -> np.ndarray:
        result = minimize(
            self.log_likelihood,
            vec,
            method="BFGS",
            options={"gtol": 1e-5, "maxiter": 1000, "disp": False},
        )
        return self.generate_q_from_r(result.x)
