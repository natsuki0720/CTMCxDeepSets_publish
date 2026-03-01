"""対角指数形式CTMCの最尤推定。"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


class LikelihoodDiagonalExp:
    """対角指数形式Q行列の負の対数尤度を最小化する。"""

    def __init__(self, data: np.ndarray, num_state: int = 4) -> None:
        if num_state < 2:
            raise ValueError("num_state は2以上である必要があります。")
        self._data = data
        self._num_state = num_state

    def generate_q_from_r(self, r_vec: np.ndarray) -> np.ndarray:
        if len(r_vec) != self._num_state - 1:
            raise ValueError("r_vec の長さは num_state - 1 と一致する必要があります。")

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
        #     raise RuntimeError(f"MLE最適化に失敗しました: {result.message}")
        return self.generate_q_from_r(result.x)
