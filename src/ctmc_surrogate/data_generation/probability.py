"""CTMC遷移確率計算モジュール。"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class CalcProbmatrix:
    """pure birthの解析式でP(Δt)を計算する。"""

    def __init__(self, eps: float = 1e-12) -> None:
        self._eps = eps

    def __call__(self, q_matrix: Tensor, delta_t: float) -> Tensor:
        if q_matrix.ndim != 2 or q_matrix.shape[0] != q_matrix.shape[1]:
            raise ValueError("q_matrix は正方行列である必要があります。")
        if delta_t < 0:
            raise ValueError("delta_t は非負である必要があります。")

        q_np = q_matrix.detach().cpu().numpy().astype(np.float64, copy=False)
        n = int(q_np.shape[0])
        lambdas = -np.diag(q_np)[:-1]

        rows = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            rows[i] = _transition_row(lambdas=lambdas, start_index=i, t=float(delta_t), eps=self._eps)
        rows[n - 1, n - 1] = 1.0
        return torch.from_numpy(rows).to(dtype=torch.float64)


def _transition_row(
    lambdas: np.ndarray, start_index: int, t: float, eps: float = 1e-12
) -> np.ndarray:
    """開始状態start_indexから時刻tの遷移確率行を返す。"""
    n_state = int(len(lambdas) + 1)
    if start_index < 0 or start_index >= n_state - 1:
        raise ValueError("start_index は 0..N-2 の範囲である必要があります。")

    lambda_ext = np.zeros(n_state, dtype=np.float64)
    lambda_ext[:-1] = np.asarray(lambdas, dtype=np.float64)

    for a in range(n_state):
        for b in range(a + 1, n_state):
            if abs(lambda_ext[a] - lambda_ext[b]) < eps:
                lambda_ext[b] += eps * (b + 1)

    p = np.zeros(n_state, dtype=np.float64)
    i = start_index
    p[i] = np.exp(-lambda_ext[i] * t)

    for j in range(i + 1, n_state):
        prefactor = 1.0
        for r in range(i, j):
            prefactor *= lambda_ext[r]

        s_ij = 0.0
        for k in range(i, j + 1):
            denom = 1.0
            for r in range(i, j + 1):
                if r == k:
                    continue
                denom *= (lambda_ext[r] - lambda_ext[k])
            s_ij += np.exp(-lambda_ext[k] * t) / denom

        p[j] = prefactor * s_ij

    p = np.where(p < 0.0, 0.0, p)
    s = p.sum()
    if s <= 0.0:
        p[:] = 0.0
        p[i] = 1.0
    else:
        p /= s
    return p
