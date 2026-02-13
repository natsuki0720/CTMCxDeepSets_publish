"""解析式による pure birth CTMC の遷移確率計算。"""

from __future__ import annotations

import numpy as np


def transition_row(
    lambdas: np.ndarray, start_index: int, t: float, eps: float = 1e-12
) -> np.ndarray:
    """開始状態 start_index から時刻 t の遷移確率ベクトルを返す。"""
    n_state = int(len(lambdas) + 1)
    if start_index < 0 or start_index >= n_state - 1:
        raise ValueError("start_index は 0..N-2 の範囲である必要があります。")

    lambda_ext = np.zeros(n_state, dtype=float)
    lambda_ext[:-1] = np.asarray(lambdas, dtype=float)

    # 同率対策: 近接する率があれば後ろ側に微小摂動を加える。
    for a in range(n_state):
        for b in range(a + 1, n_state):
            if abs(lambda_ext[a] - lambda_ext[b]) < eps:
                lambda_ext[b] += eps * (b + 1)

    p = np.zeros(n_state, dtype=float)
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
