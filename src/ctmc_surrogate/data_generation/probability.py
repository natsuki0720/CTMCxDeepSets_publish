"""CTMC遷移確率行列計算モジュール。"""

from __future__ import annotations

import torch
from torch import Tensor


class CalcProbmatrix:
    """行列指数を用いてP(Δt)=exp(QΔt)を計算する。"""

    def __call__(self, q_matrix: Tensor, delta_t: float) -> Tensor:
        if q_matrix.ndim != 2 or q_matrix.shape[0] != q_matrix.shape[1]:
            raise ValueError("q_matrix は正方行列である必要があります。")
        if delta_t < 0:
            raise ValueError("delta_t は非負である必要があります。")

        qt = q_matrix.to(dtype=torch.float64) * float(delta_t)
        return torch.matrix_exp(qt)
