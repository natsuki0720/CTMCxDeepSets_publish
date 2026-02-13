"""劣化方向のみ CTMC データセット生成器。"""

from __future__ import annotations

from typing import Any

import numpy as np

from delta_t_mixture import make_dt_mixture, sample_dt
from q_matrix import generate_q
from rng_utils import dirichlet_ones, make_rng
from transition_prob import transition_row


def generate_dataset(
    base_seed: int,
    dataset_id: int,
    n_state: int,
    num_samples: int,
    lifespan_upper: float,
    k_min: int,
    k_max: int,
    cand_low: float,
    cand_high: float,
    round_digits: int,
) -> dict[str, Any]:
    """仕様どおりに Q とサンプル列を生成して JSON 互換 dict を返す。"""
    rng = make_rng(base_seed, dataset_id)

    # 乱数消費順序: 1)Q生成
    q, lambdas = generate_q(rng, n_state=n_state, lifespan_upper=lifespan_upper)

    # 乱数消費順序: 2)Δt混合生成
    candidates, weights, rd = make_dt_mixture(
        rng=rng,
        k_min=k_min,
        k_max=k_max,
        cand_low=cand_low,
        cand_high=cand_high,
        round_digits=round_digits,
    )

    # 乱数消費順序: 3)初期分布生成
    pi = dirichlet_ones(rng, n_state - 1)

    # 乱数消費順序: 4)サンプル生成
    samples: list[dict[str, Any]] = []
    for _ in range(num_samples):
        s0_index = int(rng.choice(n_state - 1, p=pi))
        start_state = s0_index + 1

        dt = sample_dt(rng, candidates=candidates, weights=weights, round_digits=rd)

        p = transition_row(lambdas=lambdas, start_index=s0_index, t=dt)
        next_index = int(rng.choice(n_state, p=p))
        next_state = next_index + 1

        samples.append({"start": start_state, "next": next_state, "dt": dt})

    return {
        "Q": q.tolist(),
        "samples": samples,
    }
