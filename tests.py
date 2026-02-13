"""最低限の仕様テスト。"""

from __future__ import annotations

import numpy as np

from dataset_generator import generate_dataset
from delta_t_mixture import make_dt_mixture
from q_matrix import generate_q
from rng_utils import make_rng
from transition_prob import transition_row


def run_tests() -> None:
    """指定された最小要件を検証する。"""
    base_seed = 123
    dataset_id = 0
    n_state = 4
    num_samples = 5
    lifespan_upper = 10.0
    k_min, k_max = 2, 5
    cand_low, cand_high = 0.1, 3.0
    round_digits = 3

    data = generate_dataset(
        base_seed=base_seed,
        dataset_id=dataset_id,
        n_state=n_state,
        num_samples=num_samples,
        lifespan_upper=lifespan_upper,
        k_min=k_min,
        k_max=k_max,
        cand_low=cand_low,
        cand_high=cand_high,
        round_digits=round_digits,
    )

    q = np.array(data["Q"], dtype=float)
    assert q.shape == (n_state, n_state)
    for i in range(n_state - 1):
        nz = np.where(np.abs(q[i]) > 0)[0]
        assert np.array_equal(nz, np.array([i, i + 1]))
        assert q[i, i] <= 0.0
        assert np.isclose(q[i, i + 1], -q[i, i])
    assert np.allclose(q[n_state - 1], 0.0)

    # 同じ乱数消費順序で候補点を再現し、丸め後候補集合を得る。
    rng = make_rng(base_seed, dataset_id)
    _, lambdas = generate_q(rng, n_state=n_state, lifespan_upper=lifespan_upper)
    candidates, _, _ = make_dt_mixture(
        rng,
        k_min=k_min,
        k_max=k_max,
        cand_low=cand_low,
        cand_high=cand_high,
        round_digits=round_digits,
    )
    dt_set = set(np.round(candidates, round_digits).tolist())

    for s in data["samples"]:
        assert 1 <= int(s["start"]) <= n_state - 1
        assert 1 <= int(s["next"]) <= n_state
        assert float(s["dt"]) in dt_set

        p = transition_row(lambdas=lambdas, start_index=int(s["start"]) - 1, t=float(s["dt"]))
        assert np.all(p >= -1e-15)
        assert abs(float(p.sum()) - 1.0) <= 1e-10


if __name__ == "__main__":
    run_tests()
    print("tests passed")
