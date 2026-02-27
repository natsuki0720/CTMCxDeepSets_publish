"""CTMC離散DelTデータセット生成器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from .config import DatasetGenerationConfig, MultiDatasetConfig
from .delta_t import DirichletDeltaT
from .mle_diagonal_exp import LikelihoodDiagonalExp
from .probability import CalcProbmatrix
from .transition_rate import DiagonalTransitionRateMatrixGenerator


@dataclass(frozen=True)
class CTMCTransitionSample:
    """単一遷移サンプル。"""

    start_state: int
    next_state: int
    delta_t: float


@dataclass(frozen=True)
class GeneratedDataset:
    """生成済みデータセット。"""

    q_matrix: list[list[float]]
    q_mle: list[list[float]] | None
    samples: list[CTMCTransitionSample]


class DataGenerator:
    """定式化に基づき1データセットを生成する。"""

    def __init__(self, config: DatasetGenerationConfig) -> None:
        if config.num_samples <= 0:
            raise ValueError("num_samples は1以上である必要があります。")

        self._config = config
        self._matrix_generator = DiagonalTransitionRateMatrixGenerator(config.transition_rate)
        self._calc_prob = CalcProbmatrix()

    def generate_matrix(self, rng: Generator) -> GeneratedDataset:
        """Q行列と遷移サンプル群を生成する。"""
        q_matrix = self._matrix_generator.generate(rng)
        n_state = q_matrix.shape[0]

        delta_t_generator = DirichletDeltaT(self._config.delta_t, rng)
        init_state_probs = _dirichlet_ones(n_state - 1, rng)

        samples: list[CTMCTransitionSample] = []
        for _ in range(self._config.num_samples):
            start_state = int(rng.choice(n_state - 1, p=init_state_probs))
            delta_t = delta_t_generator.sample()

            prob = self._calc_prob(q_matrix, delta_t)
            next_state = int(rng.choice(n_state, p=prob[start_state]))

            samples.append(
                CTMCTransitionSample(
                    start_state=start_state + 1,
                    next_state=next_state + 1,
                    delta_t=delta_t,
                )
            )

        q_mle: list[list[float]] | None = None
        if self._config.enable_mle:
            mle_data = np.array(
                [[sample.start_state, sample.next_state, sample.delta_t] for sample in samples],
                dtype=float,
            )
            init_r = np.array(self._resolve_mle_init_r(n_state), dtype=float)
            likelihood = LikelihoodDiagonalExp(mle_data, num_state=n_state)
            q_mle_matrix = likelihood.optimize(init_r)
            q_mle = q_mle_matrix.tolist()

        return GeneratedDataset(q_matrix=q_matrix.tolist(), q_mle=q_mle, samples=samples)

    def _resolve_mle_init_r(self, n_state: int) -> list[float]:
        expected_len = n_state - 1
        if self._config.mle_init_r is None:
            return [-0.5 * (i + 1) for i in range(expected_len)]

        init_r = list(self._config.mle_init_r)
        if len(init_r) != expected_len:
            raise ValueError(f"mle_init_r の長さは {expected_len} である必要があります。")
        return init_r


def _dirichlet_ones(size: int, rng: Generator) -> np.ndarray:
    uniform = rng.random(size)
    raw = -np.log(uniform)
    return raw / raw.sum()


def generate_multiple_datasets(config: MultiDatasetConfig) -> list[GeneratedDataset]:
    """ベースシードとデータセットIDから子シードを作り独立生成する。"""
    if config.num_datasets <= 0:
        raise ValueError("num_datasets は1以上である必要があります。")

    generator = DataGenerator(config.dataset)
    results: list[GeneratedDataset] = []
    for dataset_id in range(config.num_datasets):
        child_seed = (config.base_seed * 1_000_003 + dataset_id) % (2**63 - 1)
        rng = np.random.default_rng(child_seed)
        results.append(generator.generate_matrix(rng))

    return results
