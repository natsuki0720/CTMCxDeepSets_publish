"""CTMCデータ生成の設定定義。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransitionRateConfig:
    """推移率行列生成の設定。"""

    num_states: int
    lifetime_upper: float = 100.0


@dataclass(frozen=True)
class DeltaTSamplingConfig:
    """離散DelT生成の設定。"""

    k_min: int = 2
    k_max: int = 10
    candidate_low: float = 1.0
    candidate_high: float = 100.0
    round_digits: int = 1


@dataclass(frozen=True)
class DatasetGenerationConfig:
    """1データセット生成の設定。"""

    num_samples: int
    transition_rate: TransitionRateConfig
    delta_t: DeltaTSamplingConfig = field(default_factory=DeltaTSamplingConfig)


@dataclass(frozen=True)
class MultiDatasetConfig:
    """複数データセット生成の設定。"""

    num_datasets: int
    base_seed: int
    dataset: DatasetGenerationConfig
