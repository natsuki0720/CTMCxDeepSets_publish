"""CTMCデータ生成の設定定義。"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Sequence


@dataclass(frozen=True)
class TransitionRateConfig:
    """推移率行列生成の設定。"""

    num_states: int = 4
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

    num_samples: int = 1000
    transition_rate: TransitionRateConfig = field(default_factory=TransitionRateConfig)
    delta_t: DeltaTSamplingConfig = field(default_factory=DeltaTSamplingConfig)
    enable_mle: bool = True
    mle_init_r: Sequence[float] | None = None


@dataclass(frozen=True)
class MultiDatasetConfig:
    """複数データセット生成の設定。"""

    num_datasets: int = 1
    base_seed: int = 42
    dataset: DatasetGenerationConfig = field(default_factory=DatasetGenerationConfig)
