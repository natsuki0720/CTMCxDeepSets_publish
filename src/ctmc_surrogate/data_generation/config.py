"""Configuration definitions for CTMC data generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Sequence


@dataclass(frozen=True)
class TransitionRateConfig:
    """Configuration for transition-rate matrix generation."""

    num_states: int = 4
    lifetime_upper: float = 100.0


@dataclass(frozen=True)
class DeltaTSamplingConfig:
    """Configuration for discrete DelT generation."""

    k_min: int = 2
    k_max: int = 10
    candidate_low: float = 1.0
    candidate_high: float = 100.0
    round_digits: int = 1


@dataclass(frozen=True)
class DatasetGenerationConfig:
    """Configuration for generating one dataset."""

    num_samples: int = 1000
    transition_rate: TransitionRateConfig = field(default_factory=TransitionRateConfig)
    delta_t: DeltaTSamplingConfig = field(default_factory=DeltaTSamplingConfig)
    enable_mle: bool = True
    mle_init_r: Sequence[float] | None = None


@dataclass(frozen=True)
class MultiDatasetConfig:
    """Configuration for generating multiple datasets."""

    num_datasets: int = 1
    base_seed: int = 42
    dataset: DatasetGenerationConfig = field(default_factory=DatasetGenerationConfig)
