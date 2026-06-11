"""Calibration diagnostics for the NPE posterior (NPE design note, section 4.2).

Provides four checks:

1. :func:`replication_test` — the single most direct test of the ``log K``
   injection. Feeding ``X`` and ``X (+) X`` (every sample duplicated) must shrink
   the posterior SD by ``1/sqrt(2)``; an attention-pooled encoder *without* the
   ``log K`` feature is replication-invariant and would give a ratio of 1.
2. :func:`run_sbc` / :func:`sbc_ks_test` / :func:`sbc_stratified` — simulation
   based calibration: rank statistics of the true ``z`` under the NPE posterior
   should be uniform; stratifying by ``K`` exposes band-specific miscalibration.
3. :func:`shrinkage_curve` — posterior SD vs ``K`` on log-log axes; the slope
   should approach ``-1/2`` (Bernstein-von Mises).
4. :func:`coverage_test` — empirical coverage of central credible intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.stats import kstest
from torch import nn

from ..features.sample_features import duplicate_samples, samples_to_model_input
from ..models.posterior_utils import posterior_moments, posterior_quantiles, posterior_sample

__all__ = [
    "ReplicationResult",
    "SBCResult",
    "CoverageResult",
    "ShrinkageResult",
    "replication_test",
    "posterior_rank",
    "run_sbc",
    "sbc_ks_test",
    "sbc_stratified",
    "coverage_test",
    "shrinkage_curve",
    "generate_sbc_datasets",
    "make_fixed_truth_sampler",
]


def _infer_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - models always have parameters
        return torch.device("cpu")


def _posterior_for_samples(model: nn.Module, samples: np.ndarray):
    device = _infer_device(model)
    state, delta_t, lengths = samples_to_model_input(samples, device=device)
    return model.posterior(state, delta_t, lengths)


# ----------------------------------------------------------------------
# 1. Replication test
# ----------------------------------------------------------------------
@dataclass
class ReplicationResult:
    """Per-dimension posterior SD for ``X`` vs ``X (+) X`` and their ratio."""

    sd_single: np.ndarray
    sd_double: np.ndarray
    ratio: np.ndarray  # sd_double / sd_single, expected ~ 1/sqrt(2)
    expected_ratio: float = float(1.0 / np.sqrt(2.0))

    @property
    def max_abs_error(self) -> float:
        return float(np.max(np.abs(self.ratio - self.expected_ratio)))


def replication_test(model: nn.Module, samples: np.ndarray, times: int = 2) -> ReplicationResult:
    """Check that duplicating every sample ``times`` times shrinks SD by ``1/sqrt(times)``."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            post_single = _posterior_for_samples(model, samples)
            sd_single = posterior_moments(post_single)[1].squeeze(0).cpu().numpy()

            doubled = duplicate_samples(samples, times=times)
            post_double = _posterior_for_samples(model, doubled)
            sd_double = posterior_moments(post_double)[1].squeeze(0).cpu().numpy()
    finally:
        model.train(was_training)

    ratio = sd_double / np.clip(sd_single, 1e-12, None)
    expected = float(1.0 / np.sqrt(times))
    return ReplicationResult(sd_single=sd_single, sd_double=sd_double, ratio=ratio, expected_ratio=expected)


# ----------------------------------------------------------------------
# 2. Simulation-based calibration (SBC)
# ----------------------------------------------------------------------
@dataclass
class SBCResult:
    """SBC rank statistics, one row per dataset."""

    ranks: np.ndarray  # (N, D) integer ranks in [0, num_draws]
    sample_sizes: np.ndarray  # (N,) K per dataset
    num_draws: int

    @property
    def normalized_ranks(self) -> np.ndarray:
        return self.ranks / float(self.num_draws)


def posterior_rank(posterior, z_true: np.ndarray, num_draws: int, seed: int | None = None) -> np.ndarray:
    """SBC rank of ``z_true`` per dimension: number of posterior draws below it."""
    draws = posterior_sample(posterior, num_draws, seed=seed).squeeze(1).cpu().numpy()  # (L, D)
    z = np.asarray(z_true, dtype=np.float64).reshape(1, -1)
    return np.sum(draws < z, axis=0).astype(np.int64)


def run_sbc(
    model: nn.Module,
    datasets: Sequence[tuple[np.ndarray, np.ndarray]],
    num_draws: int = 1000,
    seed: int = 0,
) -> SBCResult:
    """Compute SBC ranks over ``(samples, z_true)`` pairs."""
    was_training = model.training
    model.eval()
    ranks: list[np.ndarray] = []
    sizes: list[int] = []
    try:
        with torch.no_grad():
            for i, (samples, z_true) in enumerate(datasets):
                posterior = _posterior_for_samples(model, samples)
                ranks.append(posterior_rank(posterior, z_true, num_draws, seed=seed + i))
                sizes.append(int(np.asarray(samples).shape[0]))
    finally:
        model.train(was_training)

    return SBCResult(
        ranks=np.stack(ranks, axis=0),
        sample_sizes=np.asarray(sizes, dtype=np.int64),
        num_draws=int(num_draws),
    )


def sbc_ks_test(result: SBCResult) -> dict[str, np.ndarray]:
    """KS test of SBC ranks against the uniform distribution, per dimension."""
    normalized = result.normalized_ranks
    d = normalized.shape[1]
    stats = np.zeros(d)
    pvals = np.zeros(d)
    for k in range(d):
        ks = kstest(normalized[:, k], "uniform")
        stats[k] = float(ks.statistic)
        pvals[k] = float(ks.pvalue)
    return {"ks_statistic": stats, "p_value": pvals}


def sbc_stratified(result: SBCResult, k_edges: Sequence[float]) -> list[dict]:
    """Run :func:`sbc_ks_test` within ``K`` bands defined by ``k_edges``."""
    edges = np.asarray(k_edges, dtype=np.float64)
    reports = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (result.sample_sizes >= lo) & (result.sample_sizes < hi)
        n = int(mask.sum())
        entry: dict = {"k_low": float(lo), "k_high": float(hi), "n_datasets": n}
        if n >= 2:
            sub = SBCResult(result.ranks[mask], result.sample_sizes[mask], result.num_draws)
            ks = sbc_ks_test(sub)
            entry["ks_statistic"] = ks["ks_statistic"]
            entry["p_value"] = ks["p_value"]
        reports.append(entry)
    return reports


# ----------------------------------------------------------------------
# 3. Coverage
# ----------------------------------------------------------------------
@dataclass
class CoverageResult:
    """Empirical coverage of central credible intervals per level."""

    levels: tuple[float, ...]
    coverage_per_dim: dict[float, np.ndarray]  # level -> (D,)
    coverage_overall: dict[float, float]  # level -> scalar
    n_datasets: int


def coverage_test(
    model: nn.Module,
    datasets: Sequence[tuple[np.ndarray, np.ndarray]],
    levels: tuple[float, ...] = (0.5, 0.9),
    num_draws: int = 2000,
    seed: int = 0,
) -> CoverageResult:
    """Fraction of datasets whose true ``z`` falls inside each central credible interval."""
    was_training = model.training
    model.eval()
    quantile_levels: list[float] = []
    for level in levels:
        quantile_levels.extend([(1.0 - level) / 2.0, (1.0 + level) / 2.0])

    inside: dict[float, list[np.ndarray]] = {level: [] for level in levels}
    try:
        with torch.no_grad():
            for i, (samples, z_true) in enumerate(datasets):
                posterior = _posterior_for_samples(model, samples)
                quants = posterior_quantiles(posterior, tuple(quantile_levels), num_draws, seed=seed + i)
                z = np.asarray(z_true, dtype=np.float64)
                for level in levels:
                    lo = quants[(1.0 - level) / 2.0].squeeze(0).cpu().numpy()
                    hi = quants[(1.0 + level) / 2.0].squeeze(0).cpu().numpy()
                    inside[level].append((z >= lo) & (z <= hi))
    finally:
        model.train(was_training)

    coverage_per_dim = {level: np.mean(np.stack(inside[level], axis=0), axis=0) for level in levels}
    coverage_overall = {level: float(np.mean(np.stack(inside[level], axis=0))) for level in levels}
    return CoverageResult(
        levels=tuple(levels),
        coverage_per_dim=coverage_per_dim,
        coverage_overall=coverage_overall,
        n_datasets=len(list(datasets)),
    )


# ----------------------------------------------------------------------
# 4. Shrinkage curve
# ----------------------------------------------------------------------
@dataclass
class ShrinkageResult:
    """Posterior SD vs sample size ``K`` and fitted log-log slopes."""

    k_values: np.ndarray  # (T,)
    sd: np.ndarray  # (T, D)
    slopes: np.ndarray  # (D,) fitted d log(sd) / d log(K), expected ~ -0.5


def shrinkage_curve(
    model: nn.Module,
    sampler_fn: Callable[[int], np.ndarray],
    k_values: Sequence[int],
) -> ShrinkageResult:
    """Posterior SD as a function of ``K`` for a fixed ground truth.

    ``sampler_fn(K)`` should return an ``(K, 3)`` sample array drawn from a single
    fixed generator matrix; the BvM scaling predicts a log-log slope of ``-1/2``.
    """
    was_training = model.training
    model.eval()
    sds: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for k in k_values:
                samples = sampler_fn(int(k))
                posterior = _posterior_for_samples(model, samples)
                sds.append(posterior_moments(posterior)[1].squeeze(0).cpu().numpy())
    finally:
        model.train(was_training)

    sd = np.stack(sds, axis=0)
    log_k = np.log(np.asarray(k_values, dtype=np.float64))
    slopes = np.array([np.polyfit(log_k, np.log(np.clip(sd[:, d], 1e-12, None)), 1)[0] for d in range(sd.shape[1])])
    return ShrinkageResult(k_values=np.asarray(k_values, dtype=np.float64), sd=sd, slopes=slopes)


# ----------------------------------------------------------------------
# Convenience dataset generation for SBC / coverage
# ----------------------------------------------------------------------
def generate_sbc_datasets(
    num_datasets: int,
    num_states: int = 4,
    lifetime_upper: float = 100.0,
    k_min: int = 200,
    k_max: int = 5000,
    base_seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate fresh ``(samples, z_true)`` pairs for SBC / coverage.

    Uses the project generator with ``enable_mle=False`` (no MLE step), so a
    thousand calibration datasets are cheap to produce.
    """
    from ..data.targets import build_npe_target_from_Q
    from ..data_generation.config import DatasetGenerationConfig, TransitionRateConfig
    from ..data_generation.generator import DataGenerator

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(num_datasets):
        seed = np.random.SeedSequence([base_seed, i]).generate_state(1, dtype=np.uint64)[0]
        rng = np.random.default_rng(int(seed))
        k = int(rng.integers(k_min, k_max + 1))
        cfg = DatasetGenerationConfig(
            num_samples=k,
            transition_rate=TransitionRateConfig(num_states=num_states, lifetime_upper=lifetime_upper),
            enable_mle=False,
        )
        gen = DataGenerator(cfg).generate_matrix(rng)
        samples = np.array([[s.start_state, s.next_state, s.delta_t] for s in gen.samples], dtype=np.float64)
        z_true = build_npe_target_from_Q(np.array(gen.q_matrix))
        out.append((samples, z_true))
    return out


def make_fixed_truth_sampler(
    q_matrix: np.ndarray,
    delta_t_config=None,
    seed: int = 0,
) -> Callable[[int], np.ndarray]:
    """Return ``sampler_fn(K)`` drawing ``K`` samples from one *fixed* generator ``Q``.

    Used by :func:`shrinkage_curve`: the ground truth stays constant while ``K``
    varies, isolating the BvM ``-1/2`` shrinkage. Re-seeded per call so the
    delta-t mixture and initial-state distribution are identical across ``K``.
    """
    from ..data_generation.config import DeltaTSamplingConfig
    from ..data_generation.delta_t import DirichletDeltaT
    from ..data_generation.probability import CalcProbmatrix

    q = np.asarray(q_matrix, dtype=np.float64)
    n = q.shape[0]
    cfg = delta_t_config if delta_t_config is not None else DeltaTSamplingConfig()
    calc = CalcProbmatrix()

    def sampler(k: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dt_gen = DirichletDeltaT(cfg, rng)
        raw = -np.log(rng.random(n - 1))
        init = raw / raw.sum()
        rows = []
        for _ in range(int(k)):
            start = int(rng.choice(n - 1, p=init))
            dt = dt_gen.sample()
            prob = calc(q, dt)
            nxt = int(rng.choice(n, p=prob[start]))
            rows.append([start + 1, nxt + 1, dt])
        return np.asarray(rows, dtype=np.float64)

    return sampler
