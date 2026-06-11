"""Importance-sampling correction of the NPE posterior (NPE design note, section 5).

Because the CTMC likelihood is exact and cheap on aggregated cells, the NPE
posterior can be used as an importance *proposal* and reweighted toward the
asymptotically exact Bayesian posterior:

    w(z) ∝ exact_likelihood(z) · prior(z) / q_phi(z | X).

Two uses:

* **Defense against approximation / misspecification** — corrects NPE bias with
  self-normalized importance weights (ESS reported as a trust signal).
* **Prior swap** — supply ``prior_log_prob`` from a *different* prior than the
  ``U(1, lifetime_upper)`` used at training time to retarget the posterior with
  no retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..eval.exact_posterior import AggregatedCells, aggregate_cells, exact_loglik_z, log_prior_z

__all__ = ["ISCorrectionResult", "importance_correct", "is_correct_from_predictor"]


@dataclass
class ISCorrectionResult:
    """Self-normalized importance-sampling correction summary (z space)."""

    mean_z: np.ndarray
    std_z: np.ndarray
    cov_z: np.ndarray
    quantiles_z: dict[float, np.ndarray]
    ess: float
    n_eff_ratio: float
    weights: np.ndarray
    samples_z: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def mean_nu(self) -> np.ndarray:
        return np.exp(self.samples_z).mean(axis=0) if self.samples_z.size else np.exp(self.mean_z)

    @property
    def mean_lambda(self) -> np.ndarray:
        return np.exp(-self.samples_z).mean(axis=0) if self.samples_z.size else np.exp(-self.mean_z)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return float(np.interp(q, cw, v))


def importance_correct(
    npe_samples_z: np.ndarray,
    cells: AggregatedCells,
    npe_log_prob: np.ndarray,
    lifetime_upper: float | None = None,
    prior_log_prob: np.ndarray | None = None,
    quantiles: tuple[float, ...] = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975),
    eps: float = 1e-12,
    seed: int = 0,
) -> ISCorrectionResult:
    """Reweight NPE proposal samples toward the exact posterior.

    Args:
        npe_samples_z: ``(S, D)`` samples from ``q_phi(z | X)``.
        cells: aggregated transition cells for this dataset.
        npe_log_prob: ``(S,)`` log-density ``log q_phi(z_s | X)`` of the proposal.
        lifetime_upper: prior bound ``U`` for the default ``U(1, U)`` prior.
            Ignored when ``prior_log_prob`` is provided.
        prior_log_prob: ``(S,)`` log prior under a (possibly different) prior —
            supply this to perform a prior swap.
    """
    samples = np.asarray(npe_samples_z, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"npe_samples_z must be 2D (S, D): shape={samples.shape}")
    npe_lp = np.asarray(npe_log_prob, dtype=np.float64)

    if prior_log_prob is None:
        if lifetime_upper is None:
            raise ValueError("Provide either prior_log_prob or lifetime_upper.")
        prior_lp = np.asarray(log_prior_z(samples, lifetime_upper), dtype=np.float64)
    else:
        prior_lp = np.asarray(prior_log_prob, dtype=np.float64)

    loglik = np.asarray(exact_loglik_z(samples, cells, eps), dtype=np.float64)
    log_w = loglik + prior_lp - npe_lp

    finite = np.isfinite(log_w)
    if not np.any(finite):
        raise RuntimeError("All importance weights are zero after correction.")
    samples = samples[finite]
    log_w = log_w[finite]
    log_w -= log_w.max()
    weights = np.exp(log_w)
    weights /= weights.sum()

    ess = float(1.0 / np.sum(weights ** 2))
    n_eff_ratio = ess / weights.shape[0]

    mean_z = weights @ samples
    centered = samples - mean_z
    cov_z = (centered * weights[:, None]).T @ centered
    denom = 1.0 - np.sum(weights ** 2)
    if denom > 0:
        cov_z = cov_z / denom
    std_z = np.sqrt(np.clip(np.diag(cov_z), 0.0, None))

    quantiles_z = {
        float(q): np.array([_weighted_quantile(samples[:, k], weights, q) for k in range(samples.shape[1])])
        for q in quantiles
    }

    rng = np.random.default_rng(seed)
    idx = rng.choice(samples.shape[0], size=samples.shape[0], replace=True, p=weights)
    resampled = samples[idx]

    return ISCorrectionResult(
        mean_z=mean_z,
        std_z=std_z,
        cov_z=cov_z,
        quantiles_z=quantiles_z,
        ess=ess,
        n_eff_ratio=n_eff_ratio,
        weights=weights,
        samples_z=resampled,
    )


def is_correct_from_predictor(
    predictor,
    samples_raw: np.ndarray,
    lifetime_upper: float | None = None,
    num_samples: int = 4000,
    prior_log_prob_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    one_based: bool = True,
    seed: int = 0,
) -> ISCorrectionResult:
    """End-to-end IS correction: draw NPE proposals and reweight by the exact posterior.

    ``prior_log_prob_fn(z)`` maps ``(S, D)`` samples to ``(S,)`` log-prior values;
    pass it to swap in a different prior than the training one.
    """
    import torch

    posterior = predictor.posterior(samples_raw)
    with torch.no_grad():
        draws = posterior.sample((int(num_samples),))  # (S, 1, D)
        npe_logp = posterior.log_prob(draws)  # (S, 1)
    draws_np = draws.squeeze(1).cpu().numpy()
    npe_logp_np = npe_logp.squeeze(1).cpu().numpy()

    cells = aggregate_cells(samples_raw, n_states=predictor.n_states, one_based=one_based)
    prior_lp = None if prior_log_prob_fn is None else np.asarray(prior_log_prob_fn(draws_np), dtype=np.float64)

    return importance_correct(
        npe_samples_z=draws_np,
        cells=cells,
        npe_log_prob=npe_logp_np,
        lifetime_upper=lifetime_upper,
        prior_log_prob=prior_lp,
        seed=seed,
    )
