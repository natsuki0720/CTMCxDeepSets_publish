"""Exact CTMC posterior over log-lifetimes ``z = log nu`` (calibration ground truth).

This module computes a numerically exact reference posterior for one dataset, to
benchmark the amortized NPE posterior against (NPE design note, section 4.1).

Why it is cheap: the log-likelihood depends on the samples only through the
*aggregated* unique ``(i, j, delta_t)`` cells and their counts,

    log L(z) = sum_cells  count * log P_ij(delta_t; lambda(z)),   lambda = exp(-z),

so it is independent of ``K`` once aggregated (a handful of cells for a 4-state
serial chain). ``P_ij`` is evaluated with the analytic pure-birth formula
(:func:`ctmc_surrogate.data_generation.probability.transition_row`).

Pipeline: aggregate cells -> multi-start MAP -> numerical-Hessian Laplace
approximation -> self-normalized importance sampling with a heavy-tailed
multivariate-t proposal (ESS monitored) -> posterior moments and quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_t

from ..data_generation.probability import transition_row

__all__ = [
    "AggregatedCells",
    "ExactPosteriorResult",
    "aggregate_cells",
    "log_prior_z",
    "exact_loglik_z",
    "log_posterior_z",
    "find_map",
    "laplace_covariance",
    "exact_posterior",
]

_DEFAULT_QUANTILES = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)


@dataclass(frozen=True)
class AggregatedCells:
    """Unique ``(start, target, delta_t)`` transition cells with multiplicities."""

    start_index: np.ndarray  # (C,) zero-based start state
    target_index: np.ndarray  # (C,) zero-based target state
    delta_t: np.ndarray  # (C,)
    count: np.ndarray  # (C,) integer multiplicity
    n_states: int

    @property
    def num_cells(self) -> int:
        return int(self.start_index.shape[0])

    @property
    def total_count(self) -> int:
        return int(self.count.sum())


@dataclass
class ExactPosteriorResult:
    """Exact-posterior summary in ``z = log nu`` space."""

    map_z: np.ndarray
    mean_z: np.ndarray
    cov_z: np.ndarray
    std_z: np.ndarray
    quantiles_z: dict[float, np.ndarray]
    ess: float
    n_eff_ratio: float
    samples_z: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    laplace_cov_z: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def mean_nu(self) -> np.ndarray:
        """Posterior mean lifetime estimated from importance samples."""
        return np.exp(self.samples_z).mean(axis=0) if self.samples_z.size else np.exp(self.mean_z)

    @property
    def mean_lambda(self) -> np.ndarray:
        """Posterior mean transition rate estimated from importance samples."""
        return np.exp(-self.samples_z).mean(axis=0) if self.samples_z.size else np.exp(-self.mean_z)


def aggregate_cells(samples: np.ndarray, n_states: int, one_based: bool = True) -> AggregatedCells:
    """Aggregate ``(state_pre, state_post, delta_t)`` rows into unique counted cells."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"samples must have shape (M, 3): shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("Cannot aggregate empty samples.")

    pre = np.rint(arr[:, 0]).astype(np.int64)
    post = np.rint(arr[:, 1]).astype(np.int64)
    dt = arr[:, 2].astype(np.float64)
    if one_based:
        pre = pre - 1
        post = post - 1

    keys = np.stack([pre, post, dt], axis=1)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    return AggregatedCells(
        start_index=uniq[:, 0].astype(np.int64),
        target_index=uniq[:, 1].astype(np.int64),
        delta_t=uniq[:, 2].astype(np.float64),
        count=counts.astype(np.int64),
        n_states=int(n_states),
    )


def _loglik_single(z: np.ndarray, cells: AggregatedCells, eps: float) -> float:
    """Exact log-likelihood for one ``z`` vector via cached transition rows."""
    lambdas = np.exp(-z)
    n = cells.n_states
    cache: dict[tuple[int, float], np.ndarray] = {}
    total = 0.0
    for start, target, dt, count in zip(
        cells.start_index, cells.target_index, cells.delta_t, cells.count
    ):
        key = (int(start), float(dt))
        row = cache.get(key)
        if row is None:
            row = transition_row(lambdas=lambdas, start_index=int(start), delta_t=float(dt))
            cache[key] = row
        total += float(count) * float(np.log(row[int(target)] + eps))
    return total


def exact_loglik_z(
    z: np.ndarray, cells: AggregatedCells, eps: float = 1e-12
) -> np.ndarray | float:
    """Exact log-likelihood ``log L(z)``. Accepts a single ``(D,)`` or batched ``(S, D)``."""
    arr = np.atleast_2d(np.asarray(z, dtype=np.float64))
    out = np.array([_loglik_single(row, cells, eps) for row in arr])
    return float(out[0]) if np.ndim(z) == 1 else out


def log_prior_z(z: np.ndarray, lifetime_upper: float) -> np.ndarray | float:
    """Log prior on ``z`` induced by ``nu ~ Uniform(1, lifetime_upper)`` per dimension.

    ``p(z) = exp(z) / (U - 1)`` on ``z in (0, log U)`` and zero outside.
    """
    upper = float(np.log(lifetime_upper))
    arr = np.atleast_2d(np.asarray(z, dtype=np.float64))
    inside = np.all((arr > 0.0) & (arr < upper), axis=1)
    log_norm = np.log(float(lifetime_upper) - 1.0)
    lp = np.where(inside, arr.sum(axis=1) - arr.shape[1] * log_norm, -np.inf)
    return float(lp[0]) if np.ndim(z) == 1 else lp


def log_posterior_z(
    z: np.ndarray, cells: AggregatedCells, lifetime_upper: float, eps: float = 1e-12
) -> np.ndarray | float:
    """Unnormalized log posterior ``log L(z) + log prior(z)``."""
    lp = log_prior_z(z, lifetime_upper)
    arr = np.atleast_2d(np.asarray(z, dtype=np.float64))
    lp_arr = np.atleast_1d(lp).astype(np.float64)
    out = np.full(arr.shape[0], -np.inf)
    finite = np.isfinite(lp_arr)
    if np.any(finite):
        ll = np.array([_loglik_single(arr[i], cells, eps) for i in np.where(finite)[0]])
        out[finite] = ll + lp_arr[finite]
    return float(out[0]) if np.ndim(z) == 1 else out


def find_map(
    cells: AggregatedCells,
    lifetime_upper: float,
    n_restarts: int = 8,
    seed: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Multi-start MAP estimate of ``z`` with box constraints from the prior."""
    d = cells.n_states - 1
    upper = float(np.log(lifetime_upper))
    margin = 1e-4 * upper
    bounds = [(margin, upper - margin)] * d
    rng = np.random.default_rng(seed)

    inits = [np.full(d, upper / 2.0)]
    inits.extend(rng.uniform(margin, upper - margin, size=(max(0, n_restarts - 1), d)))

    def neg_log_post(z: np.ndarray) -> float:
        val = log_posterior_z(np.clip(z, margin, upper - margin), cells, lifetime_upper, eps)
        return -float(val)

    best_z = inits[0]
    best_obj = np.inf
    for z0 in inits:
        try:
            res = minimize(neg_log_post, z0, method="L-BFGS-B", bounds=bounds)
            cand = np.clip(res.x, margin, upper - margin)
            obj = neg_log_post(cand)
        except Exception:
            continue
        if np.isfinite(obj) and obj < best_obj:
            best_obj = obj
            best_z = cand
    return best_z


def laplace_covariance(
    z_map: np.ndarray,
    cells: AggregatedCells,
    lifetime_upper: float,
    step: float = 1e-3,
    eps: float = 1e-12,
    fallback_sd: float = 0.5,
) -> np.ndarray:
    """Laplace covariance = inverse of the numerical Hessian at the MAP.

    The curvature is taken from the *smooth* negative log-likelihood, which is
    finite for every ``z`` (``lambda = exp(-z) > 0``). The ``U(1, U)`` prior is
    linear in ``z`` inside its support, so it adds no curvature; using the
    likelihood Hessian avoids ``-inf`` when a finite-difference step crosses the
    prior boundary near a boundary MAP. The proposal scale only needs to be
    roughly right — the heavy-tailed t-proposal and exact target (which does
    include the prior truncation) correct any mismatch downstream.
    """
    d = z_map.shape[0]

    def neg_loglik(z: np.ndarray) -> float:
        return -float(exact_loglik_z(z, cells, eps))

    hess = np.zeros((d, d))
    f0 = neg_loglik(z_map)
    for i in range(d):
        for j in range(i, d):
            ei = np.zeros(d); ei[i] = step
            ej = np.zeros(d); ej[j] = step
            if i == j:
                fp = neg_loglik(z_map + ei)
                fm = neg_loglik(z_map - ei)
                hess[i, i] = (fp - 2.0 * f0 + fm) / (step * step)
            else:
                fpp = neg_loglik(z_map + ei + ej)
                fpm = neg_loglik(z_map + ei - ej)
                fmp = neg_loglik(z_map - ei + ej)
                fmm = neg_loglik(z_map - ei - ej)
                val = (fpp - fpm - fmp + fmm) / (4.0 * step * step)
                hess[i, j] = val
                hess[j, i] = val

    hess = 0.5 * (hess + hess.T)
    fallback_cov = (fallback_sd ** 2) * np.eye(d)
    if not np.all(np.isfinite(hess)):
        return fallback_cov
    try:
        evals, evecs = np.linalg.eigh(hess)
    except np.linalg.LinAlgError:
        return fallback_cov
    # Floor eigenvalues so the Hessian is positive definite before inverting.
    evals = np.clip(evals, 1e-6, None)
    cov = (evecs / evals) @ evecs.T
    cov = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(cov)):
        return fallback_cov
    return cov


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return float(np.interp(q, cw, v))


def exact_posterior(
    samples: np.ndarray,
    n_states: int,
    lifetime_upper: float,
    one_based: bool = True,
    n_importance: int = 3000,
    n_restarts: int = 8,
    proposal_df: float = 4.0,
    scale_inflate: float = 1.5,
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES,
    seed: int = 0,
    eps: float = 1e-12,
) -> ExactPosteriorResult:
    """Compute the exact posterior over ``z`` for one dataset.

    Args:
        samples: ``(M, 3)`` array of ``(state_pre, state_post, delta_t)``.
        n_states: number of CTMC states (``D = n_states - 1`` rates).
        lifetime_upper: prior upper bound ``U`` for ``nu ~ Uniform(1, U)``.
        one_based: whether state ids in ``samples`` are 1-based.
        n_importance: number of importance-sampling particles.
        n_restarts: number of MAP restarts.
        proposal_df / scale_inflate: heavy-tailed t-proposal parameters.

    Returns:
        :class:`ExactPosteriorResult` with MAP, mean, covariance, std, quantiles
        (all in ``z`` space) plus the effective sample size and resampled
        importance particles.
    """
    cells = aggregate_cells(samples, n_states=n_states, one_based=one_based)
    d = n_states - 1

    z_map = find_map(cells, lifetime_upper, n_restarts=n_restarts, seed=seed, eps=eps)
    laplace_cov = laplace_covariance(z_map, cells, lifetime_upper, eps=eps)

    proposal = multivariate_t(loc=z_map, shape=scale_inflate * laplace_cov, df=proposal_df, seed=seed)
    z_samples = np.atleast_2d(proposal.rvs(size=n_importance))
    if z_samples.shape[0] != n_importance:  # rvs squeezes for d==1
        z_samples = z_samples.reshape(n_importance, d)

    log_target = log_posterior_z(z_samples, cells, lifetime_upper, eps)
    log_proposal = proposal.logpdf(z_samples)
    log_w = np.asarray(log_target, dtype=np.float64) - np.asarray(log_proposal, dtype=np.float64)

    finite = np.isfinite(log_w)
    if not np.any(finite):
        raise RuntimeError("All importance weights are zero; check the proposal/MAP.")
    z_samples = z_samples[finite]
    log_w = log_w[finite]
    log_w -= log_w.max()
    weights = np.exp(log_w)
    weights /= weights.sum()

    ess = float(1.0 / np.sum(weights ** 2))
    n_eff_ratio = ess / weights.shape[0]

    mean_z = weights @ z_samples
    centered = z_samples - mean_z
    cov_z = (centered * weights[:, None]).T @ centered
    cov_z *= 1.0 / (1.0 - np.sum(weights ** 2))  # bias correction for weighted cov
    std_z = np.sqrt(np.clip(np.diag(cov_z), 0.0, None))

    quantiles_z = {
        float(q): np.array([_weighted_quantile(z_samples[:, k], weights, q) for k in range(d)])
        for q in quantiles
    }

    # Resample to unweighted particles for downstream use (e.g. IS correction).
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(z_samples.shape[0], size=z_samples.shape[0], replace=True, p=weights)
    resampled = z_samples[idx]

    return ExactPosteriorResult(
        map_z=z_map,
        mean_z=mean_z,
        cov_z=cov_z,
        std_z=std_z,
        quantiles_z=quantiles_z,
        ess=ess,
        n_eff_ratio=n_eff_ratio,
        samples_z=resampled,
        laplace_cov_z=laplace_cov,
    )
