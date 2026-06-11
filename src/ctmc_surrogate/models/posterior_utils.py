"""Generic helpers for the conditional posteriors returned by NPE heads.

These operate on any ``torch.distributions``-style object (the Gaussian head's
``MultivariateNormal`` or a ``zuko`` flow), using closed-form moments when
available and falling back to Monte-Carlo estimates otherwise.
"""

from __future__ import annotations

import contextlib

import torch
from torch import Tensor

__all__ = [
    "posterior_sample",
    "posterior_moments",
    "posterior_quantiles",
]


@contextlib.contextmanager
def _maybe_seed(seed: int | None):
    """Temporarily set the torch RNG seed, restoring global state afterwards."""
    if seed is None:
        yield
        return
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.random.set_rng_state(state)


def posterior_sample(posterior, num_samples: int, seed: int | None = None) -> Tensor:
    """Draw ``(num_samples, B, D)`` samples from a conditional posterior."""
    with _maybe_seed(seed), torch.no_grad():
        samples = posterior.sample((int(num_samples),))
    return samples


def posterior_moments(
    posterior, num_samples: int = 2000, seed: int | None = None
) -> tuple[Tensor, Tensor]:
    """Return ``(mean, std)`` of shape ``(B, D)`` for a conditional posterior.

    Uses the closed-form ``mean`` / ``stddev`` attributes when the distribution
    exposes them (e.g. ``MultivariateNormal``); otherwise estimates them from
    samples (e.g. a normalizing flow).
    """
    try:
        mean = posterior.mean
        std = posterior.stddev
        if mean is not None and std is not None:
            return mean.detach(), std.detach()
    except (AttributeError, NotImplementedError, RuntimeError):
        pass

    samples = posterior_sample(posterior, num_samples, seed=seed)
    return samples.mean(dim=0), samples.std(dim=0)


def posterior_quantiles(
    posterior,
    levels: tuple[float, ...],
    num_samples: int = 2000,
    seed: int | None = None,
) -> dict[float, Tensor]:
    """Return per-dimension Monte-Carlo quantiles ``{level: (B, D)}``."""
    samples = posterior_sample(posterior, num_samples, seed=seed)
    q = torch.tensor([float(x) for x in levels], dtype=samples.dtype, device=samples.device)
    quant = torch.quantile(samples, q, dim=0)  # (len(levels), B, D)
    return {float(level): quant[i].detach() for i, level in enumerate(levels)}
