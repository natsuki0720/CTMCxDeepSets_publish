"""Target-space transforms for Neural Posterior Estimation (NPE).

The point-estimate surrogate is trained to predict the transition rate
``q = lambda`` directly. The NPE estimator instead works in the *log-lifetime*
space ``z = log(nu) = log(1 / q)`` where ``nu`` is the exponential mean lifetime.
Working in ``z`` space has two advantages:

* The generative prior is ``nu ~ Uniform(1, lifetime_upper)`` (see
  :class:`~ctmc_surrogate.data_generation.transition_rate.DiagonalTransitionRateMatrixGenerator`),
  so ``z`` lives on a bounded, well-conditioned interval ``(0, log lifetime_upper)``.
* The NPE supervision label is the *generating* parameter, which is always valid
  by construction and requires no MLE step (see the NPE design note).

All functions accept NumPy arrays or Python floats and return NumPy arrays so
they compose with the existing CSV-loading pipeline, which builds torch tensors
via ``torch.as_tensor`` afterwards.
"""

from __future__ import annotations

import numpy as np

from .dataset_screening import extract_lambdas_from_Q

__all__ = [
    "lambdas_to_z",
    "z_to_lambdas",
    "z_to_nu",
    "nu_to_z",
    "build_npe_target_from_Q",
    "z_to_unbounded",
    "unbounded_to_z",
]


def lambdas_to_z(lambdas: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map transition rates ``lambda`` to log-lifetimes ``z = log(1 / lambda)``."""
    arr = np.asarray(lambdas, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError("All lambdas must be positive to take log(1 / lambda).")
    return np.log(1.0 / np.clip(arr, eps, None))


def z_to_lambdas(z: np.ndarray) -> np.ndarray:
    """Inverse of :func:`lambdas_to_z`: ``lambda = exp(-z)``."""
    return np.exp(-np.asarray(z, dtype=np.float64))


def z_to_nu(z: np.ndarray) -> np.ndarray:
    """Convert log-lifetimes to lifetimes ``nu = exp(z)``."""
    return np.exp(np.asarray(z, dtype=np.float64))


def nu_to_z(nu: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert lifetimes to log-lifetimes ``z = log(nu)``."""
    arr = np.asarray(nu, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError("All lifetimes must be positive to take log(nu).")
    return np.log(np.clip(arr, eps, None))


def build_npe_target_from_Q(Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Build the NPE supervision label ``z`` from the *true* generator matrix Q.

    Extracts the serial pure-birth rates ``lambda_i = Q[i, i+1]`` and maps them to
    log-lifetimes. Because Q is the generating matrix (never an MLE fit), the
    resulting target is always finite and inside ``(0, log lifetime_upper)``.
    """
    lambdas = extract_lambdas_from_Q(np.asarray(Q, dtype=np.float64))
    return lambdas_to_z(lambdas, eps=eps)


def z_to_unbounded(z: np.ndarray, lifetime_upper: float) -> np.ndarray:
    """Optional logit transform from ``z in (0, log U)`` to an unbounded space.

    This is the boundary-aware re-parameterization suggested in the NPE design
    note (section 2.2). Training a Gaussian head on the unbounded coordinate
    reduces the distortion caused by the prior truncation near ``nu = 1`` and
    ``nu = lifetime_upper``. ``unbounded = logit(z / log U)``.
    """
    upper = float(np.log(lifetime_upper))
    if upper <= 0:
        raise ValueError("lifetime_upper must be greater than 1.")
    u = np.clip(np.asarray(z, dtype=np.float64) / upper, 1e-6, 1.0 - 1e-6)
    return np.log(u / (1.0 - u))


def unbounded_to_z(unbounded: np.ndarray, lifetime_upper: float) -> np.ndarray:
    """Inverse of :func:`z_to_unbounded`: ``z = log U * sigmoid(unbounded)``."""
    upper = float(np.log(lifetime_upper))
    if upper <= 0:
        raise ValueError("lifetime_upper must be greater than 1.")
    sig = 1.0 / (1.0 + np.exp(-np.asarray(unbounded, dtype=np.float64)))
    return upper * sig
