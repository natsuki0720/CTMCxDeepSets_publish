"""Convert raw CTMC samples into DeepSets model input tensors."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = ["samples_to_model_input", "duplicate_samples"]


def samples_to_model_input(
    samples: np.ndarray | Tensor,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack one dataset's ``(state_pre, state_post, delta_t)`` rows into model inputs.

    Returns a batch of size 1:

    * ``state``   -> ``(1, 2, M)`` long tensor ``[pre; post]`` (raw state ids; the
      model normalizes the index base via ``input_is_one_based``).
    * ``delta_t`` -> ``(1, M)`` float tensor.
    * ``lengths`` -> ``(1,)`` long tensor equal to ``M`` (the sample count ``K``).
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"samples must have shape (M, 3): shape={arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("samples must contain at least one row.")

    pre = np.rint(arr[:, 0]).astype(np.int64)
    post = np.rint(arr[:, 1]).astype(np.int64)
    state = torch.from_numpy(np.stack([pre, post], axis=0)).unsqueeze(0).to(torch.long)
    delta_t = torch.from_numpy(arr[:, 2].astype(np.float32)).unsqueeze(0)
    lengths = torch.tensor([arr.shape[0]], dtype=torch.long)

    if device is not None:
        state = state.to(device)
        delta_t = delta_t.to(device)
        lengths = lengths.to(device)
    return state, delta_t, lengths


def duplicate_samples(samples: np.ndarray, times: int = 2) -> np.ndarray:
    """Stack a sample array ``times`` times (used by the replication test ``X (+) X``)."""
    if times < 1:
        raise ValueError("times must be at least 1.")
    arr = np.asarray(samples, dtype=np.float64)
    return np.concatenate([arr] * int(times), axis=0)
