"""collate_fn for batching variable-length sequence inputs."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .dataset import CTMCSample


def ctmc_collate_fn(samples: Sequence[CTMCSample]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Zero-pad variable-length sequences and convert them into model input format."""
    if len(samples) == 0:
        raise ValueError("Empty batches cannot be processed.")

    target_dim = int(samples[0].target.shape[0])
    lengths = torch.tensor([int(s.state.shape[1]) for s in samples], dtype=torch.long)
    batch_size = len(samples)
    max_len = int(lengths.max().item())

    state_padded = torch.zeros((batch_size, 2, max_len), dtype=torch.long)
    delta_t_padded = torch.zeros((batch_size, max_len), dtype=torch.float32)
    target_batch = torch.zeros((batch_size, target_dim), dtype=torch.float32)

    for i, sample in enumerate(samples):
        if sample.state.ndim != 2 or sample.state.shape[0] != 2:
            raise ValueError(f"samples[{i}].state must have shape (2, Li).")
        if sample.delta_t.ndim != 1:
            raise ValueError(f"samples[{i}].delta_t must be a 1D tensor with shape (Li,).")
        if sample.target.ndim != 1:
            raise ValueError(f"samples[{i}].target must be a 1D tensor with shape (output_dim,).")

        li = int(sample.state.shape[1])
        if li < 1:
            raise ValueError(f"samples[{i}] sequence length Li must be at least 1.")
        if int(sample.delta_t.shape[0]) != li:
            raise ValueError(f"samples[{i}] has mismatched sequence lengths between state and delta_t.")
        if int(sample.target.shape[0]) != target_dim:
            raise ValueError("Target dimensions do not match within the batch.")

        state_padded[i, :, :li] = sample.state.to(dtype=torch.long)
        delta_t_padded[i, :li] = sample.delta_t.to(dtype=torch.float32)
        target_batch[i] = sample.target.to(dtype=torch.float32)

    return state_padded, delta_t_padded, target_batch, lengths
