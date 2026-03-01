"""Dataset definitions for CTMC surrogate training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CTMCSample:
    """Hold one sample of state-transition sequence and supervision target."""

    state: Tensor
    delta_t: Tensor
    target: Tensor


class CTMCSurrogateDataset(Dataset[CTMCSample]):
    """Dataset for CTMC surrogate estimation with variable-length sequence inputs."""

    def __init__(
        self,
        state_list: Sequence[Tensor],
        delta_t_list: Sequence[Tensor],
        target_list: Sequence[Tensor],
    ) -> None:
        if not (len(state_list) == len(delta_t_list) == len(target_list)):
            raise ValueError(
                "Input and target label counts do not match."
                f" state={len(state_list)}, delta_t={len(delta_t_list)}, target={len(target_list)}"
            )
        if len(state_list) == 0:
            raise ValueError("Empty datasets are not allowed.")

        self._samples: list[CTMCSample] = []
        expected_target_dim: int | None = None

        for i, (state, delta_t, target) in enumerate(zip(state_list, delta_t_list, target_list)):
            if state.ndim != 2 or state.shape[0] != 2:
                raise ValueError(f"state[{i}] must have shape (2, Li).")
            if delta_t.ndim != 1:
                raise ValueError(f"delta_t[{i}] must be a 1D tensor with shape (Li,).")
            if target.ndim != 1:
                raise ValueError(f"target[{i}] must be a 1D tensor with shape (output_dim,).")

            seq_len = int(state.shape[1])
            if seq_len < 1:
                raise ValueError(f"state[{i}] sequence length Li must be at least 1.")
            if int(delta_t.shape[0]) != seq_len:
                raise ValueError(f"state[{i}] and delta_t[{i}] have mismatched sequence lengths.")

            if expected_target_dim is None:
                expected_target_dim = int(target.shape[0])
            elif expected_target_dim != int(target.shape[0]):
                raise ValueError("Please ensure target dimensions are consistent across all samples.")

            self._samples.append(
                CTMCSample(
                    state=state.to(dtype=torch.long),
                    delta_t=delta_t.to(dtype=torch.float32),
                    target=target.to(dtype=torch.float32),
                )
            )

        self._target_dim = int(self._samples[0].target.shape[0])

    @property
    def target_dim(self) -> int:
        """Return the supervision target dimension."""
        return self._target_dim

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> CTMCSample:
        return self._samples[index]
