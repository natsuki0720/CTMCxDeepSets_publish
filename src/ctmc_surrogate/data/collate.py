"""可変長系列入力をバッチ化する collate_fn。"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .dataset import CTMCSample


def ctmc_collate_fn(samples: Sequence[CTMCSample]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """可変長系列をゼロパディングし、モデル入力形式へ変換する。"""
    if len(samples) == 0:
        raise ValueError("空バッチは処理できません。")

    target_dim = int(samples[0].target.shape[0])
    lengths = torch.tensor([int(s.state.shape[1]) for s in samples], dtype=torch.long)
    batch_size = len(samples)
    max_len = int(lengths.max().item())

    state_padded = torch.zeros((batch_size, 2, max_len), dtype=torch.long)
    delta_t_padded = torch.zeros((batch_size, max_len), dtype=torch.float32)
    target_batch = torch.zeros((batch_size, target_dim), dtype=torch.float32)

    for i, sample in enumerate(samples):
        if sample.state.ndim != 2 or sample.state.shape[0] != 2:
            raise ValueError(f"samples[{i}].state は shape (2, Li) である必要があります。")
        if sample.delta_t.ndim != 1:
            raise ValueError(f"samples[{i}].delta_t は shape (Li,) の1次元テンソルである必要があります。")
        if sample.target.ndim != 1:
            raise ValueError(f"samples[{i}].target は shape (output_dim,) の1次元テンソルである必要があります。")

        li = int(sample.state.shape[1])
        if li < 1:
            raise ValueError(f"samples[{i}] の系列長 Li は 1 以上である必要があります。")
        if int(sample.delta_t.shape[0]) != li:
            raise ValueError(f"samples[{i}] の state と delta_t の系列長が一致していません。")
        if int(sample.target.shape[0]) != target_dim:
            raise ValueError("バッチ内で target 次元が一致していません。")

        state_padded[i, :, :li] = sample.state.to(dtype=torch.long)
        delta_t_padded[i, :li] = sample.delta_t.to(dtype=torch.float32)
        target_batch[i] = sample.target.to(dtype=torch.float32)

    return state_padded, delta_t_padded, target_batch, lengths
