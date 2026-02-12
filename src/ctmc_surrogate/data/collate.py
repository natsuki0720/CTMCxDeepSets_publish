"""可変長集合入力をバッチ化する collate_fn。"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .dataset import CTMCSample


def ctmc_collate_fn(samples: Sequence[CTMCSample], min_positive: float = 1e-8) -> dict[str, Tensor]:
    """可変長集合をゼロパディングし、マスク付きバッチへ変換する。"""

    if len(samples) == 0:
        raise ValueError("空バッチは処理できません。")

    feature_dim = int(samples[0].set_features.shape[1])
    target_dim = int(samples[0].target_raw.shape[0])

    lengths = torch.tensor([s.set_features.shape[0] for s in samples], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(samples)

    set_features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    set_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    target_raw = torch.zeros((batch_size, target_dim), dtype=torch.float32)

    for i, sample in enumerate(samples):
        if sample.set_features.shape[1] != feature_dim:
            raise ValueError("バッチ内で入力特徴量次元が一致していません。")
        if sample.target_raw.shape[0] != target_dim:
            raise ValueError("バッチ内で target_raw 次元が一致していません。")

        n_i = sample.set_features.shape[0]
        set_features[i, :n_i] = sample.set_features
        set_mask[i, :n_i] = True
        target_raw[i] = sample.target_raw

    target_lifetime = 1.0 / torch.clamp(target_raw, min=float(min_positive))

    return {
        "set_features": set_features,
        "set_mask": set_mask,
        "lengths": lengths,
        "target_raw": target_raw,
        "target_lifetime": target_lifetime,
    }
