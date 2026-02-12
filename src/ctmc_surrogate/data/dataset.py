"""CTMCサロゲート学習用データセット定義。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CTMCSample:
    """1サンプル分の入力集合と教師信号を保持する。"""

    set_features: Tensor
    target_raw: Tensor


class CTMCSurrogateDataset(Dataset[CTMCSample]):
    """可変長集合入力を扱うCTMCサロゲート推定用データセット。"""

    def __init__(
        self,
        set_features_list: Sequence[Tensor],
        target_raw_list: Sequence[Tensor],
        min_positive: float = 1e-8,
    ) -> None:
        if len(set_features_list) != len(target_raw_list):
            raise ValueError(
                "入力集合と教師ラベルの件数が一致しません。"
                f" set_features={len(set_features_list)}, target={len(target_raw_list)}"
            )

        if len(set_features_list) == 0:
            raise ValueError("空データセットは許可されません。")

        self._samples: list[CTMCSample] = []
        self._min_positive = float(min_positive)

        expected_target_dim: int | None = None
        for i, (features, target_raw) in enumerate(zip(set_features_list, target_raw_list)):
            if features.ndim != 2:
                raise ValueError(f"set_features[{i}] は2次元テンソルである必要があります。")
            if target_raw.ndim != 1:
                raise ValueError(f"target_raw[{i}] は1次元テンソルである必要があります。")

            if expected_target_dim is None:
                expected_target_dim = int(target_raw.shape[0])
            elif expected_target_dim != int(target_raw.shape[0]):
                raise ValueError("全サンプルで target_raw の次元を一致させてください。")

            features_tensor = features.to(dtype=torch.float32)
            target_tensor = target_raw.to(dtype=torch.float32)
            if torch.any(target_tensor <= 0.0):
                raise ValueError(
                    "target_raw には正の値のみを指定してください。"
                    f" 非正値が set index={i} に含まれます。"
                )

            self._samples.append(
                CTMCSample(set_features=features_tensor, target_raw=target_tensor)
            )

        self._feature_dim = int(self._samples[0].set_features.shape[1])
        self._target_dim = int(self._samples[0].target_raw.shape[0])

    @property
    def feature_dim(self) -> int:
        """入力特徴量次元を返す。"""

        return self._feature_dim

    @property
    def target_dim(self) -> int:
        """教師信号次元を返す。"""

        return self._target_dim

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> CTMCSample:
        return self._samples[index]

    def get_lifetime_target(self, index: int) -> Tensor:
        """寿命指標（逆数変換後）を返す。"""

        raw = self._samples[index].target_raw
        return 1.0 / torch.clamp(raw, min=self._min_positive)
