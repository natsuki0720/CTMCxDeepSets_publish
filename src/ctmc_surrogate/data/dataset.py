"""CTMCサロゲート学習用データセット定義。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CTMCSample:
    """1サンプル分の状態遷移系列と教師信号を保持する。"""

    state: Tensor
    delta_t: Tensor
    target: Tensor


class CTMCSurrogateDataset(Dataset[CTMCSample]):
    """可変長系列入力を扱うCTMCサロゲート推定用データセット。"""

    def __init__(
        self,
        state_list: Sequence[Tensor],
        delta_t_list: Sequence[Tensor],
        target_list: Sequence[Tensor],
    ) -> None:
        if not (len(state_list) == len(delta_t_list) == len(target_list)):
            raise ValueError(
                "入力と教師ラベルの件数が一致しません。"
                f" state={len(state_list)}, delta_t={len(delta_t_list)}, target={len(target_list)}"
            )
        if len(state_list) == 0:
            raise ValueError("空データセットは許可されません。")

        self._samples: list[CTMCSample] = []
        expected_target_dim: int | None = None

        for i, (state, delta_t, target) in enumerate(zip(state_list, delta_t_list, target_list)):
            if state.ndim != 2 or state.shape[0] != 2:
                raise ValueError(f"state[{i}] は shape (2, Li) である必要があります。")
            if delta_t.ndim != 1:
                raise ValueError(f"delta_t[{i}] は shape (Li,) の1次元テンソルである必要があります。")
            if target.ndim != 1:
                raise ValueError(f"target[{i}] は shape (output_dim,) の1次元テンソルである必要があります。")

            seq_len = int(state.shape[1])
            if seq_len < 1:
                raise ValueError(f"state[{i}] の系列長 Li は 1 以上である必要があります。")
            if int(delta_t.shape[0]) != seq_len:
                raise ValueError(f"state[{i}] と delta_t[{i}] の系列長が一致していません。")

            if expected_target_dim is None:
                expected_target_dim = int(target.shape[0])
            elif expected_target_dim != int(target.shape[0]):
                raise ValueError("全サンプルで target の次元を一致させてください。")

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
        """教師信号次元を返す。"""
        return self._target_dim

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> CTMCSample:
        return self._samples[index]
