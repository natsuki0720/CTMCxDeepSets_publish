"""DeepSetsベースのサロゲート回帰モデル。"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepSetsRegressor(nn.Module):
    """可変長集合入力から正値パラメータを推定するDeepSets回帰器。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        phi_hidden_dim: int = 128,
        rho_hidden_dim: int = 128,
        aggregation: str = "sum",
        min_positive: float = 1e-8,
    ) -> None:
        super().__init__()

        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"aggregation は 'sum' または 'mean' を指定してください: {aggregation}")

        self.aggregation = aggregation
        self.min_positive = float(min_positive)

        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden_dim),
            nn.ReLU(),
            nn.Linear(phi_hidden_dim, phi_hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden_dim, rho_hidden_dim),
            nn.ReLU(),
            nn.Linear(rho_hidden_dim, output_dim),
        )

    def forward_raw(self, set_features: Tensor, set_mask: Tensor) -> Tensor:
        """素の正値パラメータ（例: 遷移率や寿命率）を推定する。"""

        if set_features.ndim != 3:
            raise ValueError("set_features は [batch, set_size, feature_dim] を想定します。")
        if set_mask.ndim != 2:
            raise ValueError("set_mask は [batch, set_size] を想定します。")

        embedded = self.phi(set_features)
        mask = set_mask.unsqueeze(-1).to(dtype=embedded.dtype)
        masked = embedded * mask

        pooled = masked.sum(dim=1)
        if self.aggregation == "mean":
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled / denom

        logits = self.rho(pooled)
        return torch.nn.functional.softplus(logits) + self.min_positive

    def forward(self, set_features: Tensor, set_mask: Tensor, output_mode: str = "lifetime") -> Tensor:
        """出力モードに応じて寿命指標または生パラメータを返す。"""

        raw = self.forward_raw(set_features=set_features, set_mask=set_mask)
        if output_mode == "raw":
            return raw
        if output_mode == "lifetime":
            return 1.0 / torch.clamp(raw, min=self.min_positive)
        raise ValueError(f"output_mode は 'raw' または 'lifetime' を指定してください: {output_mode}")


def build_model(model_config: dict) -> DeepSetsRegressor:
    """設定辞書から DeepSetsRegressor を構築する。"""

    required = ["input_dim", "output_dim"]
    missing = [k for k in required if k not in model_config]
    if missing:
        raise ValueError(f"model_config に必須キーが不足しています: {missing}")

    return DeepSetsRegressor(
        input_dim=int(model_config["input_dim"]),
        output_dim=int(model_config["output_dim"]),
        phi_hidden_dim=int(model_config.get("phi_hidden_dim", 128)),
        rho_hidden_dim=int(model_config.get("rho_hidden_dim", 128)),
        aggregation=str(model_config.get("aggregation", "sum")),
        min_positive=float(model_config.get("min_positive", 1e-8)),
    )
