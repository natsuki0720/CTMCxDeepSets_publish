"""DeepSets系サロゲート回帰モデル。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DeepSetsAttnRegressor(nn.Module):
    """可変長集合入力から正値の生パラメータを推定する注意機構付きDeepSets回帰器。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        phi_hidden_dim: int = 128,
        rho_hidden_dim: int = 128,
        attn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        min_positive: float = 1e-8,
    ) -> None:
        super().__init__()
        self.min_positive = float(min_positive)
        phi_dim = int(phi_hidden_dim)
        attn_dim = int(attn_hidden_dim) if attn_hidden_dim is not None else phi_dim

        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_dim),
            nn.ReLU(),
            nn.Linear(phi_dim, phi_dim),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)

        self.att_fc = nn.Linear(phi_dim, attn_dim)
        self.att_score = nn.Linear(attn_dim, 1)

        self.rho = nn.Sequential(
            nn.Linear(phi_dim, rho_hidden_dim),
            nn.ReLU(),
            nn.Linear(rho_hidden_dim, output_dim),
        )

    def forward_raw(self, set_features: Tensor, set_mask: Tensor) -> Tensor:
        """正値制約付きの生パラメータを返す。"""
        if set_features.ndim != 3:
            raise ValueError("set_features は [batch, set_size, feature_dim] を想定します。")
        if set_mask.ndim != 2:
            raise ValueError("set_mask は [batch, set_size] を想定します。")
        if set_features.shape[:2] != set_mask.shape:
            raise ValueError("set_features と set_mask の [batch, set_size] が一致しません。")

        set_mask_bool = set_mask.to(dtype=torch.bool)
        if not torch.all(set_mask_bool.any(dim=1)):
            raise ValueError("set_mask の各バッチに少なくとも1つの有効要素が必要です。")

        x = self.drop(self.phi(set_features))

        attn_hidden = torch.tanh(self.att_fc(x))
        score = self.att_score(attn_hidden).squeeze(-1)
        score = score.masked_fill(~set_mask_bool, float("-inf"))
        weight = F.softmax(score, dim=1)

        pooled = torch.sum(x * weight.unsqueeze(-1), dim=1)
        logits = self.rho(self.drop(pooled))
        return F.softplus(logits) + self.min_positive

    def forward(self, set_features: Tensor, set_mask: Tensor) -> Tensor:
        """仕様どおり生パラメータ（raw）のみ返す。"""
        return self.forward_raw(set_features, set_mask)


# 後方互換のため公開名は維持する。
DeepSetsRegressor = DeepSetsAttnRegressor


def build_model(model_config: dict) -> DeepSetsAttnRegressor:
    """設定辞書から注意機構付きDeepSets回帰器を構築する。"""
    required = ["input_dim", "output_dim"]
    missing = [key for key in required if key not in model_config]
    if missing:
        raise ValueError(f"model_config に必須キーが不足しています: {missing}")

    return DeepSetsAttnRegressor(
        input_dim=int(model_config["input_dim"]),
        output_dim=int(model_config["output_dim"]),
        phi_hidden_dim=int(model_config.get("phi_hidden_dim", 128)),
        rho_hidden_dim=int(model_config.get("rho_hidden_dim", 128)),
        attn_hidden_dim=(
            int(model_config["attn_hidden_dim"])
            if model_config.get("attn_hidden_dim") is not None
            else None
        ),
        dropout=float(model_config.get("dropout", 0.0)),
        min_positive=float(model_config.get("min_positive", 1e-8)),
    )
