from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DeepSetsAttnRegressor(nn.Module):
    """可変長集合入力 -> raw(正)パラメータ。損失側で lifetime に変換する前提。"""

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
        H = phi_hidden_dim
        A = attn_hidden_dim if attn_hidden_dim is not None else H

        # element-wise encoder φ
        self.phi = nn.Sequential(
            nn.Linear(input_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
        )

        self.drop = nn.Dropout(dropout)

        # attention over set elements (mask-aware)
        self.att_fc = nn.Linear(H, A)
        self.att_score = nn.Linear(A, 1)

        # set-level regressor ρ
        self.rho = nn.Sequential(
            nn.Linear(H, rho_hidden_dim),
            nn.ReLU(),
            nn.Linear(rho_hidden_dim, output_dim),
        )

    def forward_raw(self, set_features: Tensor, set_mask: Tensor) -> Tensor:
        """
        set_features: (B, S, D)
        set_mask:     (B, S)  True for valid elements
        returns raw:  (B, output_dim), strictly positive
        """
        if set_features.ndim != 3:
            raise ValueError("set_features は [batch, set_size, feature_dim] を想定します。")
        if set_mask.ndim != 2:
            raise ValueError("set_mask は [batch, set_size] を想定します。")
        if set_features.shape[:2] != set_mask.shape:
            raise ValueError("set_features と set_mask の [batch, set_size] が一致しません。")

        x = self.phi(set_features)          # (B, S, H)
        x = self.drop(x)

        # attention logits
        a = torch.tanh(self.att_fc(x))      # (B, S, A)
        score = self.att_score(a).squeeze(-1)  # (B, S)

        # mask: invalid -> -inf
        score = score.masked_fill(~set_mask, float("-inf"))
        w = F.softmax(score, dim=1)         # (B, S)

        pooled = torch.sum(x * w.unsqueeze(-1), dim=1)  # (B, H)

        logits = self.rho(self.drop(pooled))            # (B, output_dim)
        raw = F.softplus(logits) + self.min_positive
        return raw

    def forward(self, set_features: Tensor, set_mask: Tensor) -> Tensor:
        # 仕様どおり raw のみ返す
        return self.forward_raw(set_features, set_mask)
