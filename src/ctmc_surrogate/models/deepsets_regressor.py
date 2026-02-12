"""状態遷移系列を入力とするDeepSets系サロゲート回帰モデル。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..constants import DEFAULT_MIN_POSITIVE


class DeepSetsVarSetsAttnRegressor(nn.Module):
    """(state, delta_t, lengths) から正値の生パラメータを推定する回帰器。"""

    def __init__(
        self,
        num_categories: int,
        embedding_dim: int,
        output_dim: int,
        token_hidden1: int = 256,
        token_hidden2: int = 512,
        output_hidden1: int = 128,
        output_hidden2: int = 64,
        dropout: float = 0.0,
        input_is_one_based: bool = True,
        min_positive: float = DEFAULT_MIN_POSITIVE,
    ) -> None:
        super().__init__()
        self.input_is_one_based = bool(input_is_one_based)
        self.min_positive = float(min_positive)

        self.embedding = nn.Embedding(
            num_embeddings=int(num_categories) + 1,
            embedding_dim=int(embedding_dim),
            padding_idx=0,
        )

        token_input_dim = 2 * int(embedding_dim) + 1
        self.fc1 = nn.Linear(token_input_dim, int(token_hidden1))
        self.ln1 = nn.LayerNorm(int(token_hidden1))
        self.drop1 = nn.Dropout(float(dropout))

        self.fc2 = nn.Linear(int(token_hidden1), int(token_hidden2))
        self.ln2 = nn.LayerNorm(int(token_hidden2))
        self.drop2 = nn.Dropout(float(dropout))

        self.att_fc = nn.Linear(int(token_hidden2), int(token_hidden2))
        self.att_score = nn.Linear(int(token_hidden2), 1)

        self.out_fc1 = nn.Linear(int(token_hidden2), int(output_hidden1))
        self.out_ln1 = nn.LayerNorm(int(output_hidden1))
        self.out_drop1 = nn.Dropout(float(dropout))

        self.out_fc2 = nn.Linear(int(output_hidden1), int(output_hidden2))
        self.out_ln2 = nn.LayerNorm(int(output_hidden2))
        self.out_drop2 = nn.Dropout(float(dropout))

        self.out_fc3 = nn.Linear(int(output_hidden2), int(output_dim))

    def _normalize_indices(self, state: Tensor) -> Tensor:
        if self.input_is_one_based:
            return torch.where(state > 0, state, torch.zeros_like(state))
        return torch.where(state >= 0, state + 1, torch.zeros_like(state))

    def forward(self, state: Tensor, delta_t: Tensor, lengths: Tensor) -> Tensor:
        """正値制約付きの生パラメータ（raw）のみ返す。"""
        if state.ndim != 3 or state.shape[1] != 2:
            raise ValueError("state は shape (B, 2, L) である必要があります。")
        if delta_t.ndim != 2:
            raise ValueError("delta_t は shape (B, L) である必要があります。")
        if lengths.ndim != 1:
            raise ValueError("lengths は shape (B,) である必要があります。")

        bsz, _, seq_len = state.shape
        if delta_t.shape != (bsz, seq_len):
            raise ValueError("delta_t の shape は state と整合する (B, L) である必要があります。")
        if lengths.shape[0] != bsz:
            raise ValueError("lengths のバッチ次元は state と一致する必要があります。")
        if torch.any(lengths < 1):
            raise ValueError("lengths の各要素は 1 以上である必要があります。")
        if int(lengths.max().item()) > seq_len:
            raise ValueError("lengths の最大値が state の系列長 L を超えています。")

        norm_idx = self._normalize_indices(state.long())
        pre_emb = self.embedding(norm_idx[:, 0, :])
        post_emb = self.embedding(norm_idx[:, 1, :])

        dt = delta_t.to(dtype=pre_emb.dtype).unsqueeze(-1)
        x = torch.cat([pre_emb, post_emb, dt], dim=-1)

        x = self.fc1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.drop2(x)

        attn_input = torch.tanh(self.att_fc(x))
        score = self.att_score(attn_input).squeeze(-1)

        positions = torch.arange(seq_len, device=lengths.device).unsqueeze(0)
        key_padding_mask = positions >= lengths.unsqueeze(1)
        score = score.masked_fill(key_padding_mask, float("-inf"))
        weight = F.softmax(score, dim=1)

        pooled = torch.sum(x * weight.unsqueeze(-1), dim=1)

        h = self.out_fc1(pooled)
        h = self.out_ln1(h)
        h = F.gelu(h)
        h = self.out_drop1(h)

        h = self.out_fc2(h)
        h = self.out_ln2(h)
        h = F.gelu(h)
        h = self.out_drop2(h)

        logits = self.out_fc3(h)
        return F.softplus(logits) + self.min_positive


def build_model(model_config: dict) -> DeepSetsVarSetsAttnRegressor:
    """設定辞書から新仕様の回帰モデルを構築する。"""
    required = ["num_categories", "embedding_dim", "output_dim"]
    missing = [key for key in required if key not in model_config]
    if missing:
        raise ValueError(f"model_config に必須キーが不足しています: {missing}")

    return DeepSetsVarSetsAttnRegressor(
        num_categories=int(model_config["num_categories"]),
        embedding_dim=int(model_config["embedding_dim"]),
        token_hidden1=int(model_config.get("token_hidden1", 256)),
        token_hidden2=int(model_config.get("token_hidden2", 512)),
        output_hidden1=int(model_config.get("output_hidden1", 128)),
        output_hidden2=int(model_config.get("output_hidden2", 64)),
        dropout=float(model_config.get("dropout", 0.0)),
        input_is_one_based=bool(model_config.get("input_is_one_based", True)),
        output_dim=int(model_config["output_dim"]),
        min_positive=float(model_config.get("min_positive", DEFAULT_MIN_POSITIVE)),
    )
