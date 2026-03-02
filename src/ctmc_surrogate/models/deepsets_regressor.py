"""DeepSets-based surrogate regressor for state-transition sequence inputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DeepSetsVarSetsAttnRegressor(nn.Module):
    """Regressor that estimates positive raw parameters from (state, delta_t, lengths)."""

    def __init__(
        self,
        num_categories: int,
        embedding_dim: int,
        output_dim: int,
        token_hidden1: int = 256,
        token_hidden2: int = 512,
        output_hidden1: int = 128,
        output_hidden2: int = 64,
        dropout: float = 0.2,
        input_is_one_based: bool = True,
    ) -> None:
        super().__init__()
        self.input_is_one_based = bool(input_is_one_based)
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
        """Return only raw parameters constrained to positive values."""
        if state.ndim != 3 or state.shape[1] != 2:
            raise ValueError("state must have shape (B, 2, L).")
        if delta_t.ndim != 2:
            raise ValueError("delta_t must have shape (B, L).")
        if lengths.ndim != 1:
            raise ValueError("lengths must have shape (B,).")

        device = state.device
        lengths = lengths.to(device)
        if delta_t.device != device:
            delta_t = delta_t.to(device)

        bsz, _, seq_len = state.shape
        if delta_t.shape != (bsz, seq_len):
            raise ValueError("delta_t shape must align with state as (B, L).")
        if lengths.shape[0] != bsz:
            raise ValueError("lengths batch dimension must match state.")
        if torch.any(lengths < 1):
            raise ValueError("Each element of lengths must be at least 1.")
        if int(lengths.max().item()) > seq_len:
            raise ValueError("The maximum value in lengths exceeds state sequence length L.")

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

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
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
        output = F.softplus(logits)
        if __debug__:
            assert torch.all(output >= 0), "softplus output is negative."
        return output


def build_model(model_config: dict) -> DeepSetsVarSetsAttnRegressor:
    """Build the new-spec regressor model from a config dictionary."""
    required = ["num_categories", "embedding_dim", "output_dim"]
    missing = [key for key in required if key not in model_config]
    if missing:
        raise ValueError(f"model_config is missing required keys: {missing}")

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
    )
