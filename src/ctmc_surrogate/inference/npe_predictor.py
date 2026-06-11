"""Load a trained NPE model from a run directory and expose a posterior API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ..features.sample_features import samples_to_model_input
from ..models import build_model
from ..models.posterior_utils import posterior_sample

__all__ = ["NPEPrediction", "NPEPredictor", "load_npe_predictor", "read_model_config"]


def _parse_scalar(raw: str):
    """Parse a value written by ``train_loop._write_yaml_like_dict``."""
    text = raw.strip()
    if text in ("true", "false"):
        return text == "true"
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def read_model_config(path: str | Path) -> dict:
    """Read the minimal ``model_config.yaml`` produced during training."""
    config: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        config[key.strip()] = _parse_scalar(value)
    return config


@dataclass
class NPEPrediction:
    """Point estimate and uncertainty derived from the NPE posterior."""

    z_mean: np.ndarray
    z_std: np.ndarray
    nu_mean: np.ndarray  # lifetime = exp(z)
    nu_std: np.ndarray
    rate_mean: np.ndarray  # q = exp(-z)
    rate_std: np.ndarray
    z_samples: np.ndarray | None = None


class NPEPredictor:
    """Wrap a trained NPE model with a convenient sample-in / posterior-out API."""

    def __init__(self, model: nn.Module, model_config: dict, device: torch.device | str = "cpu") -> None:
        if str(model_config.get("head", "point")) == "point":
            raise ValueError("NPEPredictor requires a gaussian/flow head, got head='point'.")
        self.model = model.to(device)
        self.model.eval()
        self.model_config = dict(model_config)
        self.device = torch.device(device)
        self.output_dim = int(model_config["output_dim"])

    @property
    def n_states(self) -> int:
        return self.output_dim + 1

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, device: torch.device | str = "cpu") -> "NPEPredictor":
        run_path = Path(run_dir)
        config_path = run_path / "model_config.yaml"
        weights_path = run_path / "weights" / "best_model.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"model_config.yaml not found in run dir: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"best_model.pt not found in run dir: {weights_path}")

        model_config = read_model_config(config_path)
        model = build_model(model_config)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return cls(model=model, model_config=model_config, device=device)

    def posterior(self, samples: np.ndarray | Tensor):
        """Return the conditional posterior ``q_phi(z | X)`` for one dataset."""
        state, delta_t, lengths = samples_to_model_input(samples, device=self.device)
        with torch.no_grad():
            return self.model.posterior(state, delta_t, lengths)

    def predict(
        self,
        samples: np.ndarray | Tensor,
        num_samples: int = 2000,
        return_samples: bool = False,
        seed: int | None = None,
    ) -> NPEPrediction:
        """Posterior point estimates in z / lifetime / rate space with marginal SDs.

        Lifetime (``nu``) and rate (``q``) moments are computed from samples
        because ``E[exp(z)] != exp(E[z])``.
        """
        posterior = self.posterior(samples)
        draws = posterior_sample(posterior, num_samples, seed=seed).squeeze(1).cpu().numpy()  # (S, D)
        nu = np.exp(draws)
        rate = np.exp(-draws)
        return NPEPrediction(
            z_mean=draws.mean(axis=0),
            z_std=draws.std(axis=0),
            nu_mean=nu.mean(axis=0),
            nu_std=nu.std(axis=0),
            rate_mean=rate.mean(axis=0),
            rate_std=rate.std(axis=0),
            z_samples=draws if return_samples else None,
        )


def load_npe_predictor(run_dir: str | Path, device: torch.device | str = "cpu") -> NPEPredictor:
    """Convenience wrapper for :meth:`NPEPredictor.from_run_dir`."""
    return NPEPredictor.from_run_dir(run_dir, device=device)
