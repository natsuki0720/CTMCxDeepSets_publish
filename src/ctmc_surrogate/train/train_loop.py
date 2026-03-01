"""Training loop implementation for CTMC surrogate models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    patience: int = 10
    min_delta: float = 0.0


@dataclass(frozen=True)
class TrainLoopConfig:
    """Training-loop configuration."""

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


@dataclass(frozen=True)
class TrainResult:
    """Training result summary."""

    best_epoch: int
    best_val_loss: float
    train_loss_history: list[float]
    val_loss_history: list[float]
    stopped_early: bool


class CustomLoss(nn.Module):
    """Loss function that minimizes MAE after reciprocal transform."""

    def __init__(self, epsilon: float = 1e-12) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Return reciprocal MAE with a small epsilon to prevent division by zero."""
        y_pred_inverse = 1.0 / (outputs + self.epsilon)
        y_true_inverse = 1.0 / (targets + self.epsilon)
        return torch.abs(y_pred_inverse - y_true_inverse).mean()


def _move_batch_to_device(
    batch: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state, delta_t, target, lengths = batch
    return (
        state.to(device),
        delta_t.to(device),
        target.to(device),
        lengths.to(device),
    )


def _build_optimizer(model: nn.Module, config: TrainLoopConfig) -> Optimizer:
    return Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        state, delta_t, target, lengths = _move_batch_to_device(batch, device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        pred = model(state, delta_t, lengths)
        loss = loss_fn(pred, target)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = int(state.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("No samples were provided from the DataLoader.")

    return total_loss / total_samples


def fit(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    valid_loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    config: TrainLoopConfig,
    loss_fn: nn.Module | None = None,
) -> TrainResult:
    """Run training and validation, then restore and return best weights with early stopping."""
    if int(config.epochs) < 1:
        raise ValueError("epochs must be at least 1.")
    if int(config.early_stopping.patience) < 1:
        raise ValueError("early_stopping.patience must be at least 1.")

    device = torch.device(config.device)
    model.to(device)

    criterion = loss_fn if loss_fn is not None else CustomLoss()
    optimizer = _build_optimizer(model, config)

    best_state_dict: dict[str, Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve_epochs = 0

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    for epoch in range(int(config.epochs)):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss = _run_epoch(model, valid_loader, criterion, device, optimizer=None)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        improved = val_loss < (best_val_loss - float(config.early_stopping.min_delta))
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= int(config.early_stopping.patience):
            break

    if best_state_dict is None:
        raise RuntimeError("No valid best model was recorded during training.")

    model.load_state_dict(best_state_dict)
    model.to(device)

    stopped_early = len(train_loss_history) < int(config.epochs)
    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        stopped_early=stopped_early,
    )


def save_run_artifacts(
    run_dir: str | Path,
    model: nn.Module,
    model_config: dict[str, Any],
    metrics: TrainResult,
) -> None:
    """Save training artifacts to run_dir.

    Output paths:
      - Model config: ``run_dir/model_config.yaml``
      - Training metrics: ``run_dir/metrics.json``
      - Model weights: ``run_dir/weights/best_model.pt``

    The ``weights`` directory is created automatically if it does not exist.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    weights_dir = run_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_config_path = run_path / "model_config.yaml"
    best_model_weights_path = weights_dir / "best_model.pt"
    metrics_path = run_path / "metrics.json"

    _write_yaml_like_dict(model_config_path, model_config)
    torch.save(model.state_dict(), best_model_weights_path)

    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(asdict(metrics), fp, ensure_ascii=False, indent=2)


def _write_yaml_like_dict(path: Path, value: dict[str, Any]) -> None:
    """Save a dictionary in a minimal YAML-compatible format to avoid adding dependencies."""
    lines = []
    for key, val in value.items():
        if isinstance(val, bool):
            rendered = "true" if val else "false"
        elif isinstance(val, (int, float)):
            rendered = str(val)
        else:
            rendered = f'"{val}"'
        lines.append(f"{key}: {rendered}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
