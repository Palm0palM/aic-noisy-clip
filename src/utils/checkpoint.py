"""Small checkpoints that omit the immutable official CLIP weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models import FrozenCLIPLinearClassifier


def save_checkpoint(
    path: Path,
    *,
    model: FrozenCLIPLinearClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_accuracy: float,
    model_name: str,
) -> None:
    """Save head and training state; fixed official CLIP weights stay external."""

    torch.save(
        {
            "format_version": 1,
            "architecture": "OpenAI CLIP ViT-B/32 frozen image encoder + linear classifier",
            "model_name": model_name,
            "feature_dim": model.feature_dim,
            "num_classes": model.num_classes,
            "classifier_state_dict": model.classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": int(epoch),
            "best_validation_accuracy": float(best_validation_accuracy),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: FrozenCLIPLinearClassifier,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Validate checkpoint metadata and restore the classifier and optional state."""

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise RuntimeError("Unsupported checkpoint format version.")
    if int(checkpoint["feature_dim"]) != model.feature_dim:
        raise RuntimeError("Checkpoint feature dimension does not match the model.")
    if int(checkpoint["num_classes"]) != model.num_classes:
        raise RuntimeError("Checkpoint class count does not match the model.")
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    model.assert_baseline_trainability()
    return checkpoint

