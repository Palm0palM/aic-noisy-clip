"""Small, explicit training and validation loops for the baseline."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models import FrozenCLIPLinearClassifier


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def make_dataloader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[Any]:
    """Build a deterministic DataLoader with Windows-safe options."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "worker_init_fn": _seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _move_batch(
    images: torch.Tensor, targets: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        images.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def train_one_epoch(
    model: FrozenCLIPLinearClassifier,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    """Train the linear classifier for one epoch while CLIP stays frozen."""

    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for images, targets, _paths in loader:
        images, targets = _move_batch(images, targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        loss_sum += float(loss.detach()) * batch_size
        correct += int(logits.detach().argmax(dim=1).eq(targets).sum())
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Training DataLoader yielded no samples.")
    model.assert_baseline_trainability()
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "num_samples": float(sample_count),
    }


@torch.inference_mode()
def evaluate(
    model: FrozenCLIPLinearClassifier,
    loader: DataLoader[Any],
    criterion: nn.Module,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    """Evaluate a labeled split without updating any parameter."""

    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for images, targets, _paths in loader:
        images, targets = _move_batch(images, targets, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        loss_sum += float(loss) * batch_size
        correct += int(logits.argmax(dim=1).eq(targets).sum())
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Validation DataLoader yielded no samples.")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "num_samples": float(sample_count),
    }

