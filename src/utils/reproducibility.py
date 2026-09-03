"""Reproducibility helpers."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch and select deterministic cuDNN behavior."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

