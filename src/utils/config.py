"""YAML configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and validate the baseline's required fields."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping.")

    required = {
        "seed": config.get("seed"),
        "experiment.name": config.get("experiment", {}).get("name"),
        "data.train_dir": config.get("data", {}).get("train_dir"),
        "data.test_dir": config.get("data", {}).get("test_dir"),
        "data.split_manifest": config.get("data", {}).get("split_manifest"),
        "data.val_ratio": config.get("data", {}).get("val_ratio"),
        "model.name": config.get("model", {}).get("name"),
        "model.download_root": config.get("model", {}).get("download_root"),
        "training.batch_size": config.get("training", {}).get("batch_size"),
        "training.num_workers": config.get("training", {}).get("num_workers"),
        "training.epochs": config.get("training", {}).get("epochs"),
        "training.learning_rate": config.get("training", {}).get("learning_rate"),
        "training.weight_decay": config.get("training", {}).get("weight_decay"),
        "training.amp": config.get("training", {}).get("amp"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
    if config["model"]["name"] != "ViT-B/32":
        raise ValueError("This baseline only permits OpenAI CLIP ViT-B/32.")
    return config


def project_path(value: str | Path) -> Path:
    """Resolve a configured project-relative path consistently on all platforms."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

