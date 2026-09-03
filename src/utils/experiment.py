"""Run-directory creation, logging, and artifact serialization."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import PROJECT_ROOT


def create_run_directory(experiment_name: str) -> Path:
    """Create a unique runs/<timestamp>_<experiment_name> directory."""

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_name).strip("_")
    if not safe_name:
        raise ValueError("experiment name must contain a usable character.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = PROJECT_ROOT / "runs" / f"{timestamp}_{safe_name}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def create_logger(run_dir: Path, name: str = "train") -> logging.Logger:
    """Log the same concise messages to console and train.log."""

    logger = logging.getLogger(f"aic.{name}.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Serialize configuration in a human-readable stable form."""

    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def save_json(path: Path, data: Any) -> None:
    """Serialize a JSON artifact."""

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_class_mapping(
    path: Path, class_to_idx: dict[str, int], idx_to_class: dict[int, str]
) -> None:
    """Persist both directions of the exact training label mapping."""

    save_json(
        path,
        {
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(key): value for key, value in idx_to_class.items()},
        },
    )


def load_class_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load and validate a saved bidirectional class mapping."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    class_to_idx = {
        str(key): int(value) for key, value in data["class_to_idx"].items()
    }
    idx_to_class = {
        int(key): str(value) for key, value in data["idx_to_class"].items()
    }
    expected_indices = set(range(len(class_to_idx)))
    if set(class_to_idx.values()) != expected_indices or set(idx_to_class) != expected_indices:
        raise RuntimeError("Class mapping indices must be contiguous from zero.")
    if any(idx_to_class[index] != name for name, index in class_to_idx.items()):
        raise RuntimeError("class_to_idx and idx_to_class are not exact inverses.")
    return class_to_idx, idx_to_class

