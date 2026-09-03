"""Local CSV registry for training, validation, prediction, and smoke results."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT


RUN_REGISTRY_FIELDS = (
    "recorded_at",
    "event_type",
    "run_id",
    "status",
    "config_path",
    "checkpoint_path",
    "artifact_path",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "num_samples",
    "num_predictions",
    "duration_seconds",
    "gpu_peak_allocated_gib",
    "notes",
)


def run_registry_path() -> Path:
    """Return the developer-local run registry path."""

    return PROJECT_ROOT / "runs" / "run_registry.csv"


def portable_path(value: str | Path | None) -> str:
    """Prefer a project-relative POSIX path for portable CSV records."""

    if value is None or str(value) == "":
        return ""
    path = Path(value).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_recorded_path(value: str) -> Path:
    """Resolve a path stored in a registry row."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def append_run_record(
    *,
    event_type: str,
    run_id: str,
    status: str = "success",
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    train_loss: float | None = None,
    train_accuracy: float | None = None,
    validation_loss: float | None = None,
    validation_accuracy: float | None = None,
    num_samples: int | None = None,
    num_predictions: int | None = None,
    duration_seconds: float | None = None,
    gpu_peak_allocated_gib: float | None = None,
    notes: str = "",
    registry_path: Path | None = None,
) -> dict[str, str]:
    """Append one successful local result using a stable CSV schema."""

    if event_type not in {"train", "smoke_test", "validation", "predict"}:
        raise ValueError(f"Unsupported run event type: {event_type!r}")
    path = registry_path or run_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event_type": event_type,
        "run_id": run_id,
        "status": status,
        "config_path": portable_path(config_path),
        "checkpoint_path": portable_path(checkpoint_path),
        "artifact_path": portable_path(artifact_path),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "num_samples": num_samples,
        "num_predictions": num_predictions,
        "duration_seconds": duration_seconds,
        "gpu_peak_allocated_gib": gpu_peak_allocated_gib,
        "notes": notes,
    }
    normalized = {
        key: "" if row[key] is None else str(row[key]) for key in RUN_REGISTRY_FIELDS
    }

    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if tuple(header) != RUN_REGISTRY_FIELDS:
            raise RuntimeError(f"Unexpected run registry schema: {path}")
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_REGISTRY_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(normalized)
    return normalized


def load_run_records(registry_path: Path | None = None) -> list[dict[str, str]]:
    """Read local run records in chronological file order."""

    path = registry_path or run_registry_path()
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RUN_REGISTRY_FIELDS:
            raise RuntimeError(f"Unexpected run registry schema: {path}")
        return [dict(row) for row in reader]


def _shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(1, width - 1)] + "…"


def _metric(value: str) -> str:
    if not value:
        return "-"
    return f"{float(value):.4f}"


def format_recent_runs(
    records: Iterable[dict[str, str]], *, limit: int = 10
) -> str:
    """Render recent run records as a compact Unicode table."""

    if limit <= 0:
        raise ValueError("limit must be positive.")
    recent = list(records)[-limit:][::-1]
    if not recent:
        return "暂无本地运行记录。"

    headers = ("No.", "Time", "Type", "Run", "Val Acc", "Val Loss", "Count", "Artifact")
    widths = (4, 19, 11, 28, 9, 10, 8, 36)
    rows: list[tuple[str, ...]] = []
    for index, record in enumerate(recent, start=1):
        recorded_at = record["recorded_at"].replace("T", " ")[:19]
        count = record["num_predictions"] or record["num_samples"] or "-"
        artifact = record["artifact_path"] or record["checkpoint_path"] or "-"
        rows.append(
            (
                str(index),
                recorded_at,
                record["event_type"],
                record["run_id"],
                _metric(record["validation_accuracy"]),
                _metric(record["validation_loss"]),
                count,
                artifact,
            )
        )

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [border]
    lines.append(
        "|"
        + "|".join(
            f" {_shorten(value, width):<{width}} "
            for value, width in zip(headers, widths)
        )
        + "|"
    )
    lines.append(border)
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                f" {_shorten(value, width):<{width}} "
                for value, width in zip(row, widths)
            )
            + "|"
        )
    lines.append(border)
    return "\n".join(lines)
