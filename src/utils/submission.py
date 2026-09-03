"""Test-only inference and strict competition submission validation."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import format_class_id
from src.models import FrozenCLIPLinearClassifier


@torch.inference_mode()
def predict_unlabeled(
    model: FrozenCLIPLinearClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[str], list[int]]:
    """Run inference on an unlabeled loader without any parameter update."""

    model.eval()
    filenames: list[str] = []
    predictions: list[int] = []
    for images, batch_filenames in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits = model(images)
        filenames.extend(str(name) for name in batch_filenames)
        predictions.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())
    return filenames, predictions


def write_submission(
    output_path: Path,
    *,
    filenames: Sequence[str],
    predictions: Sequence[int],
    idx_to_class: dict[int, str],
    expected_filenames: Sequence[str],
) -> None:
    """Write filename-sorted rows and verify exact test-set coverage."""

    if len(filenames) != len(predictions):
        raise ValueError("Prediction count does not match filename count.")
    if len(filenames) != len(set(filenames)):
        raise ValueError("Predictions contain duplicate filenames.")
    if set(filenames) != set(expected_filenames):
        missing = sorted(set(expected_filenames) - set(filenames))
        unexpected = sorted(set(filenames) - set(expected_filenames))
        raise ValueError(
            f"Prediction coverage mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}."
        )

    rows: list[tuple[str, str]] = []
    for filename, prediction in zip(filenames, predictions):
        if prediction not in idx_to_class:
            raise ValueError(f"Predicted class index is invalid: {prediction}")
        rows.append((filename, format_class_id(idx_to_class[prediction])))
    rows.sort(key=lambda row: row[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)

    validate_submission(
        output_path,
        expected_filenames=expected_filenames,
        valid_class_ids=idx_to_class.values(),
    )


def validate_submission(
    path: Path,
    *,
    expected_filenames: Sequence[str],
    valid_class_ids: Sequence[str],
) -> None:
    """Reject missing, duplicate, unexpected, or malformed submission rows."""

    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if any(len(row) != 2 for row in rows):
        raise ValueError("Every submission row must have exactly two columns.")

    filenames = [row[0] for row in rows]
    labels = [row[1] for row in rows]
    expected = list(expected_filenames)
    if len(filenames) != len(expected):
        raise ValueError(
            f"Submission row count {len(filenames)} != test image count {len(expected)}."
        )
    if len(filenames) != len(set(filenames)):
        raise ValueError("Submission contains duplicate filenames.")
    if set(filenames) != set(expected):
        raise ValueError("Submission filenames do not exactly match the test images.")

    valid = {format_class_id(class_id) for class_id in valid_class_ids}
    for label in labels:
        if len(label) != 4 or not label.isdigit() or label not in valid:
            raise ValueError(f"Submission contains an invalid class ID: {label!r}")

