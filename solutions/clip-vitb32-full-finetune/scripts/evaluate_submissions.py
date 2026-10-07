#!/usr/bin/env python3
"""Evaluate one or more AIC submission CSVs against organizer ground truth."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ENCODINGS = ("utf-8-sig", "gb18030", "utf-16")


def decode_csv(data: bytes) -> list[list[str]]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
            rows = list(csv.reader(io.StringIO(text)))
            if rows and len(rows[0]) >= 2:
                return rows
        except (UnicodeDecodeError, csv.Error) as exc:
            last_error = exc
    raise ValueError(f"Unable to decode CSV: {last_error}")


def normalize_id(value: str) -> str:
    return Path(value.strip().replace("\\", "/")).name


def normalize_label(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return f"{int(value):04d}"
    return value


def rows_to_mapping(rows: list[list[str]], source: str) -> dict[str, str]:
    result: dict[str, str] = {}
    duplicates: list[str] = []
    first_cell = rows[0][0].strip().lower() if rows and rows[0] else ""
    has_header = not first_cell.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    for row in rows[1 if has_header else 0 :]:
        if len(row) < 2 or not row[0].strip():
            continue
        image_id = normalize_id(row[0])
        label = normalize_label(row[1])
        if image_id in result:
            duplicates.append(image_id)
        result[image_id] = label
    if duplicates:
        raise ValueError(f"{source}: duplicate image ids, e.g. {duplicates[:5]}")
    return result


def load_csv_mapping(path: Path) -> dict[str, str]:
    return rows_to_mapping(decode_csv(path.read_bytes()), str(path))


def load_zip_mapping(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected exactly one CSV, found {names}")
        return rows_to_mapping(decode_csv(archive.read(names[0])), f"{path}!{names[0]}")


def load_category_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = decode_csv(path.read_bytes())
    header = [cell.strip() for cell in rows[0]]
    metadata: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        item = dict(zip(header, padded))
        category_id = normalize_label(item.get("category_id", padded[0]))
        metadata[category_id] = item
    return metadata


def discover_predictions(inputs: Iterable[str], truth: Path) -> list[Path]:
    found: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found.update(path.rglob("pred_results.csv"))
            found.update(path.rglob("pred_results.zip"))
        elif path.is_file():
            found.add(path)
        else:
            raise FileNotFoundError(path)
    return sorted(path for path in found if path.resolve() != truth.resolve())


def evaluate(
    truth: dict[str, str],
    prediction: dict[str, str],
    metadata: dict[str, dict[str, str]],
) -> dict[str, object]:
    truth_ids = set(truth)
    prediction_ids = set(prediction)
    missing = sorted(truth_ids - prediction_ids)
    extra = sorted(prediction_ids - truth_ids)
    if missing or extra:
        raise ValueError(
            f"ID mismatch: missing={len(missing)} extra={len(extra)} "
            f"missing_examples={missing[:3]} extra_examples={extra[:3]}"
        )

    total = len(truth)
    correct = sum(prediction[key] == label for key, label in truth.items())
    class_total: Counter[str] = Counter(truth.values())
    class_correct: Counter[str] = Counter(
        label for key, label in truth.items() if prediction[key] == label
    )
    per_class_accuracy = {
        label: class_correct[label] / count for label, count in sorted(class_total.items())
    }
    macro_accuracy = sum(per_class_accuracy.values()) / len(per_class_accuracy)

    noise_totals: Counter[str] = Counter()
    noise_correct: Counter[str] = Counter()
    for key, label in truth.items():
        noise = metadata.get(label, {}).get("noise_level", "unknown") or "unknown"
        noise_totals[noise] += 1
        noise_correct[noise] += int(prediction[key] == label)

    confusion: Counter[tuple[str, str]] = Counter(
        (truth[key], prediction[key])
        for key in truth
        if truth[key] != prediction[key]
    )
    top_confusions = [
        {"truth": true_label, "prediction": pred_label, "count": count}
        for (true_label, pred_label), count in confusion.most_common(30)
    ]

    worst_classes = []
    for label in sorted(class_total, key=lambda x: (per_class_accuracy[x], -class_total[x], x))[:50]:
        item = metadata.get(label, {})
        worst_classes.append(
            {
                "category_id": label,
                "test_samples": class_total[label],
                "correct": class_correct[label],
                "accuracy": per_class_accuracy[label],
                "natural_language_label": item.get("natural_language_label", ""),
                "noise_level": item.get("noise_level", ""),
            }
        )

    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total,
        "macro_accuracy": macro_accuracy,
        "predicted_classes": len(set(prediction.values())),
        "truth_classes": len(class_total),
        "noise_level_accuracy": {
            noise: {
                "samples": noise_totals[noise],
                "correct": noise_correct[noise],
                "accuracy": noise_correct[noise] / noise_totals[noise],
            }
            for noise in sorted(noise_totals)
        },
        "top_confusions": top_confusions,
        "worst_classes": worst_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--category-labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    truth = load_csv_mapping(args.truth)
    metadata = load_category_metadata(args.category_labels)
    prediction_paths = discover_predictions(args.predictions, args.truth)
    if not prediction_paths:
        raise ValueError("No prediction CSV or ZIP files found")

    reports: dict[str, object] = {}
    ranking = []
    for path in prediction_paths:
        prediction = load_zip_mapping(path) if path.suffix.lower() == ".zip" else load_csv_mapping(path)
        report = evaluate(truth, prediction, metadata)
        reports[str(path)] = report
        ranking.append(
            {
                "path": str(path),
                "accuracy": report["accuracy"],
                "macro_accuracy": report["macro_accuracy"],
                "correct": report["correct"],
            }
        )

    ranking.sort(key=lambda item: (item["accuracy"], item["macro_accuracy"]), reverse=True)
    result = {"ranking": ranking, "reports": reports}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(json.dumps({"ranking": ranking}, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
