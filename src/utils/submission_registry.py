"""Shared candidate promotion, captain selection, and leaderboard records."""

from __future__ import annotations

import csv
import getpass
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import PROJECT_ROOT
from .run_registry import portable_path


SUBMISSION_FIELDS = (
    "submission_id",
    "created_at",
    "created_by",
    "source_run_id",
    "source_prediction_path",
    "candidate_path",
    "sha256",
    "row_count",
    "status",
    "selected_at",
    "score",
    "rank",
    "scored_at",
    "notes",
)


def submission_root() -> Path:
    return PROJECT_ROOT / "submission"


def submission_records_path(root: Path | None = None) -> Path:
    return (root or submission_root()) / "records.csv"


def inspect_prediction_csv(path: Path) -> tuple[int, str]:
    """Validate basic submission syntax and return row count and SHA-256."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError("Prediction CSV is empty.")
    if any(len(row) != 2 for row in rows):
        raise ValueError("Every prediction row must have exactly two columns.")
    filenames = [row[0] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Prediction CSV contains duplicate filenames.")
    for _filename, class_id in rows:
        if len(class_id) != 4 or not class_id.isdigit():
            raise ValueError(f"Invalid four-digit class ID: {class_id!r}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return len(rows), digest


def _load_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SUBMISSION_FIELDS:
            raise RuntimeError(f"Unexpected submission registry schema: {path}")
        return [dict(row) for row in reader]


def load_submission_records(root: Path | None = None) -> list[dict[str, str]]:
    """Read all shared submission candidate records."""

    return _load_records(submission_records_path(root))


def _write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def promote_prediction(
    source_path: Path,
    *,
    source_run_id: str,
    expected_row_count: int | None = None,
    created_by: str | None = None,
    notes: str = "",
    root: Path | None = None,
) -> dict[str, str]:
    """Copy one validated local prediction into the shared candidate archive."""

    root = root or submission_root()
    row_count, digest = inspect_prediction_csv(source_path)
    if expected_row_count is not None and row_count != expected_row_count:
        raise ValueError(
            f"Prediction row count {row_count} differs from its run record "
            f"({expected_row_count})."
        )
    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    safe_run_id = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in source_run_id
    ).strip("_") or "run"
    submission_id = f"{timestamp}_{digest[:8]}"
    candidate_name = f"candidate_{timestamp}_{safe_run_id}_{digest[:8]}.csv"
    candidate_path = root / "candidates" / candidate_name
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_path.exists():
        raise FileExistsError(f"Candidate already exists: {candidate_path}")
    shutil.copy2(source_path, candidate_path)

    record = {field: "" for field in SUBMISSION_FIELDS}
    record.update(
        {
            "submission_id": submission_id,
            "created_at": now.isoformat(timespec="seconds"),
            "created_by": created_by or getpass.getuser(),
            "source_run_id": source_run_id,
            "source_prediction_path": portable_path(source_path),
            "candidate_path": candidate_path.relative_to(root).as_posix(),
            "sha256": digest,
            "row_count": str(row_count),
            "status": "candidate",
            "notes": notes,
        }
    )
    records_path = submission_records_path(root)
    records = _load_records(records_path)
    records.append(record)
    _write_records(records_path, records)
    return record


def select_submission(
    submission_id: str, *, root: Path | None = None
) -> tuple[dict[str, str], Path]:
    """Select one candidate and copy it to the required pred_results.csv name."""

    root = root or submission_root()
    records_path = submission_records_path(root)
    records = _load_records(records_path)
    matches = [record for record in records if record["submission_id"] == submission_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one submission record for {submission_id!r}.")

    selected = matches[0]
    candidate_path = root / selected["candidate_path"]
    row_count, digest = inspect_prediction_csv(candidate_path)
    if digest != selected["sha256"] or str(row_count) != selected["row_count"]:
        raise RuntimeError("Candidate file no longer matches its recorded hash or row count.")

    selected_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for record in records:
        if record["status"] == "selected":
            record["status"] = "candidate"
            record["selected_at"] = ""
        if record["submission_id"] == submission_id:
            record["status"] = "selected"
            record["selected_at"] = selected_at

    final_path = root / "pred_results.csv"
    shutil.copy2(candidate_path, final_path)
    _write_records(records_path, records)
    return selected, final_path


def record_platform_result(
    submission_id: str,
    *,
    score: str,
    rank: str,
    notes: str = "",
    root: Path | None = None,
) -> dict[str, str]:
    """Attach the manually observed platform score and rank to a candidate."""

    if not score.strip():
        raise ValueError("score cannot be empty.")
    if not rank.strip():
        raise ValueError("rank cannot be empty.")
    root = root or submission_root()
    records_path = submission_records_path(root)
    records = _load_records(records_path)
    matches = [record for record in records if record["submission_id"] == submission_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one submission record for {submission_id!r}.")
    record = matches[0]
    if record["status"] not in {"selected", "scored"}:
        raise RuntimeError("Platform results can only be attached to a selected candidate.")
    record["status"] = "scored"
    record["score"] = score.strip()
    record["rank"] = rank.strip()
    record["scored_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if notes.strip():
        record["notes"] = notes.strip()
    _write_records(records_path, records)
    return record


def format_submission_records(
    records: Iterable[dict[str, str]], *, limit: int = 20
) -> str:
    """Render recent shared candidates for interactive selection."""

    recent = list(records)[-limit:][::-1]
    if not recent:
        return "暂无 submission 候选记录。"
    lines = [
        " No. | Created             | Status    | Developer    | Score      | Rank   | Source run",
        "-----+---------------------+-----------+--------------+------------+--------+------------------------------",
    ]
    for index, record in enumerate(recent, start=1):
        created_at = record["created_at"].replace("T", " ")[:19]
        lines.append(
            f"{index:>4} | {created_at:<19} | {record['status']:<9} | "
            f"{record['created_by'][:12]:<12} | {(record['score'] or '-'):<10} | "
            f"{(record['rank'] or '-'):<6} | {record['source_run_id'][:30]}"
        )
    return "\n".join(lines)
