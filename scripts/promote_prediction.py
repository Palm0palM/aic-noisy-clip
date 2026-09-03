"""Promote one recent local predict result into the shared submission archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.run_registry import (
    format_recent_runs,
    load_run_records,
    resolve_recorded_path,
)
from src.utils.submission_registry import promote_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--latest", action="store_true", help="Select the latest valid predict result.")
    parser.add_argument("--index", type=int, default=None, help="Select the displayed one-based row.")
    parser.add_argument("--developer", default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    predicts = [
        record
        for record in load_run_records()
        if record["event_type"] == "predict"
        and record["status"] == "success"
        and record["artifact_path"]
        and resolve_recorded_path(record["artifact_path"]).is_file()
    ][-args.limit:][::-1]
    if not predicts:
        raise SystemExit("没有找到可用的本地 predict 结果。请先运行 src.predict。")
    print(format_recent_runs(list(reversed(predicts)), limit=len(predicts)))

    if args.latest:
        selected_index = 1
    elif args.index is not None:
        selected_index = args.index
    else:
        selected_index = int(input("请选择要加入 submission 的序号：").strip())
    if not 1 <= selected_index <= len(predicts):
        raise SystemExit(f"序号超出范围：{selected_index}")

    selected = predicts[selected_index - 1]
    record = promote_prediction(
        resolve_recorded_path(selected["artifact_path"]),
        source_run_id=selected["run_id"],
        expected_row_count=int(selected["num_predictions"]),
        created_by=args.developer,
        notes=args.notes,
    )
    print(f"已创建候选：submission/{record['candidate_path']}")
    print(f"submission_id：{record['submission_id']}")


if __name__ == "__main__":
    main()
