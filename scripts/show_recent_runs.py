"""Pretty-print recent entries from the developer-local runs registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.run_registry import format_recent_runs, load_run_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--type",
        choices=("train", "smoke_test", "validation", "predict"),
        default=None,
        dest="event_type",
    )
    args = parser.parse_args()
    records = load_run_records()
    if args.event_type:
        records = [record for record in records if record["event_type"] == args.event_type]
    print(format_recent_runs(records, limit=args.limit))


if __name__ == "__main__":
    main()

