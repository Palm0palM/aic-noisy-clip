"""Captain workflow: select a candidate, stage pred_results.csv, then record score."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.submission_registry import (
    format_submission_records,
    load_submission_records,
    record_platform_result,
    select_submission,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--index", type=int, default=None, help="Select the displayed one-based row.")
    parser.add_argument("--submission-id", default=None)
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Create pred_results.csv and exit without waiting for platform results.",
    )
    args = parser.parse_args()

    records = load_submission_records()
    displayed = records[-args.limit:][::-1]
    if not displayed:
        raise SystemExit("submission/records.csv 中暂无候选。")
    print(format_submission_records(records, limit=args.limit))

    if args.submission_id:
        submission_id = args.submission_id
    else:
        index = args.index
        if index is None:
            index = int(input("请选择队长要提交的序号：").strip())
        if not 1 <= index <= len(displayed):
            raise SystemExit(f"序号超出范围：{index}")
        submission_id = displayed[index - 1]["submission_id"]

    selected, final_path = select_submission(submission_id)
    print(f"已生成平台提交文件：{final_path}")
    print(f"文件 SHA-256：{selected['sha256']}")
    if args.stage_only:
        return

    input("请上传 pred_results.csv；平台显示结果后按 Enter 开始录入……")
    score = input("请输入平台分数：")
    rank = input("请输入当前排名：")
    notes = input("备注（可留空）：")
    record = record_platform_result(
        submission_id,
        score=score,
        rank=rank,
        notes=notes,
    )
    print(
        f"已记录：score={record['score']}，rank={record['rank']}，"
        "请提交 submission/records.csv 的变更供团队同步。"
    )


if __name__ == "__main__":
    main()

