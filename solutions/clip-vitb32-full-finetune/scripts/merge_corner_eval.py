"""Local-only: incremental view evaluation on top of recipe B (best so far).

Recipe B = six centre views + the four 1.14 five-crop corners, 77.3689 on the
test set. This asks whether the second round of corner views (their horizontal
flips, and the same corners at 1.28) adds anything on top of B, and includes the
weight-matched control GPT asked for: adding four duplicated corner views keeps
the view count and the zoom balance identical while adding no new information,
so a gain from real views over that control is spatial information, not weight.

    python scripts/merge_corner_eval.py --npz D:/AIC_calib/views512_v16ema.npz \
        --extra D:/AIC_calib/corner2_v16ema.npz
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

TRUTH = Path(r"C:\Users\CSL\Desktop\AIC文档\submission.csv")
BASE6 = [
    "center:512:1.0", "flip:512:1.0", "center:512:1.14",
    "flip:512:1.14", "center:512:1.28", "center:512:1.4",
]
C4 = ["tl:512:1.14", "tr:512:1.14", "bl:512:1.14", "br:512:1.14"]
C4F = ["tl_flip:512:1.14", "tr_flip:512:1.14", "bl_flip:512:1.14", "br_flip:512:1.14"]
C4_128 = ["tl:512:1.28", "tr:512:1.28", "bl:512:1.28", "br:512:1.28"]
B = BASE6 + C4


def load_truth() -> dict[str, int]:
    for encoding in ("gb18030", "utf-8-sig"):
        try:
            with TRUTH.open(encoding=encoding, newline="") as handle:
                rows = list(csv.reader(handle))
            mapping = {}
            for row in rows:
                if len(row) >= 2:
                    try:
                        mapping[row[0]] = int(row[1])
                    except ValueError:
                        pass
            return mapping
        except UnicodeDecodeError:
            continue
    raise SystemExit("cannot decode truth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--extra", required=True)
    args = parser.parse_args()

    first = np.load(args.npz, allow_pickle=False)
    second = np.load(args.extra, allow_pickle=False)
    files = [str(f) for f in first["files"]]
    assert files == [str(f) for f in second["files"]], "file order differs between dumps"
    probs: dict[str, np.ndarray] = {}
    for source in (first, second):
        for key in source.files:
            if key != "files":
                probs[key] = source[key]
    truth = load_truth()
    labels = np.array([truth[f] for f in files], dtype=np.int64)

    def score(views: list[str]) -> tuple[float, np.ndarray]:
        stacked = np.mean(np.stack([probs[v].astype(np.float32) for v in views], axis=0), axis=0)
        pred = stacked.argmax(1)
        return float((pred == labels).mean()), pred

    base_acc, base_pred = score(B)
    print(f"B (10 views)                        {base_acc * 100:.4f}")
    candidates = {
        "B + corner4_flip": B + C4F,
        "B + corner4@1.28": B + C4_128,
        "B + both": B + C4F + C4_128,
        "CTRL B + 4x dup corner": B + ["tl:512:1.14"] * 4,
        "CTRL B + 4x dup centre": B + ["center:512:1.14"] * 4,
        "swap: replace C4 by flip": BASE6 + C4F,
        "swap: replace C4 by 1.28": BASE6 + C4_128,
        "B + fullpad_gray": B + ["fullpad_gray:512:1.0"],
    }
    for name, views in candidates.items():
        acc, pred = score(views)
        gained = int(((pred == labels) & (base_pred != labels)).sum())
        lost = int(((pred != labels) & (base_pred == labels)).sum())
        sigma = math.sqrt(max(gained + lost, 1))
        print(
            f"{name:35s} {acc * 100:.4f}  delta {(acc - base_acc) * 100:+.4f}pp  "
            f"+{gained}/-{lost}  z {(gained - lost) / sigma:+.2f}"
        )


if __name__ == "__main__":
    main()
