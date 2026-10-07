"""Local-only analysis of a per-view dump: paired stats plus a few extensions.

Everything here is scored against the local ground truth, which by the user's
rule may only be used to score finished inference output and pick a direction.

    python scripts/analyze_view_recipes.py --npz D:/AIC_calib/views512_v16ema.npz
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

TRUTH = Path(r"C:\Users\CSL\Desktop\AIC文档\submission.csv")

BASE6 = [
    "center:512:1.0",
    "flip:512:1.0",
    "center:512:1.14",
    "flip:512:1.14",
    "center:512:1.28",
    "center:512:1.4",
]
ANCHOR8 = ["tl:512:1.0", "tc:512:1.0", "tr:512:1.0", "ml:512:1.0", "mr:512:1.0", "bl:512:1.0", "bc:512:1.0", "br:512:1.0"]
C4_114 = ["tl:512:1.14", "tr:512:1.14", "bl:512:1.14", "br:512:1.14"]
FULLPAD_G = ["fullpad_gray:512:1.0"]


def load_truth() -> dict[str, int]:
    for encoding in ("gb18030", "utf-8-sig"):
        try:
            with TRUTH.open(encoding=encoding, newline="") as handle:
                rows = list(csv.reader(handle))
            mapping = {}
            for row in rows:
                if len(row) < 2:
                    continue
                try:
                    mapping[row[0]] = int(row[1])
                except ValueError:
                    continue  # header row
            return mapping
        except UnicodeDecodeError:
            continue
    raise SystemExit("cannot decode truth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    files = [str(f) for f in data["files"]]
    truth_map = load_truth()
    truth = np.array([truth_map[f] for f in files], dtype=np.int64)

    def predict(views: list[str]) -> np.ndarray:
        stacked = np.mean(np.stack([data[v].astype(np.float32) for v in views], axis=0), axis=0)
        return stacked.argmax(1)

    def score(views: list[str]) -> tuple[np.ndarray, float]:
        pred = predict(views)
        return pred, float((pred == truth).mean())

    base_pred, base_acc = score(BASE6)
    print(f"base6 (6 views)                acc {base_acc * 100:.4f}")

    print("\nsingle-view accuracies:")
    single = {}
    for key in sorted(data.files):
        if key == "files":
            continue
        pred = data[key].astype(np.float32).argmax(1)
        single[key] = float((pred == truth).mean())
    for key, acc in sorted(single.items(), key=lambda kv: -kv[1]):
        print(f"  {key:26s} {acc * 100:.4f}")

    # controls: is the gain from spatial coverage, or just from adding four more
    # views at ratio 1.14 (i.e. re-weighting the zoom distribution)?
    candidates: dict[str, list[str]] = {
        "CTRL base6 + 4x center@1.14": BASE6 + ["center:512:1.14"] * 4,
        "CTRL base6 + 4x flip@1.14": BASE6 + ["flip:512:1.14"] * 4,
        "CTRL base6 + 4x center@1.0": BASE6 + ["center:512:1.0"] * 4,
        "CTRL base6 + 4x center@1.4": BASE6 + ["center:512:1.4"] * 4,
        "base6 + corner4@1.14": BASE6 + C4_114,
        "base6 + corner4@1.14 + flip:1.14 x3": BASE6 + C4_114 + ["flip:512:1.14"] * 3,
        "base6 + corner4@1.14 x2": BASE6 + C4_114 + C4_114,
        "base6 + corner4@1.14 + fullpad_gray": BASE6 + C4_114 + FULLPAD_G,
        "base6 + anchor8@1.0 (8 extra)": BASE6 + ANCHOR8,
        "base6 + corners@1.14, anchors@1.0": BASE6 + C4_114 + ANCHOR8,
        "corners@1.14 + base6-halfweight": BASE6 + BASE6 + C4_114,
    }
    for name, views in candidates.items():
        pred, acc = score(views)
        gained = int(((pred == truth) & (base_pred != truth)).sum())
        lost = int(((pred != truth) & (base_pred == truth)).sum())
        net = gained - lost
        # McNemar-style normal approximation on the discordant pairs
        sigma = math.sqrt(gained + lost) if gained + lost else 1.0
        z = net / sigma
        print(
            f"{name:38s} acc {acc * 100:.4f}  delta {(acc - base_acc) * 100:+.4f}pp  "
            f"gained {gained} lost {lost}  z {z:+.2f}"
        )


if __name__ == "__main__":
    main()
