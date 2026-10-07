"""Local-only: turn a per-view probability dump into candidate submissions.

The dump comes from the training machine (it only ever holds test images, never
labels); the ground truth lives on this machine and is used here purely to score
finished inference output, exactly like an official submission score.

    python scripts/score_view_recipes.py --npz D:/AIC_calib/views512_v16ema.npz \
        --out-dir D:/AIC_calib/recipes_v16ema
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

BASE6 = [
    "center:512:1.0",
    "flip:512:1.0",
    "center:512:1.14",
    "flip:512:1.14",
    "center:512:1.28",
    "center:512:1.4",
]
ANCHOR8 = ["tl:512:1.0", "tc:512:1.0", "tr:512:1.0", "ml:512:1.0", "mr:512:1.0", "bl:512:1.0", "bc:512:1.0", "br:512:1.0"]
CORNER4 = ["tl:512:1.14", "tr:512:1.14", "bl:512:1.14", "br:512:1.14"]
CORNER4_10 = ["tl:512:1.0", "tr:512:1.0", "bl:512:1.0", "br:512:1.0"]
FULLPAD = ["fullpad_edge:512:1.0", "fullpad_gray:512:1.0"]

RECIPES: dict[str, list[str]] = {
    "base6": BASE6,
    "base6_anchor8": BASE6 + ANCHOR8,
    "base6_corner4_10": BASE6 + CORNER4_10,
    "base6_corner4_114": BASE6 + CORNER4,
    "base6_fullpad_edge": BASE6 + [FULLPAD[0]],
    "base6_fullpad_gray": BASE6 + [FULLPAD[1]],
    "base6_anchor8_fullpad": BASE6 + ANCHOR8 + FULLPAD,
    "anchor8": ANCHOR8,
    "anchor8_fullpad": ANCHOR8 + FULLPAD,
    "fivecrop": ["center:512:1.14"] + CORNER4,
    "all": BASE6 + ANCHOR8 + CORNER4 + FULLPAD,
    # exploratory: half weight on the new group
    "base6_anchor8_half": BASE6 + ANCHOR8 + BASE6 + ["tl:512:1.0", "tc:512:1.0", "tr:512:1.0", "ml:512:1.0",
                                                      "mr:512:1.0", "bl:512:1.0", "bc:512:1.0", "br:512:1.0"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    files = [str(f) for f in data["files"]]
    available = {key for key in data.files if key != "files"}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, views in RECIPES.items():
        missing = [v for v in views if v not in available]
        if missing:
            print(f"[recipes] skip {name}: {missing}")
            continue
        stacked = np.mean(np.stack([data[v].astype(np.float32) for v in views], axis=0), axis=0)
        labels = stacked.argmax(1)
        path = out_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for file_name, label in zip(files, labels):
                writer.writerow([file_name, f"{int(label):04d}"])
        print(f"[recipes] {name}: {len(views)} views, conf {float(stacked.max(1).mean()):.4f} -> {path.name}")


if __name__ == "__main__":
    main()
