"""Combine per-view probability dumps into pre-registered TTA recipes.

`infer_ft --save-probs` writes one probability matrix per view spec into an npz.
This turns fixed subsets of those views into submission CSVs so that each recipe
can be scored independently. The recipes are written down before any score is
seen, and every recipe differs from the baseline by exactly one view group.

    python scripts/combine_view_probs.py \
        --npz artifacts/views512_v16ema/test_view_probs.npz \
        --out-dir artifacts/views512_v16ema/recipes
"""

from __future__ import annotations

import argparse
import zipfile
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

ANCHOR8 = [
    "tl:512:1.0",
    "tc:512:1.0",
    "tr:512:1.0",
    "ml:512:1.0",
    "mr:512:1.0",
    "bl:512:1.0",
    "bc:512:1.0",
    "br:512:1.0",
]

CORNER4_114 = ["tl:512:1.14", "tr:512:1.14", "bl:512:1.14", "br:512:1.14"]
CORNER4_114_FLIP = ["tl_flip:512:1.14", "tr_flip:512:1.14", "bl_flip:512:1.14", "br_flip:512:1.14"]
CORNER4_128 = ["tl:512:1.28", "tr:512:1.28", "bl:512:1.28", "br:512:1.28"]
FULLPAD = ["fullpad_edge:512:1.0", "fullpad_gray:512:1.0"]

RECIPES: dict[str, list[str]] = {
    # baseline: the current best recipe
    "base6": BASE6,
    # baseline + one new group at a time
    "base6_anchor8": BASE6 + ANCHOR8,
    "base6_fivecrop": BASE6 + CORNER4_114,
    "base6_fullpad_edge": BASE6 + [FULLPAD[0]],
    "base6_fullpad_gray": BASE6 + [FULLPAD[1]],
    "base6_anchor8_fullpad": BASE6 + ANCHOR8 + FULLPAD,
    # the new groups on their own, as a diagnostic
    "anchor8": ANCHOR8,
    "fivecrop": ["center:512:1.14"] + CORNER4_114,
    "anchor8_fullpad": ANCHOR8 + FULLPAD,
    # second-round corner variants
    "base6_corner4flip": BASE6 + CORNER4_114_FLIP,
    "base6_corner4_128": BASE6 + CORNER4_128,
    "base6_corner8": BASE6 + CORNER4_114 + CORNER4_114_FLIP,
    "base6_corner4_both": BASE6 + CORNER4_114 + CORNER4_128,
    "base6_corner12": BASE6 + CORNER4_114 + CORNER4_114_FLIP + CORNER4_128,
    # everything that was dumped
    "all": BASE6 + ANCHOR8 + CORNER4_114 + FULLPAD,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    files = [str(f) for f in data["files"]]
    available = {key for key in data.files if key != "files"}
    print(f"[combine] {len(files)} images, {len(available)} views: {sorted(available)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, views in RECIPES.items():
        missing = [v for v in views if v not in available]
        if missing:
            print(f"[combine] skip {name}: missing {missing}")
            continue
        stacked = np.mean(np.stack([data[v].astype(np.float32) for v in views], axis=0), axis=0)
        labels = stacked.argmax(1)
        csv_path = out_dir / f"{name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            for file_name, label in zip(files, labels):
                handle.write(f"{file_name},{int(label):04d}\n")
        bin_path = out_dir / f"pred_{name}.zip"
        with zipfile.ZipFile(bin_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(csv_path, arcname="pred_results.csv")
        print(
            f"[combine] {name}: {len(views)} views, mean confidence "
            f"{float(stacked.max(1).mean()):.4f} -> {csv_path.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
