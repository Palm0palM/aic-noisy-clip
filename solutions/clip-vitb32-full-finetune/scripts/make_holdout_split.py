"""Carve a clean, stratified hold-out split for ablation runs.

Every model produced so far was trained on the complete manifest, so no clean
signal existed for comparing recipes. This script makes a new 95/5 split: the 5%
hold-out is excluded from training in every ablation run and used only for
model comparison. Stratified per class with a fixed seed; classes with very few
images keep a minimum of 4 in training.

    python scripts/make_holdout_split.py --ratio 0.05 --seed 20260926 \
        --output artifacts/split95_seed20260926.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--min-train-per-class", type=int, default=4)
    parser.add_argument("--output", default="artifacts/split95_seed20260926.json")
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_class[int(row["label"])].append(i)

    rng = random.Random(args.seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        n_val = int(round(len(items) * args.ratio))
        n_val = max(0, min(n_val, len(items) - args.min_train_per_class))
        if len(items) <= 5:
            n_val = min(n_val, 1)
        val_idx.extend(items[:n_val])
        train_idx.extend(items[n_val:])

    train_idx.sort()
    val_idx.sort()
    payload = {"seed": args.seed, "ratio": args.ratio, "train": train_idx, "val": val_idx}
    Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({
        "train": len(train_idx), "val": len(val_idx),
        "classes_in_val": len({int(rows[i]['label']) for i in val_idx}),
        "output": args.output,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
