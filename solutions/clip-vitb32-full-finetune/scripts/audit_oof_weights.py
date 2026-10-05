"""Rebuild the reliability weights with GPT's stricter recipe, and audit what is
actually left to fix once the existing drop list is taken into account.

Conditions for a half-weight:
  * the two weak views both disagree with the manifest label, AND both point at
    the same alternative class, AND the smaller of the two margins is >= 0.5;
  * the image is not already excluded by artifacts/dedup_drop.npy.

Also reports the deduplicated near-duplicate conflict statistics (the raw pair
count double counts pairs that share several hash bands) and how many images sit
in majority-conflict clusters that are still trained with their original label.

    python scripts/audit_oof_weights.py --a ... --b ... --folds ... \
        --dedup-drop D:/AIC_calib/dedup_drop.npy --dedup-labels D:/AIC_calib/dedup_labels.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

MANIFEST = Path(r"D:\AIC-Robust-CLIP-复赛\artifacts\train_manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--dedup-drop", required=True)
    parser.add_argument("--dedup-labels", required=True)
    parser.add_argument("--out-weights", default=r"D:/AIC_calib/oof_weights_v2.npy")
    parser.add_argument("--out-report", default=r"D:/AIC_calib/oof_weights_v2_report.json")
    args = parser.parse_args()

    folds = np.array(json.loads(Path(args.folds).read_text(encoding="utf-8"))["fold"], dtype=np.int64)
    a = np.load(args.a, allow_pickle=False)
    b = np.load(args.b, allow_pickle=False)
    labels = a["labels"].astype(np.int64)
    n = len(labels)
    assert np.array_equal(labels, b["labels"])

    use_a = folds == 1
    summaries = np.zeros((2, n, 4), dtype=np.float32)
    summaries[:, use_a, :] = a["summaries"][:, use_a, :].astype(np.float32)
    summaries[:, ~use_a, :] = b["summaries"][:, ~use_a, :].astype(np.float32)

    top1 = summaries[:, :, 0].astype(np.int64)
    margin = summaries[:, :, 3]

    both_disagree = (top1[0] != labels) & (top1[1] != labels)
    same_alternative = top1[0] == top1[1]
    min_margin = np.minimum(margin[0], margin[1])
    selected = both_disagree & same_alternative & (min_margin >= 0.5)

    drop = set(int(x) for x in np.load(args.dedup_drop))
    already_dropped = np.array([i in drop for i in range(n)])
    newly_affected = selected & ~already_dropped

    dedup_labels = np.load(args.dedup_labels)
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    relabelled = np.array([int(r["label"]) for r in rows]) != dedup_labels

    report = {
        "images": int(n),
        "candidates_both_views_same_alternative_margin0.5": int(selected.sum()),
        "already_in_drop_list": int((selected & already_dropped).sum()),
        "newly_affected_by_half_weight": int(newly_affected.sum()),
        "newly_affected_share": float(newly_affected.mean()),
        "newly_affected_by_class_size": {},
        "relabelled_but_still_training": int((relabelled & ~already_dropped).sum()),
        "relabelled_total": int(relabelled.sum()),
    }
    counts = Counter(int(x) for x in labels)
    sizes = np.array([counts[int(l)] for l in labels])
    for lo, hi in [(0, 50), (50, 100), (100, 150), (150, 250), (250, 10 ** 6)]:
        mask = (sizes >= lo) & (sizes < hi)
        report["newly_affected_by_class_size"][f"{lo}-{hi if hi < 10**6 else '+'}"] = {
            "images": int(mask.sum()),
            "half_weight": int((newly_affected & mask).sum()),
        }

    weights = np.ones(n, dtype=np.float32)
    weights[newly_affected] = 0.5
    np.save(args.out_weights, weights)
    Path(args.out_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
