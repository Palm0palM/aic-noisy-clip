"""Build the dual-candidate soft targets for the large-scale supervision rebuild.

For every training image the out-of-fold teacher (the fold model that never saw
it) has two weak views. An image is a candidate when both views disagree with the
manifest label, both point at the SAME alternative class, the smaller margin is
at least --min-margin, and the image is not already in the duplicate drop list.
Those images keep the original label with at least half of the target mass:

    r_i = min_v p_v(alt) / (p_v(alt) + p_v(label))
    q_i(label) = 1 - 0.5 r_i,  q_i(alt) = 0.5 r_i

then the usual label smoothing is applied on top. Everything else keeps the plain
smoothed one-hot. Writes a float16 (n, num_classes) matrix in manifest order,
which train_ft reads via train.soft_targets + soft_target_weight=1.0.

    python scripts/build_soft_targets.py --min-margin 0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

project = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-margin", type=float, default=0.1)
    parser.add_argument("--folds", default="artifacts/folds_2.json")
    parser.add_argument("--a", default="artifacts/oof_a_probs_views.npz")
    parser.add_argument("--b", default="artifacts/oof_b_probs_views.npz")
    parser.add_argument("--drop", default="artifacts/dedup_drop.npy")
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--smoothing", type=float, default=0.15)
    parser.add_argument("--output", default="artifacts/oof_soft_targets.npy")
    args = parser.parse_args()

    import csv

    with (project / args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    n = len(labels)
    num_classes = int(labels.max()) + 1

    folds = np.array(json.loads((project / args.folds).read_text(encoding="utf-8"))["fold"], dtype=np.int64)
    a = np.load(project / args.a, allow_pickle=False)
    b = np.load(project / args.b, allow_pickle=False)
    use_a = folds == 1
    summaries = np.zeros((2, n, 4), dtype=np.float32)
    summaries[:, use_a, :] = a["summaries"][:, use_a, :].astype(np.float32)
    summaries[:, ~use_a, :] = b["summaries"][:, ~use_a, :].astype(np.float32)

    top1 = summaries[:, :, 0].astype(np.int64)
    top1_prob = summaries[:, :, 1]
    label_prob = summaries[:, :, 2]
    margin = summaries[:, :, 3]

    drop = set(int(x) for x in np.load(project / args.drop))
    keep = np.array([i not in drop for i in range(n)])

    candidate = (
        (top1[0] != labels) & (top1[1] != labels)
        & (top1[0] == top1[1])
        & (np.minimum(margin[0], margin[1]) >= args.min_margin)
        & keep
    )
    alt = np.where(candidate, top1[0], labels)
    ratio = np.minimum(
        top1_prob[0] / np.maximum(top1_prob[0] + label_prob[0], 1e-6),
        top1_prob[1] / np.maximum(top1_prob[1] + label_prob[1], 1e-6),
    )
    ratio = np.where(candidate, ratio, 0.0).astype(np.float32)

    targets = np.full((n, num_classes), 0.0, dtype=np.float32)
    rows_idx = np.arange(n)
    targets[rows_idx, labels] = 1.0 - 0.5 * ratio
    targets[rows_idx[candidate], alt[candidate]] += 0.5 * ratio[candidate]
    smoothing = float(args.smoothing)
    targets = (1.0 - smoothing) * targets + smoothing / num_classes

    np.save(project / args.output, targets.astype(np.float16))
    print(json.dumps({
        "images": n,
        "candidates": int(candidate.sum()),
        "candidate_share_of_trainable": float(candidate.sum() / max(keep.sum(), 1)),
        "ratio_mean": float(ratio[candidate].mean()) if candidate.any() else 0.0,
        "ratio_p10": float(np.quantile(ratio[candidate], 0.1)) if candidate.any() else 0.0,
        "ratio_p90": float(np.quantile(ratio[candidate], 0.9)) if candidate.any() else 0.0,
        "target_mass_check": float(targets.sum(axis=1).mean()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
