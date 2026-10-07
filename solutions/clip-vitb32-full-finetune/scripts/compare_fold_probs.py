"""Compare two out-of-fold probability dumps on the same held-out fold.

Reports overall / macro / tail-mid-head accuracy (the trainer's definition:
classes present in the fold, sorted by fold frequency, split into thirds) and
the paired flips, for one or more dumps against a reference dump.

    python scripts/compare_fold_probs.py --fold 1 \
        --reference artifacts/fast_v30_oof_a_probs.npy \
        --candidate artifacts/anchor_a_probs.npy \
        --labels-name "anchor(raw)" \
        --degraded-reference artifacts/fast_v30_oof_a_probs_degraded.npy \
        --degraded-candidate artifacts/anchor_a_probs_degraded.npy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_labels(manifest: Path) -> np.ndarray:
    with manifest.open(encoding="utf-8", newline="") as handle:
        return np.asarray([int(row["label"]) for row in csv.DictReader(handle)], dtype=np.int64)


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    correct = np.zeros(probs.shape[1], dtype=np.int64)
    total = np.zeros(probs.shape[1], dtype=np.int64)
    pred = probs.argmax(1)
    hit = (pred == labels).astype(np.int64)
    np.add.at(correct, labels, hit)
    np.add.at(total, labels, 1)
    present = total > 0
    per_class = correct[present] / np.maximum(total[present], 1)
    order = np.argsort(total[present])
    per_present = per_class[order]
    third = max(len(per_present) // 3, 1)
    return {
        "accuracy": float(correct.sum() / max(total.sum(), 1)),
        "macro_accuracy": float(per_class.mean()) if present.any() else 0.0,
        "tail_accuracy": float(per_present[:third].mean()),
        "mid_accuracy": float(per_present[third:2 * third].mean()),
        "head_accuracy": float(per_present[2 * third:].mean()),
        "mean_max_prob": float(probs.max(1).mean()),
    }


def compare(reference: Path, candidate: Path, labels: np.ndarray, mask: np.ndarray, name: str) -> dict:
    ref = np.load(reference).astype(np.float32)
    cand = np.load(candidate).astype(np.float32)
    ref_pred = ref[mask].argmax(1)
    cand_pred = cand[mask].argmax(1)
    truth = labels[mask]
    ref_ok, cand_ok = ref_pred == truth, cand_pred == truth
    w2c = int((~ref_ok & cand_ok).sum())
    c2w = int((ref_ok & ~cand_ok).sum())
    n = int(mask.sum())
    return {
        "name": name,
        "n": n,
        "reference": metrics(ref[mask], truth),
        "candidate": metrics(cand[mask], truth),
        "paired": {
            "wrong_to_correct": w2c,
            "correct_to_wrong": c2w,
            "net": w2c - c2w,
            "delta_pp": 100.0 * (w2c - c2w) / n,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--folds", default="artifacts/folds_2.json")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--labels-name", default="candidate")
    parser.add_argument("--degraded-reference", default="")
    parser.add_argument("--degraded-candidate", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    labels = read_labels(project / args.manifest)
    folds = np.asarray(json.loads((project / args.folds).read_text(encoding="utf-8"))["fold"], dtype=np.int64)
    mask = folds == args.fold

    results = {"raw": compare(project / args.reference, project / args.candidate, labels, mask, args.labels_name)}
    if args.degraded_reference and args.degraded_candidate:
        results["degraded"] = compare(
            project / args.degraded_reference, project / args.degraded_candidate,
            labels, mask, args.labels_name + " (jpeg45+blur1)",
        )

    for kind, r in results.items():
        ref, cand, paired = r["reference"], r["candidate"], r["paired"]
        print(f"--- {kind}: {r['name']} (n={r['n']}) ---")
        print(f"{'metric':>16} {'control':>10} {'candidate':>10}")
        for key in ("accuracy", "macro_accuracy", "tail_accuracy", "mid_accuracy", "head_accuracy", "mean_max_prob"):
            print(f"{key:>16} {ref[key]:>10.4f} {cand[key]:>10.4f}")
        print(f"{'w2c/c2w/net':>16} {paired['wrong_to_correct']:>4d}/{paired['correct_to_wrong']:<4d} "
              f"net {paired['net']:+d} = {paired['delta_pp']:+.4f}pp")

    if args.output:
        (project / args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[compare] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
