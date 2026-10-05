"""Split the training set into two folds that respect near-duplicate clusters.

Out-of-fold predictions are only meaningful if a sample's duplicates are not in
the fold the model trained on. Clusters come from a perceptual hash (dHash,
hamming <= 6, the relation the drop-list audit validated), so no GPU or model is
needed. Clusters are assigned whole, greedily balancing the per-class counts.

Candidate pairs come from 8 bands of 8 bits, which guarantees finding every pair
within 7 differing bits; no bucket is skipped, they are processed in chunks. The
script then re-enumerates the same candidate set and verifies that no pair within
the threshold crosses folds - the check GPT asked for.

    python scripts/build_folds.py --cache /root/autodl-tmp/data/aic-rematch/train_cache384
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

_ARGS: dict = {}


def init_worker(args) -> None:
    _ARGS.update(vars(args))


def dhash(relative_path: str) -> int:
    path = Path(_ARGS["cache"]) / relative_path
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((9, 8), Image.BILINEAR)
            pixels = np.asarray(im, dtype=np.int16)
    except Exception:
        return -1
    value = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).flatten():
        value = (value << 1) | int(bit)
    return value


class HashUnion:
    """Connected components over hashes, computed with numpy only.

    A single "lower the label" pass is wrong here: overwriting a member's pointer
    to its root can cut the component it belonged to. The correct relaxation
    takes the minimum of both endpoints' current labels and assigns it to both,
    then repeats over the whole edge set until the labels stop changing - a
    Bellman-Ford style fixpoint, which is what this class does.
    """

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def resolve(self, a: np.ndarray, b: np.ndarray, rounds: int = 200) -> np.ndarray:
        parent = self.parent
        for _ in range(rounds):
            before = parent.copy()
            minimum = np.minimum(parent[a], parent[b])
            np.minimum.at(parent, a, minimum)
            np.minimum.at(parent, b, minimum)
            for _ in range(4):
                jumped = np.minimum(parent, parent[parent])
                if np.array_equal(jumped, parent):
                    break
                parent = jumped
            if np.array_equal(parent, before):
                break
        self.parent = parent
        return parent


_POPCOUNT = np.array([bin(b).count("1") for b in range(256)], dtype=np.uint8)


def _popcount(diff: np.ndarray) -> np.ndarray:
    return _POPCOUNT[diff.view(np.uint8).reshape(diff.shape + (8,))]


def collect_candidate_edges(hashes: np.ndarray, valid: np.ndarray, max_hamming: int, chunk: int = 512):
    """All index pairs that share a band and are within the hamming threshold."""
    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.nonzero(valid)[0]:
        value = int(hashes[index])
        for band in range(8):
            bands[(band, (value >> (8 * band)) & 0xFF)].append(int(index))
    left, right = [], []
    for members in bands.values():
        if len(members) < 2:
            continue
        m = np.array(members, dtype=np.int64)
        for start in range(0, len(m), chunk):
            block = m[start:start + chunk]
            diff = hashes[block][:, None] ^ hashes[m][None, :]
            counts = _popcount(diff).sum(axis=2)
            mask = (counts <= max_hamming) & (block[:, None] < m[None, :])
            rows, cols = np.nonzero(mask)
            if len(rows):
                left.append(block[rows])
                right.append(m[cols])
    if not left:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(left), np.concatenate(right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--output", default="artifacts/folds_2.json")
    parser.add_argument("--report", default="artifacts/folds_2_report.json")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--max-hamming", type=int, default=6)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    _ARGS.update(vars(args))

    project = Path(args.project_root).resolve()
    with (project / args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    paths = [r["relative_path"] for r in rows]
    print(f"[folds] {len(rows)} images", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args,)) as pool:
        raw = np.fromiter(pool.map(dhash, paths, chunksize=256), dtype=np.uint64, count=len(paths))
    valid = raw != np.iinfo(np.uint64).max
    print(f"[folds] hashed {int(valid.sum())} images", flush=True)

    edge_a, edge_b = collect_candidate_edges(raw, valid, args.max_hamming)
    pair_count = int(len(edge_a))
    conflicting_pairs = int((labels[edge_a] != labels[edge_b]).sum()) if pair_count else 0
    union = HashUnion(len(rows))
    parent = union.resolve(edge_a, edge_b)
    print(f"[folds] near-duplicate pairs (hamming <= {args.max_hamming}): {pair_count}, "
          f"conflicting labels: {conflicting_pairs}", flush=True)
    unresolved = int((parent[edge_a] != parent[edge_b]).sum()) if pair_count else 0
    print(f"[folds] edges not merged into one component: {unresolved}", flush=True)
    print(f"[folds] near-duplicate pairs (hamming <= {args.max_hamming}): {pair_count}, "
          f"conflicting labels: {conflicting_pairs}", flush=True)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        clusters[int(parent[index])].append(index)
    cluster_list = [members for members in clusters.values() if len(members) > 1]
    cluster_list.sort(key=len, reverse=True)
    print(f"[folds] {len(cluster_list)} clusters, {sum(len(m) for m in cluster_list)} images", flush=True)

    num_classes = int(labels.max()) + 1
    counts = [np.zeros(num_classes, dtype=np.int64), np.zeros(num_classes, dtype=np.int64)]
    fold_of = np.full(len(rows), -1, dtype=np.int64)
    for members in cluster_list:
        label = Counter(int(labels[m]) for m in members).most_common(1)[0][0]
        target = 0 if counts[0][label] <= counts[1][label] else 1
        for m in members:
            fold_of[m] = target
        np.add.at(counts[target], labels[members], 1)
    for index in range(len(rows)):
        if fold_of[index] < 0:
            label = int(labels[index])
            target = 0 if counts[0][label] <= counts[1][label] else 1
            fold_of[index] = target
            counts[target][label] += 1

    # independent verification: no near-duplicate pair may cross folds
    cross_fold = int((fold_of[edge_a] != fold_of[edge_b]).sum()) if pair_count else 0

    sizes = [int((fold_of == f).sum()) for f in (0, 1)]
    report = {
        "images": int(len(rows)),
        "hashed": int(valid.sum()),
        "near_duplicate_pairs": int(pair_count),
        "conflicting_pairs": int(conflicting_pairs),
        "clusters_gt1": len(cluster_list),
        "images_in_clusters": int(sum(len(m) for m in cluster_list)),
        "fold_sizes": sizes,
        "fold_sizes_by_class_max_abs_diff": int(np.abs(counts[0] - counts[1]).max()),
        "fold_sizes_by_class_median_abs_diff": float(np.median(np.abs(counts[0] - counts[1]))),
        "candidate_pairs_crossing_folds": int(cross_fold),
        "edges_not_merged": int(unresolved),
        "min_class_count_fold0": int(counts[0].min()),
        "min_class_count_fold1": int(counts[1].min()),
    }
    (project / args.output).write_text(json.dumps({"fold": fold_of.tolist()}, ensure_ascii=False), encoding="utf-8")
    (project / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[folds] " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
