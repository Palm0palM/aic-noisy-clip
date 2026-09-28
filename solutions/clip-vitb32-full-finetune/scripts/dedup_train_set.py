"""Detect near-duplicate clusters inside the official training set.

Motivation: a held-out analysis showed that 24% of validation images have a
near-duplicate inside the training set, and exact-duplicate pairs disagree about
their label roughly one third of the time. The same structure must exist inside
the training set itself, where the conflicting pairs act as contradictory
supervision.

This script embeds every training image with a trained checkpoint, builds a
near-duplicate graph (cosine similarity above a threshold), groups it into
clusters, and reports how many images sit in a cluster with conflicting labels.
It writes an automatic, reproducible relabelling rule: inside each cluster every
member takes the majority label; clusters with no majority are dropped.

    python scripts/dedup_train_set.py --checkpoint checkpoints/v10_final4/best.pt \
        --cache /root/autodl-tmp/data/aic-rematch/train_cache384 \
        --threshold 0.97 --output artifacts/dedup_labels.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image, read_manifest


class EmbedDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform):
        self.root = root
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        return self.transform(load_image(self.root / rec["relative_path"])), rec["label"], index


@torch.no_grad()
def embed(model, root, records, transform, batch_size, workers, device, dim):
    loader = DataLoader(EmbedDataset(root, records, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    feats = np.zeros((len(records), dim), dtype=np.float32)
    labels = np.zeros(len(records), dtype=np.int64)
    for images, target, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.embed(images)
        feats[indices.numpy()] = F.normalize(out.float(), dim=-1).cpu().numpy()
        labels[indices.numpy()] = target.numpy()
    return feats, labels


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--output", default="artifacts/dedup_labels.npy")
    parser.add_argument("--drop-output", default="artifacts/dedup_drop.npy")
    parser.add_argument("--output-report", default="artifacts/dedup_report.json")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    num_classes = int(payload["num_classes"])
    which = (payload.get("metrics") or {}).get("chosen", "ema")
    state = payload.get("model" if which == "raw" else which) or payload.get("model")
    model = FTClassifier(cfg["model"]["backbone"], cfg["model"].get("revision"), num_classes,
                         head=cfg["model"].get("head", "linear"), dropout=0.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    records = read_manifest(project / args.manifest)
    size = int(payload.get("eval_size", payload.get("image_size", 448)))
    dim = int(model.vision.config.projection_dim)
    transform = T.Compose([
        T.Resize(int(round(size * 1.14)), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    print(f"[dedup] embedding {len(records)} training images at {size}px", flush=True)
    feats, labels = embed(model, Path(args.cache), records, transform, args.batch_size, args.workers, device, dim)

    feats_t = torch.from_numpy(feats).to(device).half()
    union = UnionFind(len(records))
    pair_count = conflict_pairs = 0
    max_sim = np.zeros(len(records), dtype=np.float32)
    nn_index = np.full(len(records), -1, dtype=np.int64)
    for start in range(0, len(records), args.chunk):
        stop = min(start + args.chunk, len(records))
        block = torch.from_numpy(feats[start:stop]).to(device).half()
        sims = block @ feats_t.t()
        sims[torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)] = -1.0
        top = sims.topk(2, dim=1)
        best_sim = top.values[:, 0].float().cpu().numpy()
        best_idx = top.indices[:, 0].cpu().numpy()
        max_sim[start:stop] = best_sim
        nn_index[start:stop] = best_idx
        close = best_sim >= args.threshold
        for offset in np.nonzero(close)[0]:
            i = start + int(offset)
            j = int(best_idx[offset])
            pair_count += 1
            if labels[i] != labels[j]:
                conflict_pairs += 1
            union.union(i, j)
        del sims, block
        torch.cuda.empty_cache()

    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(union.find(i), []).append(i)

    clusters = [members for members in groups.values() if len(members) > 1]
    conflict_clusters = 0
    relabeled = 0
    dropped: list[int] = []
    new_labels = labels.copy()
    for members in clusters:
        counts = Counter(int(labels[m]) for m in members)
        top_label, top_count = counts.most_common(1)[0]
        if len(counts) == 1:
            continue
        conflict_clusters += 1
        tied = [lab for lab, cnt in counts.items() if cnt == top_count]
        if len(tied) > 1:
            # no majority: the conflicting images carry contradictory supervision,
            # so they are listed for exclusion instead of relabelling
            dropped.extend(members)
            continue
        for m in members:
            if labels[m] != top_label:
                relabeled += 1
            new_labels[m] = top_label

    report = {
        "threshold": args.threshold,
        "train_images": int(len(records)),
        "images_with_near_duplicate": int((max_sim >= args.threshold).sum()),
        "near_duplicate_pairs": int(pair_count),
        "conflicting_pairs": int(conflict_pairs),
        "clusters": len(clusters),
        "clusters_with_conflicting_labels": conflict_clusters,
        "images_relabelled": int(relabeled),
        "images_dropped": int(len(dropped)),
        "max_sim_quantiles": {str(q): float(np.quantile(max_sim, q)) for q in (0.5, 0.9, 0.95, 0.99)},
    }
    print("[dedup] " + json.dumps(report, ensure_ascii=False), flush=True)
    np.save(project / args.output, new_labels.astype(np.int64))
    (project / args.output_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(project / args.drop_output, np.array(sorted(set(dropped)), dtype=np.int64))


if __name__ == "__main__":
    main()
