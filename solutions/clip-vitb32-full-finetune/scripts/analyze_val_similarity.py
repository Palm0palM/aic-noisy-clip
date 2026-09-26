"""Measure how much of the held-out validation split is near-duplicate of training.

Web-scraped fine-grained datasets often contain several images of the same
specimen from the same shoot or the same web page; those end up split across
train and validation and inflate validation accuracy relative to a separately
curated test set. This script embeds train/val images with a checkpoint and
measures, for every validation image, the maximum cosine similarity to any
training image. Validation images are then bucketed by that similarity so that
model selection can use a "test-like" subset instead of the whole split.

Outputs:
  artifacts/val_similarity_report.json  - accuracy per similarity bucket
  artifacts/val_split_testsim.json      - {"test_like": [...], "near_dup": [...]}

    python scripts/analyze_val_similarity.py \
        --checkpoint checkpoints/v10_ft288/best.pt \
        --train-cache /root/autodl-tmp/data/aic-rematch/train_cache288
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image, read_manifest, load_split


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
def embed(model, root, records, transform, batch_size, workers, device):
    loader = DataLoader(
        EmbedDataset(root, records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features = np.zeros((len(records), 512), dtype=np.float32)
    labels = np.zeros(len(records), dtype=np.int64)
    preds = np.zeros(len(records), dtype=np.int64)
    for images, target, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.embed(images)
            logits = model.head(feats).float()
        features[indices.numpy()] = F.normalize(feats.float(), dim=-1).cpu().numpy()
        labels[indices.numpy()] = target.numpy()
        preds[indices.numpy()] = logits.argmax(1).cpu().numpy()
    return features, labels, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--split", default="artifacts/split_seed20260921.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--output-report", default="artifacts/val_similarity_report.json")
    parser.add_argument("--output-split", default="artifacts/val_split_testsim.json")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    image_size = int(payload.get("eval_size", payload.get("image_size", 224)))
    which = (payload.get("metrics") or {}).get("chosen", "ema")
    state = payload.get(which) or payload.get("model")

    model = FTClassifier(
        cfg["model"]["backbone"], cfg["model"].get("revision"), int(payload["num_classes"]),
        head=cfg["model"].get("head", "linear"), dropout=0.0,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    transform = T.Compose(
        [
            T.Resize(int(round(image_size * 1.14)), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )

    records = read_manifest(project / args.manifest)
    train_idx, val_idx = load_split(project / args.split, records)
    root = Path(args.train_cache)
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    print(f"[sim] embedding {len(train_records)} train / {len(val_records)} val at {image_size}px", flush=True)

    train_feat, _, _ = embed(model, root, train_records, transform, args.batch_size, args.workers, device)
    val_feat, val_labels, val_preds = embed(model, root, val_records, transform, args.batch_size, args.workers, device)
    train_feat_t = torch.from_numpy(train_feat).to(device).half()

    max_sim = np.zeros(len(val_records), dtype=np.float32)
    chunk = 512
    for start in range(0, len(val_records), chunk):
        block = torch.from_numpy(val_feat[start:start + chunk]).to(device).half()
        sims = block @ train_feat_t.t()
        max_sim[start:start + chunk] = sims.max(dim=1).values.float().cpu().numpy()
        del sims, block
    torch.cuda.empty_cache()

    hit = (val_preds == val_labels).astype(np.float64)
    buckets = [(0.0, 0.9), (0.9, 0.95), (0.95, 0.97), (0.97, 0.99), (0.99, 1.01)]
    report = {
        "image_size": image_size,
        "checkpoint": args.checkpoint,
        "overall_accuracy": float(hit.mean()),
        "max_sim_quantiles": {str(q): float(np.quantile(max_sim, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "buckets": [],
    }
    for low, high in buckets:
        mask = (max_sim >= low) & (max_sim < high)
        if mask.sum() == 0:
            continue
        report["buckets"].append(
            {
                "range": [low, high],
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "mean_sim": float(max_sim[mask].mean()),
                "accuracy_against_noisy_labels": float(hit[mask].mean()),
            }
        )

    threshold = float(np.quantile(max_sim, args.keep_fraction))
    test_like = [int(i) for i, s in zip(val_idx, max_sim) if s <= threshold]
    near_dup = [int(i) for i, s in zip(val_idx, max_sim) if s > threshold]
    report["threshold"] = threshold
    report["test_like_count"] = len(test_like)
    report["near_dup_count"] = len(near_dup)
    report["test_like_accuracy_against_noisy_labels"] = float(
        hit[[j for j, s in enumerate(max_sim) if s <= threshold]].mean()
    )
    report["near_dup_accuracy_against_noisy_labels"] = float(
        hit[[j for j, s in enumerate(max_sim) if s > threshold]].mean()
    )

    (project / args.output_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (project / args.output_split).write_text(
        json.dumps({"test_like": test_like, "near_dup": near_dup, "threshold": threshold, "source_split": args.split},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(project / "artifacts/val_max_similarity.npy", max_sim)
    print("[sim] " + json.dumps({k: report[k] for k in ("threshold", "test_like_count", "near_dup_count")}), flush=True)


if __name__ == "__main__":
    main()
