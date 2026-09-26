"""Score inference-view combinations on the held-out split (train-side labels only).

Used to choose the TTA recipe without touching the test set. Views are combined in
probability space; every combination reported is a deterministic function of the
single checkpoint.

    python scripts/evaluate_views.py --checkpoint checkpoints/v10_ft384/best.pt \
        --cache /root/autodl-tmp/data/aic-rematch/train_cache384
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image, load_split, read_manifest


class SplitDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform):
        self.root = root
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        return self.transform(load_image(self.root / rec["relative_path"])), rec["label"]


def view_transform(kind: str, size: int, ratio: float = 1.14):
    resize = int(round(size * ratio))
    base = [T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC)]
    if kind == "center":
        return T.Compose(base + [T.CenterCrop(size), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    if kind == "flip":
        return T.Compose(base + [T.CenterCrop(size), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    if kind == "fivecrop":
        return T.Compose(base + [FiveCropTensor(size)])
    raise ValueError(kind)


class FiveCropTensor:
    """FiveCrop -> (5, 3, H, W) float tensor, normalised like CLIP."""

    def __init__(self, size: int):
        self.crop = T.FiveCrop(size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(CLIP_MEAN, CLIP_STD)

    def __call__(self, image):
        return torch.stack([self.normalize(self.to_tensor(c)) for c in self.crop(image)])


@torch.no_grad()
def probs_for(model, root, records, transform, batch_size, workers, device, num_classes):
    loader = DataLoader(SplitDataset(root, records, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    out = np.zeros((len(records), num_classes), dtype=np.float32)
    labels = np.zeros(len(records), dtype=np.int64)
    cursor = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        if images.ndim == 5:
            b, k, c, h, w = images.shape
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images.view(b * k, c, h, w)).view(b, k, -1)
            logits = logits.float().mean(1)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images).float()
        n = logits.shape[0]
        out[cursor:cursor + n] = F.softmax(logits, dim=1).cpu().numpy()
        labels[cursor:cursor + n] = target.numpy()
        cursor += n
    return out, labels


def metrics(probs, labels, class_counts):
    pred = probs.argmax(1)
    hit = (pred == labels).astype(np.float64)
    present = np.bincount(labels, minlength=len(class_counts)) > 0
    per_class = np.bincount(labels[hit == 1], minlength=len(class_counts))[present] / np.bincount(labels, minlength=len(class_counts))[present]
    order = np.argsort(class_counts[present])
    ordered = per_class[order]
    third = max(len(ordered) // 3, 1)
    return {
        "accuracy": float(hit.mean()),
        "macro_accuracy": float(per_class.mean()),
        "tail_accuracy": float(ordered[:third].mean()),
        "mid_accuracy": float(ordered[third:2 * third].mean()),
        "head_accuracy": float(ordered[2 * third:].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--split", default="artifacts/split_seed20260921.json")
    parser.add_argument("--views", default="center:384,flip:384,fivecrop:384,center:288,center:448")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--weights", default="auto")
    parser.add_argument("--output", default="")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    num_classes = int(payload["num_classes"])
    which = args.weights
    if which == "auto":
        which = (payload.get("metrics") or {}).get("chosen", "ema")
    state = payload.get("model" if which == "raw" else which) or payload.get("model")
    model = FTClassifier(cfg["model"]["backbone"], cfg["model"].get("revision"), num_classes,
                         head=cfg["model"].get("head", "linear"), dropout=0.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[views] checkpoint epoch={payload.get('epoch')} weights={which} trained@{payload.get('image_size')}px", flush=True)

    records = read_manifest(project / args.manifest)
    _, val_idx = load_split(project / args.split, records)
    val_records = [records[i] for i in val_idx]
    class_counts = np.bincount([r["label"] for r in records], minlength=num_classes).astype(np.float64)

    specs = []
    for item in args.views.split(","):
        parts = item.split(":")
        kind, size = parts[0], int(parts[1])
        ratio = float(parts[2]) if len(parts) > 2 else 1.14
        specs.append((item, kind, size, ratio))

    probs = {}
    labels = None
    for name, kind, size, ratio in specs:
        p, lab = probs_for(model, Path(args.cache), val_records, view_transform(kind, size, ratio), args.batch_size,
                           args.workers, device, num_classes)
        probs[name] = p
        labels = lab
        print(f"[views] {name}: " + json.dumps(metrics(p, labels, class_counts)), flush=True)

    report = {"checkpoint": args.checkpoint, "weights": which, "single": {}, "combinations": []}
    for name, p in probs.items():
        report["single"][name] = metrics(p, labels, class_counts)
    names = list(probs)
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            if len(combo) > 4:
                continue
            blended = np.mean([probs[n] for n in combo], axis=0)
            report["combinations"].append({"views": list(combo), **metrics(blended, labels, class_counts)})
    report["combinations"].sort(key=lambda x: -x["accuracy"])
    for entry in report["combinations"][:8]:
        print("[views] combo " + json.dumps(entry), flush=True)
    if args.output:
        (project / args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
