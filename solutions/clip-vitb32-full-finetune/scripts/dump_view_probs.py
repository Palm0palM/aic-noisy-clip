"""Dump per-view probabilities for a checkpoint on a held-out split.

Used to choose the TTA recipe on clean training-side data (never the test set).
Writes an .npz with one probability matrix per view plus the labels, so the
combination search can be done offline without re-running the model.

    python scripts/dump_view_probs.py --checkpoint checkpoints/v12/s3_576/best.pt \
        --cache /root/autodl-tmp/data/aic-rematch/train_cache384 \
        --split artifacts/split95_seed20260926.json \
        --views center:576:1.0,center:576:1.14,flip:576:1.14,center:576:1.28,center:576:1.4,center:576:1.6,fivecrop:576:1.14 \
        --output artifacts/tta_views_v12.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image, load_split, read_manifest


class ItemDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform):
        self.root = root
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        return self.transform(load_image(self.root / rec["relative_path"])), rec["label"]


class FiveCropTensor:
    def __init__(self, size: int):
        self.crop = T.FiveCrop(size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(CLIP_MEAN, CLIP_STD)

    def __call__(self, image):
        return torch.stack([self.normalize(self.to_tensor(c)) for c in self.crop(image)])


def view_transform(kind: str, size: int, ratio: float):
    resize = int(round(size * ratio))
    base = [T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC)]
    if kind == "center":
        return T.Compose(base + [T.CenterCrop(size), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    if kind == "flip":
        return T.Compose(base + [T.CenterCrop(size), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    if kind == "fivecrop":
        return T.Compose(base + [FiveCropTensor(size)])
    raise ValueError(kind)


@torch.no_grad()
def probs_for(model, root, records, transform, batch_size, workers, device, num_classes):
    loader = DataLoader(ItemDataset(root, records, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    out = np.zeros((len(records), num_classes), dtype=np.float32)
    labels = np.zeros(len(records), dtype=np.int64)
    cursor = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        if images.ndim == 5:
            b, k, c, h, w = images.shape
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images.view(b * k, c, h, w)).view(b, k, -1).float().mean(1)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images).float()
        n = logits.shape[0]
        out[cursor:cursor + n] = F.softmax(logits, dim=1).cpu().numpy()
        labels[cursor:cursor + n] = target.numpy()
        cursor += n
    return out, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--split", default="artifacts/split95_seed20260926.json")
    parser.add_argument("--views", required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", required=True)
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
    _, val_idx = load_split(project / args.split, records)
    val_records = [records[i] for i in val_idx]
    print(f"[dump] {len(val_records)} held-out images, weights={which}", flush=True)

    arrays = {}
    labels = None
    for item in args.views.split(","):
        parts = item.split(":")
        kind, size = parts[0], int(parts[1])
        ratio = float(parts[2]) if len(parts) > 2 else 1.14
        p, labels = probs_for(model, Path(args.cache), val_records, view_transform(kind, size, ratio),
                              args.batch_size, args.workers, device, num_classes)
        arrays[item] = p
        print(f"[dump] {item}: acc {float((p.argmax(1) == labels).mean()):.4f}", flush=True)

    np.savez_compressed(project / args.output, labels=labels, **arrays)
    print(f"[dump] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
