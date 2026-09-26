"""Tune the class-prior correction on a class-balanced slice of the held-out split.

The training set is long-tailed (5-248 images per class) while the official
evaluation set is described as class balanced, so the decision rule that matches
the evaluation prior is `argmax(logits - tau * log(train_prior))`. Both the
correction and the value of tau are derived from official training data only.

    python scripts/evaluate_prior_correction.py --checkpoint checkpoints/v10_ft384/best.pt \
        --cache /root/autodl-tmp/data/aic-rematch/train_cache384 --cap 20
"""

from __future__ import annotations

import argparse
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


def make_views(size: int):
    def norm():
        return [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]

    def build(ratio, flip=False):
        ops = [T.Resize(int(round(size * ratio)), interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]
        if flip:
            ops.append(T.RandomHorizontalFlip(p=1.0))
        return T.Compose(ops + norm())

    return {
        "center1.0": build(1.0),
        "center1.14": build(1.14),
        "flip1.14": build(1.14, flip=True),
        "center1.4": build(1.4),
    }


@torch.no_grad()
def probs_for(model, root, records, transform, batch_size, workers, device, num_classes):
    loader = DataLoader(ItemDataset(root, records, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    out = np.zeros((len(records), num_classes), dtype=np.float32)
    labels = np.zeros(len(records), dtype=np.int64)
    cursor = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
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
    parser.add_argument("--split", default="artifacts/split_seed20260921.json")
    parser.add_argument("--cap", type=int, default=20, help="max validation images per class")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="artifacts/prior_correction.json")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    num_classes = int(payload["num_classes"])
    size = int(payload.get("eval_size", payload.get("image_size", 384)))
    which = (payload.get("metrics") or {}).get("chosen", "ema")
    state = payload.get("model" if which == "raw" else which) or payload.get("model")
    model = FTClassifier(cfg["model"]["backbone"], cfg["model"].get("revision"), num_classes,
                         head=cfg["model"].get("head", "linear"), dropout=0.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    records = read_manifest(project / args.manifest)
    train_idx, val_idx = load_split(project / args.split, records)
    by_class: dict[int, list[int]] = {}
    for i in val_idx:
        by_class.setdefault(records[i]["label"], []).append(i)
    picked: list[int] = []
    for label, items in sorted(by_class.items()):
        picked.extend(items[: args.cap])
    picked.sort()
    sub_records = [records[i] for i in picked]
    print(f"[prior] balanced slice: {len(sub_records)} images over {len(by_class)} classes (cap {args.cap}/class)", flush=True)

    counts = np.bincount([records[i]["label"] for i in train_idx], minlength=num_classes).astype(np.float64)
    prior = counts / counts.sum()

    views = make_views(size)
    total = None
    labels = None
    single = {}
    for name, tf in views.items():
        p, lab = probs_for(model, Path(args.cache), sub_records, tf, args.batch_size, args.workers, device, num_classes)
        single[name] = float((p.argmax(1) == lab).mean())
        total = p if total is None else total + p
        labels = lab
    probs = total / len(views)
    print("[prior] views " + json.dumps(single), flush=True)
    print(f"[prior] multi-view accuracy {float((probs.argmax(1) == labels).mean()):.6f}", flush=True)

    log_prior = torch.from_numpy(np.log(np.maximum(prior, 1e-12))).float().to(device).unsqueeze(0)
    logits = torch.from_numpy(np.log(np.maximum(probs, 1e-12))).float().to(device)
    report = {"views": single, "multi_view_accuracy": float((probs.argmax(1) == labels).mean()), "scan": []}
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]:
        pred = (logits - tau * log_prior).argmax(1).cpu().numpy()
        per_class = []
        for label in np.unique(labels):
            mask = labels == label
            per_class.append(float((pred[mask] == label).mean()))
        entry = {"tau": tau, "accuracy": float((pred == labels).mean()), "macro_accuracy": float(np.mean(per_class))}
        report["scan"].append(entry)
        print("[prior] " + json.dumps(entry), flush=True)
    best = max(report["scan"], key=lambda x: x["accuracy"])
    report["best"] = best
    print("[prior] best " + json.dumps(best), flush=True)
    (project / args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
