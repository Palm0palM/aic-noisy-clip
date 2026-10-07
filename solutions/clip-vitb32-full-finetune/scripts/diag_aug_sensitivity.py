"""Controlled augmentation-sensitivity diagnostic.

Fixes the confounds of the first version: every variant is rendered at the SAME
resolution (384), from the SAME crop box and flip decision, with a per-image seed,
so that colour jitter / RandAugment / erasing / the cut-half of CutMix are the
only things that change between variants. Reports overall label agreement, the
agreement restricted to the subset the weak view already classifies correctly,
and the right->wrong and wrong->right transitions.

This measures sensitivity only; it cannot show that a training label is wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image, read_manifest  # noqa: E402

NAMES = ["weak_center_384", "rrc_384", "+colorjitter", "+randaugment", "+erase", "+cutmix_patch"]


def sample_crop(height, width, scale, ratio, rng):
    area = height * width
    for _ in range(10):
        target = area * rng.uniform(*scale)
        aspect = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
        cw = int(round(np.sqrt(target * aspect)))
        ch = int(round(np.sqrt(target / aspect)))
        if 0 < cw <= width and 0 < ch <= height:
            i = int(rng.integers(0, height - ch + 1))
            j = int(rng.integers(0, width - cw + 1))
            return j, i, cw, ch
    side = min(height, width)
    return (width - side) // 2, (height - side) // 2, side, side


class VariantDataset(Dataset):
    """One image -> the same crop under progressively stronger augmentation."""

    def __init__(self, root, records, size=384, scale=(0.35, 1.0), weak_ratio=1.14):
        self.root = Path(root)
        self.records = records
        self.size = size
        self.scale = scale
        self.weak_ratio = weak_ratio

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        image = load_image(self.root / rec["relative_path"]).convert("RGB")
        rng = np.random.default_rng(1000 + index)

        weak_resize = int(round(self.size * self.weak_ratio))
        factor = weak_resize / min(image.size)
        weak = image.resize((int(round(image.width * factor)), int(round(image.height * factor))), Image.BICUBIC)
        left = (weak.width - self.size) // 2
        top = (weak.height - self.size) // 2
        weak = weak.crop((left, top, left + self.size, top + self.size))

        j, i, cw, ch = sample_crop(image.height, image.width, self.scale, (3 / 4, 4 / 3), rng)
        crop = image.crop((j, i, j + cw, i + ch)).resize((self.size, self.size), Image.BICUBIC)

        torch.manual_seed(1000 + index)
        variants = []
        for level in range(6):
            work = crop
            if level >= 2:
                work = T.ColorJitter(0.5, 0.5, 0.5, 0.125)(work)
            if level >= 3:
                work = T.RandAugment(num_ops=2, magnitude=7, interpolation=T.InterpolationMode.BICUBIC)(work)
            tensor = TF.normalize(TF.to_tensor(work), CLIP_MEAN, CLIP_STD)
            if level >= 4:
                tensor = T.RandomErasing(p=1.0, value="random")(tensor)
            if level == 5:
                donor = load_image(self.root / self.records[int(rng.integers(0, len(self.records)))]["relative_path"]).convert("RGB")
                ds = min(donor.size)
                half = self.size // 2
                dl = int(rng.integers(0, max(ds - half, 1)))
                dt = int(rng.integers(0, max(ds - half, 1)))
                patch = donor.crop((dl, dt, dl + half, dt + half)).resize((half, half), Image.BICUBIC)
                pt = int(rng.integers(0, self.size - half + 1))
                pl = int(rng.integers(0, self.size - half + 1))
                patch = TF.normalize(TF.to_tensor(patch), CLIP_MEAN, CLIP_STD)
                tensor = tensor.clone()
                tensor[:, pt:pt + half, pl:pl + half] = patch
            variants.append(tensor)
        variants[0] = TF.normalize(TF.to_tensor(weak), CLIP_MEAN, CLIP_STD)
        return torch.stack(variants), rec["label"], index


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", default="/root/autodl-tmp/data/aic-rematch/train")
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--folds", default="artifacts/folds_2.json")
    parser.add_argument("--eval-fold", type=int, default=1)
    parser.add_argument("--images", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="artifacts/diag_aug.json")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    device = torch.device("cuda")
    payload = torch.load(project / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = FTClassifier(cfg["model"]["backbone"], cfg["model"].get("revision"), int(payload["num_classes"]),
                         head=cfg["model"].get("head", "linear"), dropout=0.0).to(device)
    model.load_state_dict(payload.get("ema") or payload.get("model"), strict=True)
    model.eval()

    folds = np.array(json.loads((project / args.folds).read_text(encoding="utf-8"))["fold"], dtype=np.int64)
    records = read_manifest(project / args.manifest)
    held = [r for r, f in zip(records, folds) if f == args.eval_fold]
    rng = np.random.default_rng(0)
    subset = [held[i] for i in sorted(rng.choice(len(held), size=min(args.images, len(held)), replace=False).tolist())]
    print(f"[diag] {len(subset)} held-out images, fold {args.eval_fold}, all variants at 384", flush=True)

    loader = DataLoader(VariantDataset(args.cache, subset), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    probs = np.zeros((len(subset), len(NAMES), int(payload["num_classes"])), dtype=np.float32)
    labels = np.zeros(len(subset), dtype=np.int64)
    for stack, label, index in loader:
        b, v, c, h, w = stack.shape
        flat = stack.view(b * v, c, h, w).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(flat).float()
        probs[index.numpy()] = logits.view(b, v, -1).softmax(-1).cpu().numpy()
        labels[index.numpy()] = label.numpy()

    base_correct = probs[:, 0].argmax(1) == labels
    report = {"images": int(len(subset)), "weak_view_label_accuracy": float(base_correct.mean()), "variants": {}}
    for vi, name in enumerate(NAMES):
        correct = probs[:, vi].argmax(1) == labels
        report["variants"][name] = {
            "label_accuracy": float(correct.mean()),
            "label_accuracy_on_weak_correct_subset": float(correct[base_correct].mean()) if base_correct.any() else None,
            "weak_correct_to_wrong": int((base_correct & ~correct).sum()),
            "weak_wrong_to_correct": int((~base_correct & correct).sum()),
            "mean_max_prob": float(probs[:, vi].max(1).mean()),
        }
    (project / args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
