"""Single-model inference with multi-view TTA for full-fine-tuned checkpoints.

Produces artifacts/<tag>/pred_results.csv (+ .zip) using one checkpoint and one
fixed inference recipe (a small set of deterministic views averaged in
probability space). Test images are only read here, never during training.

    python -m aic_clip.infer_ft --checkpoint checkpoints/v10_ft224/best.pt \
        --test-dir /root/autodl-tmp/data/aic-rematch/test \
        --views center,flip --output-dir artifacts/submission_v10a
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image
from .domain import canonicalize

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


class TestDataset(Dataset):
    def __init__(self, root: Path, files: list[str], transform, canonical_crop: bool = False):
        self.root = root
        self.files = files
        self.transform = transform
        self.canonical_crop = canonical_crop

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = load_image(self.root / self.files[index])
        if self.canonical_crop:
            image = canonicalize(image)
        return self.transform(image), index


ANCHOR_FRACTIONS = {
    "tl": (0.0, 0.0),
    "tc": (0.5, 0.0),
    "tr": (1.0, 0.0),
    "ml": (0.0, 0.5),
    "mc": (0.5, 0.5),
    "mr": (1.0, 0.5),
    "bl": (0.0, 1.0),
    "bc": (0.5, 1.0),
    "br": (1.0, 1.0),
}


class AnchorCrop:
    """Crop a size x size window placed at a relative anchor in the image.

    The image is first resized so its shorter side is `size`, so corners and
    edge anchors slide the window along the longer side - exactly the content a
    plain centre crop throws away.
    """

    def __init__(self, size: int, fx: float, fy: float):
        self.size = size
        self.fx = fx
        self.fy = fy

    def __call__(self, image):
        width, height = image.size
        left = int(round((width - self.size) * self.fx))
        top = int(round((height - self.size) * self.fy))
        return image.crop((left, top, left + self.size, top + self.size))


class FitPad:
    """Scale the whole image inside a size x size square, padding the rest."""

    def __init__(self, size: int, mode: str):
        self.size = size
        self.mode = mode

    def __call__(self, image):
        width, height = image.size
        scale = self.size / float(max(width, height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        image = image.resize((new_w, new_h), Image.BICUBIC)
        pad_l = (self.size - new_w) // 2
        pad_t = (self.size - new_h) // 2
        pad_r = self.size - new_w - pad_l
        pad_b = self.size - new_h - pad_t
        pad = (pad_l, pad_t, pad_r, pad_b)
        if self.mode == "edge":
            return TF.pad(image, pad, padding_mode="edge")
        return TF.pad(image, pad, fill=128)


def build_view(name: str, image_size: int):
    """Build a deterministic view.

    Two forms are accepted:
      * legacy names: "center", "flip", "multicrop", "resize<X>"
      * explicit specs: "kind:size:ratio", where kind is
          - center / flip: resize the shorter side to size*ratio, centre crop
          - an anchor (tl tc tr ml mc mr bl bc br): resize the shorter side to
            size*ratio, then crop a size x size window at that anchor
          - fullpad_edge / fullpad_gray: fit the entire image inside the square
            and pad the remainder
    """
    if ":" in name:
        parts = name.split(":")
        kind, size = parts[0], int(parts[1])
        ratio = float(parts[2]) if len(parts) > 2 else 1.14
        if kind in ANCHOR_FRACTIONS or (kind.endswith("_flip") and kind[:-5] in ANCHOR_FRACTIONS):
            flipped = kind.endswith("_flip")
            anchor = kind[:-5] if flipped else kind
            fx, fy = ANCHOR_FRACTIONS[anchor]
            ops = [
                T.Resize(int(round(size * ratio)), interpolation=T.InterpolationMode.BICUBIC),
                AnchorCrop(size, fx, fy),
            ]
            if flipped:
                ops.append(T.RandomHorizontalFlip(p=1.0))
            ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
            return T.Compose(ops)
        if kind.startswith("fullpad"):
            mode = kind.split("_", 1)[1] if "_" in kind else "edge"
            return T.Compose(
                [
                    FitPad(size, mode),
                    T.ToTensor(),
                    T.Normalize(CLIP_MEAN, CLIP_STD),
                ]
            )
        ops = [
            T.Resize(int(round(size * ratio)), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
        ]
        if kind == "flip":
            ops.append(T.RandomHorizontalFlip(p=1.0))
        ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
        return T.Compose(ops)

    if name == "center":
        return T.Compose(
            [
                T.Resize(int(round(image_size * 1.14)), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    if name == "flip":
        return T.Compose(
            [
                T.Resize(int(round(image_size * 1.14)), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    if name.startswith("resize"):
        ratio = float(name[6:]) / image_size
        return T.Compose(
            [
                T.Resize(int(round(image_size * ratio)), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    if name == "multicrop":
        return T.Compose(
            [
                T.Resize(int(round(image_size * 1.14)), interpolation=T.InterpolationMode.BICUBIC),
                T.FiveCrop(image_size),
                T.Lambda(lambda crops: torch.stack([T.Normalize(CLIP_MEAN, CLIP_STD)(T.ToTensor()(c)) for c in crops])),
            ]
        )
    raise ValueError(f"unknown view {name}")


@torch.no_grad()
def run_view(model, root: Path, files: list[str], transform, batch_size: int, workers: int, device, amp_dtype, num_classes: int, canonical_crop: bool = False):
    loader = DataLoader(
        TestDataset(root, files, transform, canonical_crop),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    probs = np.zeros((len(files), num_classes), dtype=np.float32)
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        if images.ndim == 5:  # multicrop -> (B, K, C, H, W)
            b, k, c, h, w = images.shape
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(images.view(b * k, c, h, w)).view(b, k, -1)
            logits = logits.float().mean(dim=1)
        else:
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(images)
            logits = logits.float()
        probs[indices.numpy()] = F.softmax(logits, dim=1).cpu().numpy()
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--views", default="center,flip")
    parser.add_argument("--weights", default="auto", choices=["auto", "raw", "ema"])
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=37444)
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--local-scale", type=float, default=1.0,
                        help="multiply the local read-out logits (diagnostic sweep)")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N test files (cheap diagnostic runs)")
    parser.add_argument("--canonical-crop", action="store_true",
                        help="apply the deterministic border crop first (models trained with it)")
    args = parser.parse_args()

    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # note: view specs may carry their own size; the checkpoint size is the fallback
    cfg = payload["config"]
    num_classes = int(payload["num_classes"])
    image_size = int(payload["image_size"])

    which = args.weights
    metrics = payload.get("metrics", {})
    if which == "auto":
        which = metrics.get("chosen", "ema") if metrics else "ema"
    state = payload[which] if which in payload else payload.get("model", payload.get("ema"))
    print(f"[infer] checkpoint epoch={payload.get('epoch')} weights={which} image_size={image_size}")

    model = FTClassifier(
        cfg["model"]["backbone"],
        cfg["model"].get("revision"),
        num_classes,
        head=cfg["model"].get("head", "linear"),
        dropout=0.0,
        feature=payload.get("feature", cfg["model"].get("feature", "projected")),
        head_dim=int(payload.get("head_dim", 0)) or None,
        local_readout=bool(payload.get("local_readout", cfg["model"].get("local_readout", False))),
        local_queries=int(payload.get("local_queries", cfg["model"].get("local_queries", 1))),
    ).to(device)
    model.load_state_dict(state, strict=True)
    if hasattr(model, "local_scale"):
        model.local_scale = float(args.local_scale)
    model.eval()

    root = Path(args.test_dir)
    files = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit:
        files = files[: args.limit]
    if args.expected_count and not args.limit and len(files) != args.expected_count:
        raise ValueError(f"found {len(files)} test images, expected {args.expected_count}")
    print(f"[infer] {len(files)} test images")

    amp_dtype = torch.bfloat16
    view_probs = {}
    for name in args.views.split(","):
        name = name.strip()
        if not name:
            continue
        probs = run_view(
            model, root, files, build_view(name, image_size), args.batch_size, args.workers, device, amp_dtype,
            num_classes, args.canonical_crop,
        )
        view_probs[name] = probs
        print(f"[infer] view {name}: done, mean max prob {probs.max(1).mean():.4f}", flush=True)

    stacked = np.mean(np.stack(list(view_probs.values()), axis=0), axis=0)
    predictions = stacked.argmax(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pred_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        for name, label in zip(files, predictions):
            handle.write(f"{name},{int(label):04d}\n")
    zip_path = out_dir / "pred_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="pred_results.csv")
    if args.save_probs:
        # float16 keeps the dump small enough to pull off the training machine
        np.savez_compressed(
            out_dir / "test_view_probs.npz",
            files=np.array(files),
            **{name: probs.astype(np.float16) for name, probs in view_probs.items()},
        )

    counts = np.bincount(predictions, minlength=num_classes)
    report = {
        "checkpoint": str(args.checkpoint),
        "weights": which,
        "epoch": payload.get("epoch"),
        "views": list(view_probs),
        "rows": len(files),
        "unique_files": len(set(files)),
        "prediction_count_min": int(counts.min()),
        "prediction_count_max": int(counts.max()),
        "mean_confidence": float(stacked.max(1).mean()),
        "train_metrics": metrics,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[infer] " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
