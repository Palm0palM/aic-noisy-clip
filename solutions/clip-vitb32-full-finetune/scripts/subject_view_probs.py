"""Per-image subject-crop view probabilities for the test set (inference only).

Companion to the fold-B diagnostic, at deployment scale: for every test image
the model's own Grad-CAM (patch tokens entering the last attention block) picks
a box, the box is squared so nothing inside it is cut away, and the crop is
re-encoded at the production size. Images whose box cannot be squared inside
the frame, is degenerate, or nearly fills the view are marked as fallbacks -
for those the caller should fall back to the plain ensemble, i.e. add no view.

This reads test images but never any label; nothing here touches training.

    python scripts/subject_view_probs.py --checkpoint checkpoints/v31/s3_576/last.pt \
        --test-dir /root/autodl-tmp/data/aic-rematch/test --size 512 \
        --output-dir artifacts/subject_test_v31
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image  # noqa: E402
from subject_crop_utils import cam_box, square_box  # noqa: E402
from aic_clip.infer_ft import IMAGE_EXTENSIONS

REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"


class TestDataset(Dataset):
    def __init__(self, root: Path, files: list[str]):
        self.root = root
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = load_image(self.root / self.files[index]).convert("RGB")
        return np.asarray(image), index


def collate_raw(batch):
    return [Image.fromarray(item[0]) for item in batch], np.asarray([item[1] for item in batch])


def centre_view(image: Image.Image, size: int, ratio: float = 1.0):
    scale = int(round(size * ratio)) / min(image.size)
    resized = image.resize(
        (int(round(image.width * scale)), int(round(image.height * scale))), Image.BICUBIC
    )
    left, top = (resized.width - size) // 2, (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def tensor(image: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)


def to_original(box, width: int, height: int, size: int):
    scale = size / min(width, height)
    off_x = (int(round(width * scale)) - size) // 2
    off_y = (int(round(height * scale)) - size) // 2
    return (
        max(0.0, (box[0] + off_x) / scale),
        max(0.0, (box[1] + off_y) / scale),
        min(float(width), (box[2] + off_x) / scale),
        min(float(height), (box[3] + off_y) / scale),
    )


@torch.no_grad()
def predict(model, device, images: list[Image.Image], size: int) -> np.ndarray:
    chunk = torch.stack([tensor(im) for im in images]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(chunk).float()
    probs = logits.softmax(-1).cpu().numpy()
    if not np.isfinite(probs).all():
        raise RuntimeError("Non-finite subject probabilities")
    return probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--files-json", default="")
    parser.add_argument("--weights", default="ema", choices=["auto", "raw", "ema"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--min-view-area", type=float, default=0.02)
    parser.add_argument("--max-view-area", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N images (smoke tests)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    if not (0 < args.tau <= 1 and args.margin >= 0 and 0 < args.min_view_area < args.max_view_area <= 1):
        parser.error("Invalid CAM or area parameters")
    if args.size <= 0 or args.size % 32 or args.batch_size <= 0 or args.limit < 0:
        parser.error("Invalid image size, batch size or image limit")

    started = time.time()
    project = Path(args.project_root).resolve()
    out = project / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "subject_view_probs.npz").exists():
        raise SystemExit("Refusing to overwrite existing subject probabilities; use a new output directory")

    root = Path(args.test_dir)
    files = (json.loads(Path(args.files_json).read_text())["files"] if args.files_json else
             sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
    if args.limit:
        files = files[: args.limit]
    print(f"[subject-test] {len(files)} images", flush=True)

    device = torch.device("cuda")
    payload = torch.load(project / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if cfg["model"].get("local_readout") or cfg["model"].get("revision") != REVISION:
        raise SystemExit("Requires the specified official CLIP revision and plain CLS head")
    which = args.weights
    if which == "auto":
        which = (payload.get("metrics") or {}).get("chosen", "raw")
    state = payload.get("model" if which == "raw" else which)
    if state is None:
        raise SystemExit(f"Requested {which} weights are absent; refusing silent fallback")
    model = FTClassifier(
        cfg["model"]["backbone"], cfg["model"].get("revision"), int(payload["num_classes"]),
        head=cfg["model"].get("head", "linear"), dropout=0.0,
        feature=payload.get("feature", cfg["model"].get("feature", "projected")),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)

    loader = DataLoader(
        TestDataset(root, files), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_raw,
    )
    probs = np.zeros((len(files), int(payload["num_classes"])), dtype=np.float32)
    fallback = np.zeros(len(files), dtype=bool)
    reason_codes = np.empty(len(files), dtype="U32")
    actual_boxes = np.full((len(files), 4), -1, dtype=np.int64)
    gradient_norms = np.zeros(len(files), dtype=np.float32)
    reasons = {}
    done = 0

    for images, rows in loader:
        first = [centre_view(im, args.size, 1.0) for im in images]
        inputs = torch.stack([tensor(im) for im in first]).to(device)
        captured = {}

        def capture(module, inputs, output):
            captured["hidden"] = output[0] if isinstance(output, tuple) else output

        handle = model.vision.vision_model.encoder.layers[-1].layer_norm1.register_forward_hook(capture)
        inputs.requires_grad_(True)
        try:
            with torch.enable_grad():
                logits = model(inputs)
                hidden = captured["hidden"]
                chosen = logits.argmax(1)
                grad = torch.autograd.grad(logits.gather(1, chosen[:, None]).sum(), hidden)[0]
        finally:
            handle.remove()
        tokens, patch_grad = hidden.detach()[:, 1:, :], grad.detach()[:, 1:, :]
        norms = patch_grad.norm(dim=(1, 2))
        if not torch.isfinite(norms).all() or not bool((norms > 1e-6).any()):
            raise RuntimeError("Broken localisation gradient path")
        gradient_norms[rows] = norms.cpu().numpy()
        grid = int(round(tokens.shape[1] ** 0.5))
        if grid * grid != tokens.shape[1]:
            raise RuntimeError("Patch grid is not square")
        cam = torch.relu((tokens * patch_grad.mean(dim=1, keepdim=True)).sum(-1)).view(-1, grid, grid)
        cams = cam.detach().cpu().numpy()
        del logits, hidden, grad, tokens, patch_grad, cam, inputs, captured, norms

        second = []
        for row, image in enumerate(images):
            index = rows[row].item()
            box, reason = cam_box(cams[row], args.tau, args.margin, args.size)
            actual = None
            if box is not None:
                raw_area = (box[2] - box[0]) * (box[3] - box[1]) / (args.size ** 2)
                original = to_original(box, image.width, image.height, args.size)
                actual = square_box(original, image.width, image.height)
                if raw_area < args.min_view_area:
                    reason = "too_small_cam_box"
                elif actual is None:
                    reason = "square_cannot_preserve_box"
                elif min(original[2] - original[0], original[3] - original[1]) < 8:
                    reason = "too_small_original_box"
                elif ((actual[2] - actual[0]) / min(image.size)) ** 2 > args.max_view_area:
                    reason = "near_full_square"
            if gradient_norms[index] <= 1e-6:
                reason = "zero_patch_gradient"
            reason_codes[index] = reason
            if reason != "ok":
                fallback[index] = True
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                reasons["roi"] = reasons.get("roi", 0) + 1
                actual_boxes[index] = actual
                second.append(image.crop(actual).resize((args.size, args.size), Image.BICUBIC))
        if second:
            use = [j for j, idx in enumerate(rows) if not fallback[idx]]
            probs[rows[use]] = predict(model, device, second, args.size)
        done += len(rows)
        if done % 2000 < args.batch_size or done == len(files):
            print(f"[subject-test] {done}/{len(files)} {(time.time() - started) / 60:.1f}min", flush=True)

    np.savez_compressed(
        out / "subject_view_probs.npz",
        files=np.array(files),
        probs=probs,
        fallback=fallback,
        reasons=reason_codes,
        boxes=actual_boxes,
        patch_grad_norm=gradient_norms,
    )
    report = {
        "images": len(files), "size": args.size, "weights": which,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.file_digest(open(project / args.checkpoint, "rb"), "sha256").hexdigest(),
        "saved_probs_dtype": str(probs.dtype), "batch_size": args.batch_size,
        "tau": args.tau, "margin": args.margin,
        "min_view_area": args.min_view_area, "max_view_area": args.max_view_area,
        "patch_grad_norm_min": float(gradient_norms.min()),
        "minutes": round((time.time() - started) / 60, 1),
        "fallback_fraction": float(fallback.mean()),
        "reasons": reasons,
        "note": "fallback rows carry zero vectors; the caller must use the plain ensemble there",
    }
    (out / "subject_view_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
