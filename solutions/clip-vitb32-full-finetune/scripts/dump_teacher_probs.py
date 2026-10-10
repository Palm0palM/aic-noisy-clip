"""Dump a teacher model's averaged TTA probabilities for the whole training pool.

Used for self-distillation: the student is trained on a blend of these soft
targets and the (label-smoothed) hard labels. Probabilities are averaged over the
same deterministic views the teacher uses at inference time, so nothing about the
test set enters here - only official training images are read.

    python scripts/dump_teacher_probs.py \
        --checkpoint checkpoints/v16/s3_576/last.pt \
        --cache /root/autodl-tmp/data/aic-rematch/train \
        --views center:576:1.0,flip:576:1.0,center:576:1.14,flip:576:1.14,center:576:1.28,center:576:1.4 \
        --output artifacts/teacher_probs_v16.npy

The result has one row per manifest row, so the student can index it with
record["index"] directly.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image, read_manifest  # noqa: E402


def degrade_image(image: Image.Image, mode: str) -> Image.Image:
    """Pre-registered mild degradations, applied before any view transform."""
    if mode == "none":
        return image
    if mode == "jpeg45_blur1":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=45)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.0))
    if mode == "canonicalize":
        # remove the flat/watermark edge bands the border detector fires on, so the
        # held-out accuracy can be read with and without them (test-set match check)
        from aic_clip.domain import canonicalize

        return canonicalize(image)
    if mode.startswith("saturation"):
        # scale the HSV saturation channel towards the test-set distribution
        factor = float(mode.split(":")[1]) if ":" in mode else 0.88
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
    raise SystemExit(f"unknown degrade mode: {mode}")


class ItemDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform, decode_cap: int = 0, degrade: str = "none"):
        self.root = root
        self.records = records
        self.transform = transform
        self.decode_cap = decode_cap
        self.degrade = degrade

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = load_image(self.root / rec["relative_path"], self.decode_cap)
        if self.degrade != "none":
            image = degrade_image(image, self.degrade)
        return self.transform(image), index


def view_transform(spec: str):
    """Use the production view builder.

    A local re-implementation used to live here and silently turned every
    non-`flip` kind - including the anchor kinds (tl/tc/bc/...) - into a centre
    crop, so an "anchor view" dump was really a duplicate of the centre view.
    Delegating to infer_ft.build_view keeps the two paths from drifting apart.
    """
    from aic_clip.infer_ft import build_view

    return build_view(spec, int(spec.split(":")[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True, help="directory holding the training images")
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--views", required=True)
    parser.add_argument("--weights", default="raw", choices=["auto", "raw", "ema"])
    parser.add_argument("--decode-cap", type=int, default=1152)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-file", default="", help="folds_2.json, to restrict the reported rate to a held-out fold")
    parser.add_argument("--eval-fold", type=int, default=-1, help="0 or 1: the fold this model did not train on")
    parser.add_argument("--degrade", default="none",
                        help="none / jpeg45_blur1 / canonicalize / saturation:<factor> (applied before the view transform)")
    parser.add_argument("--only-fold", type=int, default=-1,
                        help="run the model only on this fold's images (about half the work); "
                             "the output array keeps full manifest length with zeros elsewhere")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    device = torch.device("cuda")
    payload = torch.load(project / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    num_classes = int(payload["num_classes"])
    which = args.weights
    if which == "auto":
        which = (payload.get("metrics") or {}).get("chosen", "raw")
    state = payload.get(which) or payload.get("model")
    model = FTClassifier(
        cfg["model"]["backbone"], cfg["model"].get("revision"), num_classes,
        head=cfg["model"].get("head", "linear"), dropout=0.0,
        feature=payload.get("feature", cfg["model"].get("feature", "projected")),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Whole manifest, in manifest order, so the student can index the result with
    # record["index"] directly.
    records = read_manifest(project / args.manifest)
    subset_records = records
    if args.only_fold >= 0:
        if not args.fold_file:
            raise SystemExit("--only-fold needs --fold-file")
        folds = np.array(json.loads((project / args.fold_file).read_text(encoding="utf-8"))["fold"], dtype=np.int64)
        if len(folds) != len(records):
            raise SystemExit(f"fold file has {len(folds)} rows, manifest has {len(records)}")
        subset_records = [rec for rec, f in zip(records, folds) if int(f) == args.only_fold]
        print(f"[teacher] restricted to fold {args.only_fold}: {len(subset_records)} images", flush=True)
    labels = np.asarray([rec["label"] for rec in subset_records], dtype=np.int64)
    labels_all = np.asarray([rec["label"] for rec in records], dtype=np.int64)
    print(f"[teacher] {len(subset_records)} images, weights={which}", flush=True)

    # per-view summaries: top-1 class, its probability, the probability of the
    # manifest label, and the top1-minus-label margin. The margin is what tells
    # whether a held-out view is confidently against the label - the average
    # probability alone cannot show that.
    # The output arrays keep full manifest length regardless of which rows were
    # actually run, so downstream code can keep indexing by manifest row.
    views = [v for v in args.views.split(",") if v]
    if args.only_fold >= 0:
        run_rows = np.asarray([rec["index"] for rec in subset_records], dtype=np.int64)
    else:
        run_rows = np.arange(len(records), dtype=np.int64)
    summaries = np.full((len(views), len(records), 4), -1.0, dtype=np.float32)
    probs = np.zeros((len(records), num_classes), dtype=np.float32)
    started = time.time()
    for view_index, spec in enumerate(views):
        transform = view_transform(spec)
        loader = DataLoader(
            ItemDataset(Path(args.cache), subset_records, transform, args.decode_cap, args.degrade),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        )
        with torch.no_grad():
            for batch, indices in loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch).float()
                view_probs = logits.softmax(dim=1).cpu().numpy()
                rows = run_rows[indices.numpy()]
                probs[rows] += view_probs
                summaries[view_index, rows, 0] = view_probs.argmax(1)
                summaries[view_index, rows, 1] = view_probs.max(1)
                summaries[view_index, rows, 2] = view_probs[np.arange(len(rows)), labels[indices.numpy()]]
                summaries[view_index, rows, 3] = summaries[view_index, rows, 1] - summaries[view_index, rows, 2]
        print(f"[teacher] {spec}: {time.time() - started:.0f}s", flush=True)
    probs /= max(len(views), 1)

    # report agreement only on the fold this model was NOT trained on, so the
    # number is a genuine out-of-fold rate rather than a mixed in/out figure.
    # Rows that were not run (--only-fold) must stay out of the average: their
    # probabilities are still zero.
    mask = np.ones(len(records), dtype=bool)
    if args.fold_file:
        folds = np.array(json.loads((project / args.fold_file).read_text(encoding="utf-8"))["fold"], dtype=np.int64)
        if args.eval_fold >= 0:
            mask = folds == args.eval_fold
            print(f"[teacher] agreement below is restricted to fold {args.eval_fold} "
                  f"({int(mask.sum())} images, all held out)", flush=True)
    if args.only_fold >= 0:
        run_mask = np.zeros(len(records), dtype=bool)
        run_mask[run_rows] = True
        mask &= run_mask
        print(f"[teacher] agreement restricted further to the {int(mask.sum())} rows actually run", flush=True)
    agreement = float((probs[mask].argmax(1) == labels_all[mask]).mean())
    confidence = float(probs[mask].max(1).mean())
    print(f"[teacher] agreement with the manifest label (out-of-fold) {agreement:.4f}, "
          f"mean max prob {confidence:.4f}", flush=True)

    output = project / args.output
    np.save(output, probs.astype(np.float16))
    summary_path = output.with_name(output.stem + "_views.npz")
    np.savez_compressed(
        summary_path,
        views=np.array(views),
        summaries=summaries.astype(np.float16),
        labels=labels,
    )
    print(f"[teacher] wrote {output} ({probs.shape}) and {summary_path.name} "
          f"(per-view [top1, top1_prob, label_prob, margin])", flush=True)


if __name__ == "__main__":
    main()
