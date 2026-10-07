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
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image, read_manifest  # noqa: E402


class ItemDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform, decode_cap: int = 0):
        self.root = root
        self.records = records
        self.transform = transform
        self.decode_cap = decode_cap

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        return self.transform(load_image(self.root / rec["relative_path"], self.decode_cap)), index


def view_transform(spec: str):
    parts = spec.split(":")
    kind, size = parts[0], int(parts[1])
    ratio = float(parts[2]) if len(parts) > 2 else 1.14
    ops = [
        T.Resize(int(round(size * ratio)), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
    ]
    if kind == "flip":
        ops.append(T.RandomHorizontalFlip(p=1.0))
    ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    return T.Compose(ops)


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
    labels = np.asarray([rec["label"] for rec in subset_records], dtype=np.int64)
    print(f"[teacher] {len(subset_records)} images, weights={which}", flush=True)

    # per-view summaries: top-1 class, its probability, the probability of the
    # manifest label, and the top1-minus-label margin. The margin is what tells
    # whether a held-out view is confidently against the label - the average
    # probability alone cannot show that.
    views = [v for v in args.views.split(",") if v]
    summaries = np.full((len(views), len(subset_records), 4), -1.0, dtype=np.float32)
    probs = np.zeros((len(subset_records), num_classes), dtype=np.float32)
    started = time.time()
    for view_index, spec in enumerate(views):
        transform = view_transform(spec)
        loader = DataLoader(
            ItemDataset(Path(args.cache), subset_records, transform, args.decode_cap),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        )
        with torch.no_grad():
            for batch, indices in loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch).float()
                view_probs = logits.softmax(dim=1).cpu().numpy()
                rows = indices.numpy()
                probs[rows] += view_probs
                summaries[view_index, rows, 0] = view_probs.argmax(1)
                summaries[view_index, rows, 1] = view_probs.max(1)
                summaries[view_index, rows, 2] = view_probs[np.arange(len(rows)), labels[rows]]
                summaries[view_index, rows, 3] = summaries[view_index, rows, 1] - summaries[view_index, rows, 2]
        print(f"[teacher] {spec}: {time.time() - started:.0f}s", flush=True)
    probs /= max(len(views), 1)

    # report agreement only on the fold this model was NOT trained on, so the
    # number is a genuine out-of-fold rate rather than a mixed in/out figure
    mask = np.ones(len(subset_records), dtype=bool)
    if args.fold_file:
        folds = np.array(json.loads((project / args.fold_file).read_text(encoding="utf-8"))["fold"], dtype=np.int64)
        if args.eval_fold >= 0:
            mask = folds == args.eval_fold
            print(f"[teacher] agreement below is restricted to fold {args.eval_fold} "
                  f"({int(mask.sum())} images, all held out)", flush=True)
    agreement = float((probs[mask].argmax(1) == labels[mask]).mean())
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
