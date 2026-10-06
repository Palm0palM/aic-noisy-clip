"""Auditable, fixed-weight subject-crop diagnostic on an unseen training fold.

The localiser and classifier use exactly the same checkpoint. No optimiser or
weight update is performed. Compare two prediction views, with the additional
localisation forward/backward cost reported separately from that view count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aic_clip.train_ft import CLIP_MEAN, CLIP_STD, FTClassifier, load_image, read_manifest
from subject_crop_utils import cam_box, paired_summary, square_box

REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centre_image(image, size, ratio=1.0):
    scale = int(round(size * ratio)) / min(image.size)
    resized = image.resize((int(round(image.width * scale)), int(round(image.height * scale))), Image.Resampling.BICUBIC)
    left, top = (resized.width - size) // 2, (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def tensor(image):
    return TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)


def to_original(box, width, height, size):
    scale = size / min(width, height)
    off_x = (int(round(width * scale)) - size) // 2
    off_y = (int(round(height * scale)) - size) // 2
    return (max(0.0, (box[0] + off_x) / scale), max(0.0, (box[1] + off_y) / scale),
            min(float(width), (box[2] + off_x) / scale), min(float(height), (box[3] + off_y) / scale))


class FoldDataset(Dataset):
    def __init__(self, root, records):
        self.root, self.records = root, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        im = load_image(self.root / self.records[index]["relative_path"]).convert("RGB")
        return np.asarray(im), index


def collate_raw(batch):
    return [Image.fromarray(item[0]) for item in batch], np.asarray([item[1] for item in batch])


@torch.no_grad()
def predict(model, device, images):
    inputs = torch.stack([tensor(im) for im in images]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs).float()
    probs = logits.softmax(-1).cpu().numpy()
    if not np.isfinite(probs).all():
        raise RuntimeError("Non-finite prediction probabilities")
    return probs


def save_montage(rows, path):
    # Each row shows original+actual square, actual first view, actual second view.
    tile, caption = 190, 22
    canvas = Image.new("RGB", (3 * tile, len(rows) * (tile + caption)), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    for row, (im, first, second, box, rec) in enumerate(rows):
        original = im.copy()
        if box is not None:
            ImageDraw.Draw(original).rectangle(box, outline=(255, 64, 64), width=max(2, min(im.size) // 100))
        for col, view in enumerate((original, first, second)):
            shown = view.copy()
            shown.thumbnail((tile, tile), Image.Resampling.BICUBIC)
            canvas.paste(shown, (col * tile, row * (tile + caption)))
        draw.text((2, row * (tile + caption) + tile), f"id={rec['index']} {rec['fallback_reason'] or 'roi'}", fill="white")
        draw.text((tile + 2, row * (tile + caption) + tile), "actual center input", fill="white")
        draw.text((2 * tile + 2, row * (tile + caption) + tile), "actual second input", fill="white")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", default="/root/autodl-tmp/data/aic-rematch/train")
    parser.add_argument("--manifest", default="artifacts/train_manifest.csv")
    parser.add_argument("--folds", default="artifacts/folds_2.json")
    parser.add_argument("--dedup-drop", default="artifacts/dedup_drop.npy")
    parser.add_argument("--training-drop", default="artifacts/oof_a_drop.npy")
    parser.add_argument("--eval-fold", type=int, default=1)
    parser.add_argument("--images", type=int, default=1200)
    parser.add_argument("--reference-images", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--min-view-area", type=float, default=0.02)
    parser.add_argument("--max-view-area", type=float, default=0.95)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-budget-minutes", type=float, default=30.0)
    args = parser.parse_args()
    if not (0 < args.tau <= 1 and 0 <= args.margin and 0 < args.min_view_area < args.max_view_area <= 1):
        parser.error("Invalid CAM or area parameters")
    if min(args.images, args.reference_images, args.batch_size, args.size) <= 0 or args.size % 32:
        parser.error("Positive counts and size divisible by 32 required")

    started = time.monotonic()
    project = Path(__file__).resolve().parents[1]
    out = project / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite a nonempty output directory: {out}")
    records = read_manifest(project / args.manifest)
    folds = np.asarray(json.loads((project / args.folds).read_text(encoding="utf-8"))["fold"], dtype=np.int64)
    ids_all = np.asarray([r["index"] for r in records], dtype=np.int64)
    if len(folds) != len(records) or not np.array_equal(ids_all, np.arange(len(records))):
        raise SystemExit("Manifest IDs/fold rows must be aligned and contiguous")
    dedup = np.load(project / args.dedup_drop, allow_pickle=False)
    training_drop = np.load(project / args.training_drop, allow_pickle=False)
    held = np.flatnonzero(folds == args.eval_fold)
    train_keep = np.setdiff1d(ids_all, training_drop)
    expected_keep = np.setdiff1d(ids_all[folds != args.eval_fold], dedup)
    if len(held) == 0 or not np.array_equal(train_keep, expected_keep):
        raise SystemExit("Training-drop manifest does not match the opposite fold minus dedup")
    ids = held[np.sort(np.random.default_rng(0).choice(len(held), min(args.images, len(held)), replace=False))]
    old = held[np.sort(np.random.default_rng(0).choice(len(held), min(args.reference_images, len(held)), replace=False))]
    subset = [records[i] for i in ids]
    is_new, retained = ~np.isin(ids, old), ~np.isin(ids, dedup)
    np.save(out / "sample_ids.npy", ids)
    np.save(out / "reference_ids.npy", old)

    device = torch.device("cuda")
    payload = torch.load(project / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if cfg["model"].get("local_readout") or cfg["model"].get("revision") != REVISION:
        raise SystemExit("Requires the specified official CLIP revision with a plain CLS head")
    if payload.get("ema") is None:
        raise SystemExit("EMA weights are required; refusing silent raw-weight fallback")
    model = FTClassifier(cfg["model"]["backbone"], cfg["model"].get("revision"), int(payload["num_classes"]),
                         head=cfg["model"].get("head", "linear"), dropout=0.0,
                         feature=payload.get("feature", cfg["model"].get("feature", "projected"))).to(device)
    model.load_state_dict(payload["ema"], strict=True)
    model.eval().requires_grad_(False)
    provenance = {
        "version": "subject_crop_v3", "args": vars(args), "checkpoint_epoch": payload.get("epoch"),
        "weights": "ema", "training_drop_matches_opposite_fold": True,
        "training_drop_note": "Checked supplied drop list; checkpoint itself does not embed the historical training ID list.",
        "hashes": {name: sha256(project / path) for name, path in {
            "checkpoint": args.checkpoint, "manifest": args.manifest, "folds": args.folds,
            "dedup_drop": args.dedup_drop, "training_drop": args.training_drop,
            "script": str(Path(__file__).resolve()),
            "helpers": str(Path(__file__).with_name("subject_crop_utils.py").resolve()),
        }.items()},
        "n": len(ids), "overlap_reference": int((~is_new).sum()), "dedup_count": int((~retained).sum()),
        "new_retained_count": int((is_new & retained).sum()),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({k: provenance[k] for k in ("version", "n", "overlap_reference", "dedup_count", "new_retained_count", "checkpoint_epoch")}), flush=True)
    del payload

    loader = DataLoader(FoldDataset(Path(args.cache), subset), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_raw)
    n, classes = len(ids), int(model.head.weight.shape[0]) if hasattr(model.head, "weight") else int(cfg["data"]["num_classes"])
    full_probs = np.zeros((n, classes), dtype=np.float32)
    wide_probs, crop_probs = np.zeros_like(full_probs), np.zeros_like(full_probs)
    labels = np.asarray([r["label"] for r in subset])
    fallback = np.zeros(n, dtype=bool)
    near_full = np.zeros(n, dtype=bool)
    grad_norms = np.zeros(n, dtype=np.float32)
    reasons, montage = Counter(), []
    # Spread visual checks across the sampled manifest rather than its first classes.
    montage_indices = set(np.linspace(0, n - 1, min(n, 12), dtype=int).tolist())
    prediction_seconds, localisation_seconds = 0.0, 0.0
    done = 0
    with (out / "per_image.jsonl").open("w", encoding="utf-8") as audit:
        for images, rows in loader:
            first = [centre_image(im, args.size) for im in images]
            inputs = torch.stack([tensor(im) for im in first]).to(device).requires_grad_(True)
            captured = {}
            handle = model.vision.vision_model.encoder.layers[-1].layer_norm1.register_forward_hook(
                lambda module, ins, output: captured.__setitem__("hidden", output))
            torch.cuda.synchronize()
            tic = time.monotonic()
            try:
                with torch.enable_grad():
                    logits = model(inputs)
                    hidden = captured["hidden"]
                    chosen = logits.argmax(1)
                    grad = torch.autograd.grad(logits.gather(1, chosen[:, None]).sum(), hidden)[0]
            finally:
                handle.remove()
            tokens, patch_grad = hidden.detach()[:, 1:, :], grad.detach()[:, 1:, :]
            norm = patch_grad.norm(dim=(1, 2))
            if not torch.isfinite(norm).all() or not bool((norm > 1e-6).any()):
                raise RuntimeError("Broken localisation gradient path")
            grid = int(round(tokens.shape[1] ** 0.5))
            if grid * grid != tokens.shape[1]:
                raise RuntimeError("Patch grid is not square")
            cams = (tokens * patch_grad.mean(dim=1, keepdim=True)).sum(-1).relu().reshape(-1, grid, grid).cpu().numpy()
            grad_norms[rows] = norm.cpu().numpy()
            torch.cuda.synchronize()
            localisation_seconds += time.monotonic() - tic
            del logits, hidden, grad, tokens, patch_grad, norm, inputs, captured

            wide = [centre_image(im, args.size, 1.14) for im in images]
            second, batch_records = [], []
            for j, (im, idx) in enumerate(zip(images, rows)):
                raw_box, reason = cam_box(cams[j], args.tau, args.margin, args.size)
                reason = "" if reason == "ok" else reason
                original_box, actual_box, raw_area, actual_area = None, None, None, None
                if raw_box is not None:
                    raw_area = (raw_box[2] - raw_box[0]) * (raw_box[3] - raw_box[1]) / args.size ** 2
                    original_box = to_original(raw_box, im.width, im.height, args.size)
                    actual_box = square_box(original_box, im.width, im.height)
                    if raw_area < args.min_view_area:
                        reason = "too_small_cam_box"
                    elif actual_box is None:
                        reason = "square_cannot_preserve_box"
                    elif min(original_box[2] - original_box[0], original_box[3] - original_box[1]) < 8:
                        reason = "too_small_original_box"
                    if actual_box is not None:
                        actual_area = ((actual_box[2] - actual_box[0]) / min(im.size)) ** 2
                        if actual_area > args.max_view_area:
                            near_full[idx] = True
                            reason = "near_full_square"
                if grad_norms[idx] <= 1e-6:
                    reason = "zero_patch_gradient"
                fallback[idx] = bool(reason)
                reasons[reason or "roi"] += 1
                crop = wide[j] if reason else im.crop(actual_box).resize((args.size, args.size), Image.Resampling.BICUBIC)
                second.append(crop)
                rec = {"row": int(idx), "index": int(ids[idx]), "relative_path": subset[idx]["relative_path"],
                       "label": int(labels[idx]), "new_vs_reference": bool(is_new[idx]), "retained": bool(retained[idx]),
                       "fallback_reason": reason, "cam_box_view": raw_box, "cam_box_original": original_box,
                       "square_box_original": actual_box, "raw_area_view": raw_area, "actual_square_area_view": actual_area,
                       "patch_grad_norm": float(grad_norms[idx]), "localiser_class": int(chosen[j].item())}
                batch_records.append(rec)
                if int(idx) in montage_indices:
                    montage.append((im, first[j], crop, None if reason else actual_box, rec))
            torch.cuda.synchronize()
            tic = time.monotonic()
            full_probs[rows] = predict(model, device, first)
            wide_probs[rows] = predict(model, device, wide)
            # Reuse exactly the plain second-view probabilities for fallback rows.
            crop_probs[rows] = wide_probs[rows]
            use = [j for j, idx in enumerate(rows) if not fallback[idx]]
            if use:
                crop_probs[rows[use]] = predict(model, device, [second[j] for j in use])
            torch.cuda.synchronize()
            prediction_seconds += time.monotonic() - tic
            for rec in batch_records:
                idx = rec["row"]
                rec["plain_pred"] = int((full_probs[idx] + wide_probs[idx]).argmax())
                rec["subject_pred"] = int((full_probs[idx] + crop_probs[idx]).argmax())
                audit.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
            audit.flush()
            done += len(rows)
            if done % 200 < args.batch_size or done == n:
                print(f"[subject-v3] {done}/{n} elapsed={(time.monotonic()-started)/60:.1f}min", flush=True)
            if (time.monotonic() - started) / 60 > args.time_budget_minutes and done < n:
                raise RuntimeError("Time budget exhausted; partial per-image audit preserved, no completed score reported")

    plain = (full_probs + wide_probs) * 0.5
    subject = (full_probs + crop_probs) * 0.5
    if not np.array_equal(plain[fallback], subject[fallback]):
        raise RuntimeError("Fallback predictions differ from baseline")
    pp, sp = plain.argmax(1), subject.argmax(1)
    masks = {"all": np.ones(n, dtype=bool), "new": is_new, "retained": retained,
             "new_retained": is_new & retained, "roi_used": ~fallback, "fallback": fallback, "near_full": near_full}
    report = {"version": "subject_crop_v3", "complete": True, "images": n,
              "minutes": (time.monotonic() - started) / 60, "localisation_gpu_seconds": localisation_seconds,
              "prediction_gpu_seconds": prediction_seconds,
              "cost_note": "Paired run includes shared center, plain wide, ROI prediction and extra CAM forward/backward; unequal arm compute.",
              "fallback_reasons": dict(reasons), "patch_grad_norm_mean": float(grad_norms.mean()),
              "patch_grad_norm_min": float(grad_norms.min()), "patch_grad_zero_fraction": float((grad_norms <= 1e-6).mean()),
              "statistics_note": "Approximate paired intervals assume independent images; dedup/group dependence may widen them.",
              "groups": {name: paired_summary(pp, sp, labels, mask) for name, mask in masks.items()}}
    np.savez_compressed(out / "probabilities.npz", ids=ids, labels=labels, full=full_probs, wide=wide_probs,
                        crop=crop_probs, fallback=fallback, new=is_new, retained=retained)
    save_montage(montage, out / "actual_inputs.png")
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
