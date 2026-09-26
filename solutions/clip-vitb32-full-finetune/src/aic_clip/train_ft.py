"""V10: full fine-tuning of CLIP ViT-B/32 for noisy fine-grained classification.

Single-model, single-inference-flow pipeline. Uses only the official training
split (no test images, no test labels, no external data at any point).

Key ingredients (all standard, all reproducible from this file + config):
  * full fine-tuning of the official openai/clip-vit-base-patch32 vision tower
  * class-balanced sampling to match the (near-uniform) evaluation prior
  * RandomResizedCrop + flip + colour jitter + RandAugment + random erasing
  * mixup / cutmix on label-smoothed soft targets
  * EMA weights, BF16 autocast, separate LR for backbone and head
  * optional ELR regulariser for noisy labels
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"  # openai/clip-vit-base-patch32


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_image(path: Path, decode_cap: int = 0) -> Image.Image:
    """Decode any format Pillow understands, tolerate broken EXIF/truncation.

    decode_cap > 0 lets Pillow's JPEG draft mode decode a downscaled version,
    which is a large speed-up when training below the native image resolution.
    """
    with Image.open(path) as im:
        if decode_cap and im.format == "JPEG":
            try:
                im.draft("RGB", (decode_cap, decode_cap))
            except Exception:
                pass
        try:
            im.load()
        except Exception:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            im = Image.open(path)
            im.load()
        try:
            transposed = ImageOps.exif_transpose(im)
        except Exception:
            transposed = None
        if transposed is not None and transposed is not im:
            im = transposed
        if im.mode != "RGB":
            im = im.convert("RGB")
        else:
            im = im.copy()
    return im


class ManifestDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform, decode_cap: int = 0):
        self.root = root
        self.records = records
        self.transform = transform
        self.decode_cap = decode_cap

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = load_image(self.root / rec["relative_path"], self.decode_cap)
        return self.transform(image), rec["label"], index


def read_manifest(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "relative_path": row["relative_path"],
            "label": int(row["label"]),
            "index": int(row.get("index", i)),
        }
        for i, row in enumerate(rows)
    ]


def load_split(path: Path, records: list[dict]) -> tuple[list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup = {rec["index"]: i for i, rec in enumerate(records)}
    lookup_by_path = {rec["relative_path"]: i for i, rec in enumerate(records)}

    def resolve(items):
        out = []
        for item in items:
            if isinstance(item, int):
                if item in lookup:
                    out.append(lookup[item])
                elif 0 <= item < len(records):
                    out.append(item)
            else:
                key = str(item)
                if key in lookup_by_path:
                    out.append(lookup_by_path[key])
        return sorted(set(out))

    train_idx = resolve(payload["train"])
    val_idx = resolve(payload["val"])
    return train_idx, val_idx


def build_train_transform(image_size: int, augment: dict):
    ops = [
        T.RandomResizedCrop(
            image_size,
            scale=tuple(augment.get("rrc_scale", (0.35, 1.0))),
            ratio=tuple(augment.get("rrc_ratio", (3 / 4, 4 / 3))),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
    ]
    jitter = float(augment.get("color_jitter", 0.4))
    if jitter > 0:
        ops.append(T.ColorJitter(jitter, jitter, jitter, jitter / 4))
    if augment.get("rand_augment", True):
        ops.append(
            T.RandAugment(
                num_ops=int(augment.get("rand_augment_ops", 2)),
                magnitude=int(augment.get("rand_augment_magnitude", 9)),
                interpolation=T.InterpolationMode.BICUBIC,
            )
        )
    ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    if float(augment.get("random_erase", 0.25)) > 0:
        ops.append(T.RandomErasing(p=float(augment["random_erase"]), value="random"))
    return T.Compose(ops)


def build_eval_transform(image_size: int, resize_ratio: float = 1.14):
    resize = int(round(image_size * resize_ratio))
    return T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class CosineHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feat = F.normalize(features.float(), dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return self.logit_scale.exp() * feat @ weight.t()


class FTClassifier(nn.Module):
    def __init__(self, backbone: str, revision: str, num_classes: int, head: str = "linear", dropout: float = 0.0):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.vision = CLIPVisionModelWithProjection.from_pretrained(
            backbone, revision=revision, use_safetensors=True
        )
        dim = int(self.vision.config.projection_dim)
        self.dropout = nn.Dropout(dropout)
        if head == "cosine":
            self.head = CosineHead(dim, num_classes)
        else:
            self.head = nn.Linear(dim, num_classes)

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        configured = int(self.vision.config.image_size)
        interpolate = height != configured or width != configured
        return self.vision(pixel_values=pixel_values, interpolate_pos_encoding=interpolate).image_embeds

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(pixel_values)))

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        ]

    def head_params(self):
        return list(self.head.parameters())


def build_optimizer(model: FTClassifier, cfg: dict):
    head_ids = {id(p) for p in model.head.parameters()}
    decay_bb, nodecay_bb, decay_head, nodecay_head = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target_decay = decay_head if id(param) in head_ids else decay_bb
        target_nodecay = nodecay_head if id(param) in head_ids else nodecay_bb
        if param.ndim <= 1 or name.endswith(".bias"):
            target_nodecay.append(param)
        else:
            target_decay.append(param)
    wd = float(cfg["weight_decay"])
    return torch.optim.AdamW(
        [
            {"params": decay_bb, "lr": float(cfg["lr_backbone"]), "weight_decay": wd},
            {"params": nodecay_bb, "lr": float(cfg["lr_backbone"]), "weight_decay": 0.0},
            {"params": decay_head, "lr": float(cfg["lr_head"]), "weight_decay": wd},
            {"params": nodecay_head, "lr": float(cfg["lr_head"]), "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for key, value in model.state_dict().items():
            stored = self.state[key]
            if stored.dtype.is_floating_point:
                stored.mul_(self.decay).add_(value.detach().float(), alpha=1.0 - self.decay)
            else:
                stored.copy_(value)


# --------------------------------------------------------------------------- #
# mixup / cutmix / loss
# --------------------------------------------------------------------------- #
def one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    target = torch.full((labels.size(0), num_classes), smoothing / num_classes, device=labels.device)
    target.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing + smoothing / num_classes)
    return target


def rand_bbox(height: int, width: int, lam: float, device):
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = torch.randint(height, (1,), device=device).item(), torch.randint(width, (1,), device=device).item()
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    return y1, y2, x1, x2


def apply_mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha, prob, num_classes):
    if prob <= 0 or random.random() > prob:
        return images, targets, None
    index = torch.randperm(images.size(0), device=images.device)
    if mixup_alpha > 0 and (cutmix_alpha <= 0 or random.random() < 0.5):
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        images = images * lam + images[index] * (1.0 - lam)
    elif cutmix_alpha > 0:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        y1, y2, x1, x2 = rand_bbox(images.size(2), images.size(3), lam, images.device)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(2) * images.size(3)))
    else:
        return images, targets, None
    return images, lam * targets + (1.0 - lam) * targets[index], index


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, amp_dtype, device) -> dict:
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    loss_sum, seen = 0.0, 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        pred = logits.argmax(1)
        np.add.at(correct, labels.cpu().numpy(), (pred.cpu().numpy() == labels.cpu().numpy()).astype(np.int64))
        np.add.at(total, labels.cpu().numpy(), 1)
        loss_sum += float(loss)
        seen += labels.numel()
    present = total > 0
    per_class = correct[present] / np.maximum(total[present], 1)
    counts = total.copy()
    order = np.argsort(counts[present])
    per_present = per_class[np.argsort(counts[present])]
    third = max(len(per_present) // 3, 1)
    return {
        "accuracy": float(correct.sum() / max(seen, 1)),
        "macro_accuracy": float(per_class.mean()) if present.any() else 0.0,
        "tail_accuracy": float(per_present[:third].mean()) if len(per_present) else 0.0,
        "mid_accuracy": float(per_present[third: 2 * third].mean()) if len(per_present) else 0.0,
        "head_accuracy": float(per_present[2 * third:].mean()) if len(per_present) else 0.0,
        "loss": loss_sum / max(seen, 1),
        "classes_present": int(present.sum()),
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, epochs: float, warmup_epochs: float, steps_per_epoch: int, min_lr_ratio: float):
    total_steps = max(int(epochs * steps_per_epoch), 1)
    warmup_steps = max(int(warmup_epochs * steps_per_epoch), 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", type=int, default=0, help="limit optimiser steps for a smoke test")
    parser.add_argument("--resume", default="")
    parser.add_argument("--initialize", default="", help="load model weights from a checkpoint (weights only)")
    parser.add_argument("--init-weights", default="auto", choices=["auto", "raw", "ema"])
    parser.add_argument("--train-on-all", action="store_true",
                        help="train on the full manifest (train+holdout); holdout metrics then become in-sample and are only logged")
    parser.add_argument("--drop-indices", default="", help="npy with manifest positions to exclude from training")
    parser.add_argument("--snapshot-dir", default="", help="if set, save the raw weights after every epoch (for SWA)")
    parser.add_argument("--image-size", type=int, default=0, help="override config image size")
    parser.add_argument("--epochs", type=int, default=0, help="override config epochs")
    parser.add_argument("--tag", default="", help="suffix appended to output dir")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.image_size:
        cfg["data"]["image_size"] = args.image_size
        cfg["data"]["eval_size"] = args.image_size
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    seed = int(cfg["train"].get("seed", 20260926))
    set_seed(seed)

    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if cfg["train"].get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16

    project = Path(cfg["paths"]["project_root"]).resolve()
    manifest_path = project / cfg["data"]["manifest"]
    split_path = project / cfg["data"]["split"]
    train_root = Path(cfg["data"]["train_dir"])

    records = read_manifest(manifest_path)
    if len(records) != int(cfg["data"]["expected_train_images"]):
        raise ValueError(f"manifest size {len(records)} != expected {cfg['data']['expected_train_images']}")
    train_idx, val_idx = load_split(split_path, records)
    if not val_idx:
        raise ValueError("empty validation split")
    if args.train_on_all:
        print(f"[data] train_on_all: using {len(records)} samples (holdout metrics are in-sample)", flush=True)
        train_idx = list(range(len(records)))
    if args.drop_indices:
        drop = {int(x) for x in np.load(args.drop_indices)}
        before = len(train_idx)
        train_idx = [i for i in train_idx if i not in drop]
        print(f"[data] drop_indices: removed {before - len(train_idx)} samples "
              f"(keep {len(train_idx)})", flush=True)
    if args.snapshot_dir:
        Path(args.snapshot_dir).mkdir(parents=True, exist_ok=True)

    image_size = int(cfg["data"]["image_size"])
    eval_size = int(cfg["data"]["eval_size"])
    train_tf = build_train_transform(image_size, cfg["augment"])
    eval_tf = build_eval_transform(eval_size, float(cfg["data"].get("eval_resize_ratio", 1.14)))

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    if args.smoke:
        val_records = val_records[:1500]
    decode_cap = int(cfg["data"].get("decode_cap", 0))
    train_ds = ManifestDataset(train_root, train_records, train_tf, decode_cap)
    val_ds = ManifestDataset(train_root, val_records, eval_tf, decode_cap)

    num_classes = int(cfg["data"]["num_classes"])
    counts = np.bincount([r["label"] for r in train_records], minlength=num_classes).astype(np.float64)
    scheme = cfg["sampler"].get("scheme", "sqrt_inv")
    power = float(cfg["sampler"].get("power", 0.5))
    if scheme == "uniform":
        weights = np.ones(num_classes)
    elif scheme == "inv":
        weights = 1.0 / np.maximum(counts, 1.0)
    else:
        weights = 1.0 / np.power(np.maximum(counts, 1.0), power)
    sample_weights = weights[[r["label"] for r in train_records]]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_records),
        replacement=True,
    )

    loader_kwargs = dict(
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 4)) if int(cfg["data"]["num_workers"]) > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["data"]["batch_size"]), sampler=sampler, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["data"]["val_batch_size"]), shuffle=False, **loader_kwargs
    )

    model = FTClassifier(
        cfg["model"]["backbone"],
        cfg["model"].get("revision", DEFAULT_REVISION),
        num_classes,
        head=cfg["model"].get("head", "linear"),
        dropout=float(cfg["model"].get("dropout", 0.0)),
    ).to(device)

    if cfg["model"].get("freeze_patch_embed", False):
        for param in model.vision.vision_model.embeddings.parameters():
            param.requires_grad = False

    if args.initialize:
        payload = torch.load(args.initialize, map_location="cpu", weights_only=False)
        which = args.init_weights
        if which == "auto":
            which = (payload.get("metrics") or {}).get("chosen", "ema") if isinstance(payload.get("metrics"), dict) else "ema"
        state = payload.get(which) or payload.get("model") or payload.get("ema") or payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init] loaded {args.initialize} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    optimizer = build_optimizer(model, cfg["train"])
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        float(cfg["train"]["epochs"]),
        float(cfg["train"].get("warmup_epochs", 1.5)),
        steps_per_epoch,
        float(cfg["train"].get("min_lr_ratio", 0.01)),
    )
    ema = ModelEMA(model, float(cfg["train"].get("ema_decay", 0.9998)))

    elr_lambda = float(cfg["train"].get("elr_lambda", 0.0))
    elr_momentum = float(cfg["train"].get("elr_momentum", 0.9))
    elr_targets = None
    if elr_lambda > 0:
        elr_targets = torch.zeros(len(train_records), num_classes, device=device)

    out_dir = project / cfg["train"]["output_dir"]
    if args.tag:
        out_dir = out_dir.parent / (out_dir.name + "_" + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.json"
    history = []
    start_epoch = 0
    best_score = -1.0

    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("ema"):
            ema.state = {k: v.float() for k, v in payload["ema"].items()}
        if payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        history = payload.get("history", [])
        best_score = float(payload.get("best_score", -1.0))
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"[resume] from {args.resume} at epoch {start_epoch}", flush=True)

    epochs = float(cfg["train"]["epochs"])
    label_smoothing = float(cfg["augment"].get("label_smoothing", 0.1))
    mix_prob = float(cfg["augment"].get("mix_prob", 0.5))
    mixup_alpha = float(cfg["augment"].get("mixup", 0.2))
    cutmix_alpha = float(cfg["augment"].get("cutmix", 1.0))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    patience = int(cfg["train"].get("early_stop_patience", 4))
    log_every = int(cfg["train"].get("log_every", 50))
    since_improved = 0

    print(json.dumps({k: cfg[k] for k in ("data", "model", "train", "augment", "sampler")}, ensure_ascii=False), flush=True)
    print(f"[setup] train={len(train_records)} val={len(val_records)} steps/epoch={steps_per_epoch}", flush=True)

    for epoch in range(start_epoch, int(math.ceil(epochs))):
        model.train()
        epoch_start = time.time()
        running_loss, running_acc, seen = 0.0, 0.0, 0
        for step, (images, labels, indices) in enumerate(train_loader):
            if args.smoke and step >= args.smoke:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target = one_hot(labels, num_classes, label_smoothing)
            images, target, _ = apply_mixup_cutmix(
                images, target, mixup_alpha, cutmix_alpha, mix_prob, num_classes
            )
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(images)
            loss = soft_cross_entropy(logits.float(), target)
            if elr_lambda > 0:
                with torch.no_grad():
                    probs = logits.detach().float().softmax(dim=1)
                    elr_targets[indices] = (
                        elr_momentum * elr_targets[indices] + (1.0 - elr_momentum) * probs
                    )
                    dot = (probs * elr_targets[indices]).sum(dim=1).clamp(max=1.0 - 1e-4)
                loss = loss + elr_lambda * torch.log(1.0 - dot).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            running_loss += float(loss.detach()) * labels.numel()
            running_acc += float((logits.detach().argmax(1) == labels).float().sum())
            seen += labels.numel()
            if log_every and (step + 1) % log_every == 0:
                print(
                    f"epoch {epoch} step {step + 1}/{steps_per_epoch} "
                    f"loss {running_loss / seen:.4f} acc {running_acc / seen:.4f} "
                    f"lr {optimizer.param_groups[0]['lr']:.3e} "
                    f"{(time.time() - epoch_start) / (step + 1):.2f}s/step",
                    flush=True,
                )

        del images, labels, logits, loss
        torch.cuda.empty_cache()

        val_metrics = evaluate(model, val_loader, num_classes, amp_dtype, device)
        raw_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(device) for k, v in ema.state.items()}, strict=True)
        ema_metrics = evaluate(model, val_loader, num_classes, amp_dtype, device)
        model.load_state_dict({k: v.to(device) for k, v in raw_state.items()}, strict=True)

        use_ema = ema_metrics["macro_accuracy"] >= val_metrics["macro_accuracy"]
        chosen = ema_metrics if use_ema else val_metrics
        score = chosen["macro_accuracy"] + 0.5 * chosen["accuracy"]
        entry = {
            "epoch": epoch,
            "train": {"loss": running_loss / max(seen, 1), "accuracy": running_acc / max(seen, 1)},
            "val": val_metrics,
            "val_ema": ema_metrics,
            "chosen": "ema" if use_ema else "raw",
            "selection_score": score,
            "minutes": round((time.time() - epoch_start) / 60.0, 2),
        }
        history.append(entry)
        print("[epoch] " + json.dumps(entry, ensure_ascii=False), flush=True)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        payload = {
            "model": raw_state,
            "ema": ema.state,
            "epoch": epoch,
            "metrics": entry,
            "config": cfg,
            "image_size": image_size,
            "eval_size": eval_size,
            "num_classes": num_classes,
            "backbone": cfg["model"]["backbone"],
        }
        torch.save(payload, out_dir / "last.pt")
        if args.snapshot_dir:
            torch.save(
                {"model": raw_state, "epoch": epoch, "config": cfg,
                 "image_size": image_size, "eval_size": eval_size,
                 "num_classes": num_classes, "backbone": cfg["model"]["backbone"]},
                Path(args.snapshot_dir) / f"epoch{epoch:02d}.pt",
            )
        if score > best_score:
            best_score = score
            torch.save(payload, out_dir / "best.pt")
            since_improved = 0
            print(f"[best] epoch {epoch} score {score:.4f} ({entry['chosen']})", flush=True)
        else:
            since_improved += 1
        if patience and since_improved >= patience:
            print(f"[early-stop] no improvement for {since_improved} epochs", flush=True)
            break
        if args.smoke:
            break

    print("[done] best_score=%.4f" % best_score, flush=True)


if __name__ == "__main__":
    main()
