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
import gc
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

from .domain import TargetedCorruption, canonicalize

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
    def __init__(
        self,
        root: Path,
        records: list[dict],
        transform,
        decode_cap: int = 0,
        canonical_crop: bool = False,
        soft_targets=None,
    ):
        self.root = root
        self.records = records
        self.transform = transform
        self.decode_cap = decode_cap
        self.canonical_crop = canonical_crop
        self.soft_targets = soft_targets

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = load_image(self.root / rec["relative_path"], self.decode_cap)
        if self.canonical_crop:
            image = canonicalize(image)
        tensor = self.transform(image)
        if self.soft_targets is not None:
            return tensor, rec["label"], index, torch.from_numpy(
                np.asarray(self.soft_targets[rec["index"]], dtype=np.float32)
            )
        return tensor, rec["label"], index


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


class LongSideCap:
    """Scale an image down so its longest side does not exceed `cap`.

    Test images all have a long side of at most 500 px, so this makes the training
    images carry the same level of detail while the model input stays whatever it
    was; smaller images are left untouched and nothing is upscaled.
    """

    def __init__(self, cap: int):
        self.cap = int(cap)

    def __call__(self, image):
        width, height = image.size
        longest = max(width, height)
        if longest <= self.cap:
            return image
        scale = self.cap / float(longest)
        return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                            Image.BICUBIC)


def build_train_transform(image_size: int, augment: dict):
    ops = []
    if augment.get("long_side_cap"):
        ops.append(LongSideCap(int(augment["long_side_cap"])))
    ops += [
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
    if augment.get("targeted_corruption"):
        ops.append(TargetedCorruption(**augment["targeted_corruption"]))
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
    """CLIP vision tower plus a linear head.

    feature="projected" (default) keeps the historical behaviour: the head sees
    the 512-d cross-modal projection of the pooled token. feature="pooled" feeds
    the 768-d pooled token itself, which removes the rank bottleneck of the
    projection for a 750-way classifier (see configs/b2_768.yaml).

    local_readout=True adds a second, attention-pooled read-out of the patch
    tokens: logits = head(projected CLS) + local_head(attention pool of patches).
    The local head is zero-initialised, so the model starts exactly at the CLS
    model and learns whatever extra it can use from the spatial tokens.
    local_queries > 1 pools with several independent queries and concatenates
    the results, so the read-out can keep more than one region per image.
    """

    def __init__(
        self,
        backbone: str,
        revision: str,
        num_classes: int,
        head: str = "linear",
        dropout: float = 0.0,
        feature: str = "projected",
        head_dim: int | None = None,
        local_readout: bool = False,
        local_queries: int = 1,
    ):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.vision = CLIPVisionModelWithProjection.from_pretrained(
            backbone, revision=revision, use_safetensors=True
        )
        self.feature = feature
        if head_dim:
            dim = int(head_dim)
        elif feature == "pooled":
            dim = int(self.vision.config.hidden_size)
        else:
            dim = int(self.vision.config.projection_dim)
        self.dropout = nn.Dropout(dropout)
        if head == "cosine":
            self.head = CosineHead(dim, num_classes)
        else:
            self.head = nn.Linear(dim, num_classes)
        self.local_readout = bool(local_readout)
        self.local_queries = max(1, int(local_queries))
        # inference-time knob: does the branch carry useful signal that training
        # simply left under-weighted, or is it noise?
        self.local_scale = 1.0
        if self.local_readout:
            hidden = int(self.vision.config.hidden_size)
            self.local_norm = nn.LayerNorm(hidden)
            # flat parameter so a single-query and a k-query checkpoint stay
            # shape-compatible for k=1
            self.local_query = nn.Parameter(torch.randn(hidden * self.local_queries) * 0.02)
            self.local_head = nn.Linear(hidden * self.local_queries, num_classes)
            nn.init.zeros_(self.local_head.weight)
            nn.init.zeros_(self.local_head.bias)

    def encode(self, pixel_values: torch.Tensor):
        """One pass through the vision tower: (features, patch tokens).

        The CLS path and the local read-out share this single forward; running
        the tower twice doubles the activation memory and OOMs at 576px.
        """
        height, width = pixel_values.shape[-2:]
        configured = int(self.vision.config.image_size)
        interpolate = height != configured or width != configured
        outputs = self.vision.vision_model(
            pixel_values=pixel_values, interpolate_pos_encoding=interpolate
        )
        pooled = outputs.pooler_output
        if self.feature == "pooled":
            features = pooled
        else:
            features = self.vision.visual_projection(pooled)
        return features, outputs.last_hidden_state[:, 1:, :]

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode(pixel_values)[0]

    def pool_local(self, tokens: torch.Tensor) -> torch.Tensor:
        """Attention-pool normalised patch tokens with each query, then flatten."""
        tokens = self.local_norm(tokens)
        query = self.local_query.view(self.local_queries, -1)
        scale = tokens.shape[-1] ** -0.5
        weights = torch.softmax(torch.einsum("bnd,kd->bkn", tokens, query) * scale, dim=-1)
        pooled = torch.einsum("bkn,bnd->bkd", weights, tokens)
        return pooled.flatten(1)

    def local_attention(self, pixel_values: torch.Tensor):
        """Per-query attention over the patch tokens, plus the normalised tokens."""
        _, tokens = self.encode(pixel_values)
        tokens = self.local_norm(tokens)
        query = self.local_query.view(self.local_queries, -1)
        scale = tokens.shape[-1] ** -0.5
        weights = torch.softmax(torch.einsum("bnd,kd->bkn", tokens, query) * scale, dim=-1)
        return weights, tokens

    def embed_local(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Attention-pooled patch tokens (the extra spatial read-out)."""
        _, tokens = self.encode(pixel_values)
        return self.pool_local(tokens)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features, tokens = self.encode(pixel_values)
        logits = self.head(self.dropout(features))
        if self.local_readout:
            logits = logits + self.local_scale * self.local_head(self.dropout(self.pool_local(tokens)))
        return logits

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

    def head_side_params(self):
        """Everything that should train at the head learning rate."""
        params = list(self.head.parameters())
        if self.local_readout:
            params += list(self.local_norm.parameters())
            params += list(self.local_head.parameters())
            params.append(self.local_query)
        return params


def build_optimizer(model: FTClassifier, cfg: dict):
    """Build AdamW groups, optionally with layer-wise LR decay (LLRD).

    With llrd_gamma < 1 the vision-tower layers get lr * gamma^(n_layers - depth),
    i.e. earlier layers train more slowly. The head always uses lr_head.
    """
    head_ids = {id(p) for p in model.head_side_params()}
    wd = float(cfg["weight_decay"])
    lr_bb = float(cfg["lr_backbone"])
    gamma = float(cfg.get("llrd_gamma", 1.0))
    n_layers = len(model.vision.vision_model.encoder.layers) if gamma != 1.0 else 0

    groups: dict[tuple[float, float], list] = {}

    def add(param, lr, weight_decay):
        key = (round(lr, 12), weight_decay)
        groups.setdefault(key, []).append(param)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        decay = 0.0 if no_decay else wd
        if id(param) in head_ids:
            add(param, float(cfg["lr_head"]), decay)
            continue
        lr = lr_bb
        if gamma != 1.0:
            depth = None
            if "encoder.layers." in name:
                depth = int(name.split("encoder.layers.")[1].split(".")[0])
            elif "pre_layrnorm" in name or "embeddings" in name:
                depth = -1
            if depth is not None:
                lr = lr_bb * (gamma ** (n_layers - 1 - depth if depth >= 0 else n_layers))
        add(param, lr, decay)

    return torch.optim.AdamW(
        [{"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in groups.items()],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def apply_freeze_policy(model: FTClassifier, freeze_first_blocks: int = 0, train_last_blocks: int = 0,
                        freeze_tower: bool = False):
    """Freeze the patch embedding / early tower blocks, or keep only the last N blocks.

    freeze_tower freezes the whole image tower including the projection, leaving
    only the classifier trainable - the linear-probe stage of LP-FT, which gives
    the head a sensible direction before the tower is unfrozen.
    """
    if freeze_tower:
        for param in model.vision.parameters():
            param.requires_grad = False
        return
    tower = model.vision.vision_model
    n_layers = len(tower.encoder.layers)
    if train_last_blocks > 0:
        keep_from = n_layers - train_last_blocks
        for name, param in tower.named_parameters():
            if "encoder.layers." in name:
                depth = int(name.split("encoder.layers.")[1].split(".")[0])
                param.requires_grad = depth >= keep_from
            elif "post_layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.vision.visual_projection.requires_grad = False
        return
    if freeze_first_blocks > 0:
        for param in tower.embeddings.parameters():
            param.requires_grad = False
        for name, param in tower.named_parameters():
            if "encoder.layers." in name:
                depth = int(name.split("encoder.layers.")[1].split(".")[0])
                if depth < freeze_first_blocks:
                    param.requires_grad = False


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


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    per_sample = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    if reduction == "none":
        return per_sample
    return per_sample.mean()


def generalized_cross_entropy(logits: torch.Tensor, target: torch.Tensor, q: float = 0.7,
                              reduction: str = "mean") -> torch.Tensor:
    """GCE (Zhang & Sabuncu 2018), generalised to soft targets.

    p_y is the probability mass the model puts on the (blended) target, so the
    loss saturates once p_y is large instead of pushing it to 1 - the standard
    robust-loss behaviour under label noise.
    """
    probs = F.softmax(logits, dim=1)
    p_y = (probs * target).sum(dim=1).clamp(min=1e-6)
    per_sample = (1.0 - p_y.pow(q)) / q
    if reduction == "none":
        return per_sample
    return per_sample.mean()


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def enable_gradient_checkpointing(model: nn.Module, last_n: int = 0) -> int:
    """Recompute the last `last_n` tower layers' activations during backward.

    This transformers build declares `gradient_checkpointing` on CLIPVisionEncoder
    but never uses it in forward(), so the layer loop is replaced here. The
    memory saving is what lets the local read-out run at the same batch size as
    V16 instead of changing the recipe to fit.
    """
    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_outputs import BaseModelOutput

    encoder = model.vision.vision_model.encoder
    layers = list(encoder.layers)
    n = len(layers) if last_n <= 0 else min(last_n, len(layers))
    start = len(layers) - n

    def run_layer(layer, hidden_states, attention_mask, causal_attention_mask):
        return layer(hidden_states, attention_mask, causal_attention_mask)[0]

    def forward(inputs_embeds, attention_mask=None, causal_attention_mask=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        hidden_states = inputs_embeds
        for index, layer in enumerate(layers):
            if index >= start:
                hidden_states = checkpoint(
                    run_layer, layer, hidden_states, attention_mask, causal_attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = run_layer(layer, hidden_states, attention_mask, causal_attention_mask)
        return BaseModelOutput(last_hidden_state=hidden_states)

    encoder.forward = forward
    return n


@torch.no_grad()
def local_branch_report(model: nn.Module, loader: DataLoader, amp_dtype, device, batches: int = 6) -> None:
    """Check that the local read-out learned something, and that queries differ.

    Prints the mean pairwise cosine similarity between the per-image attention
    maps of the different queries (a value near 1 means they collapsed onto the
    same region, so a multi-query result would not test anything), and how large
    the local logits are relative to the CLS logits.
    """
    if not getattr(model, "local_readout", False):
        return
    model.eval()
    maps, local_scale, cls_scale = [], [], []
    for step, batch in enumerate(loader):
        if step >= batches:
            break
        images = batch[0].to(device, non_blocking=True)
        weights, _ = model.local_attention(images)
        flat = weights.float().reshape(weights.shape[0], weights.shape[1], -1)
        flat = flat - flat.mean(dim=-1, keepdim=True)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        maps.append(flat.cpu())
        with torch.autocast("cuda", dtype=amp_dtype):
            cls_logits = model.head(model.embed(images))
            local_logits = model.local_head(model.embed_local(images))
        cls_scale.append(cls_logits.float().abs().mean().item())
        local_scale.append(local_logits.float().abs().mean().item())
    if not maps:
        return
    stacked = torch.cat(maps, dim=0)  # (B, k, N)
    k = stacked.shape[1]
    sims = []
    for i in range(k):
        for j in range(i + 1, k):
            sims.append(float((stacked[:, i] * stacked[:, j]).sum(-1).mean()))
    ratio = float(np.mean(local_scale) / max(np.mean(cls_scale), 1e-6))
    print(
        f"[local] queries={k} mean pairwise attention cosine "
        f"{np.mean(sims) if sims else float('nan'):+.3f} (min {min(sims) if sims else float('nan'):+.3f}), "
        f"|local logits| / |cls logits| {ratio:.3f}",
        flush=True,
    )
    model.train()


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
    parser.add_argument("--snapshot-keep", type=int, default=5, help="rolling window of epoch snapshots to keep")
    parser.add_argument("--image-size", type=int, default=0, help="override config image size")
    parser.add_argument("--epochs", type=int, default=0, help="override config epochs")
    parser.add_argument("--tag", default="", help="suffix appended to output dir")
    parser.add_argument("--sample-weight-file", default="",
                        help="npy of per-manifest-row weights (e.g. out-of-fold reliability)")
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
    canonical_crop = bool(cfg["data"].get("canonical_crop", False))
    soft_target_weight = float(cfg["train"].get("soft_target_weight", 0.0))
    soft_targets = None
    if soft_target_weight > 0:
        soft_path = project / cfg["train"]["soft_targets"]
        soft_targets = np.load(soft_path, mmap_mode="r")
        if len(soft_targets) != len(records):
            raise ValueError(f"soft targets {soft_path} have {len(soft_targets)} rows, manifest has {len(records)}")
        print(f"[soft] distilling from {soft_path} with weight {soft_target_weight}", flush=True)
    train_ds = ManifestDataset(train_root, train_records, train_tf, decode_cap, canonical_crop, soft_targets)
    val_ds = ManifestDataset(train_root, val_records, eval_tf, decode_cap, canonical_crop)
    resolution_schedule = cfg["data"].get("resolution_schedule") or []
    resolution_schedule = sorted((int(ep), int(sz)) for ep, sz in resolution_schedule)

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
        feature=cfg["model"].get("feature", "projected"),
        head_dim=cfg["model"].get("head_dim"),
        local_readout=bool(cfg["model"].get("local_readout", False)),
        local_queries=int(cfg["model"].get("local_queries", 1)),
    ).to(device)

    if cfg["model"].get("grad_checkpoint", 0):
        # 576px at batch 80 already sits at the memory ceiling; the local
        # read-out needs a few hundred MB more, so recompute tower activations
        # instead of changing the batch size (which would confound the recipe).
        last_n = cfg["model"].get("grad_checkpoint_layers", 4)
        checkpointed = enable_gradient_checkpointing(model, int(last_n))
        print(f"[model] gradient checkpointing on the last {checkpointed} tower layers", flush=True)

    if cfg["model"].get("freeze_patch_embed", False):
        for param in model.vision.vision_model.embeddings.parameters():
            param.requires_grad = False
    apply_freeze_policy(
        model,
        freeze_first_blocks=int(cfg["model"].get("freeze_first_blocks", 0)),
        train_last_blocks=int(cfg["model"].get("train_last_blocks", 0)),
        freeze_tower=bool(cfg["model"].get("freeze_tower", False)),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total:.1%})", flush=True)

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
    robust_q = float(cfg["train"].get("robust_loss_q", 0.0))
    filter_after = int(cfg["train"].get("drop_high_loss_after", -1))
    filter_rate = float(cfg["train"].get("drop_high_loss_rate", 0.0))
    # "loss": current behaviour, weight the mixed sample's loss by the anchor
    # sample's weight. "target": scale each sample's target before mixing, which
    # is the correct treatment of a mixed pair under per-sample reliabilities.
    weight_mode = str(cfg["train"].get("sample_weight_mode", "loss"))
    sample_weight = None
    loss_tracker = None
    loss_counter = None
    if args.sample_weight_file:
        external = np.load(args.sample_weight_file).astype(np.float32)
        if len(external) != len(records):
            raise ValueError(f"weight file has {len(external)} rows, manifest has {len(records)}")
        # the weight file is in manifest order, but the dataset hands out positions
        # inside train_records, so it has to be re-indexed here or every weight
        # lands on the wrong image once any sample has been dropped
        sample_weight = external[np.asarray(train_idx, dtype=np.int64)]
        print(f"[weights] external sample weights: min {sample_weight.min():.3f}, "
              f"mean {sample_weight.mean():.4f}, below 1.0: {int((sample_weight < 1.0).sum())}", flush=True)
    if filter_after >= 0 and filter_rate > 0:
        sample_weight = np.ones(len(train_records), dtype=np.float32)
        loss_tracker = np.zeros(len(train_records), dtype=np.float64)
        loss_counter = np.zeros(len(train_records), dtype=np.float64)
        print(f"[noise] high-loss filtering from epoch {filter_after}, dropping "
              f"{filter_rate:.0%} of the seen samples each round", flush=True)
    if robust_q > 0:
        print(f"[noise] robust loss: generalised cross entropy q={robust_q}", flush=True)
    label_smoothing = float(cfg["augment"].get("label_smoothing", 0.1))
    mix_prob = float(cfg["augment"].get("mix_prob", 0.5))
    mixup_alpha = float(cfg["augment"].get("mixup", 0.2))
    cutmix_alpha = float(cfg["augment"].get("cutmix", 1.0))
    # optional override for the last few epochs: train with strong augmentation
    # first, then let the final epochs sharpen on less ambiguous targets
    late_epochs = int(cfg["augment"].get("late_epochs", 0))
    late_mix_prob = cfg["augment"].get("late_mix_prob", None)
    late_label_smoothing = cfg["augment"].get("late_label_smoothing", None)
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    patience = int(cfg["train"].get("early_stop_patience", 4))
    log_every = int(cfg["train"].get("log_every", 50))
    since_improved = 0

    print(json.dumps({k: cfg[k] for k in ("data", "model", "train", "augment", "sampler")}, ensure_ascii=False), flush=True)
    print(f"[setup] train={len(train_records)} val={len(val_records)} steps/epoch={steps_per_epoch}", flush=True)

    for epoch in range(start_epoch, int(math.ceil(epochs))):
        if resolution_schedule:
            target_size = resolution_schedule[0][1]
            for start_ep, size in resolution_schedule:
                if epoch >= start_ep:
                    target_size = size
            if target_size != image_size:
                image_size = target_size
                eval_size = target_size
                print(f"[res] epoch {epoch}: switching training resolution to {image_size}px", flush=True)
                del train_loader
                gc.collect()
                train_tf = build_train_transform(image_size, cfg["augment"])
                train_ds = ManifestDataset(train_root, train_records, train_tf, decode_cap, canonical_crop, soft_targets)
                sampler = WeightedRandomSampler(
                    torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(train_records),
                    replacement=True,
                )
                train_loader = DataLoader(
                    train_ds, batch_size=int(cfg["data"]["batch_size"]), sampler=sampler, drop_last=True, **loader_kwargs
                )
                val_ds = ManifestDataset(train_root, val_records, build_eval_transform(eval_size, float(cfg["data"].get("eval_resize_ratio", 1.14))), decode_cap, canonical_crop)
                val_loader = DataLoader(val_ds, batch_size=int(cfg["data"]["val_batch_size"]), shuffle=False, **loader_kwargs)
        epoch_mix_prob, epoch_label_smoothing = mix_prob, label_smoothing
        if late_epochs > 0 and epoch >= int(math.ceil(epochs)) - late_epochs:
            if late_mix_prob is not None:
                epoch_mix_prob = float(late_mix_prob)
            if late_label_smoothing is not None:
                epoch_label_smoothing = float(late_label_smoothing)
            print(
                f"[aug] epoch {epoch}: mix_prob {epoch_mix_prob} label_smoothing {epoch_label_smoothing}",
                flush=True,
            )
        model.train()
        epoch_start = time.time()
        running_loss, running_acc, seen = 0.0, 0.0, 0
        for step, batch in enumerate(train_loader):
            if args.smoke and step >= args.smoke:
                break
            if soft_targets is None:
                images, labels, indices = batch
                teacher = None
            else:
                images, labels, indices, teacher = batch
                teacher = teacher.to(device, non_blocking=True).float()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target = one_hot(labels, num_classes, epoch_label_smoothing)
            if teacher is not None:
                target = soft_target_weight * teacher + (1.0 - soft_target_weight) * target
            batch_weight = None
            if sample_weight is not None:
                batch_weight = torch.from_numpy(sample_weight[indices.numpy()]).to(device, non_blocking=True)
                if weight_mode == "target":
                    # scale each sample's target by its reliability BEFORE the mix,
                    # so a mixed pair carries lambda*w_i for image i and (1-lambda)*w_j
                    # for image j, instead of lumping both under w_i
                    target = target * batch_weight.unsqueeze(1)
            images, target, _ = apply_mixup_cutmix(
                images, target, mixup_alpha, cutmix_alpha, epoch_mix_prob, num_classes
            )
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(images)
            if robust_q > 0:
                per_sample = generalized_cross_entropy(logits.float(), target, robust_q, reduction="none")
            else:
                per_sample = soft_cross_entropy(logits.float(), target, reduction="none")
            if batch_weight is None:
                loss = per_sample.mean()
            elif weight_mode == "target":
                loss = per_sample.sum() / target.sum().clamp(min=1.0)
            else:
                loss = (per_sample * batch_weight).sum() / batch_weight.sum().clamp(min=1.0)
            if loss_tracker is not None:
                with torch.no_grad():
                    rows = indices.numpy()
                    values = per_sample.detach().float().cpu().numpy()
                    if sample_weight is None:
                        np.add.at(loss_tracker, rows, values)
                        np.add.at(loss_counter, rows, 1.0)
                    else:
                        np.add.at(loss_tracker, rows, values * sample_weight[rows])
                        np.add.at(loss_counter, rows, sample_weight[rows])
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

        # high-loss filtering: samples the model has consistently found hard are
        # the prime suspects for mislabelling, so they stop contributing gradient
        if loss_tracker is not None and epoch >= filter_after:
            observed = loss_counter > 0
            mean_loss = loss_tracker[observed] / np.maximum(loss_counter[observed], 1.0)
            threshold = float(np.quantile(mean_loss, 1.0 - filter_rate))
            sample_weight[observed] = (mean_loss <= threshold).astype(np.float32)
            dropped = int((sample_weight[observed] == 0).sum())
            print(f"[noise] epoch {epoch}: dropped {dropped} of {int(observed.sum())} seen samples "
                  f"above loss {threshold:.4f}", flush=True)
            loss_tracker *= 0.0
            loss_counter *= 0.0

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
            "feature": cfg["model"].get("feature", "projected"),
            "local_readout": bool(cfg["model"].get("local_readout", False)),
            "local_queries": int(cfg["model"].get("local_queries", 1)),
        }
        torch.save(payload, out_dir / "last.pt")
        if args.snapshot_dir:
            snapshot_dir = Path(args.snapshot_dir)
            torch.save(
                {"model": raw_state, "epoch": epoch, "config": cfg,
                 "image_size": image_size, "eval_size": eval_size,
                 "num_classes": num_classes, "backbone": cfg["model"]["backbone"]},
                snapshot_dir / f"epoch{epoch:02d}.pt",
            )
            if args.snapshot_keep > 0:
                saved = sorted(snapshot_dir.glob("epoch*.pt"))
                for stale in saved[:-args.snapshot_keep]:
                    stale.unlink(missing_ok=True)
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

    local_branch_report(model, train_loader, amp_dtype, device)
    print("[done] best_score=%.4f" % best_score, flush=True)


if __name__ == "__main__":
    main()
