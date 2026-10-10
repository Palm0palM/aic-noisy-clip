"""Input-space localizer ported from the teammate's P3 package (v19_detail).

Faithful port of their `views.py`: letterbox the whole image onto a square
canvas filled with the CLIP mean colour, take the gradient of the Top1 - Top2
logit margin with respect to the input pixels, average-pool it over 32-pixel
cells, and turn the heatmap into a subject box and a broad detail box.

Their constants are written in canvas pixels (32-pixel cell, +64 px context,
55 % minimum coverage of each original dimension) and apply unchanged on our
512 canvas: the pooled grid simply becomes 16 x 16 instead of their 18 x 18
on 576. The quantile mapping, valid-region mask and clip rules are identical.

Used only for the A/B/C localizer comparison. Reads images, never labels;
never touches training.

    from detail_views import letterbox, whole_view, crop_view, input_saliency

    subject_box, detail_box, valid = boxes_from_heatmap(heat, w, h, 512)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from aic_clip.train_ft import CLIP_MEAN, CLIP_STD

FILL = tuple(round(255 * x) for x in CLIP_MEAN)
CELL = 32


def letterbox(image: Image.Image, size: int, fill=None) -> Image.Image:
    """Scale the whole image inside a size x size canvas, pad the remainder."""
    fill = FILL if fill is None else fill
    scale = size / max(image.size)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image.resize((width, height), Image.BICUBIC),
                 ((size - width) // 2, (size - height) // 2))
    return canvas


def tensor(image: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)


def whole_view(image: Image.Image, size: int) -> torch.Tensor:
    return tensor(letterbox(image, size))


def crop_original(image: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = (float(v) for v in box)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("Invalid normalized crop")
    coords = (int(x0 * image.width), int(y0 * image.height),
              max(int(x0 * image.width) + 1, round(x1 * image.width)),
              max(int(y0 * image.height) + 1, round(y1 * image.height)))
    return image.crop(coords)


def crop_view(image: Image.Image, box, size: int) -> torch.Tensor:
    return tensor(letterbox(crop_original(image, box), size))


def boxes_from_heatmap(heat, width: int, height: int, size: int):
    """Map letterbox saliency to original coordinates.

    Returns (subject_box, detail_box, valid). Mirrors the teammate's
    conservative fallback: a degenerate heatmap yields full-image boxes and
    valid=False so the caller can count and audit those rows.
    """
    a = np.asarray(heat, dtype=np.float64)
    if a.shape != (size // CELL, size // CELL) or not np.isfinite(a).all():
        raise ValueError("Invalid saliency grid")
    a = np.maximum(a, 0)
    scale = size / max(width, height)
    rw, rh = round(width * scale), round(height * scale)
    px, py = (size - rw) // 2, (size - rh) // 2
    xs, ys = np.arange(a.shape[1]) * CELL + CELL // 2, np.arange(a.shape[0]) * CELL + CELL // 2
    valid_region = ((xs[None, :] >= px) & (xs[None, :] < px + rw)
                    & (ys[:, None] >= py) & (ys[:, None] < py + rh))
    a *= valid_region
    if a.sum() <= 1e-12 or np.count_nonzero(valid_region) < 4:
        return [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], False

    def quantiles(mass, coords):
        cdf = np.cumsum(mass) / mass.sum()
        return [coords[min(len(coords) - 1, np.searchsorted(cdf, q))] for q in (.1, .9)]

    x0, x1 = quantiles(a.sum(0), xs)
    y0, y1 = quantiles(a.sum(1), ys)
    cx, cy = ((x0 + x1) / 2 - px) / scale, ((y0 + y1) / 2 - py) / scale
    bw = max(.55 * width, (x1 - x0 + 64) / scale)
    bh = max(.55 * height, (y1 - y0 + 64) / scale)
    bw, bh = min(width, bw), min(height, bh)
    left = np.clip(cx - bw / 2, 0, width - bw)
    top = np.clip(cy - bh / 2, 0, height - bh)
    subject = [left / width, top / height, (left + bw) / width, (top + bh) / height]

    smooth = F.avg_pool2d(torch.from_numpy(a)[None, None], 3, 1, 1)[0, 0].numpy() * valid_region
    iy, ix = np.unravel_index(smooth.argmax(), smooth.shape)
    cx, cy = (xs[ix] - px) / scale, (ys[iy] - py) / scale
    bw, bh = .6 * width, .6 * height
    left = np.clip(cx - bw / 2, 0, width - bw)
    top = np.clip(cy - bh / 2, 0, height - bh)
    detail = [left / width, top / height, (left + bw) / width, (top + bh) / height]
    return subject, detail, True


def input_saliency(model, pixels: torch.Tensor) -> np.ndarray:
    """Top1-Top2 margin gradient times pixel, pooled to the 32-pixel grid.

    ``pixels`` must require gradients; the model's parameters must not.
    Runs under the caller's autocast and enable_grad contexts.
    """
    logits = model(pixels)
    f = logits.float()
    top = f.topk(2, dim=1).indices
    margin = f.gather(1, top[:, :1]) - f.gather(1, top[:, 1:])
    grad = torch.autograd.grad(margin.sum(), pixels, only_inputs=True)[0]
    heat = F.avg_pool2d((grad.float() * pixels.detach().float()).abs().sum(1, keepdim=True),
                        CELL, CELL)
    if not torch.isfinite(heat).all():
        raise FloatingPointError("Nonfinite input saliency")
    return heat[:, 0].detach().cpu().numpy()
