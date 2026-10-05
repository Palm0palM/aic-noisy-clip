"""Source-domain artefact handling: conservative border canonicalisation and
targeted corruption augmentation.

Diagnosis behind these: internal metrics sit 4-5pp above the official test score,
and the gap behaves like overfitting to source-domain cues (watermarks, frames,
composition) rather than to the classes. Two interventions are compared in the
domain experiment:

  * canonicalize()  - deterministically crop flat/black/white/scan-line borders,
                      applied identically at train and inference time;
  * TargetedCorruption - train-time simulation of the corruptions the frozen
                      audit found the model to be sensitive to.

Both are automatic and reproducible, and both use official training images only.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FLAT_STD_TOL = 3.0        # a row/column is "flat" below this grey-level std
DIFF_TOL = 12.0           # ... and must differ from the image mean by this much,
DARK_LEVEL = 8.0          # ... or be near black,
BRIGHT_LEVEL = 247.0      # ... or near white
MAX_BORDER_FRACTION = 0.08
MIN_KEEP_FRACTION = 0.75


def _band_length(stds: np.ndarray, means: np.ndarray, global_mean: float, limit: int) -> int:
    length = 0
    for index in range(min(limit, len(stds))):
        flat = stds[index] <= FLAT_STD_TOL
        distinctive = (
            abs(means[index] - global_mean) >= DIFF_TOL
            or means[index] <= DARK_LEVEL
            or means[index] >= BRIGHT_LEVEL
        )
        if flat and distinctive:
            length = index + 1
        else:
            break
    return length


def detect_border_crop(image: Image.Image, max_fraction: float = MAX_BORDER_FRACTION) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) pixels to remove. Conservative by design."""
    width, height = image.size
    short = min(width, height)
    if short < 64:
        return (0, 0, 0, 0)
    scale = 256.0 / short if short > 256 else 1.0
    small = image.convert("L").resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.BILINEAR
    )
    gray = np.asarray(small, dtype=np.float32)
    global_mean = float(gray.mean())
    limit_rows = max(1, int(gray.shape[0] * max_fraction))
    limit_cols = max(1, int(gray.shape[1] * max_fraction))

    top = _band_length(gray.std(axis=1), gray.mean(axis=1), global_mean, limit_rows)
    bottom = _band_length(gray.std(axis=1)[::-1], gray.mean(axis=1)[::-1], global_mean, limit_rows)
    left = _band_length(gray.std(axis=0), gray.mean(axis=0), global_mean, limit_cols)
    right = _band_length(gray.std(axis=0)[::-1], gray.mean(axis=0)[::-1], global_mean, limit_cols)

    inverse = 1.0 / scale
    crop = [int(round(value * inverse)) for value in (left, top, right, bottom)]
    max_side = max(1, int(round(short * max_fraction)))
    crop = [min(value, max_side) for value in crop]
    # never remove more than a quarter of the image on any axis
    crop[0] = min(crop[0], int(width * (1 - MIN_KEEP_FRACTION)))
    crop[2] = min(crop[2], int(width * (1 - MIN_KEEP_FRACTION)))
    crop[1] = min(crop[1], int(height * (1 - MIN_KEEP_FRACTION)))
    crop[3] = min(crop[3], int(height * (1 - MIN_KEEP_FRACTION)))
    return tuple(max(0, value) for value in crop)


def canonicalize(image: Image.Image, enabled: bool = True) -> Image.Image:
    if not enabled:
        return image
    left, top, right, bottom = detect_border_crop(image)
    if left + top + right + bottom == 0:
        return image
    width, height = image.size
    box = (left, top, width - right, height - bottom)
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return image
    return image.crop(box)


class TargetedCorruption:
    """Apply at most one corruption, decided per sample with a fixed total budget.

    jpeg        - recompress with quality drawn uniformly from [40, 95]
    blur        - gaussian sigma drawn uniformly from [0.1, 1.2]
    edge_mask   - cover one 0-6% side band with that band's own mean colour
    """

    def __init__(
        self,
        jpeg_p: float = 0.20,
        blur_p: float = 0.10,
        edge_p: float = 0.10,
        total_p: float = 0.35,
        seed: int = 0,
    ):
        self.jpeg_p = jpeg_p
        self.blur_p = blur_p
        self.edge_p = edge_p
        self.total_p = total_p
        self.rng = random.Random(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.rng.random() >= self.total_p:
            return image
        draw = self.rng.random()
        if draw < self.jpeg_p:
            quality = self.rng.randint(40, 95)
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB").copy()
        if draw < self.jpeg_p + self.blur_p:
            return image.filter(ImageFilter.GaussianBlur(self.rng.uniform(0.1, 1.2)))
        if draw < self.jpeg_p + self.blur_p + self.edge_p:
            canvas = image.copy()
            painter = ImageDraw.Draw(canvas)
            width, height = canvas.size
            band = max(1, int(round(min(width, height) * self.rng.uniform(0.0, 0.06))))
            box = [
                (0, 0, width, band),
                (0, height - band, width, height),
                (0, 0, band, height),
                (width - band, 0, width, height),
            ][self.rng.randrange(4)]
            region = np.asarray(canvas.crop(box)).reshape(-1, 3)
            fill = tuple(int(value) for value in region.mean(0))
            painter.rectangle(box, fill=fill)
            return canvas
        return image
