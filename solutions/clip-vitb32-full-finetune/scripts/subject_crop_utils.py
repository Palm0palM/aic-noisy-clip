"""CPU helpers for auditable, single-model subject-crop diagnostics.

The confidence interval is a descriptive independent-image normal approximation.
It does not account for training variation, repeated inspection, or duplicate groups.
"""

from __future__ import annotations

import math

import numpy as np


def cam_box(cam, tau, margin, view_size):
    """Return the padded box around high-mass CAM cells and a reason string.

    Coordinates are floating-point pixels in the model's square input view.
    An invalid heatmap returns ``(None, reason)``; invalid configuration raises.
    This is a bounding box around selected cells, not a minimum-area box search.
    """
    tau, margin, view_size = float(tau), float(margin), float(view_size)
    if not math.isfinite(tau) or not 0.0 < tau <= 1.0:
        raise ValueError("tau must be finite and in (0, 1]")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if not math.isfinite(view_size) or view_size <= 0:
        raise ValueError("view_size must be finite and positive")

    values = np.asarray(cam, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return None, "invalid_shape"
    if not np.isfinite(values).all():
        return None, "nonfinite_heatmap"
    if (values < 0).any():
        return None, "negative_heatmap"
    maximum = float(values.max())
    if maximum == 0:
        return None, "zero_heatmap"
    if float(values.max() - values.min()) <= maximum * 1e-6:
        return None, "constant_heatmap"

    # Rescaling makes summation safe for arbitrarily large/small finite CAM units.
    flat = (values / maximum).ravel()
    order = np.argsort(-flat, kind="stable")
    cumulative = np.cumsum(flat[order])
    count = min(flat.size, int(np.searchsorted(cumulative, tau * cumulative[-1])) + 1)
    rows, columns = np.divmod(order[:count], values.shape[1])
    x0, x1 = float(columns.min()), float(columns.max() + 1)
    y0, y1 = float(rows.min()), float(rows.max() + 1)
    pad_x, pad_y = margin * (x1 - x0), margin * (y1 - y0)
    cell_x = view_size / values.shape[1]
    cell_y = view_size / values.shape[0]
    box = (
        max(0.0, (x0 - pad_x) * cell_x),
        max(0.0, (y0 - pad_y) * cell_y),
        min(view_size, (x1 + pad_x) * cell_x),
        min(view_size, (y1 + pad_y) * cell_y),
    )
    return box, "ok"


def square_box(box, width, height):
    """Expand and translate a rectangle to an integer, in-bounds square.

    The square contains the entire supplied rectangle. Return None when these
    requirements are impossible, rather than clipping away part of the subject.
    """
    if box is None:
        return None
    if not all(math.isfinite(float(v)) for v in (width, height)):
        return None
    if int(width) != width or int(height) != height or width <= 0 or height <= 0:
        return None
    width, height = int(width), int(height)
    coordinates = np.asarray(box, dtype=np.float64)
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        return None
    left, top, right, bottom = map(float, coordinates)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None

    # First enclose fractional edges. Integer-centre arithmetic then guarantees
    # that subsequent square expansion cannot remove an edge during rounding.
    x0, y0, x1, y1 = math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
    side = max(x1 - x0, y1 - y0)
    if side > min(width, height):
        return None
    square_x0 = (x0 + x1 - side) // 2
    square_y0 = (y0 + y1 - side) // 2
    square_x0 = min(max(0, square_x0), width - side)
    square_y0 = min(max(0, square_y0), height - side)
    return square_x0, square_y0, square_x0 + side, square_y0 + side


def paired_summary(plain_pred, subject_pred, labels, mask=None):
    """Summarize paired predictions, with accuracies as fractions and deltas as pp.

    The CI uses the sample variance of per-image {-1, 0, +1} correctness changes.
    It assumes independent images; it is not evidence of equivalence when there
    are few/no discordant pairs. With fewer than two images the CI is unavailable.
    """
    plain_pred, subject_pred, labels = map(np.asarray, (plain_pred, subject_pred, labels))
    if any(a.ndim != 1 for a in (plain_pred, subject_pred, labels)):
        raise ValueError("predictions and labels must be one-dimensional")
    if not (len(plain_pred) == len(subject_pred) == len(labels)):
        raise ValueError("predictions and labels must have equal lengths")
    if mask is None:
        selected = np.ones(len(labels), dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.ndim != 1 or len(selected) != len(labels):
            raise ValueError("mask must be one-dimensional and match labels")

    plain_correct = plain_pred[selected] == labels[selected]
    subject_correct = subject_pred[selected] == labels[selected]
    n = int(plain_correct.size)
    wrong_to_correct = int((~plain_correct & subject_correct).sum())
    correct_to_wrong = int((plain_correct & ~subject_correct).sum())
    net = wrong_to_correct - correct_to_wrong
    result = {
        "n": n,
        "plain_accuracy": float(plain_correct.mean()) if n else None,
        "subject_accuracy": float(subject_correct.mean()) if n else None,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "net_correct": net,
        "delta_pp": float(100.0 * net / n) if n else None,
        "approx95_ci_pp": None,
    }
    if n >= 2:
        variance = max(0.0, (wrong_to_correct + correct_to_wrong - net * net / n) / (n - 1))
        half_width_pp = 1.959963984540054 * math.sqrt(variance / n) * 100.0
        delta = result["delta_pp"]
        result["approx95_ci_pp"] = [max(-100.0, delta - half_width_pp), min(100.0, delta + half_width_pp)]
    return result
