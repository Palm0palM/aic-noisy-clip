"""Per-row validation of the generated soft-target file (GPT's request)."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

project = Path("/root/autodl-tmp/projects/AIC-Robust-CLIP-复赛")
rows = list(csv.DictReader(open(project / "artifacts/train_manifest.csv", encoding="utf-8")))
labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
targets = np.load(project / "artifacts/oof_soft_targets.npy", mmap_mode="r")
n, k = targets.shape
smoothing = 0.15

sums = np.zeros(n, dtype=np.float64)
label_mass = np.zeros(n, dtype=np.float64)
for start in range(0, n, 4096):
    block = np.asarray(targets[start:start + 4096], dtype=np.float32)
    sums[start:start + len(block)] = block.sum(axis=1)
    label_mass[start:start + len(block)] = block[np.arange(len(block)), labels[start:start + len(block)]]

pre = (label_mass - smoothing / k) / (1.0 - smoothing)
digest = hashlib.sha256((project / "artifacts/oof_soft_targets.npy").read_bytes()).hexdigest()
report = {
    "rows": int(n), "classes": int(k),
    "row_sum_min": float(sums.min()), "row_sum_max": float(sums.max()), "row_sum_mean": float(sums.mean()),
    "rows_off_by_more_than_1e-3": int((np.abs(sums - 1.0) > 1e-3).sum()),
    "label_mass_presmoothing_min": float(pre.min()), "label_mass_presmoothing_mean": float(pre.mean()),
    "label_mass_postsmoothing_min": float(label_mass.min()), "label_mass_postsmoothing_mean": float(label_mass.mean()),
    "sha256": digest,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
