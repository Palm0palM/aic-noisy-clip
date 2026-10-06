"""Fix the L2-SP strength from measured gradient magnitudes (no parameter scan).

The anchor gradient is exactly alpha*(theta-theta0), so its size at any state is
alpha times the drift. What it should be compared against is the task gradient
at a state the model actually reaches. That state is taken from the finished
fold-A control (checkpoints/v30_oof_a), and the task gradient norm is measured
by running a handful of real training batches from that state.

Reports the recommended alpha for a few target ratios g_anchor / g_task and
writes the numbers to json so the choice is on the record.

    python scripts/probe_anchor_scale.py --config configs/anchor_a/s1_384.yaml \
        --control checkpoints/v30_oof_a/last.pt --batches 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import (  # noqa: E402
    ManifestDataset, FTClassifier, anchor_drift, apply_mixup_cutmix,
    build_train_transform, one_hot, read_manifest, soft_cross_entropy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--control", required=True, help="a finished run at the same recipe")
    parser.add_argument("--drop-indices", default="artifacts/oof_a_drop.npy")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--output", default="artifacts/anchor_scale.json")
    args = parser.parse_args()

    import yaml

    project = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((project / args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda")

    records = read_manifest(project / cfg["data"]["manifest"])
    train_idx = list(range(len(records)))
    drop = {int(x) for x in np.load(project / args.drop_indices)}
    train_idx = [i for i in train_idx if i not in drop]
    train_records = [records[i] for i in train_idx]
    num_classes = int(cfg["data"]["num_classes"])
    print(f"[probe] {len(train_records)} training samples", flush=True)

    model = FTClassifier(
        cfg["model"]["backbone"], cfg["model"].get("revision"), num_classes,
        head=cfg["model"].get("head", "linear"),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        feature=cfg["model"].get("feature", "projected"),
    ).to(device)
    official = {name: param.detach().clone() for name, param in model.vision.named_parameters()}

    payload = torch.load(project / args.control, map_location="cpu", weights_only=False)
    state = payload.get("ema") or payload.get("model")
    model.load_state_dict(state, strict=True)
    drift = anchor_drift(model, official)
    print(f"[probe] backbone drift of the control: ||theta-theta0|| = {drift:.4f}", flush=True)

    transform = build_train_transform(int(cfg["data"]["image_size"]), cfg["augment"])
    dataset = ManifestDataset(
        Path(cfg["data"]["train_dir"]), train_records, transform,
        decode_cap=int(cfg["data"].get("decode_cap", 0)),
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["data"]["batch_size"]), shuffle=True, drop_last=True,
        num_workers=int(cfg["data"]["num_workers"]), pin_memory=True,
    )
    model.train()
    smoothing = float(cfg["augment"].get("label_smoothing", 0.1))
    mixup_alpha = float(cfg["augment"].get("mixup", 0.2))
    cutmix_alpha = float(cfg["augment"].get("cutmix", 1.0))
    mix_prob = float(cfg["augment"].get("mix_prob", 0.8))

    norms = []
    for step, batch in enumerate(loader):
        if step >= args.batches:
            break
        images, labels, _ = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target = one_hot(labels, num_classes, smoothing)
        images, target, _, _ = apply_mixup_cutmix(images, target, mixup_alpha, cutmix_alpha, mix_prob, num_classes)
        logits = model(images)
        loss = soft_cross_entropy(logits.float(), target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        total = 0.0
        for param in model.vision.parameters():
            if param.grad is not None:
                total += float(param.grad.detach().float().pow(2).sum())
        norms.append(total ** 0.5)
    task_norm = float(np.mean(norms))
    print(f"[probe] task gradient norm over {len(norms)} batches: mean {task_norm:.3e} "
          f"(min {min(norms):.3e}, max {max(norms):.3e})", flush=True)

    ratios = [0.05, 0.10, 0.20]
    report = {
        "control": args.control, "batches": len(norms), "drift": drift,
        "task_grad_norm_mean": task_norm, "task_grad_norm_min": min(norms), "task_grad_norm_max": max(norms),
        "alpha_by_target_ratio": {
            str(r): (r * task_norm / drift) for r in ratios
        },
        "note": "alpha is the coefficient of (1/2)||theta-theta0||^2; the anchor "
                "gradient norm at this state is alpha*drift",
    }
    (project / args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
