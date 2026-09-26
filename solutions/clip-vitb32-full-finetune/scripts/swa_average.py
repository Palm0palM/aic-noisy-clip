"""Average the weights of the last N training epochs into one checkpoint (SWA).

Stochastic weight averaging produces a single weight file and a single inference
flow, so it stays inside the "one model, one inference pipeline" constraint.

    python scripts/swa_average.py --snapshots checkpoints/v10_swa --last 3 \
        --reference checkpoints/v10_final4/best.pt --output checkpoints/v10_swa/swa.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--reference", required=True, help="checkpoint whose metadata (config etc.) is reused")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="model", choices=["model", "ema"])
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshots)
    files = sorted(p for p in snapshot_dir.glob("epoch*.pt") if p.name != "swa.pt")
    if len(files) < 2:
        raise SystemExit(f"need at least 2 snapshots, found {len(files)} in {snapshot_dir}")
    chosen = files[-args.last:]
    print("[swa] averaging " + ", ".join(p.name for p in chosen))

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    keys = reference[args.mode].keys()

    acc = {k: torch.zeros_like(reference[args.mode][k], dtype=torch.float32) for k in keys}
    for path in chosen:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload[args.mode]
        for k in keys:
            acc[k] += state[k].float() / len(chosen)

    out = dict(reference)
    out[args.mode] = {k: v.to(reference[args.mode][k].dtype) for k, v in acc.items()}
    out["swa"] = {"snapshots": [p.name for p in chosen], "mode": args.mode}
    out["metrics"] = {"chosen": args.mode}
    torch.save(out, args.output)
    print(f"[swa] wrote {args.output}")


if __name__ == "__main__":
    main()
