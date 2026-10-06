"""Checks required before trusting the L2-SP anchor (GPT's two conditions).

  1. alpha = 0 is a strict no-op: the anchor term contributes exactly zero to
     every parameter gradient, so a run with the code present but alpha = 0 is
     identical to the original training.
  2. alpha > 0 pulls towards the reference: the penalty's gradient equals
     alpha * (theta - theta0) elementwise, and one gradient step reduces the
     distance to theta0.

Uses a tiny stand-in for the backbone: the real code paths (anchor_penalty /
anchor_drift) with a module exposing .vision.named_parameters().

Run: python scripts/verify_anchor_grad.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import anchor_drift, anchor_penalty  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Module()
        self.vision.embed = nn.Linear(4, 4)
        self.vision.proj = nn.Linear(4, 2)
        self.head = nn.Linear(2, 3)


def main() -> None:
    torch.manual_seed(0)
    model = Tiny()
    inputs = torch.randn(5, 4)

    def task_loss():
        return (model.head(model.vision.proj(model.vision.embed(inputs))).sum()) ** 2

    # 1) alpha = 0 must be a no-op
    for param in model.parameters():
        param.grad = None
    task_loss().backward()
    plain = {name: param.grad.clone() for name, param in model.named_parameters()}

    ref = {name: param.detach().clone() for name, param in model.vision.named_parameters()}
    for param in model.parameters():
        param.grad = None
    zero_term = anchor_penalty(model, ref, 0.0)
    print(f"[anchor] alpha=0 penalty value: {float(zero_term.detach())}")
    (task_loss() + zero_term).backward()
    worst = max(
        float((param.grad - plain[name]).abs().max())
        for name, param in model.named_parameters()
    )
    print(f"[anchor] alpha=0 max |grad difference| over all params: {worst:.3e}")
    assert float(zero_term.detach()) == 0.0 and worst == 0.0, "alpha=0 is not a no-op"

    # 2) alpha > 0: gradient equals alpha*(theta-theta0) and the step pulls back
    alpha = 3.0
    with torch.no_grad():
        for name, param in model.vision.named_parameters():
            param.add_(torch.randn_like(param) * 0.05)  # simulate drift
    drift_before = anchor_drift(model, ref)
    for param in model.parameters():
        param.grad = None
    penalty = anchor_penalty(model, ref, alpha)
    penalty.backward()
    auto = torch.cat([p.grad.reshape(-1) for _, p in model.vision.named_parameters()])
    ana = torch.cat(
        [(alpha * (p.detach() - ref[name])).reshape(-1)
         for name, p in model.vision.named_parameters()]
    )
    print(f"[anchor] ||theta-theta0|| = {drift_before:.4f}, penalty = {float(penalty):.4f} "
          f"(expected {0.5 * alpha * drift_before ** 2:.4f})")
    print(f"[anchor] max |autograd - alpha*(theta-theta0)| = {float((auto - ana).abs().max()):.3e}")
    assert torch.allclose(auto, ana, atol=1e-5), "anchor gradient is not alpha*(theta-theta0)"

    with torch.no_grad():
        for name, param in model.vision.named_parameters():
            param.add_(param.grad, alpha=-1e-3)  # a gradient-descent step
    drift_after = anchor_drift(model, ref)
    print(f"[anchor] drift before step {drift_before:.4f} -> after step {drift_after:.4f}")
    assert drift_after < drift_before, "anchor step does not reduce the distance to theta0"
    print("[anchor] all checks passed")


if __name__ == "__main__":
    main()
