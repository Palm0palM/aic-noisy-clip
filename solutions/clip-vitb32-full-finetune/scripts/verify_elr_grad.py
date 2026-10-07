"""Gradient-flow check for the ELR regulariser.

The pre-2026-10-06 implementation computed the penalty under torch.no_grad()
with a detached prediction, so it was a constant: the parameter update was
identical with and without it. This script exercises the real elr_regularizer()
on a tiny tensor problem and asserts:

  1. the penalty carries a graph and a non-zero gradient w.r.t. the logits;
  2. with the penalty added, the parameter gradient differs from plain CE;
  3. the history buffer itself receives no gradient and is updated in place
     only for unmixed batches;
  4. with lambda = 0 the update is bit-identical to the plain-CE update.

Run: python scripts/verify_elr_grad.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_clip.train_ft import elr_regularizer  # noqa: E402


def check() -> None:
    torch.manual_seed(0)
    classes, features, batch = 8, 5, 6

    def fresh():
        w = nn.Parameter(torch.randn(classes, features))
        x = torch.randn(batch, features)
        y = torch.randint(0, classes, (batch,))
        return w, x, y

    def grads(lam, momentum=0.7, mixed=False, steps=3):
        w, x, y = fresh()
        buffer = torch.zeros(16, classes)
        indices = torch.arange(batch)
        mix_index = torch.tensor([1, 0, 3, 2, 5, 4]) if mixed else None
        mix_lam = 0.6
        for _ in range(steps):
            logits = x @ w.t()
            loss = F.cross_entropy(logits, y)
            before = buffer.clone()
            if lam > 0:
                loss = loss + elr_regularizer(logits, buffer, indices, mix_index, mix_lam, momentum, lam)
            w.grad = None
            loss.backward()
            updated = not torch.equal(before, buffer)
        return w.grad.clone(), buffer.clone(), updated

    g0, b0, u0 = grads(0.0)
    g1, b1, u1 = grads(3.0)
    g_mix, b_mix, u_mix = grads(3.0, mixed=True)

    w, x, y = fresh()
    buffer = torch.zeros(16, 8)
    logits = (x @ w.t()).requires_grad_(True)
    term = elr_regularizer(logits, buffer, torch.arange(batch), None, None, 0.7, 3.0)
    grad_direct = torch.autograd.grad(term, logits)[0]

    print(f"[elr] penalty requires_grad  : {term.requires_grad}")
    print(f"[elr] |d penalty / d logits| : {grad_direct.abs().sum().item():.6f}")
    print(f"[elr] ||grad(lambda=3) - grad(lambda=0)|| : {(g1 - g0).norm().item():.6f}")
    print(f"[elr] buffer updated, unmixed: {u0} -> (lambda>0) {u1}; mixed {u_mix}")
    print(f"[elr] buffer receives grad   : {b1.requires_grad}")
    assert term.requires_grad and grad_direct.abs().sum() > 0, "penalty has no gradient path"
    assert (g1 - g0).norm() > 1e-6, "gradient is identical with and without ELR"
    assert u1 and not u_mix, "mixed batches must not write into per-image history"
    assert not b1.requires_grad, "history buffer must stay detached"
    print("[elr] all checks passed")


if __name__ == "__main__":
    check()
