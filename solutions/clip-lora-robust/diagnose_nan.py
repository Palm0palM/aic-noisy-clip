"""最小复现诊断：少量样本跑若干 step，逐 step 报告数值健康状态，定位 NaN 来源。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import MemmapTrainDataset
from src.model import CLIPProtoClassifier
from src.robust import generalized_cross_entropy

cfg = yaml.safe_load(open("configs/smoke.yaml", encoding="utf-8"))
device = "cuda"
torch.manual_seed(0)

df = pd.read_csv(cfg["data"]["index_cache"])
labels = df["label"].to_numpy()
ds = MemmapTrainDataset(cfg["data"]["train_mm"], len(df), labels, train=True)
idx = np.arange(512)
loader = DataLoader(Subset(ds, idx), batch_size=64, shuffle=True, num_workers=0)

model = CLIPProtoClassifier(
    num_classes=500, clip_name=cfg["model"]["clip_name"],
    lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
    lora_targets=("q_proj", "v_proj"), train_ln=True,
    logit_scale_init=cfg["model"]["logit_scale_init"],
).to(device)

# 参考特征（等价 train.py 的零样本缓存）
model.eval()
ref = {}
with torch.no_grad():
    for img, y, i in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, f = model(img.to(device))
        for j, k in enumerate(i):
            ref[int(k)] = f[j].float().cpu()

optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=5e-4, weight_decay=1e-4)

model.train()
step = 0
for img, y, i in loader:
    img = img.to(device)
    y = y.to(device)
    assert torch.isfinite(img).all(), "input image not finite"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, f = model(img)
        logits = logits.float()
        loss_sup = generalized_cross_entropy(logits, y, q=0.7)
        f_ref = torch.stack([ref[int(k)] for k in i]).to(device)
        loss_drift = F.mse_loss(f.float(), f_ref)
        loss = loss_sup + 0.5 * loss_drift

    g_ok = torch.isfinite(f).all().item()
    l_ok = torch.isfinite(logits).all().item()
    print(f"step {step:02d} | img[{img.min():.2f},{img.max():.2f}] "
          f"logits[{logits.min():.2f},{logits.max():.2f}] finite={l_ok} "
          f"f_norm={f.float().norm(dim=-1).mean():.3f} finite={g_ok} "
          f"sup={loss_sup.item():.4f} drift={loss_drift.item():.4f} "
          f"scale={model.logit_scale.exp().item():.2f}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad:
        print(f"  !! non-finite grads in: {bad[:5]}")
    gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
    print(f"  grad_norm(pre-clip)={gn:.4f}", flush=True)
    optimizer.step()
    step += 1
    if step >= 8:
        break

print("[diag] finished without crash")
