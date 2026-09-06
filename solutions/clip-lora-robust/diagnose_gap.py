"""诊断 val/test 差距来源与特征质量，为大升级定方向。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import MemmapTrainDataset, stratified_split
from src.model import CLIPProtoClassifier
from src.robust import gmm_clean_prob

cfg = yaml.safe_load(open("configs/prelim.yaml", encoding="utf-8"))
device = "cuda"

df = pd.read_csv(cfg["data"]["index_cache"])
labels = df["label"].to_numpy()
train_idx, val_idx = stratified_split(labels, cfg["data"]["val_ratio"], cfg["seed"])

ds = MemmapTrainDataset(cfg["data"]["train_mm"], len(df), labels, train=False)

model = CLIPProtoClassifier(
    num_classes=500, clip_name=cfg["model"]["clip_name"],
    lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
    lora_targets=("q_proj", "v_proj"), train_ln=True,
    logit_scale_init=cfg["model"]["logit_scale_init"],
).to(device)
ckpt = torch.load("outputs/prelim/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"[load] best.pt epoch={ckpt['epoch']} val_acc={ckpt['val_acc']:.4f}")


@torch.no_grad()
def forward_all(indices, bs=384):
    loader = DataLoader(Subset(ds, indices), batch_size=bs, shuffle=False,
                        num_workers=0, pin_memory=True)
    outs, feats = [], []
    for img, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, f = model(img.to(device))
        outs.append(lo.float().cpu())
        feats.append(f.half().cpu())
    return torch.cat(outs), torch.cat(feats)


# ---- 1) 训练集预测一致率 -> 估计真实噪声率 ----
tr_logits, tr_feats = forward_all(train_idx)
tr_pred = tr_logits.argmax(-1)
tr_y = torch.from_numpy(labels[train_idx])
agree = (tr_pred == tr_y).float().mean().item()
print(f"[noise] train agreement(pred==given_label) = {agree:.4f}")
# 模型高置信子集上的一致率（更接近真实噪声率）
conf = F.softmax(tr_logits, -1).max(-1).values
for thr in [0.9, 0.7, 0.5]:
    m = conf > thr
    print(f"  conf>{thr}: n={int(m.sum())} agreement={(tr_pred[m] == tr_y[m]).float().mean():.4f}")

# ---- 2) 干净 val 上的真实准确率估计 ----
va_logits, va_feats = forward_all(val_idx)
va_y = torch.from_numpy(labels[val_idx])
va_pred = va_logits.argmax(-1)
print(f"[val] raw val acc = {(va_pred == va_y).float().mean():.4f}")
# val 中模型与标签一致的可信干净样本
clean_val = va_pred == va_y
print(f"[val] self-consistent samples: {int(clean_val.sum())}/{len(val_idx)}")

# ---- 3) 特征质量: 用 GMM clean 的训练样本建类均值原型, 在 val 上评估 ----
losses = F.cross_entropy(tr_logits, tr_y, reduction="none").numpy()
prob = gmm_clean_prob(losses)
clean_tr = torch.from_numpy(prob > 0.7)
print(f"[proto] clean train samples: {int(clean_tr.sum())}")
proto = torch.zeros(500, tr_feats.shape[1])
for c in range(500):
    m = (tr_y == c) & clean_tr
    if m.any():
        proto[c] = tr_feats[m].float().mean(0)
proto = F.normalize(proto, dim=-1)
sim = va_feats.float() @ proto.t()
knn_acc = (sim.argmax(-1) == va_y).float().mean().item()
print(f"[proto] class-mean(clean) val acc = {knn_acc:.4f}")

# ---- 4) 各类别难度: 找出 val acc 最低的类别 ----
per_cls = {}
for c in range(500):
    m = va_y == c
    if m.any():
        per_cls[c] = (va_pred[m] == c).float().mean().item()
worst = sorted(per_cls.items(), key=lambda kv: kv[1])[:10]
best = sorted(per_cls.items(), key=lambda kv: kv[1])[-5:]
print("[per-cls] worst 10:", [(c, f"{a:.2f}") for c, a in worst])
print("[per-cls] best 5:", [(c, f"{a:.2f}") for c, a in best])
np.save("outputs/prelim/diag_train_cleanprob.npy", prob)
