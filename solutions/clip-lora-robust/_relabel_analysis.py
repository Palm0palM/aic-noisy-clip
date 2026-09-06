"""离线分析冠军模型训练集 logits，确定自蒸馏重打标阈值。"""
import numpy as np
import pandas as pd
import torch

C = 500
SEED = 42
VAL_RATIO = 0.05

d = torch.load(r"D:\AIC\outputs\train_logits_352real.pt", map_location="cpu",
               weights_only=False)
logits = d["logits"].float()
labels = d["labels"].numpy()
N = len(labels)

df = pd.read_csv(r"D:\AIC\cache\train_index_256.csv") if False else None
# 复现服务器 stratified_split(seed=42, val_ratio=0.05)
rng = np.random.RandomState(SEED)
train_idx, val_idx = [], []
for c in np.unique(labels):
    idx = np.where(labels == c)[0]
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * VAL_RATIO)))
    val_idx.extend(idx[:n_val])
    train_idx.extend(idx[n_val:])
train_mask = torch.zeros(N, dtype=torch.bool)
train_mask[torch.tensor(train_idx)] = True
print(f"N={N} train={len(train_idx)} val={len(val_idx)}")

y = torch.from_numpy(labels)
pred_raw = logits.argmax(-1)
print(f"raw agreement: {(pred_raw == y).float().mean():.4f}")


def corrected(T, tau):
    x = logits / T
    if tau > 0:
        prior = x.softmax(-1).mean(0).clamp_min(1e-8)
        x = x + tau * (-torch.log(prior * C))
    return x


print("\n=== sweep: T x tau -> 不同置信阈值下的重打标数量 (仅训练集部分) ===")
for T in (1.0, 3.0, 6.0):
    for tau in (0.0, 1.0, 2.0):
        x = corrected(T, tau)
        p = x.softmax(-1)
        pv, pc = p.max(-1)
        diff = (pc != y) & train_mask
        line = f"T={T:<3} tau={tau:<3}"
        for thr in (0.7, 0.8, 0.9, 0.95):
            n = int(((pv > thr) & diff).sum())
            line += f"  >={thr}:{n:6d}"
        # 重打标目标类别分布偏斜
        sel = (pv > 0.9) & diff
        if sel.any():
            vc = torch.bincount(pc[sel], minlength=C)
            line += f"   [0.9] target-class max/min={int(vc.max())}/{int(vc[vc > 0].min())}"
        print(line)

# 选定候选配置后看 val 表现
print("\n=== val 子集诊断 (T=6, tau 扫描) ===")
vm = ~train_mask
for tau in (0.0, 1.0, 2.0):
    x = corrected(6.0, tau)
    p = x.softmax(-1)
    pv, pc = p.max(-1)
    acc_given = (pc[vm] == y[vm]).float().mean().item()
    diff = (pc != y) & vm
    for thr in (0.8, 0.9):
        n = int(((pv > thr) & diff).sum())
        print(f"tau={tau}: val_acc_vs_given={acc_given:.4f}  val高置信分歧(>{thr})={n}")
