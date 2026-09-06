"""模拟器稳健性检查: 权重封顶 / 排除稀有类后，top 组合形态是否保持。
并给出候选组合的测试分布形状检查。"""
import torch

V = r"D:\AIC\outputs\val_views"
meta = torch.load(rf"{V}\meta.pt", map_location="cpu", weights_only=False)
val_labels = meta["val_labels"]
sizes = [352, 384, 448, 512, 576]
logits = {s: torch.load(rf"{V}\logits_{s}.pt", map_location="cpu",
                        weights_only=False).float() for s in sizes}
N, C = logits[352].shape

cls_count = torch.bincount(val_labels, minlength=C).float()
print(f"val 类计数: min={int(cls_count.min())} p25={cls_count.float().quantile(.25):.0f} "
      f"median={cls_count.median():.0f} max={int(cls_count.max())}")
print(f"计数<=2 的类数: {(cls_count <= 2).sum()}, <=3: {(cls_count <= 3).sum()}")


def make_weights(mode):
    if mode == "full":
        w = 1.0 / cls_count[val_labels]
    elif mode == "cap":  # 封顶: 权重不超过中位数类权重的3倍
        w = 1.0 / cls_count[val_labels].clamp_min(1)
        med = w.median()
        w = w.clamp(max=3 * med)
    elif mode == "drop3":  # 排除 val 计数<=3 的类的样本
        keep = cls_count[val_labels] > 3
        w = torch.where(keep, 1.0 / cls_count[val_labels], torch.tensor(0.0))
        w = w / w.sum() * keep.sum()
        return w, keep
    return w, torch.ones(N, dtype=torch.bool)


def sim_acc(w, T, tau, mode="full"):
    sw, keep = make_weights(mode)
    blend = sum(wi * logits[s] for wi, s in zip(w, sizes))
    x = blend / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    pred = (x + tau * (-torch.log(prior * C))).argmax(-1)
    correct = ((pred == val_labels).float() * sw)[keep]
    return float(correct.sum() / keep.sum())


cands = {
    "sim_top_exact": (0.3, 0.35, 0.05, 0.05, 0.25),
    "smoothed": (0.3, 0.3, 0.1, 0.1, 0.2),
    "5537(anchor73.90)": (0.25, 0.25, 0.15, 0.35, 0.0),
    "equal4": (0.25, 0.25, 0.25, 0.25, 0.0),
    "equal5": (0.2, 0.2, 0.2, 0.2, 0.2),
    "v5_heavy_monotone": (0.25, 0.2, 0.15, 0.1, 0.3),
}
print(f"\n{'combo':20s} {'full':>7s} {'cap':>7s} {'drop3':>7s}")
for name, w in cands.items():
    row = [sim_acc(w, 5.0, 1.4, m) for m in ["full", "cap", "drop3"]]
    print(f"{name:20s} {row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f}")

# cap 模式下的完整网格（更稳健口径）
wgrid = [round(0.05 + 0.05 * k, 2) for k in range(9)]
res = []
for w1 in wgrid:
    for w2 in wgrid:
        for w3 in wgrid:
            for w4 in wgrid:
                w5 = round(1 - w1 - w2 - w3 - w4, 2)
                if 0.0 <= w5 <= 0.45:
                    res.append((sim_acc((w1, w2, w3, w4, w5), 5.0, 1.4, "cap"),
                                (w1, w2, w3, w4, w5)))
res.sort(reverse=True)
print(f"\n=== cap 口径 Top 10 ===")
for a, w in res[:10]:
    print(f"w={w} sim={a:.4f}")

# 两种口径 top-50 的平均权重（形态稳健性）
import numpy as np
top_full = []
wgrid_res = res  # cap grid
# 重算 full grid
res_full = []
for w1 in wgrid:
    for w2 in wgrid:
        for w3 in wgrid:
            for w4 in wgrid:
                w5 = round(1 - w1 - w2 - w3 - w4, 2)
                if 0.0 <= w5 <= 0.45:
                    res_full.append((sim_acc((w1, w2, w3, w4, w5), 5.0, 1.4, "full"),
                                     (w1, w2, w3, w4, w5)))
res_full.sort(reverse=True)
print(f"\n=== 两口径 top-30 平均权重 ===")
for tag, rr in [("full", res_full), ("cap", wgrid_res)]:
    avg = np.mean([r[1] for r in rr[:30]], axis=0)
    print(f"{tag}: {np.round(avg, 3)}")
