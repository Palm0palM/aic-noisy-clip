"""测试条件模拟器：类频倒数加权(模拟均衡测试集) + T5/t1.4 校正管线。
先验证能否复现已知测试排序: 5537=73.9016 > 6365=73.8535 > mv3=73.2086。"""
import numpy as np
import torch

V = r"D:\AIC\outputs\val_views"
meta = torch.load(rf"{V}\meta.pt", map_location="cpu", weights_only=False)
val_labels = meta["val_labels"]
sizes = [352, 384, 448, 512, 576]
logits = {s: torch.load(rf"{V}\logits_{s}.pt", map_location="cpu",
                        weights_only=False).float() for s in sizes}
N, C = logits[352].shape

# 类频倒数权重: 让有效类别分布均匀（模拟 500x50 均衡测试集）
cls_count = torch.bincount(val_labels, minlength=C).float()
w_samp = 1.0 / cls_count[val_labels]
w_samp = w_samp / w_samp.sum() * N


def sim_acc(w, T=5.0, tau=1.4):
    blend = sum(wi * logits[s] for wi, s in zip(w, sizes))
    x = blend / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    pred = (x + tau * adj).argmax(-1)
    correct = (pred == val_labels).float()
    return float((correct * w_samp).sum() / N)


# 1) 验证模拟器: 已知测试排序 5537(73.90) > 6365(73.85) > mv3(73.21)
known = {
    "v1only(T6t16测试71.25)": ((1.0, 0, 0, 0, 0), 6.0, 1.6, 71.2501),
    "mv3 424(73.21)": ((0.4, 0.2, 0.4, 0, 0), 5.0, 1.4, 73.2086),
    "mv4 5537(73.90)": ((0.25, 0.25, 0.15, 0.35, 0), 5.0, 1.4, 73.9016),
    "mv4 6365(73.85)": ((0.3, 0.15, 0.3, 0.25, 0), 5.0, 1.4, 73.8535),
    "mv4 equal(?)": ((0.25, 0.25, 0.25, 0.25, 0), 5.0, 1.4, None),
}
print("=== 模拟器验证（sim_acc vs 真实测试分） ===")
for name, (w, T, tau, real) in known.items():
    print(f"{name:22s} sim={sim_acc(w, T, tau):.4f}  real={real}")

# 2) 权重网格搜索（测试条件模拟口径）
wgrid = [round(0.05 + 0.05 * k, 2) for k in range(9)]
grid = []
for w1 in wgrid:
    for w2 in wgrid:
        for w3 in wgrid:
            for w4 in wgrid:
                w5 = round(1 - w1 - w2 - w3 - w4, 2)
                if 0.0 <= w5 <= 0.45:
                    grid.append((w1, w2, w3, w4, w5))

res = []
for w in grid:
    res.append((sim_acc(w), w))
res.sort(reverse=True)
print(f"\n=== 测试模拟口径 Top 15（共 {len(grid)} 组合） ===")
for a, w in res[:15]:
    print(f"w={w} sim={a:.4f}")

print("\n=== 按 v5 权重分桶 ===")
for w5b in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
    sub = [r for r in res if abs(r[1][4] - w5b) < 1e-9]
    if sub:
        print(f"w5={w5b:.2f}: best sim={sub[0][0]:.4f} w={sub[0][1]}")

# 3) 已知组合在模拟器下的差距 vs 测试分的差距（相关性初判）
print("\n=== 锚点组合 sim 差与 real 差 ===")
base = sim_acc((0.25, 0.25, 0.15, 0.35, 0))
for name, (w, T, tau, real) in list(known.items())[1:]:
    if real:
        print(f"{name:18s} sim_diff={sim_acc(w, T, tau) - base:+.4f} "
              f"real_diff={real - 73.9016:+.4f}")
