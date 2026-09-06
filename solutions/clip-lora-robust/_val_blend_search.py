"""val 集上的融合权重网格搜索：用真实精度判定 v5 价值与最优权重。
口径: raw acc（全部 val）与 cln acc（零样本一致子集，2669 张）。"""
import itertools

import torch

V = r"D:\AIC\outputs\val_views"
meta = torch.load(rf"{V}\meta.pt", map_location="cpu", weights_only=False)
val_labels = meta["val_labels"]
zs_pred = meta["zs_val_pred"]
clean = zs_pred == val_labels
print(f"val n={len(val_labels)} clean={int(clean.sum())}")

sizes = [352, 384, 448, 512, 576]
logits = {s: torch.load(rf"{V}\logits_{s}.pt", map_location="cpu",
                        weights_only=False).float() for s in sizes}

# 误判相关性：视图间错误的重叠度（越低越互补）
for a, b in itertools.combinations(sizes, 2):
    ea = logits[a].argmax(-1) != val_labels
    eb = logits[b].argmax(-1) != val_labels
    both = float((ea & eb).float().mean())
    print(f"err({a},{b}) overlap={both:.4f}  err_a={ea.float().mean():.4f} "
          f"err_b={eb.float().mean():.4f}")


def acc_of(blend):
    pred = blend.argmax(-1)
    return (float((pred == val_labels).float().mean()),
            float((pred[clean] == val_labels[clean]).float().mean()))


# 基线: 已验证的组合
combos = {
    "v1 only": [(1.0, 0, 0, 0, 0)],
    "v1v2 50/50": [(0.5, 0.5, 0, 0, 0)],
    "mv3 .4/.2/.4": [(0.4, 0.2, 0.4, 0, 0)],
    "mv4 5537": [(0.25, 0.25, 0.15, 0.35, 0)],
    "mv4 6365": [(0.3, 0.15, 0.3, 0.25, 0)],
    "equal4": [(0.25, 0.25, 0.25, 0.25, 0)],
}
print("\n=== 已验证组合的 val 表现 ===")
for name, ws in combos.items():
    blend = sum(w * logits[s] for w, s in zip(ws[0], sizes))
    a, c = acc_of(blend)
    print(f"{name:16s} acc={a:.4f} cln={c:.4f}")

# 全量网格: 5 视图, 步长 0.05, w∈[0.05,0.45]
wgrid = [round(0.05 + 0.05 * k, 2) for k in range(9)]
grid = []
for w1 in wgrid:
    for w2 in wgrid:
        for w3 in wgrid:
            for w4 in wgrid:
                w5 = round(1 - w1 - w2 - w3 - w4, 2)
                if 0.0 <= w5 <= 0.45:
                    grid.append((w1, w2, w3, w4, w5))
print(f"\n网格组合数: {len(grid)}")

res = []
for w in grid:
    blend = sum(wi * logits[s] for wi, s in zip(w, sizes))
    pred = blend.argmax(-1)
    a = (pred == val_labels).float().mean().item()
    c = (pred[clean] == val_labels[clean]).float().mean().item()
    res.append((a, c, w))

res.sort(key=lambda r: (r[1], r[0]), reverse=True)
print("\n=== cln 口径 Top 12 ===")
for a, c, w in res[:12]:
    print(f"w={w} acc={a:.4f} cln={c:.4f}")
res.sort(key=lambda r: (r[0], r[1]), reverse=True)
print("\n=== raw acc 口径 Top 12 ===")
for a, c, w in res[:12]:
    print(f"w={w} acc={a:.4f} cln={c:.4f}")

# v5 权重的影响曲线（其余权重自由）
print("\n=== 按 v5 权重分桶的最优表现 ===")
for w5b in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    sub = [r for r in res if abs(r[2][4] - w5b) < 1e-9]
    if not sub:
        continue
    best = max(sub, key=lambda r: (r[1], r[0]))
    print(f"w5={w5b:.2f}: best cln={best[1]:.4f} acc={best[0]:.4f} w={best[2]} "
          f"(n={len(sub)})")
