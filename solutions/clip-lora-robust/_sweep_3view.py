"""三视图 TTA 扫描：v1(352标准) + v2(384裁352, 8%放大) + v3(448裁352, 27%放大)。
已知锚点：50/50 v1+v2 @T6/t1.6 = 72.0992（分布 98/16）。"""
import itertools
import zipfile

import pandas as pd
import torch

C = 500
v1 = torch.load(r"D:\AIC\outputs\logits_champ_352_tta.pt", map_location="cpu",
                weights_only=False).float()
v2 = torch.load(r"D:\AIC\outputs\logits_champ_zoom_tta.pt", map_location="cpu",
                weights_only=False).float()
v3 = torch.load(r"D:\AIC\outputs\logits_champ_zoom2_tta.pt", map_location="cpu",
                weights_only=False).float()

p1, p2, p3 = v1.argmax(-1), v2.argmax(-1), v3.argmax(-1)
print(f"v1~v2 agreement: {(p1 == p2).float().mean():.4f}")
print(f"v1~v3 agreement: {(p1 == p3).float().mean():.4f}")
print(f"v2~v3 agreement: {(p2 == p3).float().mean():.4f}")

mv2_pred_csv = pd.read_csv(r"D:\AIC\outputs\submissions_G2\pred_champ_mv5050_T6_t16.csv",
                           header=None)
mv2_pred = torch.tensor(mv2_pred_csv[1].astype(int).values)


def correct(logits, T, t):
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    return (x + t * adj).argmax(-1)


# 权重组合（w1+w2+w3=1），步长 0.1；w1 保证 >=0.3（标准视图是全视野基准）
weights = []
for w1 in [0.3, 0.4, 0.5, 0.6]:
    for w2 in [0.1, 0.2, 0.3, 0.4]:
        w3 = round(1 - w1 - w2, 2)
        if 0.05 <= w3 <= 0.4:
            weights.append((w1, w2, w3))

Ts = [5, 6, 7]
taus = [1.2, 1.4, 1.6, 1.8, 2.0]

results = []
for w1, w2, w3 in weights:
    logits = w1 * v1 + w2 * v2 + w3 * v3
    for T in Ts:
        for t in taus:
            pred = correct(logits, T, t)
            v = torch.bincount(pred, minlength=C)
            mx, mn, nc = int(v.max()), int(v[v > 0].min()), int((v > 0).sum())
            if nc != 500:
                continue
            ag = (pred == mv2_pred).float().mean().item()
            results.append((w1, w2, w3, T, t, mx, mn, ag))

# 已验证形状：98/16 (72.0992)
sweet = [r for r in results if 85 <= r[5] <= 115 and 12 <= r[6] <= 25]
sweet.sort(key=lambda r: abs(r[5] - 98) + abs(r[6] - 16) + 3 * (1 - r[7]))
print(f"\n共 {len(sweet)} 个合格组合，前 15：")
print(f"{'w1':>4s} {'w2':>4s} {'w3':>4s} {'T':>2s} {'tau':>4s} {'max':>4s} {'min':>3s} {'agree72':>7s}")
for r in sweet[:15]:
    print(f"{r[0]:4.1f} {r[1]:4.1f} {r[2]:4.1f} {r[3]:2d} {r[4]:4.1f} {r[5]:4d} {r[6]:3d} {r[7]:7.4f}")

# 生成提交：前3个不同权重形态
base = mv2_pred_csv.copy()
made = []
for r in sweet[:5]:
    key = (r[0], r[1], r[2])
    if key in [m[0] for m in made]:
        continue
    logits = r[0] * v1 + r[1] * v2 + r[2] * v3
    pred = correct(logits, r[3], r[4])
    v = torch.bincount(pred, minlength=C)
    df = base.copy()
    df[1] = [f"{int(p):04d}" for p in pred.tolist()]
    name = (f"pred_mv3_{int(r[0]*10)}{int(r[1]*10)}{int(r[2]*10)}_T{r[3]}"
            f"_t{str(r[4]).replace('.', '')}")
    csv_path = rf"D:\AIC\outputs\submissions_G2\{name}.csv"
    df.to_csv(csv_path, index=False, header=False)
    with zipfile.ZipFile(rf"D:\AIC\outputs\submissions_G2\{name}.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, "pred_results.csv")
    made.append((key, name))
    print(f"made {name}: max={int(v.max())} min={int(v[v > 0].min())} "
          f"agree72={r[7]:.4f}")
    if len(made) >= 3:
        break
