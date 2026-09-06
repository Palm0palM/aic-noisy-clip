"""四视图 TTA 扫描：v1(352) + v2(384裁352) + v3(448裁352) + v4(512裁352)。
已知锚点：3视图 0.4/0.2/0.4 @T5/t1.4 = 73.2086（分布 97/17）。"""
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
v4 = torch.load(r"D:\AIC\outputs\logits_champ_zoom3_tta.pt", map_location="cpu",
                weights_only=False).float()

preds = [v.argmax(-1) for v in (v1, v2, v3, v4)]
for i, j in itertools.combinations(range(4), 2):
    print(f"v{i+1}~v{j+1} agreement: {(preds[i] == preds[j]).float().mean():.4f}")

mv3_pred_csv = pd.read_csv(
    r"D:\AIC\outputs\submissions_G2\pred_mv3_424_T5_t14.csv", header=None)
mv3_pred = torch.tensor(mv3_pred_csv[1].astype(int).values)


def correct(logits, T, t):
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    return (x + t * adj).argmax(-1)


# 权重网格：步长0.1，w_i ∈ [0.05, 0.45]，和为1
wgrid = [round(0.05 + 0.1 * k, 2) for k in range(5)]  # 0.05..0.45
weights = []
for w1 in wgrid:
    for w2 in wgrid:
        for w3 in wgrid:
            w4 = round(1 - w1 - w2 - w3, 2)
            if 0.05 <= w4 <= 0.45:
                weights.append((w1, w2, w3, w4))

Ts = [4, 4.5, 5, 5.5, 6]
taus = [1.0, 1.2, 1.4, 1.6, 1.8]
print(f"\n权重组合数: {len(weights)}, 网格大小: {len(weights) * len(Ts) * len(taus)}")

results = []
for w1, w2, w3, w4 in weights:
    logits = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    for T in Ts:
        for t in taus:
            pred = correct(logits, T, t)
            v = torch.bincount(pred, minlength=C)
            mx, mn, nc = int(v.max()), int(v[v > 0].min()), int((v > 0).sum())
            if nc != 500:
                continue
            ag = (pred == mv3_pred).float().mean().item()
            results.append((w1, w2, w3, w4, T, t, mx, mn, ag))

sweet = [r for r in results if 85 <= r[6] <= 110 and 12 <= r[7] <= 26]
sweet.sort(key=lambda r: abs(r[6] - 97) + abs(r[7] - 17) + 4 * (1 - r[8]))
print(f"合格组合 {len(sweet)} 个，前 20：")
print(f"{'w1':>4s} {'w2':>4s} {'w3':>4s} {'w4':>4s} {'T':>4s} {'tau':>4s} "
      f"{'max':>4s} {'min':>3s} {'agree73':>7s}")
for r in sweet[:20]:
    print(f"{r[0]:4.2f} {r[1]:4.2f} {r[2]:4.2f} {r[3]:4.2f} {r[4]:4.1f} "
          f"{r[5]:4.1f} {r[6]:4d} {r[7]:3d} {r[8]:7.4f}")

# 生成提交：选不同权重形态的前 3 个
made = []
for r in sweet[:12]:
    key = (r[0], r[1], r[2], r[3])
    if key in [m[0] for m in made]:
        continue
    logits = r[0] * v1 + r[1] * v2 + r[2] * v3 + r[3] * v4
    pred = correct(logits, r[4], r[5])
    v = torch.bincount(pred, minlength=C)
    df = mv3_pred_csv.copy()
    df[1] = [f"{int(p):04d}" for p in pred.tolist()]
    name = (f"pred_mv4_{int(r[0]*20)}{int(r[1]*20)}{int(r[2]*20)}{int(r[3]*20)}"
            f"_T{str(r[4]).replace('.', '')}_t{str(r[5]).replace('.', '')}")
    csv_path = rf"D:\AIC\outputs\submissions_G2\{name}.csv"
    df.to_csv(csv_path, index=False, header=False)
    with zipfile.ZipFile(rf"D:\AIC\outputs\submissions_G2\{name}.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, "pred_results.csv")
    made.append((key, name))
    print(f"made {name}: max={int(v.max())} min={int(v[v > 0].min())} "
          f"agree73={r[8]:.4f}")
    if len(made) >= 3:
        break
