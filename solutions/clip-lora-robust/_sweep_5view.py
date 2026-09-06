"""五视图 TTA 扫描：v1(352) v2(384) v3(448) v4(512) v5(576) 全部裁 352。
已知锚点：4视图 0.25/0.25/0.15/0.35 @T5/t1.4 = 73.9016（97/17）。"""
import itertools
import zipfile

import pandas as pd
import torch

C = 500
paths = [
    r"D:\AIC\outputs\logits_champ_352_tta.pt",
    r"D:\AIC\outputs\logits_champ_zoom_tta.pt",
    r"D:\AIC\outputs\logits_champ_zoom2_tta.pt",
    r"D:\AIC\outputs\logits_champ_zoom3_tta.pt",
    r"D:\AIC\outputs\logits_champ_zoom4_tta.pt",
]
views = [torch.load(p, map_location="cpu", weights_only=False).float() for p in paths]

preds = [v.argmax(-1) for v in views]
for i, j in itertools.combinations(range(5), 2):
    print(f"v{i+1}~v{j+1}: {(preds[i] == preds[j]).float().mean():.4f}", end="  ")
print()

mv4_pred_csv = pd.read_csv(r"D:\AIC\outputs\submissions_G2\pred_mv4_5537_T5_t14.csv",
                           header=None)
mv4_pred = torch.tensor(mv4_pred_csv[1].astype(int).values)


def correct(logits, T, t):
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    return (x + t * adj).argmax(-1)


# 网格：步长 0.05, w_i ∈ [0.05, 0.40]，和为 1
wgrid = [round(0.05 + 0.05 * k, 2) for k in range(8)]
weights = []
for w1 in wgrid:
    for w2 in wgrid:
        for w3 in wgrid:
            for w4 in wgrid:
                w5 = round(1 - w1 - w2 - w3 - w4, 2)
                if 0.05 <= w5 <= 0.40:
                    weights.append((w1, w2, w3, w4, w5))
print(f"权重组合数: {len(weights)}")

Ts = [4.5, 5, 5.5]
taus = [1.2, 1.4, 1.6]
results = []
for w in weights:
    logits = sum(wi * v for wi, v in zip(w, views))
    for T in Ts:
        for t in taus:
            pred = correct(logits, T, t)
            v = torch.bincount(pred, minlength=C)
            mx, mn, nc = int(v.max()), int(v[v > 0].min()), int((v > 0).sum())
            if nc != 500:
                continue
            ag = (pred == mv4_pred).float().mean().item()
            results.append((w, T, t, mx, mn, ag))

sweet = [r for r in results if 85 <= r[3] <= 110 and 12 <= r[4] <= 26]
sweet.sort(key=lambda r: abs(r[3] - 97) + abs(r[4] - 17) + 5 * (1 - r[5]))
print(f"合格 {len(sweet)}，前 15：")
print(f"{'w1':>5s} {'w2':>5s} {'w3':>5s} {'w4':>5s} {'w5':>5s} {'T':>4s} "
      f"{'tau':>4s} {'max':>4s} {'min':>3s} {'ag73.9':>6s}")
for r in sweet[:15]:
    w = r[0]
    print(f"{w[0]:5.2f} {w[1]:5.2f} {w[2]:5.2f} {w[3]:5.2f} {w[4]:5.2f} "
          f"{r[1]:4.1f} {r[2]:4.1f} {r[3]:4d} {r[4]:3d} {r[5]:6.4f}")

made = []
for r in sweet[:20]:
    if r[0] in made:
        continue
    logits = sum(wi * v for wi, v in zip(r[0], views))
    pred = correct(logits, r[1], r[2])
    v = torch.bincount(pred, minlength=C)
    df = mv4_pred_csv.copy()
    df[1] = [f"{int(p):04d}" for p in pred.tolist()]
    ws = "".join(str(int(x * 20)).zfill(2) for x in r[0])
    name = f"pred_mv5_{ws}_T{str(r[1]).replace('.', '')}_t{str(r[2]).replace('.', '')}"
    csv_path = rf"D:\AIC\outputs\submissions_G2\{name}.csv"
    df.to_csv(csv_path, index=False, header=False)
    with zipfile.ZipFile(rf"D:\AIC\outputs\submissions_G2\{name}.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, "pred_results.csv")
    made.append(r[0])
    print(f"made {name}: max={int(v.max())} min={int(v[v > 0].min())} ag={r[5]:.4f}")
    if len(made) >= 3:
        break
