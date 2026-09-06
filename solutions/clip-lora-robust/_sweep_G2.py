"""G2(384px) logits 的 temp x tau 扫描 + 提交生成，围绕冠军验证过的 98/17 分布形状。"""
import os
import zipfile

import pandas as pd
import torch

l_g2 = torch.load(r"D:\AIC\outputs\logits_G2_384_tta.pt", map_location="cpu",
                  weights_only=False).float()
C = l_g2.shape[1]

raw_pred = l_g2.argmax(-1)
vc = torch.bincount(raw_pred, minlength=C)
print(f"G2(384) raw: max={int(vc.max())} min={int(vc.min())} "
      f"skew={vc.max().item() / vc[vc > 0].min().item():.0f}x")

champ = pd.read_csv(r"D:\AIC\outputs\pred_champ_T6_t16.csv", header=None)

Ts = [3, 4, 5, 6, 7, 8]
taus = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]


def correct(logits, T, t):
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    return (x + t * adj).argmax(-1)


print(f"\n{'T':>3s} {'tau':>4s} {'max':>4s} {'min':>3s} {'cls':>4s}")
rows = []
for T in Ts:
    for t in taus:
        pred = correct(l_g2, T, t)
        v = torch.bincount(pred, minlength=C)
        mx, mn, nc = int(v.max()), int(v[v > 0].min()), int((v > 0).sum())
        ag = (champ[1].astype(int).values == pred.numpy()).mean()
        rows.append((T, t, mx, mn, nc, ag))

# 冠军分布: T6 t1.6 -> max=98 min=17 (validated 71.2501)
sweet = [r for r in rows if 80 <= r[2] <= 120 and 10 <= r[3] <= 30 and r[4] == 500]
sweet.sort(key=lambda r: abs(r[2] - 98) + abs(r[3] - 17))
for r in sweet[:12]:
    print(f"T={r[0]:2d} tau={r[1]:3.1f}  max={r[2]:3d} min={r[3]:2d} agree_champ={r[5]:.4f}")

base = pd.read_csv(r"D:\AIC\outputs\pred_raw_G2_384.csv", header=None)
assert len(base) == l_g2.shape[0]
os.makedirs(r"D:\AIC\outputs\submissions_G2", exist_ok=True)


def make(T, t):
    pred = correct(l_g2, T, t)
    v = torch.bincount(pred, minlength=C)
    df = base.copy()
    df[1] = [f"{int(p):04d}" for p in pred.tolist()]
    name = f"pred_G2_T{T}_t{str(t).replace('.', '')}"
    csv_path = rf"D:\AIC\outputs\submissions_G2\{name}.csv"
    df.to_csv(csv_path, index=False, header=False)
    with zipfile.ZipFile(rf"D:\AIC\outputs\submissions_G2\{name}.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, "pred_results.csv")
    print(f"made {name}: max={int(v.max())} min={int(v[v > 0].min())}")


print("\n=== 生成提交 ===")
for r in sweet[:3]:
    make(r[0], r[1])
