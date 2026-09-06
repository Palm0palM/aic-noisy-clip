"""G1 logits 的 temp x tau 扫描 + 提交文件生成。"""
import os
import zipfile

import pandas as pd
import torch

l_g1 = torch.load(r"D:\AIC\outputs\logits_G1_tta.pt", map_location="cpu",
                  weights_only=False).float()
l_champ = torch.load(r"D:\AIC\outputs\prelim_run2_352real\logits_tta.pt",
                     map_location="cpu", weights_only=False).float() \
    if os.path.exists(r"D:\AIC\outputs\prelim_run2_352real\logits_tta.pt") else None
C = l_g1.shape[1]
print(f"G1 logits: {tuple(l_g1.shape)}  champion logits: {tuple(l_champ.shape) if l_champ is not None else None}")

raw_pred = l_g1.argmax(-1)
vc = torch.bincount(raw_pred, minlength=C)
print(f"G1 raw: max={int(vc.max())} min={int(vc.min())} skew={vc.max().item() / vc[vc > 0].min().item():.0f}x")
if l_champ is not None:
    cp = l_champ.argmax(-1)
    print(f"raw agreement with champion: {(raw_pred == cp).float().mean():.4f}")

Ts = [3, 4, 5, 6, 7, 8, 10]
taus = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]


def correct(logits, T, t):
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    return (x + t * adj).argmax(-1)


# 冠军参考: 352real T6 t1.6 (71.2501)
champ_pred = None
if l_champ is not None:
    champ_pred = correct(l_champ, 6, 1.6)

print(f"\n{'T':>3s} {'tau':>4s} {'max':>4s} {'min':>3s} {'cls':>4s} {'agree_champ':>10s}")
rows = []
for T in Ts:
    for t in taus:
        pred = correct(l_g1, T, t)
        v = torch.bincount(pred, minlength=C)
        mx, mn, nc = int(v.max()), int(v[v > 0].min()), int((v > 0).sum())
        ag = float((pred == champ_pred).float().mean()) if champ_pred is not None else 1.0
        rows.append((T, t, mx, mn, nc, ag))
        print(f"{T:3d} {t:4.1f} {mx:4d} {mn:3d} {nc:4d} {ag:10.4f}")

# 冠军分布参考: T6 t1.6 -> max=120 min=7
sweet = [r for r in rows if 60 <= r[2] <= 140 and 6 <= r[3] <= 40 and r[4] == 500]
sweet.sort(key=lambda r: (abs(r[2] - 98) + abs(r[3] - 17), -r[5]))
print(f"\n=== sweet-zone (围绕冠军分布 max~98 min~17) {len(sweet)} 个 ===")
for r in sweet[:10]:
    print(f"T={r[0]:2d} tau={r[1]:3.1f}  max={r[2]:3d} min={r[3]:2d} agree_champ={r[5]:.4f}")

base = pd.read_csv(r"D:\AIC\outputs\pred_raw_G1.csv", header=None)
assert len(base) == l_g1.shape[0]
os.makedirs(r"D:\AIC\outputs\submissions_G1", exist_ok=True)


def make(T, t):
    pred = correct(l_g1, T, t)
    v = torch.bincount(pred, minlength=C)
    df = base.copy()
    df[1] = [f"{int(p):04d}" for p in pred.tolist()]
    name = f"pred_G1_T{T}_t{str(t).replace('.', '')}"
    csv_path = rf"D:\AIC\outputs\submissions_G1\{name}.csv"
    df.to_csv(csv_path, index=False, header=False)
    with zipfile.ZipFile(rf"D:\AIC\outputs\submissions_G1\{name}.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, "pred_results.csv")
    print(f"made {name}: max={int(v.max())} min={int(v[v > 0].min())} "
          f"agree_champ={float((pred == champ_pred).float().mean()) if champ_pred is not None else 1:.4f}")


print("\n=== 生成提交 ===")
for r in sweet[:4]:
    make(r[0], r[1])
