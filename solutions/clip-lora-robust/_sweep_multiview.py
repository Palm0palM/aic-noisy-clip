"""冠军模型多视图 TTA 离线评估：352标准视图 + 384缩放视图的各种组合。"""
import torch

C = 500
l_std = torch.load(r"D:\AIC\outputs\logits_champ_352_tta.pt", map_location="cpu",
                   weights_only=False).float()
l_zoom = torch.load(r"D:\AIC\outputs\logits_champ_zoom_tta.pt", map_location="cpu",
                    weights_only=False).float()

# 两视图本身的分歧
p_std, p_zoom = l_std.argmax(-1), l_zoom.argmax(-1)
print(f"std vs zoom raw agreement: {(p_std == p_zoom).float().mean():.4f}")

champ = None  # 冠军提交口径 T6/t1.6


def correct(logits, T, t):
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    return (x + t * adj).argmax(-1)


# 冠军提交(71.2501)的预测作为参照
import pandas as pd
champ_csv = pd.read_csv(r"D:\AIC\outputs\pred_champ_T6_t16.csv", header=None)
champ_pred = torch.tensor(champ_csv[1].astype(int).values)

# 视图组合: 0=仅标准(基线), 1=标准+缩放各0.5, 2=0.7/0.3, 3=0.3/0.7
combos = {
    "std100": (1.0, 0.0),
    "std70zoom30": (0.7, 0.3),
    "5050": (0.5, 0.5),
    "std30zoom70": (0.3, 0.7),
    "zoom100": (0.0, 1.0),
}

Ts = [4, 5, 6, 7, 8]
taus = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

print(f"\n{'combo':14s} {'T':>3s} {'tau':>4s} {'max':>4s} {'min':>3s} {'agree_champ':>10s}")
results = []
for name, (w1, w2) in combos.items():
    logits = w1 * l_std + w2 * l_zoom
    for T in Ts:
        for t in taus:
            pred = correct(logits, T, t)
            v = torch.bincount(pred, minlength=C)
            mx, mn, nc = int(v.max()), int(v[v > 0].min()), int((v > 0).sum())
            if nc != 500:
                continue
            ag = (pred == champ_pred).float().mean().item()
            results.append((name, T, t, mx, mn, ag))

# 冠军最优形状 max=98 min=17
sweet = [r for r in results if 80 <= r[3] <= 120 and 10 <= r[4] <= 30]
sweet.sort(key=lambda r: (abs(r[3] - 98) + abs(r[4] - 17)))
for r in sweet[:18]:
    print(f"{r[0]:14s} T={r[1]:2d} tau={r[2]:3.1f}  max={r[3]:3d} min={r[4]:2d} agree_champ={r[5]:.4f}")
