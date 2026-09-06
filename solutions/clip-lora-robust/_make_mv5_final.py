"""按模拟器选定的 5 视图权重生成提交：T/tau 微调 + 分布形状验证。"""
import zipfile

import pandas as pd
import torch

V = r"D:\AIC\outputs\val_views"
meta = torch.load(rf"{V}\meta.pt", map_location="cpu", weights_only=False)
val_labels = meta["val_labels"]
sizes = [352, 384, 448, 512, 576]
vl = {s: torch.load(rf"{V}\logits_{s}.pt", map_location="cpu",
                    weights_only=False).float() for s in sizes}

tl = {s: torch.load(p, map_location="cpu", weights_only=False).float()
      for s, p in zip(sizes, [
          r"D:\AIC\outputs\logits_champ_352_tta.pt",
          r"D:\AIC\outputs\logits_champ_zoom_tta.pt",
          r"D:\AIC\outputs\logits_champ_zoom2_tta.pt",
          r"D:\AIC\outputs\logits_champ_zoom3_tta.pt",
          r"D:\AIC\outputs\logits_champ_zoom4_tta.pt"])}
C = 500

cands = {
    "A_simtop": (0.3, 0.35, 0.05, 0.05, 0.25),
    "B_smooth": (0.3, 0.3, 0.1, 0.1, 0.2),
}


def sim(w, T, tau):
    blend = sum(wi * vl[s] for wi, s in zip(w, sizes))
    x = blend / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    pred = (x + tau * (-torch.log(prior * C))).argmax(-1)
    return float((pred == val_labels).float().mean())


def test_pred(w, T, tau):
    blend = sum(wi * tl[s] for wi, s in zip(w, sizes))
    x = blend / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    return (x + tau * (-torch.log(prior * C))).argmax(-1)


anchor = pd.read_csv(r"D:\AIC\outputs\submissions_G2\pred_mv4_5537_T5_t14.csv",
                     header=None)
anchor_pred = torch.tensor(anchor[1].astype(int).values)

print("=== T/tau 微调（val 模拟口径） ===")
best = {}
for name, w in cands.items():
    rows = []
    for T in [4.5, 5.0, 5.5]:
        for tau in [1.2, 1.4, 1.6]:
            rows.append((sim(w, T, tau), T, tau))
    rows.sort(reverse=True)
    best[name] = rows[0]
    print(f"{name} {w}: best T={rows[0][1]} tau={rows[0][2]} sim={rows[0][0]:.4f}")
    for a, T, tau in rows[:3]:
        print(f"   T={T} tau={tau}: {a:.4f}")

print("\n=== 测试分布形状与与锚点(73.90)分歧 ===")
made = []
for name, w in cands.items():
    _, T, tau = best[name]
    pred = test_pred(w, T, tau)
    v = torch.bincount(pred, minlength=C)
    mx, mn = int(v.max()), int(v[v > 0].min())
    ag = float((pred == anchor_pred).float().mean())
    print(f"{name}: T={T} tau={tau} max={mx} min={mn} agree={ag:.4f} "
          f"(diff {int((pred != anchor_pred).sum())})")

    df = anchor.copy()
    df[1] = [f"{int(p):04d}" for p in pred.tolist()]
    name_out = (f"pred_mv5_{name}"
                f"_T{str(T).replace('.', '')}_t{str(tau).replace('.', '')}")
    csv_path = rf"D:\AIC\outputs\submissions_G2\{name_out}.csv"
    df.to_csv(csv_path, index=False, header=False)
    with zipfile.ZipFile(rf"D:\AIC\outputs\submissions_G2\{name_out}.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, "pred_results.csv")
    print(f"  -> {name_out}.zip")
    made.append(name_out)
