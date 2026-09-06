"""冠军模型多视图 TTA 提交：50/50 融合，T6/t1.6（分布 98/16）。"""
import zipfile

import pandas as pd
import torch

C = 500
l_std = torch.load(r"D:\AIC\outputs\logits_champ_352_tta.pt", map_location="cpu",
                   weights_only=False).float()
l_zoom = torch.load(r"D:\AIC\outputs\logits_champ_zoom_tta.pt", map_location="cpu",
                    weights_only=False).float()
logits = 0.5 * l_std + 0.5 * l_zoom

T, t = 6.0, 1.6
x = logits / T
prior = x.softmax(-1).mean(0).clamp_min(1e-8)
pred = (x + t * (-torch.log(prior * C))).argmax(-1)
v = torch.bincount(pred, minlength=C)
print(f"multiview 5050 T={T} tau={t}: max={int(v.max())} min={int(v[v > 0].min())} "
      f"classes={int((v > 0).sum())}")

base = pd.read_csv(r"D:\AIC\outputs\pred_champ_T6_t16.csv", header=None)
assert len(base) == len(pred)
champ_pred = torch.tensor(base[1].astype(int).values)
print(f"agree with champion 71.2501: {(pred == champ_pred).float().mean():.4f} "
      f"(differs on {int((pred != champ_pred).sum())} images)")

df = base.copy()
df[1] = [f"{int(p):04d}" for p in pred.tolist()]
csv_path = r"D:\AIC\outputs\submissions_G2\pred_champ_mv5050_T6_t16.csv"
df.to_csv(csv_path, index=False, header=False)
with zipfile.ZipFile(r"D:\AIC\outputs\submissions_G2\pred_champ_mv5050_T6_t16.zip", "w",
                     zipfile.ZIP_DEFLATED) as z:
    z.write(csv_path, "pred_results.csv")
print(f"saved -> {csv_path}.zip")
