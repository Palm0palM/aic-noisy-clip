"""四视图补充候选：从 3 视图冠军权重 (0.4/0.2/0.4) 自然延伸出含 v4 的组合。"""
import zipfile

import pandas as pd
import torch

C = 500
views = [torch.load(p, map_location="cpu", weights_only=False).float() for p in [
    r"D:\AIC\outputs\logits_champ_352_tta.pt",
    r"D:\AIC\outputs\logits_champ_zoom_tta.pt",
    r"D:\AIC\outputs\logits_champ_zoom2_tta.pt",
    r"D:\AIC\outputs\logits_champ_zoom3_tta.pt",
]]

mv3_csv = pd.read_csv(r"D:\AIC\outputs\submissions_G2\pred_mv3_424_T5_t14.csv",
                      header=None)
mv3_pred = torch.tensor(mv3_csv[1].astype(int).values)


def correct(logits, T, t):
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    return (x + t * (-torch.log(prior * C))).argmax(-1)


candidates = [
    (0.30, 0.15, 0.30, 0.25, 5.0, 1.4),  # 3视图冠军权重的自然延伸
    (0.25, 0.15, 0.25, 0.35, 5.0, 1.4),  # v4 更重
    (0.25, 0.25, 0.25, 0.25, 5.0, 1.4),  # 等权
]
for w in candidates:
    logits = sum(wi * v for wi, v in zip(w[:4], views))
    pred = correct(logits, w[4], w[5])
    v = torch.bincount(pred, minlength=C)
    ag = (pred == mv3_pred).float().mean().item()
    print(f"w={w[:4]} T={w[4]} tau={w[5]}: max={int(v.max())} "
          f"min={int(v[v > 0].min())} agree73={ag:.4f} "
          f"(diff {int((pred != mv3_pred).sum())})")

    df = mv3_csv.copy()
    df[1] = [f"{int(p):04d}" for p in pred.tolist()]
    name = (f"pred_mv4_{int(w[0]*20)}{int(w[1]*20)}{int(w[2]*20)}{int(w[3]*20)}"
            f"_T{str(w[4]).replace('.', '')}_t{str(w[5]).replace('.', '')}")
    csv_path = rf"D:\AIC\outputs\submissions_G2\{name}.csv"
    df.to_csv(csv_path, index=False, header=False)
    with zipfile.ZipFile(rf"D:\AIC\outputs\submissions_G2\{name}.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, "pred_results.csv")
    print(f"  -> {name}.zip")
