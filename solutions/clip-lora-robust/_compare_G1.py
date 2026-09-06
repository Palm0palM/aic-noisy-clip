"""对比 G1 候选提交与冠军提交的预测一致性。"""
import pandas as pd

champ = pd.read_csv(r"D:\AIC\outputs\pred_champ_T6_t16.csv", header=None)
print(f"champion rows: {len(champ)}, sample:\n{champ.head(3)}")

for name in ("pred_G1_T3_t12", "pred_G1_T4_t12", "pred_G1_T5_t14", "pred_G1_T3_t10"):
    g = pd.read_csv(rf"D:\AIC\outputs\submissions_G1\{name}.csv", header=None)
    agree = (g[1].astype(str) == champ[1].astype(str)).mean()
    print(f"{name}: agree with champion = {agree:.4f}  (differs on {int((1 - agree) * len(g))} images)")
