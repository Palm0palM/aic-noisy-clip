import pandas as pd

old = pd.read_csv(r"D:\AIC\outputs\prelim\pred_results.csv", header=None, names=["id", "y_old"])
new = pd.read_csv(r"D:\AIC\outputs\prelim_r4\pred_results.csv", header=None, names=["id", "y_new"])

id_c, y_old, y_new = "id", "y_old", "y_new"
m = old[[id_c, y_old]].merge(new[[id_c, y_new]], on=id_c)
same = (m[y_old] == m[y_new]).sum()
print(f"agreement with 64.34 submission: {same/len(m):.4f} ({same}/{len(m)})")

co = m[y_old].value_counts()
cn = m[y_new].value_counts()
print(f"old: n_classes={len(co)} min={co.min()} max={co.max()} std={co.std():.1f}")
print(f"new: n_classes={len(cn)} min={cn.min()} max={cn.max()} std={cn.std():.1f}")

chg = (cn - co).abs().sort_values(ascending=False)
print("top-8 classes with biggest prediction-count change:")
print(chg.head(8).to_string())
