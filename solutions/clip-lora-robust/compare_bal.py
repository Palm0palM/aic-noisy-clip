import pandas as pd

files = {
    "raw(未校正)": r"D:\AIC\outputs\prelim_r4\pred_results.csv",
    "bal50": r"D:\AIC\outputs\prelim_r4\pred_bal50.csv",
    "bal100": r"D:\AIC\outputs\prelim_r4\pred_bal100.csv",
    "old_64.34": r"D:\AIC\outputs\prelim\pred_results.csv",
}
expect = 24967 / 500
dfs = {}
for name, p in files.items():
    df = pd.read_csv(p, header=None, names=["id", "y"])
    dfs[name] = df
    vc = df["y"].value_counts()
    print(f"{name:14s} n_cls={len(vc)} min={vc.min()} max={vc.max()} "
          f"std={vc.std():.1f} (期望每类={expect:.1f})")

base = dfs["raw(未校正)"]
for name in ["bal50", "bal100"]:
    m = base.merge(dfs[name], on="id", suffixes=("_a", "_b"))
    agree = (m["y_a"] == m["y_b"]).mean()
    print(f"raw vs {name}: agreement={agree:.4f}")
