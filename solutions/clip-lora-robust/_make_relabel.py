"""生成自蒸馏重打标文件：冠军模型高置信预测覆盖给定标签。"""
import numpy as np
import pandas as pd
import torch

C = 500
T, TAU = 3.0, 1.0

d = torch.load(r"D:\AIC\outputs\train_logits_352real.pt", map_location="cpu",
               weights_only=False)
logits = d["logits"].float()
labels = d["labels"].numpy()
N = len(labels)

x = logits / T
prior = x.softmax(-1).mean(0).clamp_min(1e-8)
x = x + TAU * (-torch.log(prior * C))
p = x.softmax(-1)
pv, pc = p.max(-1)
y = torch.from_numpy(labels)

rows = []
for thr in (0.7, 0.8, 0.9):
    sel = ((pv > thr) & (pc != y)).numpy()
    n = int(sel.sum())
    idx = np.where(sel)[0]
    out = pd.DataFrame({"idx": idx, "new_label": pc[sel].numpy()})
    fn = rf"D:\AIC\relabel_t3_tau1_thr{str(thr).replace('.', '')}.csv"
    out.to_csv(fn, index=False)
    vc = torch.bincount(pc[sel], minlength=C)
    src = torch.bincount(y[sel], minlength=C)
    print(f"thr={thr}: n={n}  target max/min={int(vc.max())}/{int(vc[vc > 0].min())}  "
          f"source max/min={int(src.max())}/{int(src[src > 0].min())} -> {fn}")
