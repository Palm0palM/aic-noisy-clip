"""新模型(scale035) vs 冠军 val 视图对比 + 融合搜索。
冠军锚点: B_smooth sim2=0.8590 (real 73.9216)。"""
import glob
import os
import sys

import torch

V = r"D:\AIC\outputs\val_views"      # 冠军
V2 = sys.argv[1] if len(sys.argv) > 1 else r"D:\AIC\outputs\val_views2"

meta = torch.load(rf"{V}\meta.pt", map_location="cpu", weights_only=False)
val_labels = meta["val_labels"]
clean = meta["zs_val_pred"] == val_labels
C = 500


def load_views(d):
    return {os.path.basename(f)[7:-3]: torch.load(f, map_location="cpu",
                                                  weights_only=False).float()
            for f in sorted(glob.glob(rf"{d}\logits_*.pt"))}


champ = load_views(V)
new = load_views(V2)
print(f"champ views: {list(champ)}")
print(f"new   views: {list(new)}")
sizes = [s for s in ("352", "384", "448", "512", "576") if s in new]


def sim(views, w, T=5.0, tau=1.4, mask=None):
    blend = sum(wi * views[k] for k, wi in w.items() if wi > 0)
    m = torch.ones(len(val_labels), dtype=torch.bool) if mask is None else mask
    x = blend[m] / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    pred = (x + tau * (-torch.log(prior * C))).argmax(-1)
    return float((pred == val_labels[m]).float().mean())


def sim2(views, w, T=5.0, tau=1.4):
    return (sim(views, w, T, tau) + sim(views, w, T, tau, clean)) / 2


print(f"\n=== 单视图对比 (acc_raw / acc_cln) ===")
e_c = champ["352"].argmax(-1) != val_labels
for s in sizes:
    pn, pc = new[s].argmax(-1), champ[s].argmax(-1)
    a_n = float((pn == val_labels).float().mean())
    c_n = float((pn[clean] == val_labels[clean]).float().mean())
    a_c = float((pc == val_labels).float().mean())
    ov = float(((pn != val_labels) & e_c).float().mean())
    print(f"{s}: new acc={a_n:.4f} cln={c_n:.4f} | champ acc={a_c:.4f} | "
          f"err_overlap(new352vs_champ352)={ov:.4f} "
          f"agree={(pn == pc).float().mean():.4f}")

print(f"\n=== B_smooth 权重复用（新模型直接套用冠军权重） ===")
b = {"352": .3, "384": .3, "448": .1, "512": .1, "576": .2}
b_new = {k: v for k, v in b.items() if k in new}
print(f"champ: sim2={sim2(champ, b):.4f}   new: sim2={sim2(new, b_new):.4f}")


def greedy(views, base_w, pool, rounds=5):
    cur = {k: v for k, v in base_w.items() if v > 0}
    cur_score = sim2(views, cur)
    hist = [("", cur_score)]
    for rd in range(rounds):
        best = None
        for k in pool:
            if cur.get(k, 0) >= 0.95:
                continue
            for a in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
                cand = {kk: vv * (1 - a) for kk, vv in cur.items()}
                cand[k] = cand.get(k, 0) + a
                s = sim2(views, cand)
                if best is None or s > best[0]:
                    best = (s, k, a, cand)
        if best is None or best[0] <= cur_score + 1e-5:
            break
        cur_score, k, a, cur = best
        hist.append((f"+{k}@{a:.2f}", cur_score))
    return cur, cur_score, hist


pool = list(sizes)
w1, s1, h1 = greedy(new, {"352": .3, "384": .3, "448": .1, "512": .1, "576": .2}, pool)
w2, s2, h2 = greedy(new, {"352": .5, "384": .5}, pool)
print(f"\n=== 新模型贪心 ===")
print(f"from-B:  sim2={s1:.4f} path={[f'{p}:{v:.4f}' for p, v in h1]}")
print(f"         w={w1}")
print(f"scratch: sim2={s2:.4f} path={[f'{p}:{v:.4f}' for p, v in h2]}")
print(f"         w={w2}")

best_w = w1 if s1 >= s2 else w2
rows = sorted(((sim2(new, best_w, T, tau), T, tau)
               for T in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5)
               for tau in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)), reverse=True)
print(f"\nT/tau sweep top5: {[(f'{a:.4f}', T, tau) for a, T, tau in rows[:5]]}")
print(f"\n[判定] 冠军 B_smooth sim2=0.8590; 新模型最优 sim2={max(s1, s2):.4f} "
      f"({max(s1, s2) - 0.8590:+.4f})")
