"""角点视图评估：双口径(raw+cln) T/tau 模拟器 + 贪心视图选择。
锚点: B_smooth=73.9216, 5537=73.9016, A_simtop=73.8695, 6365=73.8535, mv3=73.2086, v1=71.2501"""
import glob
import os

import torch

V = r"D:\AIC\outputs\val_views"
meta = torch.load(rf"{V}\meta.pt", map_location="cpu", weights_only=False)
val_labels = meta["val_labels"]
clean = meta["zs_val_pred"] == val_labels
C = 500

views = {}
for f in sorted(glob.glob(rf"{V}\logits_*.pt")):
    views[os.path.basename(f)[7:-3]] = torch.load(f, map_location="cpu",
                                                  weights_only=False).float()
print(f"views({len(views)}): {list(views)}")


def sim(w, T=5.0, tau=1.4, mask=None):
    blend = sum(wi * views[k] for k, wi in w.items() if wi > 0)
    m = torch.ones(len(val_labels), dtype=torch.bool) if mask is None else mask
    x = blend[m] / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    pred = (x + tau * (-torch.log(prior * C))).argmax(-1)
    return float((pred == val_labels[m]).float().mean())


def sim2(w, T=5.0, tau=1.4):
    return (sim(w, T, tau) + sim(w, T, tau, clean)) / 2


anchors = [
    ("v1(T6t16)", {"352": 1.0}, 6.0, 1.6, 71.2501),
    ("mv3", {"352": .4, "384": .2, "448": .4}, 5.0, 1.4, 73.2086),
    ("6365", {"352": .3, "384": .15, "448": .3, "512": .25}, 5.0, 1.4, 73.8535),
    ("5537", {"352": .25, "384": .25, "448": .15, "512": .35}, 5.0, 1.4, 73.9016),
    ("A_simtop", {"352": .3, "384": .35, "448": .05, "512": .05, "576": .25},
     5.0, 1.4, 73.8695),
    ("B_smooth", {"352": .3, "384": .3, "448": .1, "512": .1, "576": .2},
     5.0, 1.4, 73.9216),
]
print(f"\n{'anchor':10s} {'raw':>7s} {'cln':>7s} {'avg':>7s} {'real':>8s}")
for name, w, T, tau, real in anchors:
    r, c = sim(w, T, tau), sim(w, T, tau, clean)
    print(f"{name:10s} {r:7.4f} {c:7.4f} {(r + c) / 2:7.4f} {real:8.4f}")

corners = [k for k in views if k.startswith("c")]
if not corners:
    print("\n[no corner views yet — 上传 infer_val_corners.py 结果后重跑]")
    raise SystemExit

print("\n=== corner 视图单体（vs v1 错误重叠） ===")
e1 = views["352"].argmax(-1) != val_labels
for k in corners:
    e = views[k].argmax(-1) != val_labels
    print(f"{k:10s} acc={1 - float(e.float().mean()):.4f} "
          f"err_overlap_v1={float((e & e1).float().mean()):.4f}")


def greedy(base_w, pool, rounds=4):
    cur = {k: v for k, v in base_w.items() if v > 0}
    cur_score = sim2(cur)
    print(f"\n--- greedy from {cur} (start sim2={cur_score:.4f}) ---")
    for rd in range(rounds):
        best = None
        for k in pool:
            if cur.get(k, 0) >= 0.95:
                continue
            for a in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
                cand = {kk: vv * (1 - a) for kk, vv in cur.items()}
                cand[k] = cand.get(k, 0) + a
                s = sim2(cand)
                if best is None or s > best[0]:
                    best = (s, k, a, cand)
        if best is None or best[0] <= cur_score + 1e-5:
            break
        cur_score, k, a, cur = best
        print(f"round {rd + 1}: +{k}@{a:.2f} -> sim2={cur_score:.4f}  w={cur}")
    return cur, cur_score


pool = corners + ["352", "384", "448", "512", "576"]
b_base = {"352": .3, "384": .3, "448": .1, "512": .1, "576": .2}
w1, s1 = greedy(b_base, pool)
w2, s2 = greedy({"352": 0.5, "384": 0.5}, pool)

base_score = sim2(b_base)
print(f"\nB_smooth sim2={base_score:.4f}")
print(f"greedy-from-B:  sim2={s1:.4f} ({s1 - base_score:+.4f}) w={w1}")
print(f"greedy-scratch: sim2={s2:.4f} ({s2 - base_score:+.4f}) w={w2}")

best_w = w1 if s1 >= s2 else w2
rows = sorted(((sim2(best_w, T, tau), T, tau)
               for T in (4.0, 4.5, 5.0, 5.5, 6.0)
               for tau in (1.0, 1.2, 1.4, 1.6, 1.8)), reverse=True)
print(f"\nbest_w T/tau top5: {[(f'{a:.4f}', T, tau) for a, T, tau in rows[:5]]}")
