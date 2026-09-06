import os, glob, random
from PIL import Image

random.seed(0)
root = 'train_orig'
dirs = sorted(os.listdir(root))
files = []
for d in dirs[:500]:
    p = os.path.join(root, d)
    fs = glob.glob(os.path.join(p, '*.jpg')) + glob.glob(os.path.join(p, '*.png')) + glob.glob(os.path.join(p, '*.jpeg'))
    files.extend(fs)
random.shuffle(files)
sample = files[:400]
ws, hs, shorts, longs = [], [], [], []
bad = 0
for f in sample:
    try:
        with Image.open(f) as im:
            w, h = im.size
        ws.append(w); hs.append(h)
        shorts.append(min(w, h)); longs.append(max(w, h))
    except Exception:
        bad += 1

def pct(a, q):
    a = sorted(a); import math
    return a[min(len(a)-1, int(q*len(a)))]

print(f'sampled={len(ws)} bad={bad}')
print(f'width  min/p10/p50/p90/max = {min(ws)}/{pct(ws,0.1)}/{pct(ws,0.5)}/{pct(ws,0.9)}/{max(ws)}')
print(f'height min/p10/p50/p90/max = {min(hs)}/{pct(hs,0.1)}/{pct(hs,0.5)}/{pct(hs,0.9)}/{max(hs)}')
print(f'short  min/p10/p50/p90/max = {min(shorts)}/{pct(shorts,0.1)}/{pct(shorts,0.5)}/{pct(shorts,0.9)}/{max(shorts)}')
print(f'long   min/p10/p50/p90/max = {min(longs)}/{pct(longs,0.1)}/{pct(longs,0.5)}/{pct(longs,0.9)}/{max(longs)}')
frac_below_384 = sum(1 for s in shorts if s < 384) / len(shorts)
frac_below_448 = sum(1 for s in shorts if s < 448) / len(shorts)
print(f'fraction short-side < 384: {frac_below_384:.3f}')
print(f'fraction short-side < 448: {frac_below_448:.3f}')
