import os, glob, random
from PIL import Image

random.seed(0)
root = r'F:\初赛数据集\train'
if not os.path.isdir(root):
    print('NOT_FOUND', root)
    raise SystemExit(0)
dirs = sorted(os.listdir(root))
files = []
for d in dirs:
    p = os.path.join(root, d)
    if not os.path.isdir(p):
        continue
    fs = glob.glob(os.path.join(p, '*.jpg')) + glob.glob(os.path.join(p, '*.png')) + glob.glob(os.path.join(p, '*.jpeg'))
    files.extend(fs)
print('total files:', len(files), 'classes:', len(dirs))
random.shuffle(files)
sample = files[:500]
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
    a = sorted(a)
    return a[min(len(a)-1, int(q*len(a)))]

print(f'sampled={len(ws)} bad={bad}')
print(f'width  min/p10/p50/p90/max = {min(ws)}/{pct(ws,0.1)}/{pct(ws,0.5)}/{pct(ws,0.9)}/{max(ws)}')
print(f'height min/p10/p50/p90/max = {min(hs)}/{pct(hs,0.1)}/{pct(hs,0.5)}/{pct(hs,0.9)}/{max(hs)}')
print(f'short  min/p10/p50/p90/max = {min(shorts)}/{pct(shorts,0.1)}/{pct(shorts,0.5)}/{pct(shorts,0.9)}/{max(shorts)}')
print(f'long   min/p10/p50/p90/max = {min(longs)}/{pct(longs,0.1)}/{pct(longs,0.5)}/{pct(longs,0.9)}/{max(longs)}')
print(f'frac short<384: {sum(1 for s in shorts if s<384)/len(shorts):.3f}')
print(f'frac short<448: {sum(1 for s in shorts if s<448)/len(shorts):.3f}')
print(f'frac long<448:  {sum(1 for s in longs if s<448)/len(longs):.3f}')
