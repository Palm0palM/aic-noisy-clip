"""逐阶段检查 memmap 数据管线，找出非有限值产生的环节。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2

from src.data import CLIP_MEAN, CLIP_STD

mm = np.memmap("cache/train_u8_256.npy", dtype=np.uint8, mode="r",
               shape=(103218, 3, 256, 256))
df = pd.read_csv("cache/train_index_256.csv")

# 1) 原始 memmap 数据
for i in [0, 1, 100, 50000, 103217]:
    a = mm[i]
    print(f"raw[{i}] dtype={a.dtype} min={a.min()} max={a.max()} mean={a.mean():.1f}")

# 2) 逐变换检查
tf_steps = [
    ("todtype", v2.ToDtype(torch.float32, scale=True)),
    ("rrc", v2.RandomResizedCrop(224, scale=(0.5, 1.0),
                                 interpolation=v2.InterpolationMode.BICUBIC, antialias=True)),
    ("flip", v2.RandomHorizontalFlip()),
    ("jitter", v2.ColorJitter(0.2, 0.2, 0.2, 0.05)),
    ("norm", v2.Normalize(CLIP_MEAN, CLIP_STD)),
]

torch.manual_seed(0)
bad = 0
for trial in range(200):
    i = int(np.random.randint(0, 103218))
    x = torch.from_numpy(mm[i].copy())
    for name, tf in tf_steps:
        x = tf(x)
        if not torch.isfinite(x).all():
            print(f"[trial {trial}] idx={i} NON-FINITE after '{name}': "
                  f"min={x.min()} max={x.max()} nan={torch.isnan(x).sum().item()}")
            bad += 1
            break
print(f"checked 200 random samples, bad={bad}")

# 3) 重点怀疑 jitter：对固定样本暴力枚举多次
x0 = torch.from_numpy(mm[0].copy())
x0 = v2.ToDtype(torch.float32, scale=True)(x0)
x0 = v2.CenterCrop(224)(x0)
jit = v2.ColorJitter(0.2, 0.2, 0.2, 0.05)
cnt = 0
for t in range(500):
    y = jit(x0)
    if not torch.isfinite(y).all():
        cnt += 1
        if cnt == 1:
            print(f"jitter produced non-finite at iter {t}: "
                  f"nan={torch.isnan(y).sum().item()} inf={torch.isinf(y).sum().item()}")
print(f"jitter stress: {cnt}/500 non-finite")
