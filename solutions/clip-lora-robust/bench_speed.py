"""定位数据管道瓶颈：分别测试单线程取数、多worker加载、GPU前向速度。"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import TrainDataset, build_train_index, get_eval_transform
from src.model import CLIPProtoClassifier

cfg = yaml.safe_load(open("configs/smoke.yaml", encoding="utf-8"))
df = build_train_index(cfg["data"]["train_dir"], cfg["data"]["index_cache"])
df = df.groupby("label", group_keys=False).head(20).reset_index(drop=True)
ds = TrainDataset(df, get_eval_transform())

# (a) 单线程 __getitem__
t0 = time.time()
for i in range(300):
    _ = ds[i]
print(f"(a) single-thread getitem: {300 / (time.time() - t0):.0f} img/s")

# (b) DataLoader 不同 worker 数
for nw in (0, 4, 8):
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    t0, n = time.time(), 0
    for img, y, _ in loader:
        n += len(y)
        if n >= 2000:
            break
    print(f"(b) dataloader workers={nw}: {n / (time.time() - t0):.0f} img/s")

# (c) GPU 前向
device = "cuda"
model = CLIPProtoClassifier(num_classes=500, clip_name=cfg["model"]["clip_name"]).to(device)
model.eval()
x = torch.randn(128, 3, 224, 224, device=device)
with torch.no_grad():
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
dt = (time.time() - t0) / 10
print(f"(c) gpu forward batch=128: {128 / dt:.0f} img/s ({dt * 1000:.0f} ms/batch)")
