"""将缓存 JPEG(短边256) 打包为 numpy memmap (N,3,256,256) uint8，中心方裁剪后缩放。
训练/推理直接读 memmap，避免每 epoch 十万次文件打开 + PIL 解码 + worker 递归开销。
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
SIZE = int(os.environ.get("MM_SIZE", "256"))  # spawn 子进程经环境变量继承尺寸


def load_square_256(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        if s != SIZE:
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)  # (256,256,3) HWC


def fill_range(args):
    paths, mm_path, total, start, end = args
    mm = np.memmap(mm_path, dtype=np.uint8, mode="r+", shape=(total, 3, SIZE, SIZE))
    fails = []
    for i in range(start, end):
        try:
            mm[i] = load_square_256(paths[i]).transpose(2, 0, 1)  # CHW
        except Exception:
            mm[i] = 0
            fails.append(paths[i])
    return fails


def pack(paths, mm_path, workers=12):
    total = len(paths)
    mm = np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=(total, 3, SIZE, SIZE))
    del mm
    chunk = 2000
    ranges = [(paths, str(mm_path), total, s, min(s + chunk, total))
              for s in range(0, total, chunk)]
    t0, done, fails = time.time(), 0, []
    with Pool(workers) as pool:
        for f in pool.imap_unordered(fill_range, ranges):
            fails.extend(f)
            done += 1
            n = min(done * chunk, total)
            print(f"[pack] {n}/{total} ({n / (time.time() - t0):.0f} img/s)", flush=True)
    return fails


def main():
    import pandas as pd
    import sys
    cache = Path("cache")

    if len(sys.argv) > 1 and sys.argv[1] == "352":
        os.environ["MM_SIZE"] = "352"
        global SIZE
        SIZE = 352

    # ---- 训练集：顺序与 train_index_256.csv 完全一致 ----
    df = pd.read_csv(cache / "train_index_256.csv")
    train_mm = cache / f"train_u8_{SIZE}.npy"
    if not train_mm.exists():
        print(f"[train] packing {len(df)} images at {SIZE}px", flush=True)
        fails = pack(df["path"].tolist(), train_mm)
        json.dump(fails, open(cache / f"memmap_bad_train_{SIZE}.json", "w"))
        print(f"[train] done, fails={len(fails)}", flush=True)

    # ---- 测试集：按文件名排序，与 TestDataset 一致 ----
    test_dir = Path(os.environ.get("TEST_DIR", r"D:\download\test"))
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_paths = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in exts],
                        key=lambda p: p.name)
    test_mm = cache / f"test_u8_{SIZE}.npy"
    if not test_mm.exists():
        print(f"[test] packing {len(test_paths)} images at {SIZE}px", flush=True)
        fails = pack([str(p) for p in test_paths], test_mm)
        json.dump(fails, open(cache / f"memmap_bad_test_{SIZE}.json", "w"))
        if not (cache / "test_names.json").exists():
            json.dump([p.name for p in test_paths], open(cache / "test_names.json", "w"))
        print(f"[test] done, fails={len(fails)}", flush=True)


if __name__ == "__main__":
    main()
