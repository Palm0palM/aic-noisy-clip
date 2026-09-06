"""数据体检：扫描训练集全部图片，记录不可读文件到 cache/bad_images.json。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path

import yaml

from src.data import build_train_index, verify_images


def main():
    cfg = yaml.safe_load(open("configs/prelim.yaml", encoding="utf-8"))
    df = build_train_index(cfg["data"]["train_dir"], cfg["data"]["index_cache"])
    print(f"[scan] {len(df)} images in index")

    out = Path(cfg["data"]["bad_list"])
    t0 = time.time()
    bad = verify_images(df["path"].tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(bad, open(out, "w", encoding="utf-8"))
    print(f"[scan] done in {time.time() - t0:.1f}s, bad={len(bad)} -> {out}")


if __name__ == "__main__":
    main()
