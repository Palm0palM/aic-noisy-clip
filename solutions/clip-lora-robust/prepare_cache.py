"""一次性预处理：训练集从U盘(F:)缩放到SSD(D:)，短边256 + JPEG，多进程加速。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC = Path(r"F:\初赛数据集\train")
DST = Path(r"D:\AIC\cache\train_256")
SHORT_SIDE = 256
QUALITY = 90
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_one(args):
    src_path, dst_path = args
    try:
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            return (src_path, True)
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) > SHORT_SIDE:
                if w <= h:
                    nw, nh = SHORT_SIDE, max(1, round(h * SHORT_SIDE / w))
                else:
                    nh, nw = SHORT_SIDE, max(1, round(w * SHORT_SIDE / h))
                im = im.resize((nw, nh), Image.BICUBIC)
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            im.save(dst_path, "JPEG", quality=QUALITY)
        return (src_path, True)
    except Exception:
        return (src_path, False)


def main():
    t0 = time.time()
    tasks = []
    for class_dir in sorted(SRC.iterdir()):
        if not class_dir.is_dir():
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                dst = DST / class_dir.name / (p.stem + ".jpg")
                tasks.append((str(p), str(dst)))
    print(f"[prepare] {len(tasks)} images to process", flush=True)

    bad, done = [], 0
    with Pool(processes=12) as pool:
        for src_path, ok in pool.imap_unordered(process_one, tasks, chunksize=64):
            done += 1
            if not ok:
                bad.append(src_path)
            if done % 5000 == 0:
                el = time.time() - t0
                print(f"[prepare] {done}/{len(tasks)} "
                      f"({done / el:.0f} img/s, bad={len(bad)}, "
                      f"eta={el / done * (len(tasks) - done) / 60:.1f}min)", flush=True)

    Path("cache").mkdir(exist_ok=True)
    json.dump(bad, open("cache/bad_images.json", "w", encoding="utf-8"))
    el = time.time() - t0
    print(f"[prepare] DONE {done} images in {el / 60:.1f}min, bad={len(bad)}", flush=True)


if __name__ == "__main__":
    main()
