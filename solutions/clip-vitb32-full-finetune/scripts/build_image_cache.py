"""Build a deterministic short-side-resized image cache for fast training.

The official training images are large (median short side 480 px, up to ~2900 px).
Decoding them at full size dominates the data pipeline. This script writes a
resized copy (short side <= --short-side, never upscaled) into a cache directory
that mirrors the original class/filename layout, so every downstream run decodes
~20x fewer pixels.

Deterministic and reproducible: same input, same parameters -> same cache.
Run from the project root:

    python scripts/build_image_cache.py \
        --train-dir /root/autodl-tmp/data/aic-rematch/train \
        --manifest artifacts/train_manifest.csv \
        --cache-dir /root/autodl-tmp/data/aic-rematch/train_cache288 \
        --short-side 288 --quality 90 --workers 14
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

_ARGS = {}


def init_worker(args):
    _ARGS.update(vars(args))


def process_one(relative_path: str) -> int:
    src = Path(_ARGS["train_dir"]) / relative_path
    dst = Path(_ARGS["cache_dir"]) / relative_path
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            if im.format == "JPEG":
                try:
                    im.draft("RGB", (_ARGS["draft_cap"], _ARGS["draft_cap"]))
                except Exception:
                    pass
            im.load()
            try:
                transposed = ImageOps.exif_transpose(im)
            except Exception:
                transposed = None
            if transposed is not None and transposed is not im:
                im = transposed
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                canvas = Image.new("RGB", im.size, (255, 255, 255))
                canvas.paste(im, mask=im.split()[-1])
                im = canvas
            elif im.mode != "RGB":
                im = im.convert("RGB")
            short = min(im.size)
            if short > _ARGS["short_side"]:
                scale = _ARGS["short_side"] / float(short)
                new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
                im = im.resize(new_size, Image.BICUBIC)
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            im.save(tmp, format="JPEG", quality=_ARGS["quality"], optimize=False)
            tmp.replace(dst)
        return 1
    except Exception as exc:  # noqa: BLE001 - decoder errors vary
        print(f"[cache] FAILED {relative_path}: {exc!r}", flush=True)
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--short-side", type=int, default=288)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.draft_cap = args.short_side * 2

    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        rows = [r["relative_path"] for r in csv.DictReader(handle)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[cache] {len(rows)} images -> {args.cache_dir} (short side <= {args.short_side})", flush=True)

    done = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args,)) as pool:
        for i, result in enumerate(pool.map(process_one, rows, chunksize=64), 1):
            if result > 0:
                done += 1
            elif result == 0:
                skipped += 1
            else:
                failed += 1
            if i % 10000 == 0:
                print(f"[cache] {i}/{len(rows)} written={done} skipped={skipped} failed={failed}", flush=True)
    print(f"[cache] done written={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
