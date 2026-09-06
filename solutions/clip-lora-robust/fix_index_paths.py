"""重写 cache/train_index_256.csv 中的图片路径为当前服务器路径。

在 AIC 目录下运行: python fix_index_paths.py [AIC根目录，默认当前目录]
原始 CSV 的路径是 Windows 格式（D:\AIC\cache\train_256\类名\文件名），
此脚本取每行的 类名/文件名 两级，拼接到新根目录，保证与本地行序完全一致。
"""
import sys
from pathlib import Path

import pandas as pd


def main():
    csv = Path("cache/train_index_256.csv")
    df = pd.read_csv(csv)
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    def remap(p):
        cls, name = p.replace("\\", "/").split("/")[-2:]
        return str(base / "cache" / "train_256" / cls / name)

    df["path"] = df["path"].map(remap)
    df.to_csv(csv, index=False)

    missing = [p for p in df["path"].head(50) if not Path(p).exists()]
    total_missing = sum(not Path(p).exists() for p in df["path"].sample(500, random_state=0))
    print(f"rewrote {len(df)} paths -> {base / 'cache' / 'train_256'}")
    if missing:
        print(f"WARN: {len(missing)} of first 50 missing, e.g. {missing[0]}")
    else:
        print(f"spot check OK (sampled 500, missing={total_missing})")


if __name__ == "__main__":
    main()
