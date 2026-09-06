"""数据加载：训练集(按类别文件夹)、测试集(平铺 jpg)、memmap 高速数据集、变换与索引缓存。"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 赛题说明: 部分图片截断但 PIL 可读

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_train_index(train_dir: str, cache_csv: str) -> pd.DataFrame:
    """扫描 train/类别名/*.jpg，生成 (path, label, class_name) 索引并缓存。"""
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)
    train_dir = Path(train_dir)
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    rows = []
    for label, d in enumerate(class_dirs):
        for p in d.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                rows.append((str(p), label, d.name))
    df = pd.DataFrame(rows, columns=["path", "label", "class_name"])
    Path(cache_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def verify_images(paths, num_workers_note=False):
    """逐一用 PIL 验证可读性，返回坏文件列表。首次运行用，结果另行缓存。"""
    bad = []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im.convert("RGB")
        except Exception:
            bad.append(p)
        if (i + 1) % 20000 == 0:
            print(f"  verified {i + 1}/{len(paths)}, bad={len(bad)}")
    return bad


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def get_eval_transform():
    # CLIP 官方预处理: 短边 bicubic resize 到 224 后中心裁剪
    return transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class TrainDataset(Dataset):
    """返回 (img, label, idx)；idx 用于按样本记录 loss / 干净标记。"""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].to_numpy()
        self.labels = df["label"].to_numpy().astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i]), i


class TestDataset(Dataset):
    """返回 (img, filename)；文件名需与测试集完全一致(含大小写与扩展名)。"""

    def __init__(self, test_dir: str, transform=None):
        test_dir = Path(test_dir)
        self.paths = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in IMG_EXTS],
                            key=lambda p: p.name)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        with Image.open(p) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, p.name


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int):
    """按类别分层划分 train/val 索引。"""
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


# ---------------- memmap 高速数据集（prepare_memmap.py 产物） ----------------

MM_SHAPE = lambda n, size=256: (n, 3, size, size)


class _SafeColorJitter:
    """自实现 brightness/contrast/saturation 抖动（无 hue）。

    torchvision v2.ColorJitter 组合通道时在个别样本上产生全 NaN（实测约 1.5%），
    会污染梯度使训练发散；此处仅用逐像素仿射/lerp 运算，数值上始终有限。
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.b, self.c, self.s = brightness, contrast, saturation

    def __call__(self, x):  # x: float32 CHW in [0,1]
        if self.b:
            x = x * (1.0 + torch.empty(1).uniform_(-self.b, self.b).item())
        if self.c:
            m = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - m) * (1.0 + torch.empty(1).uniform_(-self.c, self.c).item()) + m
        if self.s:
            g = x.mean(dim=0, keepdim=True)
            x = torch.lerp(g.expand_as(x), x,
                           1.0 + torch.empty(1).uniform_(-self.s, self.s).item())
        return x.clamp_(0.0, 1.0)


def _make_train_tf(crop: int, rrc_scale=(0.5, 1.0)):
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(crop, scale=rrc_scale, interpolation=v2.InterpolationMode.BICUBIC),
        v2.RandomHorizontalFlip(),
        _SafeColorJitter(0.2, 0.2, 0.2),
        v2.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def _make_eval_tf(crop: int):
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.CenterCrop(crop),
        v2.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class MemmapTrainDataset(Dataset):
    """读 (N,3,mm_size,mm_size) uint8 memmap；惰性打开以兼容 Windows spawn。返回 (img, label, idx)。"""

    def __init__(self, mm_path: str, n: int, labels: np.ndarray, train: bool,
                 mm_size: int = 256, img_size: int = 224, rrc_scale=(0.5, 1.0)):
        self.mm_path = str(mm_path)
        self.n = n
        self.mm_size = mm_size
        self.labels = labels.astype(np.int64)
        self.tf = (_make_train_tf(img_size, rrc_scale) if train
                   else _make_eval_tf(img_size))
        self._mm = None

    def _get(self):
        if self._mm is None:
            self._mm = np.memmap(self.mm_path, dtype=np.uint8, mode="r",
                                 shape=MM_SHAPE(self.n, self.mm_size))
        return self._mm

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = torch.from_numpy(self._get()[i].copy())  # uint8 CHW
        return self.tf(img), int(self.labels[i]), i


class MemmapTestDataset(Dataset):
    """测试集 memmap；返回 (img, filename)。"""

    def __init__(self, mm_path: str, names: list, mm_size: int = 256, img_size: int = 224):
        self.mm_path = str(mm_path)
        self.names = names
        self.mm_size = mm_size
        self.tf = _make_eval_tf(img_size)
        self._mm = None

    def _get(self):
        if self._mm is None:
            self._mm = np.memmap(self.mm_path, dtype=np.uint8, mode="r",
                                 shape=MM_SHAPE(len(self.names), self.mm_size))
        return self._mm

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = torch.from_numpy(self._get()[i].copy())
        return self.tf(img), self.names[i]
