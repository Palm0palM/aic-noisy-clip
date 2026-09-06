"""val 集角点视图（空间多样性）：中心方图 resize(S) 后取 4 角 352 裁剪，S∈{448,576}。
与既有 5 个尺度视图共用冠军 EMA 权重 + flip TTA。"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
import torch
import yaml
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data import stratified_split
from src.model import CLIPProtoClassifier

ImageFile.LOAD_TRUNCATED_IMAGES = True

CFG = yaml.safe_load(open("configs/prelim_run2_352real.yaml", encoding="utf-8"))
DEVICE = "cuda"
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

df = pd.read_csv("cache/train_index_256.csv")
paths = df["path"].tolist()
labels = df["label"].to_numpy()
_, val_idx = stratified_split(labels, CFG["data"]["val_ratio"], CFG["seed"])
val_labels = torch.from_numpy(labels[val_idx]).long()

meta = torch.load("outputs/val_views/meta.pt", weights_only=False)
clean = meta["zs_val_pred"] == val_labels
print(f"[data] val={len(val_idx)} clean={int(clean.sum())}", flush=True)


class CornerDS(Dataset):
    def __init__(self, idx_list, size, pos):
        self.paths = [paths[i] for i in idx_list]
        self.size, self.pos = size, pos
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            im = im.resize((self.size, self.size), Image.BICUBIC)
            off = self.size - 352
            box = {"tl": (0, 0, 352, 352), "tr": (off, 0, off + 352, 352),
                   "bl": (0, off, 352, off + 352),
                   "br": (off, off, off + 352, off + 352)}[self.pos]
            return self.tf(im.crop(box))


@torch.no_grad()
def forward(model, ds, bs=128):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=6,
                        pin_memory=True)
    model.eval()
    outs = []
    for img in loader:
        img = img.to(DEVICE, non_blocking=True)
        x = torch.cat([img, img.flip(-1)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        logits = logits.float()
        outs.append((logits[:len(img)] + logits[len(img):]) / 2)
    return torch.cat(outs).cpu()


mcfg = CFG["model"]
model = CLIPProtoClassifier(
    num_classes=CFG["data"]["num_classes"], clip_name=mcfg["clip_name"],
    lora_rank=mcfg["lora_rank"], lora_alpha=mcfg["lora_alpha"],
    lora_dropout=mcfg["lora_dropout"], lora_targets=tuple(mcfg["lora_targets"]),
    train_ln=mcfg["train_ln"], logit_scale_init=mcfg["logit_scale_init"],
    img_size=mcfg["img_size"]).to(DEVICE)
ck = torch.load("outputs/prelim_run2_352real/best.pt", map_location=DEVICE,
                weights_only=False)
use = ck.get("best_from", "student")
model.load_state_dict(ck["ema"] if (use == "ema" and "ema" in ck) else ck["model"])
print(f"[model] loaded best.pt weights={use} epoch={ck.get('epoch')}", flush=True)

for size in (448, 576):
    for pos in ("tl", "tr", "bl", "br"):
        logits = forward(model, CornerDS(val_idx.tolist(), size, pos))
        torch.save(logits, f"outputs/val_views/logits_c{size}_{pos}.pt")
        pred = logits.argmax(-1)
        acc = float((pred == val_labels).float().mean())
        cln = float((pred[clean] == val_labels[clean]).float().mean())
        print(f"[corner c{size}_{pos}] acc={acc:.4f} cln={cln:.4f}", flush=True)
print("[done]", flush=True)
