"""在 val 集上对 5 个缩放视图做 TTA 推理，保存 logits 供本地权重搜索。
复现训练口径: val 划分 = stratified_split(labels, 0.05, seed)；
cln 子集 = 类均值原型零样本预测与给定标签一致的样本。"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json

import numpy as np
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
SIZES = [352, 384, 448, 512, 576]
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

df = pd.read_csv("cache/train_index_256.csv")
paths = df["path"].tolist()
labels = df["label"].to_numpy()
train_idx, val_idx = stratified_split(labels, CFG["data"]["val_ratio"], CFG["seed"])
val_labels = torch.from_numpy(labels[val_idx]).long()
print(f"[data] train={len(train_idx)} val={len(val_idx)}", flush=True)


class ViewDS(Dataset):
    """原图 -> 中心方裁剪 -> resize(SIZE) -> CenterCrop(352) -> CLIP 归一化。"""

    def __init__(self, idx_list, size):
        self.paths = [paths[i] for i in idx_list]
        self.size = size
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(352),
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
            if s != self.size:
                im = im.resize((self.size, self.size), Image.BICUBIC)
            return self.tf(im)


@torch.no_grad()
def forward(model, ds, tta=True, bs=128, want_feats=False):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=6,
                        pin_memory=True)
    model.eval()
    outs, feats = [], []
    for img in loader:
        img = img.to(DEVICE, non_blocking=True)
        x = torch.cat([img, img.flip(-1)]) if tta else img
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, f = model(x)
        logits = logits.float()
        f = f.float()
        if tta:
            logits = (logits[:len(img)] + logits[len(img):]) / 2
            f = (f[:len(img)] + f[len(img):]) / 2
        outs.append(logits.cpu())
        if want_feats:
            feats.append(f.cpu())
    return (torch.cat(outs), torch.cat(feats)) if want_feats else torch.cat(outs)


mcfg = CFG["model"]

# ---------- 1) 零样本: 类均值原型, 视图 352, 无 TTA（复现训练 init 口径） ----------
zs_model = CLIPProtoClassifier(
    num_classes=CFG["data"]["num_classes"], clip_name=mcfg["clip_name"],
    lora_rank=mcfg["lora_rank"], lora_alpha=mcfg["lora_alpha"],
    lora_dropout=mcfg["lora_dropout"], lora_targets=tuple(mcfg["lora_targets"]),
    train_ln=mcfg["train_ln"], logit_scale_init=mcfg["logit_scale_init"],
    img_size=mcfg["img_size"]).to(DEVICE)
with torch.no_grad():
    _, tr_feats = forward(zs_model, ViewDS(train_idx.tolist(), 352),
                          tta=False, bs=192, want_feats=True)
    tr_y = torch.from_numpy(labels[train_idx]).long()
    proto = torch.zeros(500, tr_feats.shape[1])
    for c in range(500):
        m = tr_y == c
        if m.any():
            proto[c] = tr_feats[m].mean(0)
    zs_model.prototypes.data = proto.to(DEVICE)
    zs_val_logits = forward(zs_model, ViewDS(val_idx.tolist(), 352), tta=False, bs=192)
zs_val_pred = zs_val_logits.argmax(-1)
clean_mask = zs_val_pred == val_labels
print(f"[zs] val acc={float((zs_val_pred == val_labels).float().mean()):.4f} "
      f"clean_subset={int(clean_mask.sum())}/{len(val_labels)}", flush=True)
del zs_model
torch.cuda.empty_cache()

# ---------- 2) 冠军模型 5 视图 TTA ----------
model = CLIPProtoClassifier(
    num_classes=CFG["data"]["num_classes"], clip_name=mcfg["clip_name"],
    lora_rank=mcfg["lora_rank"], lora_alpha=mcfg["lora_alpha"],
    lora_dropout=mcfg["lora_dropout"], lora_targets=tuple(mcfg["lora_targets"]),
    train_ln=mcfg["train_ln"], logit_scale_init=mcfg["logit_scale_init"],
    img_size=mcfg["img_size"]).to(DEVICE)
ck = torch.load("outputs/prelim_run2_352real/best.pt", map_location=DEVICE,
                weights_only=False)
use = ck.get("best_from", "student")
state = ck["ema"] if (use == "ema" and "ema" in ck) else ck["model"]
model.load_state_dict(state)
print(f"[model] loaded best.pt weights={use} epoch={ck.get('epoch')}", flush=True)

os.makedirs("outputs/val_views", exist_ok=True)
torch.save({"val_idx": val_idx, "val_labels": val_labels,
            "zs_val_pred": zs_val_pred}, "outputs/val_views/meta.pt")
for size in SIZES:
    logits = forward(model, ViewDS(val_idx.tolist(), size), tta=True, bs=128)
    torch.save(logits, f"outputs/val_views/logits_{size}.pt")
    pred = logits.argmax(-1)
    acc = float((pred == val_labels).float().mean())
    acc_cln = float((pred[clean_mask] == val_labels[clean_mask]).float().mean())
    print(f"[view {size}] val acc={acc:.4f} cln={acc_cln:.4f}", flush=True)
print("[done]", flush=True)
