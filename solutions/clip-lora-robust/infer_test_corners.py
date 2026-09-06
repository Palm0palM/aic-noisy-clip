"""test 集角点视图推理：从 448/576 测试 memmap 取 4 角 352 裁剪，冠军 EMA，flip TTA。"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.model import CLIPProtoClassifier

CFG = yaml.safe_load(open("configs/prelim_run2_352real.yaml", encoding="utf-8"))
DEVICE = "cuda"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

names = json.load(open("cache/test_names.json", encoding="utf-8"))
N = len(names)
print(f"[data] test N={N}", flush=True)


class MMCorner(Dataset):
    def __init__(self, mm_path, size, pos):
        self.mm = np.memmap(mm_path, dtype=np.uint8, mode="r",
                            shape=(N, 3, size, size))
        self.size, self.pos = size, pos

    def __len__(self):
        return N

    def __getitem__(self, i):
        img = torch.from_numpy(self.mm[i].copy()).float() / 255.0
        off = self.size - 352
        r0 = 0 if self.pos in ("tl", "tr") else off
        c0 = 0 if self.pos in ("tl", "bl") else off
        img = img[:, r0:r0 + 352, c0:c0 + 352]
        return (img - MEAN) / STD


@torch.no_grad()
def forward(model, ds, bs=288):
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
        logits = forward(model, MMCorner(f"cache/test_u8_{size}.npy", size, pos))
        torch.save(logits, f"outputs/prelim_run2_352real/logits_tc{size}_{pos}.pt")
        print(f"[test corner c{size}_{pos}] saved "
              f"outputs/prelim_run2_352real/logits_tc{size}_{pos}.pt", flush=True)
print("[done]", flush=True)
