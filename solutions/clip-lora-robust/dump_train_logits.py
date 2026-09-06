"""用冠军模型对全量训练集做 eval 视图 + 水平翻转 TTA 推理，保存 logits 供标签精化分析。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import MemmapTrainDataset, build_train_index
from src.model import CLIPProtoClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prelim_run2_352real.yaml")
    ap.add_argument("--ckpt", default="outputs/prelim_run2_352real/best.pt")
    ap.add_argument("--out", default="outputs/train_logits_352real.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda"

    df = build_train_index(cfg["data"]["train_dir"], cfg["data"]["index_cache"])
    labels = df["label"].to_numpy()

    mcfg = cfg["model"]
    model = CLIPProtoClassifier(
        num_classes=cfg["data"]["num_classes"],
        clip_name=mcfg["clip_name"],
        lora_rank=mcfg["lora_rank"], lora_alpha=mcfg["lora_alpha"],
        lora_dropout=mcfg["lora_dropout"], lora_targets=tuple(mcfg["lora_targets"]),
        train_ln=mcfg["train_ln"], logit_scale_init=mcfg["logit_scale_init"],
        img_size=mcfg.get("img_size", 224),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    use = ckpt.get("best_from", "student")
    state = ckpt["ema"] if (use == "ema" and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"[dump] ckpt={args.ckpt} weights={use} epoch={ckpt.get('epoch')} "
          f"val_acc={ckpt.get('val_acc'):.4f}", flush=True)

    ds = MemmapTrainDataset(cfg["data"]["train_mm"], len(df), labels, train=False,
                            mm_size=cfg["data"].get("mm_size", 256),
                            img_size=mcfg.get("img_size", 224))
    loader = DataLoader(ds, batch_size=cfg["infer"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)

    all_logits = []
    with torch.no_grad():
        for b, (img, y, idx) in enumerate(loader):
            img = img.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(img)
                logits_f, _ = model(torch.flip(img, dims=[-1]))
                logits = logits + logits_f
            all_logits.append(logits.float().cpu())
            if (b + 1) % 50 == 0:
                print(f"[dump] {b + 1}/{len(loader)}", flush=True)
    all_logits = torch.cat(all_logits)
    assert all_logits.shape == (len(df), cfg["data"]["num_classes"]), \
        f"shape mismatch {tuple(all_logits.shape)} vs ({len(df)}, C)"

    pred = all_logits.argmax(-1)
    y_t = torch.from_numpy(labels)
    agree = (pred == y_t).float().mean().item()
    print(f"[dump] train agreement with given labels: {agree:.4f}")
    torch.save({"logits": all_logits.half(), "labels": torch.from_numpy(labels)}, args.out)
    print(f"[dump] saved -> {args.out}")


if __name__ == "__main__":
    main()
