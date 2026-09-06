"""推理：加载 checkpoint，对测试集预测并生成 pred_results.csv 及提交 zip。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import json
import zipfile
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import MemmapTestDataset
from src.model import CLIPProtoClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prelim.yaml")
    parser.add_argument("--ckpt", default="outputs/prelim/best.pt")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--weights", default=None, choices=["student", "ema"],
                        help="默认按 checkpoint 记录的 best_from 选择")
    parser.add_argument("--tta", action="store_true", help="水平翻转 TTA（单一模型双视图平均）")
    parser.add_argument("--balance_tau", type=float, default=0.0,
                        help="类别均衡校正强度(0=关闭)：测试集已知类别均衡，对预测类分布做先验校正")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--save_logits", type=str, default="")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    use = args.weights or ckpt.get("best_from", "student")
    state = ckpt["ema"] if (use == "ema" and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"[infer] loaded {args.ckpt} (weights={use} epoch={ckpt.get('epoch')}, "
          f"val_acc={ckpt.get('val_acc'):.4f}, val_acc_cln={ckpt.get('val_acc_cln')}) "
          f"tta={args.tta} balance_tau={args.balance_tau}")

    names = json.load(open(cfg["data"]["test_names"], encoding="utf-8"))
    ds = MemmapTestDataset(cfg["data"]["test_mm"], names,
                           mm_size=cfg["data"].get("mm_size", 256),
                           img_size=cfg["model"].get("img_size", 224))
    print(f"[infer] test images: {len(ds)}")
    loader = DataLoader(ds, batch_size=cfg["infer"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)

    out_csv = Path(args.out_csv or cfg["infer"]["out_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_logits, all_names = [], []
    with torch.no_grad():
        for img, names in loader:
            img = img.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, _ = model(img)
                if args.tta:
                    logits_f, _ = model(torch.flip(img, dims=[-1]))
                    logits = logits + logits_f
            all_logits.append(logits.float().cpu())
            all_names.extend(names)
    all_logits = torch.cat(all_logits)

    if args.save_logits:
        torch.save(all_logits, args.save_logits)
        print(f"[infer] saved logits -> {args.save_logits} shape={tuple(all_logits.shape)}")
    if args.temp != 1.0:
        all_logits = all_logits / args.temp
    if args.cap > 0:
        N, C = all_logits.shape
        k = 64
        probs = all_logits.softmax(-1)
        topv, topi = probs.topk(k, dim=1)
        order = torch.argsort(topv.flatten(), descending=True)
        pred = torch.full((N,), -1, dtype=torch.long)
        counts = torch.zeros(C, dtype=torch.long)
        flat_c = topi.flatten()
        for idx in order.tolist():
            i = idx // k
            c = int(flat_c[idx])
            if pred[i] == -1 and counts[c] < args.cap:
                pred[i] = c
                counts[c] += 1
        left = (pred == -1).nonzero(as_tuple=True)[0]
        for i in left.tolist():
            avail = (counts < args.cap).nonzero(as_tuple=True)[0]
            c = int(avail[all_logits[i, avail].argmax()])
            pred[i] = c
            counts[c] += 1
        print(f"[infer] cap={args.cap}: dist max={int(counts.max())} min={int(counts.min())} classes={int((counts > 0).sum())}")
        all_logits = torch.zeros_like(all_logits)
        all_logits[torch.arange(N), pred] = 10.0

    if args.balance_tau > 0:
        # 测试集类别均衡：用预测平均概率估计类先验偏移，向均匀分布校正
        prior = all_logits.softmax(-1).mean(0).clamp_min(1e-8)
        adj = -torch.log(prior * all_logits.shape[1])  # log(uniform / prior)
        all_logits = all_logits + args.balance_tau * adj
        print(f"[infer] balance: pred prior max/min={prior.max().item() / prior.min().item():.1f}x")

    n_written = 0
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        for name, p in zip(all_names, all_logits.argmax(-1)):
            f.write(f"{name},{int(p):04d}\n")
            n_written += 1

    print(f"[infer] wrote {n_written} lines -> {out_csv}")

    zip_path = out_csv.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname="pred_results.csv")
    print(f"[infer] zipped -> {zip_path}")


if __name__ == "__main__":
    main()
