"""初赛训练入口：CLIP ViT-B/32 + LoRA + 原型头，GMM 噪声划分 + EMA 伪标签自训练。

性能设计：参考特征与教师伪标签均按 epoch 预计算（eval 视图），
训练主循环只做单次学生前向 + 反向，避免每 batch 三次前向。
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import (MemmapTrainDataset, build_train_index, stratified_split)
from src.model import CLIPProtoClassifier
from src.robust import (EMA, build_teacher, generalized_cross_entropy,
                        gmm_clean_prob, soft_cross_entropy)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def log(msg, log_file):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@torch.no_grad()
def forward_eval(model, dataset, indices, batch_size, device, num_workers):
    """eval transform 下对指定索引前向，返回 (logits, features, labels)（CPU 张量）。"""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    all_logits, all_feats, all_labels = [], [], []
    for img, y, _ in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, f = model(img)
        all_logits.append(logits.float().cpu())
        all_feats.append(f.half().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_feats), torch.cat(all_labels)


@torch.no_grad()
def evaluate(model, dataset, val_idx, cfg, device):
    logits, _, labels = forward_eval(model, dataset, val_idx, cfg["train"]["eval_batch_size"],
                                     device, cfg["data"]["num_workers"])
    return (logits.argmax(-1) == labels).float().mean().item()


@torch.no_grad()
def val_metrics(model, dataset, val_idx, cfg, device, zs_pred, val_labels,
                orig_val_labels=None):
    """val 三重指标: acc(含噪标签) / acc_cln(零样本判定为干净的子集) / acc_arb(零样本仲裁)。

    val 标签含噪且噪声可被模型习得 -> 原始 acc 会高估干净测试精度并奖励噪声模仿。
    acc_cln 只在零样本与给定标签一致的(大概率为干净)样本上计分，是测试精度的更好代理。
    orig_val_labels 非空时(标签精化模式)额外返回对原始给定标签的精度用于纵向对比。
    """
    logits, _, _ = forward_eval(model, dataset, val_idx, cfg["train"]["eval_batch_size"],
                                device, cfg["data"]["num_workers"])
    pred = logits.argmax(-1)
    acc = (pred == val_labels).float().mean().item()
    clean = zs_pred == val_labels
    acc_cln = (pred[clean] == val_labels[clean]).float().mean().item() if clean.any() else 0.0
    arb = torch.where(clean, val_labels, zs_pred)
    acc_arb = (pred == arb).float().mean().item()
    if orig_val_labels is not None:
        acc_orig = (pred == orig_val_labels).float().mean().item()
        return acc, acc_cln, acc_arb, acc_orig
    return acc, acc_cln, acc_arb


def cosine_lr(step, total_steps, base_lr, warmup_steps, min_ratio):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * p)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prelim.yaml")
    parser.add_argument("--init_from", default=None,
                        help="从已有 checkpoint 接力初始化（剔除分辨率相关的 pos-embed 键）")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "run_config.yaml")
    log_file = out_dir / "log.txt"

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[env] device={device} torch={torch.__version__} cuda={torch.version.cuda}", log_file)

    # ---------- 数据 ----------
    df = build_train_index(cfg["data"]["train_dir"], cfg["data"]["index_cache"])
    bad_path = Path(cfg["data"]["bad_list"])
    if bad_path.exists():
        bad = set(json.load(open(bad_path, encoding="utf-8")))
        df = df[~df["path"].isin(bad)].reset_index(drop=True)
        log(f"[data] excluded {len(bad)} unreadable images", log_file)
    labels_orig = df["label"].to_numpy()
    labels = labels_orig.copy()
    relabel_file = cfg["data"].get("relabel_file")
    if relabel_file:
        rdf = pd.read_csv(relabel_file)
        labels[rdf["idx"].to_numpy()] = rdf["new_label"].to_numpy()
        log(f"[data] relabeled {len(rdf)} samples from {relabel_file}", log_file)
    # 按原始标签分层划分：保证与未精化运行的 train/val 成员完全一致、指标可比
    train_idx, val_idx = stratified_split(labels_orig, cfg["data"]["val_ratio"], cfg["seed"])
    max_per_class = cfg["data"].get("max_per_class")
    if max_per_class:  # 冒烟测试: 每类限采样（保持 df 与 memmap 行对齐，仅裁剪索引）
        def _cap(indices, k):
            cnt, out = {}, []
            for i in indices:
                c = labels[i]
                if cnt.get(c, 0) < k:
                    cnt[c] = cnt.get(c, 0) + 1
                    out.append(i)
            return np.array(out, dtype=np.int64)
        train_idx = _cap(train_idx, max_per_class)
        val_idx = _cap(val_idx, max(2, max_per_class // 5))
        log(f"[data] smoke mode: max {max_per_class} per class", log_file)
    n_train = len(train_idx)
    log(f"[data] train={n_train} val={len(val_idx)} classes={cfg['data']['num_classes']}", log_file)

    mm_path = cfg["data"]["train_mm"]
    mm_size = cfg["data"].get("mm_size", 256)
    img_size = cfg["model"].get("img_size", 224)
    expect = len(df) * 3 * mm_size * mm_size
    assert os.path.getsize(mm_path) == expect, \
        f"memmap 大小不匹配: {mm_path}（memmap 行序须与 {cfg['data']['index_cache']} 一致）"
    rrc_scale = tuple(cfg["train"].get("rrc_scale", [0.5, 1.0]))
    train_ds = MemmapTrainDataset(mm_path, len(df), labels, train=True,
                                  mm_size=mm_size, img_size=img_size,
                                  rrc_scale=rrc_scale)
    eval_ds = MemmapTrainDataset(mm_path, len(df), labels, train=False,
                                 mm_size=mm_size, img_size=img_size)

    # ---------- 模型 ----------
    mcfg = cfg["model"]
    model = CLIPProtoClassifier(
        num_classes=cfg["data"]["num_classes"],
        clip_name=mcfg["clip_name"],
        lora_rank=mcfg["lora_rank"], lora_alpha=mcfg["lora_alpha"],
        lora_dropout=mcfg["lora_dropout"], lora_targets=tuple(mcfg["lora_targets"]),
        train_ln=mcfg["train_ln"], logit_scale_init=mcfg["logit_scale_init"],
        img_size=img_size,
    ).to(device)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        state = {k: v for k, v in ck["model"].items()
                 if "position_embedding" not in k and "position_ids" not in k}
        missing, unexpected = model.load_state_dict(state, strict=False)
        log(f"[init_from] {args.init_from} (epoch={ck.get('epoch')}, val_acc="
            f"{ck.get('val_acc'):.4f}) skipped pos-embed, missing={len(missing)}", log_file)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] trainable params={n_params / 1e6:.2f}M", log_file)

    tcfg = cfg["train"]
    num_classes = cfg["data"]["num_classes"]
    N = len(df)

    # ---------- 零样本前向：原型初始化 + 参考特征缓存（等价零样本，因 LoRA 初始为 0） ----------
    t0 = time.time()
    zs_logits, zs_feats, zs_labels = forward_eval(
        model, eval_ds, train_idx, tcfg["eval_batch_size"], device, cfg["data"]["num_workers"])
    proto = torch.zeros(num_classes, zs_feats.shape[1])
    for c in range(num_classes):
        mask = zs_labels == c
        if mask.any():
            proto[c] = zs_feats[mask].float().mean(0)
    model.prototypes.data = proto.to(device)

    ref_feats = torch.zeros(N, zs_feats.shape[1], dtype=torch.float16)
    ref_feats[train_idx] = zs_feats  # 冻结参考特征（漂移约束目标）
    # 初始模型(接力时为上游 checkpoint，否则为零样本)预测：标签精化的抗噪先验
    zs_alpha = tcfg.get("zs_alpha", 0.0)
    if zs_alpha > 0:
        p = F.softmax(zs_logits.float(), dim=-1).pow(2)
        zs_sharp = torch.zeros(N, num_classes, dtype=torch.float16)
        zs_sharp[train_idx] = (p / p.sum(dim=-1, keepdim=True)).half()
        del p
    del zs_logits
    log(f"[init] prototypes + ref features ({time.time() - t0:.1f}s)", log_file)

    # val 的初始模型预测：干净子集判定与仲裁基准（训练全程不变，作为无噪参照）
    zs_val_logits, _, _ = forward_eval(model, eval_ds, val_idx, tcfg["eval_batch_size"],
                                       device, cfg["data"]["num_workers"])
    zs_val_pred = zs_val_logits.argmax(-1)
    val_labels_t = torch.from_numpy(labels[val_idx])
    val_labels_orig_t = torch.from_numpy(labels_orig[val_idx])
    del zs_val_logits
    acc0, acc0_cln, acc0_arb, acc0_orig = val_metrics(
        model, eval_ds, val_idx, cfg, device, zs_val_pred, val_labels_t, val_labels_orig_t)
    log(f"[init] class-mean prototype val: acc={acc0:.4f} acc_cln={acc0_cln:.4f} "
        f"acc_arb={acc0_arb:.4f} acc_orig={acc0_orig:.4f}", log_file)
    torch.cuda.empty_cache()  # 释放 eval 阶段的缓存碎片，为训练激活腾出连续显存

    # ---------- 优化 ----------
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    steps_per_epoch = (n_train + tcfg["batch_size"] - 1) // tcfg["batch_size"]
    total_steps = steps_per_epoch * tcfg["epochs"]
    ema = EMA(model, decay=tcfg["ema_decay"])

    soft_targets = torch.zeros(N, num_classes, dtype=torch.float16)
    soft_conf = torch.zeros(N)
    # init 时模型即仲裁基准自身，acc0_cln 恒为 1，不能作为 best 起点
    best_sel, global_step = -1.0, 0
    select_metric = tcfg.get("select_metric", "cln")  # best.pt 选模指标: "acc"=原始精度, "cln"=干净子集精度
    loss_mode = tcfg.get("loss_mode", "dividemix")  # "run2": 8/30 原配方（硬划分+置信门控+0.5权重GCE，无mixup）
    clean_thr = tcfg.get("clean_prob_thr", 0.6)
    conf_thr = tcfg.get("pseudo_conf_thr", 0.75)
    sample_mode = torch.zeros(N, dtype=torch.int64)  # run2 模式: 0=干净CE, 1=伪标签, 2=弱GCE

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=tcfg["batch_size"],
                              shuffle=True, num_workers=cfg["data"]["num_workers"],
                              pin_memory=True, drop_last=True, persistent_workers=True)

    for epoch in range(tcfg["epochs"]):
        # ---- warm-up 后定期：GMM 干净概率 + EMA 教师标签精化（DivideMix 风格） ----
        if epoch >= tcfg["warmup_epochs"] and (epoch - tcfg["warmup_epochs"]) % tcfg["gmm_every"] == 0:
            t0 = time.time()
            torch.cuda.empty_cache()  # 释放训练缓存，为教师深拷贝腾出连续显存
            logits, _, _ = forward_eval(model, eval_ds, train_idx, tcfg["eval_batch_size"],
                                        device, cfg["data"]["num_workers"])
            y_tr = torch.from_numpy(labels[train_idx])
            losses = F.cross_entropy(logits, y_tr, reduction="none").numpy()
            prob = gmm_clean_prob(losses)
            est_noise = 1 - prob.mean()

            teacher = build_teacher(model, ema).to(device)
            t_logits, _, _ = forward_eval(teacher, eval_ds, train_idx, tcfg["eval_batch_size"],
                                          device, cfg["data"]["num_workers"])
            t_prob = F.softmax(t_logits, dim=-1)
            # sharpen(T=0.5): 压低教师预测熵，增强伪标签指导性
            t_sharp = t_prob.pow(2)
            t_sharp = t_sharp / t_sharp.sum(dim=-1, keepdim=True)
            soft_conf[train_idx] = t_prob.max(dim=-1).values
            if loss_mode == "run2":
                # 8/30 原配方：干净样本 CE / 噪声样本高置信伪标签 / 其余 0.5 权重 GCE
                clean = prob > clean_thr
                pseudo = (~clean) & (soft_conf[train_idx].numpy() > conf_thr)
                mode_np = np.full(n_train, 2, dtype=np.int64)
                mode_np[clean] = 0
                mode_np[pseudo] = 1
                sample_mode[train_idx] = torch.from_numpy(mode_np)
                soft_targets[train_idx] = t_sharp.half()
            else:
                # 标签精化: y* = w·onehot(y) + (1-w)·[(1-a)·sharpen(teacher) + a·先验]
                # 先验(零样本/上游模型)不受训练集噪声影响，抑制教师自确认导致的系统性噪声漂移
                w = torch.from_numpy(prob).float().unsqueeze(1)
                onehot = F.one_hot(y_tr, num_classes).float()
                if zs_alpha > 0:
                    prior = (1.0 - zs_alpha) * t_sharp + zs_alpha * zs_sharp[train_idx].float()
                else:
                    prior = t_sharp
                refined = w * onehot + (1.0 - w) * prior
                soft_targets[train_idx] = refined.half()
            del teacher
            torch.cuda.empty_cache()
            if loss_mode == "run2":
                log(f"[gmm] epoch {epoch} run2: clean={int(clean.sum())} pseudo={int(pseudo.sum())} "
                    f"weak={n_train - int(clean.sum()) - int(pseudo.sum())} "
                    f"est_noise_rate={est_noise:.3f} pseudo_conf_mean={soft_conf[train_idx].mean():.3f} "
                    f"({time.time() - t0:.1f}s)", log_file)
            else:
                log(f"[gmm] epoch {epoch}: clean(w>0.5)={int((prob > 0.5).sum())}/{n_train} "
                    f"est_noise_rate={est_noise:.3f} pseudo_conf_mean={soft_conf[train_idx].mean():.3f} "
                    f"({time.time() - t0:.1f}s)", log_file)

        # ---- 训练一个 epoch（仅学生前向 + 反向） ----
        model.train()
        lam = tcfg["drift_lambda"] * max(0.1, 1.0 - global_step / max(1, total_steps))
        ep_loss, ep_n = 0.0, 0
        t0 = time.time()

        for img, y, idx in train_loader:
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx_np = idx.numpy()

            lr = cosine_lr(global_step, total_steps, tcfg["lr"],
                           tcfg["warmup_steps"], tcfg["min_lr_ratio"])
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(tcfg["amp"] and device == "cuda")):
                if epoch < tcfg["warmup_epochs"]:
                    logits, f = model(img)
                    logits = logits.float()
                    loss_sup = generalized_cross_entropy(logits, y, q=tcfg["gce_q"])
                    f_ref = ref_feats[idx_np].float().to(device)
                else:
                    if loss_mode == "run2":
                        # 8/30 原配方：无 mixup，按硬划分三路损失
                        logits, f = model(img)
                        logits = logits.float()
                        m_t = sample_mode[idx_np].to(device)
                        ce_i = F.cross_entropy(logits, y, reduction="none")
                        p_y = F.softmax(logits, dim=-1).gather(1, y.unsqueeze(1)).squeeze(1)
                        gce_i = (1.0 - p_y.pow(tcfg["gce_q"])) / tcfg["gce_q"]
                        soft_i = -(soft_targets[idx_np].float().to(device)
                                   * F.log_softmax(logits, dim=-1)).sum(-1)
                        loss_i = torch.where(m_t == 0, ce_i,
                                             torch.where(m_t == 1, soft_i, 0.5 * gce_i))
                        loss_sup = loss_i.mean()
                        f_ref = ref_feats[idx_np].float().to(device)
                    else:
                        # mixup（DivideMix 风格, alpha=0.2）+ 精化软标签统一监督
                        lam_m = float(np.random.beta(0.2, 0.2))
                        perm = torch.randperm(img.size(0), device=device)
                        img_m = lam_m * img + (1.0 - lam_m) * img[perm]
                        logits, f = model(img_m)
                        logits = logits.float()
                        st = soft_targets[idx_np].float().to(device)
                        st_m = lam_m * st + (1.0 - lam_m) * st[perm]
                        loss_sup = soft_cross_entropy(logits, st_m)
                        f_ref = (lam_m * ref_feats[idx_np].float()
                                 + (1.0 - lam_m) * ref_feats[idx_np[perm.cpu().numpy()]].float()).to(device)

                loss_drift = F.mse_loss(f.float(), f_ref)
                loss = loss_sup + lam * loss_drift

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
            optimizer.step()
            ema.update(model)
            global_step += 1
            ep_loss += loss.item() * len(y)
            ep_n += len(y)

        # ---- 验证与保存：按 select_metric 在学生/EMA 中选优 ----
        acc, acc_cln, acc_arb, acc_orig = val_metrics(
            model, eval_ds, val_idx, cfg, device, zs_val_pred, val_labels_t, val_labels_orig_t)
        teacher = build_teacher(model, ema).to(device)
        acc_e, acc_cln_e, acc_arb_e, acc_orig_e = val_metrics(
            teacher, eval_ds, val_idx, cfg, device, zs_val_pred, val_labels_t, val_labels_orig_t)
        del teacher
        torch.cuda.empty_cache()
        log(f"[epoch {epoch:02d}] loss={ep_loss / ep_n:.4f} lr={lr:.2e} lam={lam:.3f} "
            f"val: acc={acc:.4f} cln={acc_cln:.4f} arb={acc_arb:.4f} orig={acc_orig:.4f} | "
            f"ema: acc={acc_e:.4f} cln={acc_cln_e:.4f} arb={acc_arb_e:.4f} "
            f"orig={acc_orig_e:.4f} time={time.time() - t0:.1f}s", log_file)

        ckpt = {"model": model.state_dict(), "ema": ema.shadow,
                "epoch": epoch, "val_acc": acc, "val_acc_cln": acc_cln, "config": cfg}
        torch.save(ckpt, out_dir / "last.pt")
        if select_metric == "acc":
            s_stu, s_ema = acc, acc_e
        else:
            s_stu, s_ema = acc_cln, acc_cln_e
        cand_acc, cand_from = (s_stu, "student") if s_stu >= s_ema else (s_ema, "ema")
        if cand_acc > best_sel:
            best_sel = cand_acc
            ckpt["best_from"] = cand_from
            torch.save(ckpt, out_dir / "best.pt")
            log(f"  -> new best ({cand_from} {select_metric}={best_sel:.4f}, "
                f"raw acc={acc if cand_from == 'student' else acc_e:.4f})", log_file)

    log(f"[done] best val {select_metric} = {best_sel:.4f}", log_file)


if __name__ == "__main__":
    main()
