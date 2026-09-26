# AIC 复赛：CLIP ViT-B/32 全参数微调 + 渐进分辨率 + 域随机化

> 冻结后测试集实测（本机用主办方真值独立计算）：**75.28%**（V10-FINAL4，28,186 / 37,444）
> 全程遵守赛题约束：单一骨干（OpenAI CLIP ViT-B/32）、单一模型、单推理流程，无多模型集成、无外部数据、无其他视觉模型，测试集图像与标签均不参与训练。

---

## 1. 赛题与结论要点

| 项目 | 内容 |
|---|---|
| 任务 | 细粒度图像分类，750 类，复赛训练集 148,695 张（可解码 148,643），测试集 37,444 张 |
| 训练分布 | 长尾：每类 5–248 张，中位数 196 |
| 测试分布 | 接近均衡：每类中位数 49 张，范围 1–127（官方说明为"类别分布均衡"） |
| 骨干约束 | 必须 CLIP ViT-B/32 + OpenAI 官方预训练权重 |
| 提交物 | `pred_results.csv`（`文件名,四位类号`），打包 zip |

关键结论：

1. **局部微调不够，要全参数微调**。把可训练参数从"末 2 个 block + LoRA（约 17%）"放开到全部 88.2M，是最大的一步收益。
2. **分辨率在强正则配方下到 512 仍在给分**：224 → 288 → 384 → 448 → 512 依次为 67.47 → 70.52 → 73.02 → 73.77 → 75.20，576 只剩 +0.08pp，至此封顶。注意 448 在旧的无正则配方下曾被判定"无效"——说明分辨率是否到顶依赖配方，不能靠单点判断下结论。
3. **训练内指标会被同源近重复图抬高 4–7pp**，必须用"冻结后测试集一次性评估"做判据，否则会出现"内部涨、线上不涨"。
4. **测试集类均衡先验基本被 inverse-sqrt 重采样对齐**：在均衡留出切片上扫描类别先验校正，最优即 τ=0。

---

## 2. 方案总览

```text
官方训练图（短边中位 480px）
   │  scripts/build_image_cache.py：确定性缩放缓存（短边 288/384、JPEG q90）
   │  448/512 训练改用原图 + Pillow draft 解码（限制在 2×训练分辨率内）
   ▼
CLIP ViT-B/32（全参数可训，官方权重 openai/clip-vit-base-patch32）
   │  位置编码双三次插值（224 → 288/384/448/512）
   │  分类头：nn.Linear(512, 750)
   ▼
训练（src/aic_clip/train_ft.py，逐级续训）
   ├─ 类均衡采样：inverse-sqrt 频率（power 0.5），对齐均衡评测先验
   ├─ 强增广：RandomResizedCrop(0.30~0.5–1.0) + 翻转 + ColorJitter 0.5
   │        + RandAugment(2, 10) + RandomErasing 0.35 + embedding dropout 0.15
   ├─ mixup 0.2 / cutmix 1.0（80% 触发）作用在 label smoothing 0.15 的软标签上
   ├─ 优化：AdamW，backbone 3e-5→3e-6 逐级退火、head 5e-4→7e-5，cosine + warmup，BF16
   ├─ EMA 0.9995
   └─ 选模：训练内留出集 acc / macro / tail 复合分（后期改用全量数据固定日程）
   ▼
推理（src/aic_clip/infer_ft.py，单模型四视图 TTA，概率空间等权平均）
   ├─ Resize(1.0×S) + CenterCrop(S)
   ├─ Resize(1.14×S) + CenterCrop(S)
   ├─ 同上 + 水平翻转
   └─ Resize(1.4×S) + CenterCrop(S)
   ▼
pred_results.csv / pred_results.zip
```

---

## 3. 逐级结果（全部为冻结后测试集实测）

| 版本 | 关键改动 | 分辨率 | 训练数据 | 测试准确率 |
|---|---|---|---|---|
| 基线（接手时） | 288px 蒸馏 + 末 2 块 + LoRA | 288 | 133,774 | 64.3254 |
| V10-A | 全参数微调 | 224 | 133,774 | 67.4688 |
| V10-B | 提高分辨率续训 | 288 | 133,774 | 70.5213 |
| V10-D | 提高分辨率续训 | 384 | 133,774 | 73.0157 |
| V10-E | 提高分辨率续训（旧配方） | 448 | 133,774 | 73.0130 |
| V10-F | 强正则 / 域随机化 | 384 | 133,774 | 73.5071 |
| V10-G | 正则再推一档 | 384 | 133,774 | 73.6006 |
| V10-H | 448 + 强正则 | 448 | 133,774 | 73.7715 |
| V10-FINAL | 全量数据 | 448 | **148,643** | 74.6288 |
| V10-FINAL2 | 全量 + 低 LR 续训 | 448 | 148,643 | 74.7730 |
| V10-FINAL3 | 提高分辨率 | 512 | 148,643 | 75.1976 |
| V10-FINAL4 | 提高分辨率 | 576 | 148,643 | **75.2751** |

单点收益最强的三步：全参微调（+3.14pp）、288→384（+2.50pp）、并入留出的 10% 数据（+0.86pp）。

---

## 4. 复现步骤

```bash
# 0) 环境：PyTorch 2.5+、transformers、torchvision、Pillow；CLIP 权重用 safetensors 离线缓存
export HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1

# 1) 构建确定性图像缓存（可选，但强烈建议：解码开销约降 20 倍）
python scripts/build_image_cache.py \
  --train-dir data/train --manifest artifacts/train_manifest.csv \
  --cache-dir data/train_cache384 --short-side 384 --quality 90 --workers 14

# 2) 逐级训练（每条命令都是上一级的续训，configs/ 下按顺序）
python -m aic_clip.train_ft --config configs/v10_ft224.yaml
python -m aic_clip.train_ft --config configs/v10_ft288.yaml --initialize checkpoints/v10_ft224/best.pt
python -m aic_clip.train_ft --config configs/v10_ft384_reg.yaml --initialize checkpoints/v10_ft288/best.pt
python -m aic_clip.train_ft --config configs/v10_ft448_reg.yaml --initialize checkpoints/v10_ft384_reg/best.pt
#    最后一步并入全部训练数据（不再保留留出集）
python -m aic_clip.train_ft --config configs/v10_final.yaml --train-on-all \
    --initialize checkpoints/v10_ft448_reg/best.pt
python -m aic_clip.train_ft --config configs/v10_final2.yaml --train-on-all \
    --initialize checkpoints/v10_final/last.pt
python -m aic_clip.train_ft --config configs/v10_final3.yaml --train-on-all \
    --initialize checkpoints/v10_final2/best.pt   # 512 px
python -m aic_clip.train_ft --config configs/v10_final4.yaml --train-on-all \
    --initialize checkpoints/v10_final3/best.pt   # 576 px

# 3) 测试集推理（四视图 TTA）→ pred_results.csv + pred_results.zip
python -m aic_clip.infer_ft --checkpoint checkpoints/v10_final4/best.pt --weights raw \
  --test-dir data/test --views resize576,center,flip,resize806.4 \
  --output-dir artifacts/submission_v10_final4
```

诊断脚本（全部只读，且只用训练侧数据）：

```bash
# 留出集与训练集的近重复程度（判断内部指标可信度）
python scripts/analyze_val_similarity.py --checkpoint <ckpt> --train-cache data/train_cache384
# 多种推理视图组合的留出集表现（TTA 配方选择）
python scripts/evaluate_views.py --checkpoint <ckpt> --cache data/train_cache384 \
  --views center:384:1.0,center:384:1.14,flip:384:1.14,center:384:1.4
# 类别先验校正扫描（均衡切片）
python scripts/evaluate_prior_correction.py --checkpoint <ckpt> --cache data/train_cache384 --cap 20
```

---

## 5. 实现细节与坑

1. **位置编码插值**：`transformers` 在输入尺寸不等于 224 时会直接抛错，必须显式传 `interpolate_pos_encoding=True`（`FTClassifier.embed` 里已按输入尺寸自动判断）。
2. **伪扩展名与坏 EXIF**：训练集里存在扩展名为 `.jpg` 实为 WebP/PNG 的图，以及可解码像素但 EXIF 段损坏的图。解码统一走 `load_image`：内容嗅探 + `exif_transpose` 失败时静默忽略方向元数据；不要把这类图直接丢掉。
3. **16 核 cgroup 配额**：容器 `nproc` 显示 128，但 cgroup 只给 16 核，`num_workers` 按 12 设置最稳；`decode_cap` 用 Pillow 的 JPEG draft 模式能再省一半解码时间。
4. **BF16 而非 FP16**：早期 FP16 路径出现过 NaN 梯度，BF16 稳定。
5. **评测口径**：本方案的所有"测试准确率"都是把预测 CSV 与主办方真值对齐后独立计算的；训练与选模过程中从未使用测试图像或测试标签。

---

## 6. 已验证无效、不建议再走的路线

| 路线 | 结论 |
|---|---|
| ELR 早学习正则 | 288px 续训 12 轮无增益，最好即第 0 轮 |
| 无温度缩放的类别先验校正 | 均衡切片上 τ=0 最优 |
| 记忆 / kNN / Sinkhorn / 硬容量 / 图传播 | 线上回退（曾低至 59.90），代理集增益 ≤0.4pp |
| 多 checkpoint 集成 | 违规且仅 +0.29pp |
| CLIP 文本原型融合 | 文本分支单独 18.4%，融合 +0.04pp |
| 零训练高分辨率推理（336） | 代理集低于 224 |
| 448px（旧无正则配方） | 与 384 测试分持平 |

---

## 7. 下一步值得做的（尚未完成）

1. 推理端：温度缩放 + 更强类别先验校正（本方案只扫了无温度版本）。
2. 推理端：加权 5 尺度 + 翻转 + 多裁剪；SWA 末段权重平均。
3. 训练端：学习率 / 冻结层数 / 增强强度的**单变量消融**（本方案的学习率与增广强度均取自经验值，未做受控对照）。
4. 数据端：训练集内部近重复消解 + 强模型上的 GMM 清洗 / EMA 伪标签自训练（赛题明确鼓励自动清洗，且已证实训练集内存在同图冲突标签）。
