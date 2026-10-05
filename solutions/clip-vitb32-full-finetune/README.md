# AIC 复赛：CLIP ViT-B/32 全参数微调 + 阶梯分辨率 + 折外可靠性降权

> 冻结后测试集实测（本机用主办方真值独立计算）：**77.4810%**（V31，29,012 / 37,444）
> 全程遵守赛题约束：单一骨干（OpenAI CLIP ViT-B/32，revision `c237dc49a33fc61debc9276459120b7eac67e7ef`）、单一模型、单推理流程，无多模型集成、无外部数据、无其他视觉模型，测试集图像与标签均不参与训练。

---

## 1. 赛题与结论要点

| 项目 | 内容 |
|---|---|
| 任务 | 细粒度图像分类，750 类，复赛训练集 148,695 张（可解码 148,643），测试集 37,444 张 |
| 训练分布 | 长尾：每类 5–248 张，中位数 196；含大量近重复图，近重复对中 77% 标签互相矛盾 |
| 测试分布 | 接近均衡：每类中位数 49 张；全部测试图长边 ≤ 500px |
| 骨干约束 | 必须 CLIP ViT-B/32 + OpenAI 官方预训练权重 |
| 提交物 | `pred_results.csv`（`文件名,四位类号`），打包 zip |

关键结论：

1. **局部微调不够，要全参数微调**。把可训练参数从"末 2 个 block + LoRA（约 17%）"放开到全部 88.2M，是最大的一步收益。
2. **分辨率阶梯到 576 封顶**：384 → 448 → 576 每级都在给分，推理端再多给尺寸只余噪声。
3. **推理端是唯一稳定出收益的一侧**：单模型十视图 TTA（六中心 + 四角裁）在最好权重之上再给 +0.41pp；视图家族已经饱和（翻转角、1.28 角、八角锚点、letterbox 都不再贡献）。
4. **训练内验证集不预测测试分，甚至反相关**（六次复现）。所有结论必须用"冻结后测试集一次性评估"做判据。
5. **训练集近重复消解 + 折外可靠性降权**是训练侧唯一复现出正向信号的方法（+0.11pp），直接降权的收益随覆盖扩大不一定增长。

---

## 2. 方案总览

```text
官方训练图（短边中位 480px）
   │
   ▼
CLIP ViT-B/32（全参数可训，官方权重，位置编码双三次插值）
   │  分类头：nn.Linear(512, 750)（对比过 cosine 头，−0.64pp）
   ▼
三段阶梯训练（src/aic_clip/train_ft.py，逐级续训，14/8/6 轮 384/448/576）
   ├─ 类均衡采样：inverse-sqrt 频率（power 0.5），对齐均衡评测先验
   ├─ 强增广：RRC 0.35–1.0 + 翻转 + ColorJitter 0.5 + RandAugment(2,7)
   │        + RandomErasing 0.3；mixup 0.2 / cutmix 1.0，mix_prob 0.8
   ├─ 标签平滑 0.15
   ├─ 优化：AdamW，LLRD γ=0.8（层间学习率衰减），cosine + warmup，BF16
   ├─ EMA 0.9995
   └─ 折外可靠性降权：2,481 张可疑样本权重 ×0.5（见 §4）
   ▼
推理（src/aic_clip/infer_ft.py，单模型十视图 TTA，概率空间等权平均）
   ├─ center:512:1.0  flip:512:1.0
   ├─ center:512:1.14 flip:512:1.14
   ├─ center:512:1.28 center:512:1.4
   └─ tl/tr/bl/br:512:1.14（四个五裁剪角视图，1.14 倍短边缩放后取角）
   ▼
pred_results.csv / pred_results.zip
```

视图语法为 `kind:size:ratio`；`ratio` 表示短边缩放倍数，锚点族（tl/tc/.../br）在缩放后按 1/7 网格取裁，角视图补上中心裁剪看不到的边角区域。

---

## 3. 逐级结果（全部为冻结后测试集实测）

| 版本 | 关键改动 | 分辨率 | 测试准确率 |
|---|---|---|---|
| 基线（接手时） | 288px 蒸馏 + 末 2 块 + LoRA | 288 | 53.28 → 64.3254 |
| V10-A | 全参数微调 | 224 | 67.4688 |
| V10-B | 提高分辨率续训 | 288 | 70.5213 |
| V10-D | 提高分辨率续训 | 384 | 73.0157 |
| V10-FINAL | 全量数据 | 448 | 74.6288 |
| V10-FINAL5 | 剔除近重复矛盾标签 + 末 3 轮 SWA | 576 | 75.5128 |
| V13 | 受控消融后修正学习率，重做阶梯 | 576 | 76.5890 |
| V16 | 阶梯 14/8/6 + LLRD γ=0.8 | 576 | 76.91 |
| V16 + EMA | EMA 权重代替 raw | 576 | 77.0163 |
| + 六视图挪到 512 | 推理尺寸 | 512 | 77.12 |
| + 四个 1.14 角视图 | 十视图 TTA | 512 | 77.40 |
| **V31** | **+ 折外可靠性降权（2,481 张 ×0.5）** | 512 | **77.4810** |

轨迹：53.28 → 67.47 → 73.02 → 74.63 → 75.51 → 76.59 → 76.74 → 76.91 → 77.12 → 77.40 → 77.48。

### 受控消融（`configs/v11/`，384px、6 轮、单变量）

| 配置 | 留出集 acc | macro | tail |
|---|---|---|---|
| 全参 lr 3e-5 + RA7 | **72.11** | 71.24 | 67.61 |
| 冻结前 6 层 lr 3e-5 | 72.02 | 71.16 | 67.85 |
| 全参 lr 3e-5 | 71.92 | 71.06 | 67.40 |
| 只训后 6 层 lr 3e-5 | 71.66 | 70.81 | 67.35 |
| 全参 lr 1e-5 | 70.39 | 69.52 | 65.91 |
| 全参 lr 1e-4 | 68.35 | 67.48 | 63.37 |

学习率是唯一大幅拉开差距的维度；冻结策略与增强强度都在 0.4pp 以内。但注意此消融在 384px 规模上做，其结论不能外推到阶梯规模（LLRD 是唯一例外：作为"尖峰附近的超参微调"它复现了，见 §5）。

---

## 4. 训练集近重复消解与折外可靠性降权（V31）

### 4.1 近重复结构

用 dHash（9×8 灰度差分哈希，注意不是 pHash）扫描全训练集：

| 指标 | 数值 |
|---|---|
| 汉明距离 ≤ 6 的带内近重复对 | 298,575 |
| 去重后唯一对 | 64,400（其中 77% 标签互相矛盾） |
| 删除名单 | 4,071 张（dHash 核验 98.7% 有近同伙伴） |

### 4.2 分折

`scripts/build_folds.py`：dHash ≤6 建边 → Bellman-Ford 式标签传播合并连通分量（8,530 簇 / 24,723 张）→ 按类贪心分折（每类差 ≤1）。两折 74,508 / 74,135，独立核验跨折近重复对 = 0。

### 4.3 折外教师与降权名单

1. 两个 384px × 14 轮模型各训一折（`configs/v30/oof_{a,b}.yaml`）。
2. 固定 `last.pt + ema`，弱视图 center/flip@448 导出折外概率（`scripts/dump_teacher_probs.py`）。
3. 筛选（`scripts/audit_oof_weights.py`）：两视图都反对原标签 + 指向同一替代类 + 较小 margin ≥0.5 + 不在删除名单 → 2,481 张，权重 0.5，写 `oof_weights_v2.npy`。

### 4.4 训练端接线

`train_ft.py --sample-weight-file` + `sample_weight_mode: "target"`：先按可靠性缩放软标签再 mixup，按目标总质量归一化。

注意一个已经踩过的坑：权重文件按 manifest 排列，而数据集内部只会见到 `train_records` 的子集，加载时必须用 `external[train_idx]` 重新索引，否则删除样本后会全体错位。

### 4.5 覆盖范围对照

| 阈值 | 样本数 | 占训练集 | 结果 |
|---|---|---|---|
| margin ≥0.5（V31） | 2,481 | 1.72% | **+0.11pp** |
| margin ≥0.25（V32，w=0.25） | — | — | 与 0.5 打平 |
| margin ≥0.1（V34，软目标双候选监督） | 19,867 | 13.74% | 见 §7 待办 |

结论边界：净增 42 张、p≈0.11，属小幅正向信号，尚未做种子复验。

---

## 5. 训练侧生产规模结果（全部相对 V16/V31 主控）

| 臂 | 改动 | 结果 |
|---|---|---|
| V17 | 多训 8 轮 | −0.17 |
| V18 | 轮数回 10/6/4 | −0.77（纯轮数效应） |
| V19 | 末段 512px | −0.25（batch 混淆） |
| V21 | 加 640 段 | −0.24 |
| V22 | 局部 token 读出 | −0.045（诊断：注意力熵 0.997） |
| V23-B | 末段 mix_prob 0.3 | −0.21/−0.28 |
| V25 | 阶段一缓存短边 576 | −0.20/−0.37 |
| V26 | γ=1.0（取消 LLRD） | **−1.00**（反过来证明 LLRD 值 1pp） |
| V27 | LP-FT（先线性探测） | 打平（+0.05/−0.03） |
| V28 | γ=0.75 | −0.16/−0.21（0.8 是尖峰） |
| V29 | cosine 头 | −0.64/−0.67 |
| **V31** | **折外降权 0.5** | **+0.11/+0.11** |
| V32 | 折外降权 0.25 | 与 0.5 打平 |
| V33 | 末段训练图长边 cap500 | 持平（+0.03/−0.03） |

### 推理端细扫

| 实验 | 结果 |
|---|---|
| 六视图 576 → 512 | +0.16 |
| EMA 替代 raw | +0.05 |
| 四个 1.14 角视图 | +0.25（对照：重复任一现有视图四遍全部掉分 → 是新增覆盖而非重加权） |
| 翻转角视图 / 1.28 角 / 八角 1.0 锚点 / letterbox | −0.05 ~ −0.28，全部不贡献 |
| 尺寸细扫 480 / 512 / 544 | 77.3181 / 77.4810 / 77.5051（+0.024 属噪声，按 <0.2pp 平手规则不切换） |

---

## 6. 复现步骤

```bash
# 0) 环境：PyTorch 2.5+、transformers、torchvision、Pillow；CLIP 权重用 safetensors 离线缓存
export HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1

# 1) 构建训练清单（可解码性核验、扩展名嗅探）
python scripts/build_manifest.py --train-dir data/train --out artifacts/train_manifest.csv

# 2) 近重复分折（可选，仅 V31 降权需要）
python scripts/build_folds.py            # -> artifacts/folds_2.json；跨折近重复对必须为 0

# 3) 折外教师（仅 V31 降权需要，两折各一个 384px/14 轮模型）
python -m aic_clip.train_ft --config configs/v30/oof_a.yaml
python -m aic_clip.train_ft --config configs/v30/oof_b.yaml
python scripts/dump_teacher_probs.py     # -> artifacts/oof_teacher_*.npz
python scripts/audit_oof_weights.py      # -> artifacts/oof_weights_v2.npy

# 4) 三段阶梯训练（V31 配方，每段是上一段的续训）
python -m aic_clip.train_ft --config configs/v31/s1_384.yaml
python -m aic_clip.train_ft --config configs/v31/s2_448.yaml --initialize checkpoints/v31/s1_384/last.pt
python -m aic_clip.train_ft --config configs/v31/s3_576.yaml --initialize checkpoints/v31/s2_448/last.pt \
    --sample-weight-file artifacts/oof_weights_v2.npy

# 5) 十视图 TTA 推理（EMA 权重，@512）→ pred_results.csv + pred_results.zip
V="center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,center:512:1.28,center:512:1.4"
V="$V,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14"
python -m aic_clip.infer_ft --checkpoint checkpoints/v31/s3_576/last.pt \
    --test-dir data/test --views "$V" --weights ema --output-dir artifacts/submission_v31

# 6) 打分（仅用于训练完全结束后的方向判断，绝不回流训练）
python scripts/evaluate_submissions.py --pred artifacts/submission_v31/pred_results.csv \
    --truth /path/to/submission.csv     # -> 77.4810
```

诊断与视图选择工具（全部只读，测试图只读不写）：

```bash
# 逐视图概率导出 + 离线配方评分（避免重复前向）
python -m aic_clip.infer_ft ... --save-probs
python scripts/combine_view_probs.py --npz artifacts/views512_v16ema/test_view_probs.npz --out-dir recipes/
python scripts/score_view_recipes.py --help
python scripts/analyze_view_recipes.py --help

# 受控增强敏感性诊断（同分辨率、同裁剪框、同种子，逐级加增强）
python scripts/diag_aug_sensitivity.py --checkpoint checkpoints/v30/oof_a/last.pt \
    --eval-fold 1 --images 1200 --output artifacts/diag_aug.json
```

---

## 7. 实现细节与坑

1. **位置编码插值**：`transformers` 在输入尺寸不等于 224 时会直接抛错，必须显式传 `interpolate_pos_encoding=True`（`FTClassifier.embed` 里已按输入尺寸自动判断）。
2. **伪扩展名与坏 EXIF**：训练集里存在扩展名为 `.jpg` 实为 WebP/PNG 的图，以及可解码像素但 EXIF 段损坏的图。解码统一走 `load_image`：内容嗅探 + `exif_transpose` 失败时静默忽略方向元数据。
3. **`evaluate()` 必须 `@torch.no_grad()`**：576px/batch128 下无梯度上下文会建完整计算图，是 V22 OOM 的真正根因（表象是"视觉塔跑两遍"）。
4. **梯度检查点**：本环境的 transformers 版本声明了 `gradient_checkpointing` 却没有在视觉编码器里使用，需要自己替换层循环为 `torch.utils.checkpoint(..., use_reentrant=False)`。
5. **样本权重索引**：见 §4.4，`external[train_idx]` 重索引是必须的。
6. **BF16 而非 FP16**：早期 FP16 路径出现过 NaN 梯度。
7. **评测口径**：所有"测试准确率"都是把预测 CSV 与主办方真值对齐后独立计算的；测试标签只在训练完全完成后用于打分与方向判断，从未进入训练。
8. **单模型单流程**：SWA / EMA 都是同一份权重的加工，不是多模型集成；十视图 TTA 是同一模型的多视图前向平均，属单一推理流程。

---

## 8. 已验证无效（不建议再走）

| 路线 | 结论 |
|---|---|
| ELR 早学习正则 | 288px 续训 12 轮无增益 |
| 无温度缩放的类别先验校正 | 均衡切片上 τ=0 最优 |
| 记忆 / kNN / Sinkhorn / 硬容量 / 图传播 | 线上回退，代理集增益 ≤0.4pp |
| 多 checkpoint 集成 | 违规且仅 +0.29pp |
| CLIP 文本原型融合 | 文本分支单独 18.4%，融合 +0.04pp |
| 局部 token 读出（V22 族） | 注意力近均匀（熵 0.997），放大扫描单调掉分 |
| 视图家族扩展 | 翻转角、1.28 角、八角锚点、letterbox 全部中性或负 |
| 训练侧去边框 | 检测器命中的是平坦边缘（含黑白摄影背景），依据不足 |
| 末段尺寸/正则小改（V19/V21/V23/V25/V33） | 全部 −0.2 ~ 持平 |

---

## 9. 已知局限与下一步

1. **折外降权的收益强度未做种子复验**：+0.11pp 属小幅正向信号，p≈0.11。
2. **大范围软监督重建（V34）**：把双候选软目标覆盖从 2,481 张扩到 19,867 张（r_i = min_v p(alt)/(p(alt)+p(lab))，原标签保留 ≥50%），正在评估。
3. **主体裁剪重编码**：V22 的局部池化是在已有特征上操作，与"裁出主体重新送入同一模型"检验的问题不同，尚未实现。
4. **推理端温度缩放 / 类别先验**：本方案只扫了无温度版本。
5. **训练侧模型选择**：训练内验证集不可信，当前实际采用的仍是固定日程 + 末段权重，缺少可信的在线选模信号。
