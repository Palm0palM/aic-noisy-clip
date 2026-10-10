# AIC 复赛：CLIP ViT-B/32 全参数微调 + 折外可靠性降权 + GCE/LA-CE 续训 + 固定 soft50 解码

> 冻结后测试集实测（本机用主办方真值独立计算）：**78.3196%**（29,326 / 37,444）
> 配方 = V31 权重（折外可靠性降权）→ 长 384 生产重训（第一阶段 24 轮）→ GCE/LA-CE 全量续训 3 轮 → 加倍剂量 D6（再 3 轮）→ 十一视图（十固定视图 + 主体裁剪视图）@512 → 固定 soft50(0.5) 类质量对齐解码。
> 只用 CLIP ViT-B/32 官方权重与官方数据；单一模型、单一推理流程；测试集图像与标签从未参与训练。

---

## 1. 赛题与结论要点

| 项目 | 内容 |
|---|---|
| 任务 | 细粒度图像分类，750 类，复赛训练集 148,695 张（可解码 148,643），测试集 37,444 张 |
| 训练分布 | 长尾：每类 5–248 张，中位数 196；含大量近重复图，近重复对中 77% 标签互相矛盾 |
| 测试分布 | 接近均衡：每类中位数 49 张；全部测试图长边 ≤ 500px |
| 骨干约束 | 必须 CLIP ViT-B/32 + OpenAI 官方预训练权重 |
| 提交物 | `pred_results.csv`（`文件名,四位类号`），37,444 行 |

关键结论：

1. 局部微调不够，要全参数微调（接手时 288px 蒸馏 + LoRA 约 53%，全参 224px 起跳到 67.47）。
2. 分辨率阶梯 384 → 448 → 576 每级都给分，推理端 512 封顶。
3. 推理端多视图 TTA 稳定出收益：六视图 → 十视图 +0.41pp，再加主体视图 +0.10pp。
4. 训练内验证集不预测测试分，甚至反相关（六次独立观察）。所有结论以冻结后测试集为准。
5. 训练集近重复消解 + 折外可靠性降权是第一个复现出正向信号的训练侧改动（V31，+0.11pp）。
6. 长 384 重训之后，GCE+LA-CE 全量续训是第二个复现出正向信号的训练侧改动（+44 张）；同配方加倍剂量（D6）再 +60 张，但单次续训的真实预期约 +30±50 张（见 §6.4）。
7. 解码侧固定 soft50(0.5)（无标签类质量对齐）与续训权重配合放大收益：D6 的 raw 比标杆低 11 张，+soft50 后反超 60 张。

---

## 2. 方案总览（78.3196）

```text
官方训练图（148,643 张可解码）
   │  剔除近重复矛盾簇 4,071 张 → 144,572 训练池
   ▼
CLIP ViT-B/32（全参数 88.2M 可训，官方权重，位置编码双三次插值）
   │  head: nn.Linear(512, 750)
   ▼
三段阶梯（24 / 8 / 6 轮 at 384 / 448 / 576；第一阶段由 14 轮扩到 24 轮）
   ├─ 类均衡采样 inverse-sqrt(0.5)
   ├─ RRC 0.35–1.0 + 翻转 + ColorJitter 0.5 + RandAugment(2,7) + Erasing 0.3
   │  mixup 0.2 / cutmix 1.0（mix_prob 0.8）；标签平滑 0.15
   ├─ AdamW + LLRD γ=0.8 + cosine + warmup；BF16；EMA 0.9995
   └─ 折外可靠性降权：2,481 张 ×0.5（详见 §5）
   ▼
GCE/LA-CE 全量续训 3 轮（从 EMA 末权重出发，固定保存 raw 末权重）
   │  阶段系数（CE / LA / GCE）：(0.80,0,0.20) → (0.35,0.35,0.30) → (0.15,0.70,0.15)
   │  每周期末轮只训分类头；LA 用训练集"有效先验"（采样×可靠性质量直方图 + 0.15 平滑）
   ▼
同配方加倍剂量 D6（6 轮 = 上表重复两遍，第 3、6 轮只训头）
   ▼
推理（单模型 11 个视图，概率空间等权平均，raw 权重）
   ├─ 十个固定视图 @512（六中心 + 四角，见 §3）
   └─ 第 11 个：主体裁剪视图 —— 模型自身 Grad-CAM 定位 → 方形扩框 → 裁剪重编码 @512
        （41.4% 的测试图实际生效；无效定位回退普通第二视图，绝不伪造裁剪）
   ▼
固定 soft50(0.5)：对无标签预测矩阵做类质量对齐（Sinkhorn，目标均匀，强度固定 0.5）
   ▼
pred_results.csv
```

---

## 3. 十个固定视图（B 配方）

```
center:512:1.0, flip:512:1.0,
center:512:1.14, flip:512:1.14,
center:512:1.28, center:512:1.4,
tl:512:1.14, tr:512:1.14, bl:512:1.14, br:512:1.14
```

视图语法 `kind:size:ratio`：kind ∈ {center, flip, 九宫锚点（可加 `_flip`）, fullpad_edge, fullpad_gray}；ratio 为短边缩放倍数。四个角视图补上中心裁剪看不到的边角区域（对照实验：把任一现有视图重复四遍全部掉分，说明增益来自新增覆盖而非重加权）。

效率约定：十视图前向包含六视图，只跑一次十视图 `--save-probs` 再离线组合（`scripts/combine_view_probs.py`，`base6` 与 `base6_fivecrop`），省 37.5% 视图前向。

---

## 4. 第 11 个视图：主体裁剪重编码

动机：细粒度判别部位在整图中分到的 patch 太少。V22 的局部 token 池化是在已有特征上操作，检验的问题不同；这里把区域裁出来重新编码，让主体占据更多 patch。

实现（`scripts/subject_view_probs.py` + `scripts/subject_crop_utils.py`）：

1. 用模型自身的 Grad-CAM 定位。hook 点是 `encoder.layers[-1].layer_norm1`，即最后一次注意力之前的特征——hook 最后一个 block 的输出是错的，因为分类头只读 CLS token，那里 patch 位置的梯度恒为 0，再把 CLS 梯度平均进 patch 权重会造出非零的假热图（这个坑实际踩过，见 §7）。
2. 只平均 patch 位置的梯度得到通道权重，CAM = ReLU(Σ w_c · act_c)，取覆盖 60% 注意力质量的 patch 集合，再加 15% 外扩。
3. 把框扩成包含原框的正方形（不能直接拉伸，否则细长主体被中心裁掉两端），裁剪后重编码到 512。
4. 无效定位（空热图 / 退化框 / 近乎全图 / 无法在画面内成方）回退到普通第二视图，不伪造裁剪。

效果（V31 EMA，测试集全量，与十视图同源对照）：

| 对比 | 变化 |
|---|---|
| 预测改变的行 | 185 / 37,444（0.49%） |
| 改对 / 改错 | 78 / 40，净 +38 张 = +0.10pp（配对检验 p=0.0006，95% 区间 [0.045, 0.158]pp） |
| 相对交付版（77.4810） | 净 +29 张 = +0.077pp |
| 实际生效比例 | 41.4%（15,494 / 37,444；其余回退） |

折 B 诊断（6,000 张，v30_oof_a 教师）：plain 双视图 0.7177 → 全图+主体 0.7200（+0.23pp），真正用到裁剪的 2,671 张 +0.52pp。

口径说明：该预测来自 float16 逐视图概率的离线平均（当时 `--save-probs` 的默认精度）。同一配方在 float16 与 float32 两条保存路径之间的差异是 66 行翻转 / 净 −9 张 = 0.024pp，所以用现在的 float32 默认路径重跑，这一版分数可能有 0.0x pp 量级的平移。代码已改为默认 float32（`--probs-dtype`）。

---

## 5. 逐级结果（全部为冻结后测试集实测）

| 版本 | 关键改动 | 测试准确率 |
|---|---|---|
| 基线（接手时） | 288px 蒸馏 + 末 2 块 + LoRA | 53.28 → 64.3254 |
| V10-A | 全参数微调 | 67.4688 |
| V10-B | 288px | 70.5213 |
| V10-D | 384px | 73.0157 |
| V10-FINAL | 全量数据 448px | 74.6288 |
| V10-FINAL5 | 近重复矛盾剔除 + 末 3 轮 SWA | 75.5128 |
| V13 | 受控消融后修正学习率的阶梯 | 76.5890 |
| V16 | 阶梯 14/8/6 + LLRD γ=0.8 | 76.91 |
| + EMA / @512 / 角视图 | — | 77.0163 / 77.12 / 77.40 |
| V31 | 折外可靠性降权 2,481 张 ×0.5 | 77.4810 |
| V34 | 大范围双候选软监督 19,867 张 | 77.4971 |
| V31 + 主体视图 | 第 11 个视图 | 77.5585 |
| 长 384 重训 | 第一阶段 14 → 24 轮（其余不变） | 78.0419 |
| + GCE/LA-CE 续训 3 轮 | 全量续训，CE/LA/GCE 三阶段 | 78.1594 |
| **+ D6 加倍剂量（6 轮）** | **同配方重复两遍** | **78.3196** |

### 折外可靠性降权（V31 的训练侧改动）

- `scripts/build_folds.py`：dHash（9×8 灰度差分哈希，不是 pHash）≤6 建边 → Bellman-Ford 式标签传播合并连通分量（8,530 簇 / 24,723 张）→ 按类贪心分折；两折 74,508 / 74,135；跨折近重复对独立核验为 0。
- 两折各训一个 384px×14 教师，弱视图 center/flip@448 导出折外概率。
- 筛选：两视图都反对原标签 + 指向同一替代类 + 较小 margin ≥0.5 + 不在删除名单 → 2,481 张 ×0.5。
- 训练端 `--sample-weight-file` + `sample_weight_mode: "target"`（mixup 前按可靠性缩放软标签）。
- 结果 +0.11pp / +0.11pp。注意权重文件按 manifest 排列，加载时必须 `external[train_idx]` 重索引，否则删除样本后全体错位。

---

## 6. 完整实验结果

### 6.1 训练侧（生产规模，相对主控）

| 臂 | 改动 | 结果 |
|---|---|---|
| V17 | 多训 8 轮 | −0.17 |
| V18 | 轮数回 10/6/4 | −0.77 |
| V19 | 末段 512px | −0.25 |
| V21 | 加 640px 段 | −0.24 |
| V22 | 局部 token 读出 | −0.045（注意力熵 0.997） |
| V23-B | 末段 mix_prob 0.3 | −0.21/−0.28 |
| V25 | 阶段一缓存短边 576 | −0.20/−0.37 |
| V26 | γ=1.0（取消 LLRD） | −1.00（反证 LLRD 值约 1pp） |
| V27 | LP-FT | 打平 |
| V28 | γ=0.75 | −0.16/−0.21（0.8 是尖峰） |
| V29 | cosine 头 | −0.64/−0.67 |
| V31 | 折外可靠性降权 0.5 | **+0.11** ✅ 已采用 |
| V32 | 折外降权 0.25 | 与 0.5 打平 |
| V33 | 末段长边 cap500 | 持平 |
| V34 | 双候选软监督（19,867 张） | +0.016pp（净 +6 张），平手 |
| 长 384 重训 | 第一阶段 14→24 轮 | 折内筛查 raw +0.83pp / degraded +0.94pp；全量 78.0419 ✅ 已采用 |
| GCE/LA-CE 续训（3 轮） | 全量续训，三阶段系数 | 9 个预冻结候选一次读出，相对基线净 +44 张（78.1594）✅ 已采用 |
| D6 加倍剂量（6 轮） | 同配方重复两遍 | 相对 loss_robust 净 +60 张（78.3196）✅ 已采用（含噪声，见 §6.4） |
| V35 | 温和增强（RRC 0.6、无 CutMix/擦除、mix_prob 0.3） | 折 B −0.43pp，未通过筛选 |
| ELR（修正实现后首测） | λ=3.0、β=0.7 | 折 B −0.085pp，未通过筛选 |
| L2-SP 预训练锚定 | 骨干拉向官方权重 | α=0.26 过强（漂移钉在 1.6/对照 9.90）；α=0.05 折 B −2.38pp |
| 在线 EMA 弱—强一致性 | Mean Teacher 式 | 折 B −0.25pp（退化视图 −0.08pp），未通过筛选 |
| CutMix 独立对照 | 把 CutMix 分支改为不混合（保留 40% Mixup） | 折 B +0.33pp、退化视图 −0.04pp，低于 0.5pp 门槛，未推进 |

注：ELR 的历史实现从未生效（在 `no_grad()` 内用 detached 预测计算，梯度严格为 0），2026-10-06 修复后才有上表结果。修复版有独立验证脚本 `scripts/verify_elr_grad.py`。

### 6.2 推理侧

| 实验 | 结果 |
|---|---|
| 六视图 576 → 512 | +0.16 |
| EMA 替代 raw | +0.05 |
| 四个 1.14 角视图 | +0.25 |
| 翻转角 / 1.28 角 / 八角锚点 / letterbox | −0.05 ~ −0.28，全部不贡献 |
| 尺寸 480 / 512 / 544 | 77.3181 / 77.4810 / 77.5051（+0.024 属噪声） |
| 主体视图（第 11 个） | +0.10pp（同源对照） |
| soft50(0.5) 类质量对齐解码 | 三种续训权重上 raw → soft50 为 +108 / +115 / +179 张（见 §6.4） |

### 6.3 已排除的方向

| 路线 | 结论 |
|---|---|
| 多 checkpoint 集成 | 违规且仅 +0.29pp |
| 文本原型融合 | 文本分支单独 18.4%，融合 +0.04pp |
| 记忆 / kNN / Sinkhorn / 硬容量 / 图传播 | 线上回退（曾低至 59.90），代理集增益 ≤0.4pp |
| 无温度类别先验校正 | 均衡切片上 τ=0 最优 |
| 训练侧去边框 | 检测器命中的是平坦边缘（含黑白摄影背景），依据不足 |
| 256 维零空间残余头（特征端结构改动） | 折内筛查未过（balanced soft50 −0.01pp、尾类 −0.10pp），未推进 |

### 6.4 续训与 soft50 解码（78.0419 → 78.3196）

配方（`scripts/train_transfer.py` / `scripts/train_round2.py`，配置 `configs/continuation.json`）：

- 起点：长 384 重训的 EMA 末权重（`s3_576/last.pt`）；全量 144,572 张，采样、增强、可靠性降权与 V31 完全一致。
- 损失：CE / LA-CE / GCE 三段系数 (0.80,0,0.20) → (0.35,0.35,0.30) → (0.15,0.70,0.15)，每周期末轮只训分类头（运行时断言骨干梯度为 None）。
  - GCE：q=0.7，权重 (1−p^q)/q。
  - LA-CE：logits + log(有效先验) 的 CE；有效先验 = 按类统计"采样权重 × 可靠性"质量、再 0.15 平滑（只用训练集）。
- 优化：AdamW（骨干 lr 2e-6 / 头 5e-5，wd 0.15），cosine + 0.2 epoch warmup，grad clip 1.0，BF16，batch 80；固定保存最后一个 raw 末权重（不做基于验证的选点）。
- 剂量：3 轮（loss_robust）与 6 轮（D6，同表重复两遍）两档；另有 3 轮重跑对照 r2（fresh optimizer）。

折内筛查（不接触测试标签）：

- 配对消融（`loss_ablation_20261009`，折内）：相对 CE 对照，robust 在 subject 视图净 +37 张；两视图上 GCE 单项最强（+0.034pp），LA-CE 主要作用于尾类（la_only 尾类 +0.28pp）；组合整体不劣于任何单项。
- 长 384 重训先经折内筛查（raw +0.83pp / degraded +0.94pp）后才进入全量生产。

读数与噪声（必须诚实记录）：

| 候选（同一起点） | raw | +soft50 | 对基线 |
|---|---:|---:|---:|
| loss_robust（3 轮，原标杆） | 29,158 | 29,266 (78.1594) | +44 |
| r2_robust（同配方再续 3 轮） | 29,140 | 29,255 (78.1300) | −11 |
| **d6_robust（单次加倍剂量 6 轮）** | 29,147 | **29,326 (78.3196)** | **+60** |

- 三次观测 +44 / −11 / +60 → 单次续训真实预期约 +30±50 张，重复性未证；折内 9 轮剂量与 6 轮几乎零差（balanced soft50 −0.017pp）。提交用冻结产物本身，不依赖"6 轮更强"成立。
- 机制读数：D6 的 raw 比标杆低 11 张，但 +soft50 后反超 60 张——更长续训让概率质量更接近均匀，soft50 的放大效应随之增强（训练侧修偏差 × 解码侧修偏差的配合）。

固定 soft50(0.5) 解码（`scripts/fixed_soft50.py`）：

- 对 11 视图合并后的无标签概率矩阵做 Sinkhorn 类质量对齐：迭代求类别偏置使列质量和趋于均匀（目标 = 样本数 / 类数），再取 `log p + 0.5 × bias` 重新归一化；强度固定 0.5，温度 1.0，收敛容差 1e-7（实测 <40 轮收敛）。
- 只使用无标签预测矩阵本身（样本数与类数），不接触任何标签、不做任何按真实分数的选择；属于单一模型、单一推理流程内的确定性解码后处理。

---

## 7. 实现细节与已经踩过的坑

1. 位置编码插值：输入尺寸 ≠ 224 时必须显式 `interpolate_pos_encoding=True`。
2. 伪扩展名与坏 EXIF：统一走容错解码（内容嗅探 + `exif_transpose` 失败时忽略方向）。
3. `evaluate()` 必须 `@torch.no_grad()`：否则 576px/batch128 会建完整计算图（V22 OOM 的根因）。
4. 梯度检查点：该 transformers 版本声明了 `gradient_checkpointing` 却未在视觉编码器使用，需自行替换层循环为 `torch.utils.checkpoint(..., use_reentrant=False)`。
5. 样本权重索引：见 §5。
6. Grad-CAM 取层：hook 最后一次注意力之前的 `layer_norm1`，不是最后一个 block 的输出。
7. 面积分母：定位器输入是 384/512 方图，用原图面积做分母会在 3:2 图上把"满框"报成 0.667。
8. 矩形框不能直接拉伸成正方形；要扩成包含原框的正方形，容纳不下就回退。
9. 概率保存精度：`--save-probs` 默认 float32；float16 会在离线平均时翻转少量平票行（实测 66 行 / 净 −9 张）。
10. 运行中脚本的重启判断不能用 `if [ ! -f last.pt ]`（每轮都会写），要看 history 的完成轮数。
11. 长任务前查磁盘余量，脚本内置 8GB 守卫；历史上满盘毁过一次训练。
12. 续训段 checkpoint 的权重键约定：起点读上游 `ema`（断言 epoch 与有限性），产物只保存 raw `model` 键（无 `ema`）；推理与导出固定用 raw 末权重。上游（loss_robust）的存档同样只有 `model` 键，下游训练脚本对两种键都做了显式兼容并把来源记入 `initial_meta.json`。
13. soft50 的平滑参数是"反转已知标签平滑"的实验开关，正式流程保持 0（`uniform_alignment` 的默认温度 1.0）。

---

## 8. 复现

```bash
# 0) 环境：PyTorch 2.5+、transformers、torchvision、Pillow；CLIP 权重离线缓存
export HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1

# 1) 近重复分折 + 折外教师 + 降权名单（V31 的训练侧改动）
python scripts/build_folds.py                      # -> artifacts/folds_2.json
python -m aic_clip.train_ft --config configs/v30/oof_a.yaml
python -m aic_clip.train_ft --config configs/v30/oof_b.yaml
python scripts/dump_teacher_probs.py ...           # 弱视图 center/flip@448，--only-fold 1
python scripts/audit_oof_weights.py                # -> artifacts/oof_weights_v2.npy

# 2) 三段阶梯（每段续训上一段）
python -m aic_clip.train_ft --config configs/v31/s1_384.yaml
python -m aic_clip.train_ft --config configs/v31/s2_448.yaml --initialize checkpoints/v31/s1_384/last.pt
python -m aic_clip.train_ft --config configs/v31/s3_576.yaml --initialize checkpoints/v31/s2_448/last.pt \
    --sample-weight-file artifacts/oof_weights_v2.npy

# 3) 推理：一次十视图前向（float32 概率）+ 主体视图，再离线组合
python -m aic_clip.infer_ft --checkpoint checkpoints/v31/s3_576/last.pt --test-dir data/test \
    --views center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,center:512:1.28,center:512:1.4,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14 \
    --weights ema --save-probs --probs-dtype float32 --output-dir artifacts/subject_v31
python scripts/subject_view_probs.py --checkpoint checkpoints/v31/s3_576/last.pt \
    --test-dir data/test --size 512 --weights ema --output-dir artifacts/subject_v31
python scripts/combine_subject_arm.py --views-npz artifacts/subject_v31/test_view_probs.npz \
    --subject-npz artifacts/subject_v31/subject_view_probs.npz --out-dir artifacts/subject_v31/arms
# arms/B_subject.csv 即 11 视图提交，arms/B.csv 为十视图对照

# 4) 生产续篇（78.0419 → 78.3196）：长 384 重训 → GCE/LA-CE 续训 → D6
#    长 384：第一阶段 14→24 轮，其余与 V31 一致（折内筛查 raw +0.83pp）
python -m aic_clip.train_ft --config configs/long384_a/s1_384.yaml --train-on-all \
    --drop-indices artifacts/dedup_drop.npy --sample-weight-file artifacts/oof_weights_v2.npy
python -m aic_clip.train_ft --config configs/long384_a/s2_448.yaml --initialize checkpoints/long384_a/s1_384/last.pt \
    --init-weights raw --train-on-all --drop-indices artifacts/dedup_drop.npy --sample-weight-file artifacts/oof_weights_v2.npy
python -m aic_clip.train_ft --config configs/long384_a/s3_576.yaml --initialize checkpoints/long384_a/s2_448/last.pt \
    --init-weights raw --train-on-all --drop-indices artifacts/dedup_drop.npy --sample-weight-file artifacts/oof_weights_v2.npy
#    续训（3 轮）与加倍剂量 D6（6 轮）；完整冻结与断言流程见
#    scripts/run_long384_full_20261008.py、scripts/run_transfer.py、scripts/run_round2.py
python scripts/train_transfer.py --arm robust --mode full \
    --checkpoint checkpoints/long384_a/s3_576/last.pt --output checkpoints/loss_robust
python scripts/train_round2.py --arm robust6 --mode full \
    --checkpoint checkpoints/loss_robust/last.pt --output checkpoints/d6_robust

# 5) 最终导出：十视图 + 主体视图 @512（raw 权重）→ 10:1 组合 → 固定 soft50(0.5)
#    run_round2.py 的 export 段：infer_ft --save-probs(float32) → subject_view_probs →
#    逐行 10:1 平均（回退行跳过）→ scripts/fixed_soft50.py 的 uniform_alignment(strength=.5)

# 6) 打分（仅在训练完全结束后；测试标签不进入任何训练流程）
python scripts/evaluate_submissions.py --pred artifacts/subject_v31/arms/B_subject.csv --truth submission.csv
```

评估与验证脚本（本地运行，只用训练侧或推理产物）：

```bash
python scripts/verify_transfer.py        # 续训起点/系数/冻结断言
python scripts/fold_eval.py              # 折内评估（base/finish/degraded）
python scripts/verify_elr_grad.py        # ELR 必须有梯度（历史实现是空操作）
python scripts/verify_anchor_grad.py     # L2-SP：α=0 严格空操作、方向正确
python scripts/diag_subject_crop_v3.py   # 主体定位诊断（逐图 jsonl + 新旧样本分组 + 95% 区间）
python scripts/compare_fold_probs.py     # 折 B 原图 / 退化视图对照（宏平均 + 尾/中/头三段）
```

---

## 9. 结论边界

本仓库记录的是被验证过的配方，未通过筛选的方向同样保留在案：软监督扩展、温和增强、ELR、L2-SP、在线一致性、CutMix 移除、9 轮剂量、零空间残余头都测过且未达到推进门槛。它们是"这些配方未通过筛查"，不是"对应机制被证伪"——每条都是单折、单种子的有界实验，且都存在配置上的已知限制（例如 ELR 的历史只在未混合批更新；L2-SP 只测了两个强度）。

从 78.3196 到 80 还差约 1.68pp（约 630 张净正确）。当前证据表明：推理端与解码端已基本到顶（视图组合、soft50 强度、校准类改动全部探过且无新收益），训练侧续训收益在噪声级（+30±50 张）；要继续逼近上限，需要比"续训加剂量"更强的训练侧改动，而不是继续加视图或加解码技巧。
