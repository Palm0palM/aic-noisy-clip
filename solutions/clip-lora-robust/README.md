# AIC 初赛：噪声标签细粒度图像分类（CLIP ViT-B/32 + LoRA + 噪声鲁棒自训练）

> 最终成绩：**73.92 分**（初赛排行榜），从基线 64.34 分提升 **+9.58**。
> 方案全程遵守赛题约束：**单一骨干网络 CLIP ViT-B/32、单一模型、单推理流程**，无多模型融合、无外部数据、无商业闭源模型。

---

## 1. 赛题与规则要点

| 项目 | 内容 |
|---|---|
| 任务 | 细粒度图像分类，**500 类**，测试集约 25,000 张（类均衡，每类约 50 张） |
| 训练数据 | 约 10.3 万张图，按 `train/<4位类号>/<图片>` 组织；**标签含大量噪声**（GMM 估计噪声率约 25%~35%） |
| 骨干约束 | **必须使用 CLIP ViT-B/32**，禁止更换骨干、禁止多模型/多骨干融合 |
| 提交物 | 单个 CSV（图片名 + 4 位补零类标签），打包为 `pred_results.csv` 的 zip |
| 参考方向 | 赛题官方鼓励噪声标签学习方向（TrustCLIP、鲁棒微调等） |

关键规则推论：测试集**类均衡**这一先验可在推理端显式利用（先验校正）；同一模型的多尺度/翻转视图平均属于"单推理流程 TTA"，合规。

## 2. 方案总览

```
原始图像 (短边中位 480px)
   │  prepare_memmap.py：中心方图 → 352px → uint8 memmap（高速随机读取）
   ▼
CLIP ViT-B/32（冻结主体）
   │  LoRA(rank=8) 注入 q_proj/v_proj（12 层 × 2 = 24 个 Linear）
   │  LayerNorm 放开训练；位置编码双三次插值 7×7 → 11×11（适配 352px）
   │  分类头：类均值原型头（初始化自零样本特征）
   ▼
噪声鲁棒自训练（train.py，20 epoch）
   ├─ warmup 3 epoch：GCE 鲁棒损失 + 特征漂移约束（抑制灾难性遗忘）
   ├─ 每 2 epoch：全量训练图前向 → 逐样本 CE 损失 → GMM 二分量拟合
   │     → 干净样本概率 prob；干净集(prob≥0.6)用硬标签，
   │       分歧高置信集用 EMA 教师伪标签（锐化 T=2，置信≥0.75），
   │       其余噪声样本以 0.5 权重 GCE 弱监督（不丢弃）
   ├─ EMA 教师（decay=0.999）：学生参数滑动平均，生成伪标签
   └─ 选模：验证集 acc（学生 / EMA 取最优）
   ▼
推理（infer.py + 多尺度 TTA）
   ├─ 5 个缩放视图：352 / 384→裁352 / 448→裁352 / 512→裁352 / 576→裁352
   │   （每视图含水平翻转 TTA），logits 按 0.3/0.3/0.1/0.1/0.2 加权
   ├─ 温度缩放 T=5（软化 logits，使先验校正可翻转边缘样本）
   └─ 类先验均衡校正：logits += tau · (-log(prior·C))，tau=1.4
   ▼
pred_results.csv（zip 提交）
```

## 3. 核心技术点

1. **LoRA 参数高效微调**：仅训练 0.59M 参数（LoRA + LayerNorm + 原型头），骨干冻结，抗过拟合噪声。
2. **位置编码插值**（`src/model.py::_interpolate_pos_encoding`）：CLIP 原生 224px（7×7 patch 网格）双三次插值到 352px（11×11），class token 不变——分辨率是本赛最大杠杆（224→352 实测 +2.9 分）。
3. **GMM 噪声划分**（`src/robust.py::gmm_clean_prob`）：对每 epoch 全量训练图的逐样本 CE 损失拟合两个高斯分量（干净/噪声），输出干净后验概率，无需任何干净验证集。
4. **EMA 教师 + 伪标签自训练**：学生参数 EMA（0.999）构成教师；教师预测平方锐化后作为软标签；分歧高置信样本改用伪标签，低置信噪声样本降权弱监督而非丢弃。
5. **GCE 鲁棒损失**（广义交叉熵，q=0.7）：对错误标签的梯度有界，warmup 阶段防止模型早期记忆噪声。
6. **特征漂移约束**（drift_lambda=0.5 余弦退火）：训练特征与冻结参考编码器特征对齐，抑制 LoRA 微调时的灾难性遗忘。
7. **memmap 数据管线**（`prepare_memmap.py` + `src/data.py::MemmapTrainDataset`）：图像预处理一次后存为 uint8 内存映射，DataLoader 零解码开销，训练 IO 不再瓶颈（>1800 img/s 打包）。
8. **多尺度 TTA + 推理端均衡校正**（夺冠关键，+2.67 分）：
   - 5 个缩放视图 logits 加权平均（放大视图提供不同的裁剪上下文，错误互补）；
   - 温度 T=5 软化后，按测试集类均衡先验做加性校正 `adj = -log(prior·C)`，tau=1.4；
   - 权重用验证集模拟器（raw + 零样本干净子集双指标）贪心搜索选出。

## 4. 目录结构

```
AIC/
├── train.py                  # 训练主循环：GMM 划分 / EMA 教师 / 伪标签 / 漂移约束
├── infer.py                  # 测试推理：--tta --temp --balance_tau --cap --save_logits
├── prepare_memmap.py         # 图像 → 中心方图 → 指定尺寸 uint8 memmap
├── dump_train_logits.py      # 导出全量训练集 logits（重打标分析用）
├── requirements.txt
├── configs/
│   ├── prelim_run2_352real.yaml   # ★ 冠军配置（352px）
│   ├── prelim_352real_zoom{,2,3,4}.yaml  # 384/448/512/576 缩放视图推理配置
│   ├── prelim_run2_384.yaml        # 384px 重训实验（分辨率对照）
│   ├── prelim_runG1.yaml           # 自蒸馏重打标重训实验
│   └── prelim_scale035.yaml        # RRC scale(0.35) 尺度鲁棒重训实验
├── src/
│   ├── model.py              # CLIPProtoClassifier：LoRA 注入 / pos-embed 插值 / 原型头 / 参考编码器
│   ├── lora.py               # LoRALinear 与 CLIP vision 注入工具
│   ├── data.py               # 索引构建 / 分层划分 / memmap 数据集 / 变换
│   └── robust.py             # GMM 干净概率 / EMA / 教师构建 / GCE / 软交叉熵
├── infer_val_views.py        # 验证集多尺度视图推理（权重模拟器输入）
├── infer_val_views2.py       # 同上，可指定任意 checkpoint（新模型对照用）
├── infer_val_corners.py      # 验证集角点裁剪视图（空间多样性实验，已证伪）
└── _*.py                     # 本地分析/扫描/补丁脚本（见下表，研究过程记录）
```

本地分析脚本（`_` 前缀，均为离线后处理，不参与训练）：

| 脚本 | 作用 |
|---|---|
| `_sweep.py` / `_sweep_multiview.py` / `_sweep_3view/4view/5view.py` | temp×tau 网格扫描、多视图权重搜索 |
| `_make_multiview.py` / `_make_mv4_extra.py` / `_make_mv5_final.py` | 按扫描结果生成融合提交 CSV/zip |
| `_make_sub.py` | 单模型提交生成（4 位补零标签 + zip 打包） |
| `_test_sim.py` / `_test_sim_robust.py` | 验证集"测试条件模拟器"（raw + 干净子集双指标） |
| `_corner_eval.py` / `_eval_scale035.py` | 角点视图 / scale035 模型的对照评估 |
| `_make_relabel.py` / `_relabel_analysis.py` | 自蒸馏重打标生成与分析（实验方向，已证伪） |
| `_patch_cap.py` / `_patch_rrc.py` / `_patch_save_logits.py` | 对 infer.py / data.py / train.py 的确定性补丁 |
| `_imgsize.py` / `diag_data.py` / `diagnose_gap.py` 等 | 数据尺寸统计与诊断工具 |

## 5. 环境与复现

### 5.1 环境

```bash
pip install -r requirements.txt   # torch / torchvision / transformers / scikit-learn / numpy / pandas / pillow / tqdm / pyyaml
```
- GPU：单卡（开发于 RTX 5060 / AutoDL RTX 4090 级，显存 ≥10GB；352px batch=160 约占 9GB）
- CLIP 权重：`openai/clip-vit-base-patch32` 放至 `cache/clip-vit-base-patch32/`（国内可用 `HF_ENDPOINT=https://hf-mirror.com`）

### 5.2 数据准备

```bash
# 目录约定：train_orig/<0000-0499>/<img>.jpg，test/<img>.jpg
# 1) 构建训练索引（路径 + 标签）
python -c "from src.data import build_train_index; build_train_index('train_orig', 'cache/train_index_256.csv')"

# 2) 打包 memmap（训练 352 + 测试 352）
MM_SIZE=352 python prepare_memmap.py
# 多尺度 TTA 还需测试集 384/448/512/576 memmap：
MM_SIZE=384 TEST_DIR=test python prepare_memmap.py   # 重复 448/512/576
```

### 5.3 训练（冠军配方）

```bash
python train.py --config configs/prelim_run2_352real.yaml
# 产物：outputs/prelim_run2_352real/best.pt（含 student / ema 权重，约 20 epoch / 60 分钟）
```

### 5.4 推理

```bash
# 单视图（352，含翻转 TTA + 温度 + 先验校正）
python infer.py --config configs/prelim_run2_352real.yaml \
  --ckpt outputs/prelim_run2_352real/best.pt --weights ema \
  --tta --temp 6 --balance_tau 1.6 \
  --save_logits outputs/prelim_run2_352real/logits_tta.pt

# 缩放视图（每个尺寸一个 zoom 配置，img_size 仍为 352，memmap 中心裁剪形成放大）
python infer.py --config configs/prelim_352real_zoom.yaml  --ckpt .../best.pt --weights ema --tta --save_logits .../logits_zoom_tta.pt   # 384
python infer.py --config configs/prelim_352real_zoom2.yaml ...  # 448
python infer.py --config configs/prelim_352real_zoom3.yaml ...  # 512
python infer.py --config configs/prelim_352real_zoom4.yaml ...  # 576
```

### 5.5 多视图融合与提交

离线加载 5 个 `logits_*.pt`，按权重 `0.3/0.3/0.1/0.1/0.2`（352/384/448/512/576）加权，再过 `T=5, tau=1.4` 校正，argmax 后写成 4 位补零标签、打包 `pred_results.csv` 的 zip（参考 `_make_mv5_final.py`）。

**冠军提交参数**：5 视图权重 0.3/0.3/0.1/0.1/0.2，T=5，tau=1.4 → 预测分布 max=95 / min=17（与测试集类均衡先验吻合）。

## 6. 实验记录（真实提交分数）

| 提交 | 方案 | 分数 |
|---|---|---|
| 基线 | 初版 pipeline | 64.34 |
| pred_f352_tau30 | 352 单视图 + 强均衡校正 | 70.45 |
| pred_r352_T6_t16 | 冠军模型单视图（温度 6 + tau 1.6） | **71.25** |
| pred_champ_mv5050 | +384 缩放视图（2 视图 50/50） | 72.10 |
| pred_mv3_424 | +448 视图（3 视图 0.4/0.2/0.4） | 73.21 |
| pred_mv4_5537 | +512 视图（4 视图 0.25/0.25/0.15/0.35） | 73.90 |
| **pred_mv5_B_smooth** | **+576 视图（5 视图 0.3/0.3/0.1/0.1/0.2, T5/t1.4）** | **73.92** |

### 已验证失败/无效的方向（避免重复踩坑）

| 方向 | 结果 | 原因 |
|---|---|---|
| 自蒸馏重打标重训（Run G1） | 70.19（-1.06） | 模型高置信预测覆盖标签会放大确认偏误；细粒度混淆中模型会"自信地错" |
| 384px 重训（Run G2） | 70.20（-1.05） | 分辨率红利在 352px 耗尽（训练图短边中位 480，384 插值扰动更大） |
| 角点裁剪视图（空间 TTA） | val 模拟器 +0.002 | 互补性有但总量太小，不具提交价值 |
| RRC scale(0.35) 尺度鲁棒重训（Run H） | val 模拟器 -0.45 | 放大视图弱并非训练尺度覆盖不足，CLIP LoRA 本身跨尺度鲁棒；增强过强拖慢噪声拟合 |
| 容量约束贪心分配（--cap） | 69.46 | 强制每类等量比先验校正更差，会强行翻转正确预测 |

## 7. 在团队仓库中的位置与运行说明

本目录 `solutions/clip-lora-robust/` 是一套**自包含方案**，独立于仓库根目录的线性 baseline 骨架（`src/`），不修改团队的数据划分 manifest 与提交流程：

- 训练 / 推理入口就在本目录下（`train.py`、`infer.py`、`prepare_memmap.py`），运行方式见上文第 5 节，工作目录即本目录；
- 依赖与根 baseline 基本一致（`requirements.txt` 已附，额外用到 `pyyaml`）；
- 数据约定：本目录下 `train_orig/<类号>/*.jpg`、`test/*.jpg`，CLIP 权重放 `cache/clip-vit-base-patch32/`（也可改配置指向根目录的 `models/`）；
- 产物（`cache/*.npy`、`outputs/`、`*.pt`）已由本目录 `.gitignore` 排除，不进入 Git；
- 夺冠提交 CSV（73.9216）已通过团队 `scripts/promote_prediction.py` 流程进入 `submission/candidates/`，队长可用 `scripts/manage_submission.py` 选用。
